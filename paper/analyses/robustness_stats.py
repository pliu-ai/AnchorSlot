#!/usr/bin/env python3
"""Robustness statistics from the official evaluator's stored per-crop counts (no re-scoring).

1. Reproduces the official semantic / instance / overall scores for every method (hard check).
2. Few-object sensitivity: instance and overall scores without instance classes that have
   <= 10 ground-truth objects in the 17 validation crops.
3. Paired crop bootstrap (10,000 resamples) of the pooled overall-score difference
   AnchorSlot minus each baseline, recomputed with the official pooling on every resample.
4. Downstream quantification: per crop-class relative object-count error (instance classes)
   and relative volume error (31 atomic labels), on pairs scored for every method.

Pooling follows the evaluator: semantic IoU from voxel counts pooled over crops; instance F1 from
pooled matched-object counts and H from pooled normalized Hausdorff sums; 'skipped_too_many_instances'
pairs contribute zero counts. Output: robustness_stats.json next to this script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RUNS = Path("/projects/weilab/liupeng/runs/Anchorslot")
METHODS = {
    "flat": RUNS / "evaluation_matrix/table_vi_current/flat_nnunet/full_annotated/4nm",
    "cb_flat": RUNS / "evaluation_matrix/flat_nnunet_class_balanced/full_annotated/4nm",
    "binary": RUNS / "cellmap_paper_refresh_20260921/evaluation/binary",
    "ternary": RUNS / "cellmap_paper_refresh_20260921/evaluation/ternary",
    "anchorslot": RUNS / "evaluation_matrix/anchorslot_primary/full_annotated/4nm",
}
ATOMIC = ["ecs", "pm", "cyto", "nucpl", "mito_mem", "mito_lum", "mito_ribo", "golgi_mem", "golgi_lum",
          "ves_mem", "ves_lum", "endo_mem", "endo_lum", "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
          "er_mem", "er_lum", "eres_mem", "eres_lum", "ne_mem", "ne_lum", "np_out", "np_in",
          "hchrom", "echrom", "mt_out", "mt_in", "perox_mem", "perox_lum"]
OUT = Path(__file__).with_name("robustness_stats.json")
RNG_SEED, N_BOOT = 20260929, 10000


def load(method):
    d = json.loads((METHODS[method] / "latest_official_metrics_extended.json").read_text())
    crops = {c: v for c, v in d.items() if c.startswith("crop")}
    official = {k: d[k] for k in ("overall_semantic_score", "overall_instance_score", "overall_score")}
    return crops, official


def is_instance(row):
    return "hausdorff_norm_sum" in row


def scores(crops, crop_list, drop_instance=()):
    sem, inst = {}, {}
    for crop in crop_list:
        for cls, r in crops[crop].items():
            if is_instance(r):
                if cls in drop_instance:
                    continue
                a = inst.setdefault(cls, [0, 0, 0, 0.0, 0])
                a[0] += r["tp"]; a[1] += r["fp"]; a[2] += r["fn"]; a[3] += r["hausdorff_norm_sum"]; a[4] += r["n_hausdorff"]
            else:
                a = sem.setdefault(cls, [0, 0, 0])
                a[0] += r["tp"]; a[1] += r["fp"]; a[2] += r["fn"]
    s_sem = np.mean([tp / (tp + fp + fn) if tp + fp + fn else 1.0 for tp, fp, fn in sem.values()])
    per_inst = []
    for tp, fp, fn, hs, nh in inst.values():
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        per_inst.append(math.sqrt(f1 * (hs / nh if nh else 1.0)))
    s_inst = float(np.mean(per_inst))
    return float(s_sem), s_inst, math.sqrt(s_sem * s_inst)


def main():
    data = {m: load(m) for m in METHODS}
    crop_list = sorted(data["anchorslot"][0])
    out = {"methods": {k: str(v) for k, v in METHODS.items()}, "reproduction": {}}
    for m, (crops, official) in data.items():
        sem, inst, ov = scores(crops, crop_list)
        ok = all(abs(a - b) < 1e-9 for a, b in zip((sem, inst, ov), official.values()))
        out["reproduction"][m] = dict(semantic=sem, instance=inst, overall=ov, official=official, exact=ok)
        if not ok:
            raise SystemExit(f"{m}: recomputed {(sem, inst, ov)} != official {official}")

    # Few-object classes from ground-truth object counts (tp + fn), taken from AnchorSlot's scored pairs
    # plus any object counted by another method where AnchorSlot's pair was skipped.
    gt_objects = {}
    for crop in crop_list:
        for cls, r in data["anchorslot"][0][crop].items():
            if is_instance(r):
                counts = [d[0][crop][cls]["tp"] + d[0][crop][cls]["fn"] for d in data.values()]
                gt_objects[cls] = gt_objects.get(cls, 0) + max(counts)
    few = sorted(c for c, n in gt_objects.items() if n <= 10)
    out["gt_objects"] = gt_objects
    out["few_object_classes"] = few
    out["without_few_object_classes"] = {
        m: dict(zip(("semantic", "instance", "overall"), scores(crops, crop_list, drop_instance=few)))
        for m, (crops, _) in data.items()}

    rng = np.random.default_rng(RNG_SEED)
    boot = {m: [] for m in METHODS if m != "anchorslot"}
    for _ in range(N_BOOT):
        sample = list(rng.choice(crop_list, size=len(crop_list), replace=True))
        ref = scores(data["anchorslot"][0], sample)[2]
        for m in boot:
            boot[m].append(ref - scores(data[m][0], sample)[2])
    out["bootstrap"] = {"seed": RNG_SEED, "resamples": N_BOOT, "unit": "crop (paired)",
                        "delta_overall_anchorslot_minus": {
                            m: dict(point=out["reproduction"]["anchorslot"]["overall"] - out["reproduction"][m]["overall"],
                                    ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                    fraction_positive=float(np.mean(np.array(v) > 0)))
                            for m, v in boot.items()}}

    # Downstream quantification on pairs scored for every method.
    count_err, vol_err = {m: [] for m in METHODS}, {m: [] for m in METHODS}
    for crop in crop_list:
        for cls, r in data["anchorslot"][0][crop].items():
            rows = {m: d[0][crop].get(cls) for m, d in data.items()}
            if any(x is None for x in rows.values()):
                continue
            if is_instance(r):
                if any(x["status"] != "scored" for x in rows.values()):
                    continue
                n_gt = r["tp"] + r["fn"]
                if n_gt == 0:
                    continue
                for m, x in rows.items():
                    count_err[m].append(abs((x["tp"] + x["fp"]) - n_gt) / n_gt)
            elif cls in ATOMIC:
                v_gt = r["tp"] + r["fn"]
                if v_gt == 0:
                    continue
                for m, x in rows.items():
                    vol_err[m].append(abs((x["tp"] + x["fp"]) - v_gt) / v_gt)
    out["quantification"] = {
        m: dict(count_pairs=len(count_err[m]), median_abs_rel_count_error=float(np.median(count_err[m])),
                mean_abs_rel_count_error=float(np.mean(count_err[m])),
                volume_pairs=len(vol_err[m]), median_abs_rel_volume_error=float(np.median(vol_err[m])))
        for m in METHODS}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("few_object_classes", "without_few_object_classes", "bootstrap", "quantification")}, indent=1))


if __name__ == "__main__":
    main()
