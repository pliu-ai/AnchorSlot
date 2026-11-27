#!/usr/bin/env python3
"""Measurement reliability: uncertainty of object-count errors and false detections in object-free crops.

A. Object-count error (official stored counts, same pairs as robustness_stats.py): for every crop-class pair scored for
   all methods with >= 1 ground-truth object, |predicted - true| / true objects. Paired crop bootstrap (10,000
   resamples) of the median and mean error, AnchorSlot minus each baseline, plus the signed error (over- vs.
   under-counting).
B. False detections where a crop is annotated for an instance class but contains no such object. The pinned
   evaluator scores only classes present in a crop, so these pairs never enter the official score.
   - Annotation comes from CellMap's train_crop_manifest.csv; all 17 validation crops annotate all 48 classes.
   - A pair counts as object-free when neither the class nor any of its atomic parts is in the locked truth store.
   - Predicted objects are 26-connected components of the class mask of each method's hard label map, as in the
     evaluator, also after the uniform 1st- and 10th-percentile size filters of uniform_filter.py.
   - Paired crop bootstrap of AnchorSlot minus each baseline for the share of object-free pairs with any false
     object and for the number of false objects.
Output: reliability_stats.json next to this script.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cc3d
import numpy as np
import zarr

from robustness_stats import METHODS, RNG_SEED, N_BOOT, is_instance, load

HERE = Path(__file__).resolve().parent
RUNS = Path("/projects/weilab/liupeng/runs/Anchorslot")
REPO = Path("/projects/weilab/liupeng/code/projects/AnchorSlot")
LOCK = RUNS / "evaluation_protocol/high_4nm/truth_lock.json"
TRAIN_MANIFEST = Path("/projects/weilab/liupeng/code/projects/cellmap-segmentation-challenge-upstream-0300239/"
                      "src/cellmap_segmentation_challenge/utils/train_crop_manifest.csv")
sys.path.insert(0, str(REPO))
from nnunetv2.evaluation.cellmap_challenge import (  # noqa: E402
    PARENT_FROM_ATOMIC, _load_hard_segmentation, entries_by_crop, hard_mask_for_class, index_predictions,
    read_manifest)

INSTANCE = ["mito", "nuc", "cell", "perox", "ves", "np", "endo", "ld", "mt", "lyso"]
FILTERS = {"raw": None, "p1": "size_thresholds.json", "p10": "size_thresholds_p10.json"}
OUT = HERE / "reliability_stats.json"


def ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# --- A. count error ------------------------------------------------------------------------------------------------
def count_errors(data, crop_list):
    """{method: {crop: [(abs_rel, signed_rel, cls)]}} on pairs scored for every method with >= 1 true object."""
    out = {m: defaultdict(list) for m in METHODS}
    for crop in crop_list:
        for cls, r in data["anchorslot"][crop].items():
            if not is_instance(r):
                continue
            rows = {m: data[m][crop].get(cls) for m in METHODS}
            if any(x is None or x["status"] != "scored" for x in rows.values()):
                continue
            n_gt = r["tp"] + r["fn"]
            if n_gt == 0:
                continue
            for m, x in rows.items():
                signed = ((x["tp"] + x["fp"]) - n_gt) / n_gt
                out[m][crop].append((abs(signed), signed, cls))
    return out


def count_error_stats(errors, crop_list, rng):
    def summary(values):
        a = np.array([v[0] for v in values]); s = np.array([v[1] for v in values])
        return dict(pairs=len(values), median_abs=float(np.median(a)), mean_abs=float(np.mean(a)),
                    median_signed=float(np.median(s)), overcount_pairs=int((s > 0).sum()),
                    undercount_pairs=int((s < 0).sum()), exact_pairs=int((s == 0).sum()))

    point = {m: summary([v for c in crop_list for v in errors[m][c]]) for m in METHODS}
    per_class = {m: {} for m in METHODS}
    for m in METHODS:
        by_cls = defaultdict(list)
        for c in crop_list:
            for v in errors[m][c]:
                by_cls[v[2]].append(v[0])
        per_class[m] = {k: dict(pairs=len(v), median_abs=float(np.median(v))) for k, v in sorted(by_cls.items())}
    boot = {m: {"median": [], "mean": []} for m in METHODS if m != "anchorslot"}
    for _ in range(N_BOOT):
        sample = rng.choice(crop_list, size=len(crop_list), replace=True)
        vals = {m: np.array([v[0] for c in sample for v in errors[m][c]]) for m in METHODS}
        for m in boot:
            boot[m]["median"].append(np.median(vals["anchorslot"]) - np.median(vals[m]))
            boot[m]["mean"].append(np.mean(vals["anchorslot"]) - np.mean(vals[m]))
    delta = {m: {stat: dict(point=point["anchorslot"][f"{stat}_abs"] - point[m][f"{stat}_abs"], ci95=ci(v),
                            fraction_anchorslot_lower=float(np.mean(np.array(v) < 0)))
                 for stat, v in b.items()} for m, b in boot.items()}
    return dict(point=point, per_class=per_class, delta_anchorslot_minus=delta)


# --- B. false detections in object-free annotated pairs ---------------------------------------------------------------
def object_free_pairs(truth, crop_list):
    annotated = defaultdict(set)
    with TRAIN_MANIFEST.open() as f:
        for row in csv.DictReader(f):
            annotated["crop" + row["crop_name"]].add(row["class_label"])
    pairs, inconsistent = [], []
    for crop in crop_list:
        present = set(truth[crop].array_keys())
        for cls in INSTANCE:
            if cls not in annotated[crop] or cls in present:
                continue
            parts = set(PARENT_FROM_ATOMIC.get(cls, (cls,)))
            (inconsistent if parts & present else pairs).append((crop, cls))
    return pairs, inconsistent


def min_voxels(thresholds, cls, voxel_size):
    if thresholds is None:
        return 1
    return max(1, int(math.ceil(thresholds[cls] / float(np.prod(voxel_size)))))


def false_detections(pairs, lock, crop_list):
    thresholds = {name: (None if f is None else {c: v["threshold_nm3"] for c, v in
                                                 json.loads((HERE / f).read_text())["classes"].items()})
                  for name, f in FILTERS.items()}
    entries = entries_by_crop(read_manifest(lock["manifest"]))
    files = {m: index_predictions(json.loads((p / "evaluation_provenance.json").read_text())["predictions"])
             for m, p in METHODS.items()}
    by_crop = defaultdict(list)
    for crop, cls in pairs:
        by_crop[crop].append(cls)
    records = []
    for crop in crop_list:
        if crop not in by_crop:
            continue
        shape = entries[crop][0].shape
        voxel_sizes = {tuple(e.voxel_size) for e in entries[crop]}
        assert len(voxel_sizes) == 1, (crop, voxel_sizes)
        voxel = voxel_sizes.pop()
        for m in METHODS:
            seg = _load_hard_segmentation(files[m][crop], shape)
            for cls in by_crop[crop]:
                mask = hard_mask_for_class(seg, cls).astype(np.uint8)
                labels, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
                sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0, dtype=np.int64)
                rec = dict(crop=crop, cls=cls, method=m, voxels=int(mask.sum()))
                for name, thr in thresholds.items():
                    rec[f"objects_{name}"] = int((sizes >= min_voxels(thr, cls, voxel)).sum())
                records.append(rec)
            del seg
        print("crop done:", crop, flush=True)
    return records


def false_detection_stats(records, crop_list, rng):
    table = {(r["crop"], r["cls"], r["method"]): r for r in records}
    pairs = sorted({(r["crop"], r["cls"]) for r in records})
    stats = {}
    for name in FILTERS:
        key = f"objects_{name}"
        per_method = {}
        for m in METHODS:
            counts = [table[(c, k, m)][key] for c, k in pairs]
            per_cls = defaultdict(int)
            for (c, k), n in zip(pairs, counts):
                per_cls[k] += n
            per_method[m] = dict(pairs=len(pairs), pairs_with_false_objects=int(sum(n > 0 for n in counts)),
                                 share_with_false_objects=float(np.mean([n > 0 for n in counts])),
                                 false_objects=int(sum(counts)), false_objects_by_class=dict(sorted(per_cls.items())))
        crop_pairs = defaultdict(list)
        for c, k in pairs:
            crop_pairs[c].append(k)
        crops = sorted(crop_pairs)
        boot = {m: {"share": [], "objects": []} for m in METHODS if m != "anchorslot"}
        for _ in range(N_BOOT):
            sample = rng.choice(crops, size=len(crops), replace=True)
            vals = {m: np.array([table[(c, k, m)][key] for c in sample for k in crop_pairs[c]]) for m in METHODS}
            for m in boot:
                boot[m]["share"].append(np.mean(vals["anchorslot"] > 0) - np.mean(vals[m] > 0))
                boot[m]["objects"].append(vals["anchorslot"].sum() - vals[m].sum())
        delta = {m: {"share": dict(point=per_method["anchorslot"]["share_with_false_objects"]
                                   - per_method[m]["share_with_false_objects"], ci95=ci(b["share"]),
                                   fraction_anchorslot_lower=float(np.mean(np.array(b["share"]) < 0))),
                     "objects": dict(point=per_method["anchorslot"]["false_objects"] - per_method[m]["false_objects"],
                                     ci95=ci(b["objects"]),
                                     fraction_anchorslot_lower=float(np.mean(np.array(b["objects"]) < 0)))}
                 for m, b in boot.items()}
        stats[name] = dict(per_method=per_method, delta_anchorslot_minus=delta)
    return stats


def main() -> None:
    lock = json.loads(LOCK.read_text())
    truth = zarr.open_group(lock["truth_store"], mode="r")
    data = {m: load(m)[0] for m in METHODS}
    crop_list = sorted(data["anchorslot"])
    rng = np.random.default_rng(RNG_SEED)
    out = {"methods": {k: str(v) for k, v in METHODS.items()}, "seed": RNG_SEED, "resamples": N_BOOT,
           "count_error": count_error_stats(count_errors(data, crop_list), crop_list, rng)}
    pairs, inconsistent = object_free_pairs(truth, crop_list)
    records = false_detections(pairs, lock, crop_list)
    out["object_free"] = dict(pairs=len(pairs), crops=len({c for c, _ in pairs}),
                              excluded_inconsistent=[list(p) for p in inconsistent],
                              filters={k: (v or "none") for k, v in FILTERS.items()},
                              stats=false_detection_stats(records, crop_list, rng), records=records)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    summary = {"count_error_point": {m: {k: v[k] for k in ("median_abs", "mean_abs", "median_signed")}
                                     for m, v in out["count_error"]["point"].items()},
               "count_error_delta_vs_cb": out["count_error"]["delta_anchorslot_minus"]["cb_flat"],
               "object_free_pairs": len(pairs), "inconsistent": inconsistent,
               "false_detections": {f: {m: {k: v[k] for k in ("pairs_with_false_objects", "false_objects")}
                                        for m, v in s["per_method"].items()}
                                    for f, s in out["object_free"]["stats"].items()},
               "false_detection_delta_vs_cb": {f: s["delta_anchorslot_minus"]["cb_flat"]
                                               for f, s in out["object_free"]["stats"].items()}}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
