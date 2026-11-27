#!/usr/bin/env python3
"""Offline decodes for the reviewer ablations (2026-10-27): reconstruction weight w, hysteresis alpha, and the
new arms. One dump case per call; writes <out-root>/<variant>/<case>.nii.gz.

Variant grammar: [H]G<w>[a<A>]
  G<w>        geometric mix s^(1-w) h^w of the structured/flat score s and the group-head score h (paper: w = 0.5);
              G0 uses score s alone (fusion disabled; training may still use a group head).
  HG<w>a<A>   hysteresis (paper: A = 3): core = G<w>; dynamic scores x A form the relaxed map; relaxed voxels of a
              dynamic label are kept only where 6-connected to a core component of that label.

Dump kinds:
  as    AnchorSlot group-head dump (predict_dump_gh.py): s = legacy merge (mean background/anchors, each dynamic label
        from its own query run), h = group head; decode_lib functions, unchanged. G0.5 / HG0.5a3 reproduce the
        paper's Gmix0.5 / HGmix0.5a3 (checked voxel-wise with --reference-root).
  nq    as, for the non-query-as-background arm (predict_dump_nq.py): the background score is the minimum of the
        per-run background probabilities (a voxel is background only if no query run claims it), because averaging
        runs that were all trained to call other groups' organelles background would inflate it ~12/13 there.
        --nq-background mean keeps the paper's mean instead (reported as a secondary decode).
  flat  flat + group-head dump (predict_dump_flatgh.py, logits): s = softmax of the 32 flat logits, h(bg / anchor a)
        = group-head probability, h(c in group g) = Q_g * s_c / sum_{c' in g} s_c'; the export receives log scores,
        so G0 equals the stock nnU-Net flat prediction (argmax is invariant to the per-voxel log-sum-exp).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

DECODE_LIB = Path(__file__).parent
if str(DECODE_LIB) not in sys.path:
    sys.path.insert(0, str(DECODE_LIB))
import decode_lib as dl  # noqa: E402

NAME = re.compile(r"(H?)G([0-9.]+)(?:a([0-9.]+))?")
EPS = 1e-8


def parse(v: str):
    m = NAME.fullmatch(v)
    if not m or (m.group(1) == "H") != (m.group(3) is not None):
        raise ValueError(f"bad variant {v!r}: use G<w> or HG<w>a<A>")
    w, alpha = float(m.group(2)), (float(m.group(3)) if m.group(1) else None)
    if not 0 <= w <= 1 or (alpha is not None and alpha < 1):
        raise ValueError(f"invalid weight/alpha: {v}")
    return w, alpha


# --- score maps -----------------------------------------------------------------------------------------------

def as_scores(p_all: np.ndarray, meta: list, w: float, nq_background: str | None) -> tuple[np.ndarray, list]:
    """32 x ... AnchorSlot scores (probability scale) and the dynamic labels."""
    compact = [c for c in meta if c["kind"] not in ("group_head", "background_run")]
    head_idx = [c["index"] for c in meta if c["kind"] == "group_head"]
    p = p_all[: len(compact)].astype(np.float32)
    s = dl.legacy_scores(p, compact)
    if nq_background == "min":
        run_idx = [c["index"] for c in sorted((c for c in meta if c["kind"] == "background_run"), key=lambda c: c["group"])]
        if len(run_idx) != 13:
            raise RuntimeError("nq dump needs 13 background_run channels")
        s[0] = p_all[run_idx].astype(np.float32).min(axis=0)
    if w > 0:
        s = dl.gmix_scores(s, p, compact, p_all[head_idx].astype(np.float32), weight=w)
    return s, dl.dynamic_labels(compact)


def flat_groups() -> tuple[list, dict]:
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
        DYNAMIC_GROUP_SPECS, FIXED_OUTPUT_TO_ORIGINAL)
    anchors = [int(label) for _, label in sorted(FIXED_OUTPUT_TO_ORIGINAL.items())]
    groups = {int(s.group_id): [int(x) for x in s.original_labels] for s in DYNAMIC_GROUP_SPECS}
    return anchors, groups


def flat_log_scores(z: np.ndarray, w: float) -> tuple[np.ndarray, list]:
    """32 x ... log scores for the flat + group-head dump (z = 50 logits) and the dynamic labels."""
    anchors, groups = flat_groups()
    zf = torch.from_numpy(z[:32].astype(np.float32))
    log_s = torch.log_softmax(zf, dim=0)
    dyn = sorted(c for labels in groups.values() for c in labels)
    if w > 0:
        log_q = torch.log_softmax(torch.from_numpy(z[32:].astype(np.float32)), dim=0)
        log_h = torch.empty_like(log_s)
        log_h[0] = log_q[0]
        for i, a in enumerate(anchors):
            log_h[a] = log_q[1 + i]
        for g, labels in groups.items():
            share = log_s[labels] - torch.logsumexp(log_s[labels], dim=0, keepdim=True)
            log_h[labels] = log_q[5 + g] + share
        log_s = (1 - w) * log_s + w * log_h
    return log_s.numpy(), dyn


# --- export ---------------------------------------------------------------------------------------------------

class Exporter:
    def __init__(self, model_folder: Path, config: str, props: dict, kind: str):
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        self.convert = convert_predicted_logits_to_segmentation_with_correct_shape
        self.plans = PlansManager(json.loads((model_folder / "plans.json").read_text()))
        self.cm = self.plans.get_configuration(config)
        if kind == "flat":
            self.lm = self.plans.get_label_manager(json.loads((model_folder / "dataset.json").read_text()))
        else:   # the 32-label identity manager of the AnchorSlot predictor (as decode_variants.py)
            sys.path.insert(0, str(Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / "scripts"))
            import predict_structured_conditional as psc
            self.lm = psc._build_original_32_identity_label_manager()
        self.props = props

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        seg = self.convert(torch.from_numpy(np.ascontiguousarray(scores)), self.plans, self.cm, self.lm, self.props,
                           return_probabilities=False, num_threads_torch=4)
        return np.asarray(seg, dtype=np.uint8)


# --- driver ---------------------------------------------------------------------------------------------------

def run_case(a, case: str) -> dict:
    meta = json.loads((a.dump / "channels.json").read_text())
    arr = np.load(a.dump / f"{case}.npy", mmap_mode="r")
    props = pickle.load(open(a.dump / f"{case}.pkl", "rb"))
    export = Exporter(a.model_folder, a.config, props, a.kind)
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    rw = SimpleITKIO()
    report_path = a.out_root / "_reports" / f"{case}.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"case": case, "kind": a.kind, "variants": {}}
    scores_cache, core_cache = {}, {}

    def scores(w):
        if w not in scores_cache:
            scores_cache.clear()
            core_cache.clear()
            if a.kind == "flat":
                scores_cache[w] = flat_log_scores(arr, w)
            else:
                scores_cache[w] = as_scores(arr, meta, w, a.nq_background if a.kind == "nq" else None)
        return scores_cache[w]

    def core(w):
        if w not in core_cache:
            core_cache[w] = export(scores(w)[0])
        return core_cache[w]

    for v in a.variants:
        out = a.out_root / v / f"{case}.nii.gz"
        if out.exists() and v in report["variants"]:
            continue
        t0 = time.time()
        w, alpha = parse(v)
        if alpha is None:
            seg = core(w)
        else:
            s, dyn = scores(w)
            relaxed_scores = s.copy()
            if a.kind == "flat":   # log scores
                relaxed_scores[dyn] += np.float32(np.log(alpha))
            else:
                relaxed_scores = dl.scale_dynamic(s, dyn, alpha)
            seg = dl.hysteresis(core(w), export(relaxed_scores), dyn)
            del relaxed_scores
        out.parent.mkdir(parents=True, exist_ok=True)
        rw.write_seg(seg, str(out), props)
        info = {"seconds": round(time.time() - t0, 1), "foreground_voxels": int((seg > 0).sum())}
        if a.reference_root is not None and (a.reference_root / a.reference_map.get(v, v) / f"{case}.nii.gz").exists():
            ref, _ = rw.read_seg(str(a.reference_root / a.reference_map.get(v, v) / f"{case}.nii.gz"))
            info["differs_from_reference"] = int((np.asarray(ref[0], dtype=np.uint8) != seg).sum())
        if info.get("differs_from_reference", 0) != 0:
            raise RuntimeError(f"reference reproduction failed: {case} {v}: {info}")
        report["variants"][v] = info
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(report_path)
        print(f"{case} {v}: {info}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("as", "nq", "flat"), required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--config", required=True, help="3d_lowres_large_patch (AnchorSlot) or 3d_lowres_large_patch_bs4 (flat)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--nq-background", choices=("min", "mean"), default="min")
    ap.add_argument("--reference-root", type=Path, default=None, help="existing variant folders to compare against")
    ap.add_argument("--case-index", type=int, default=None)
    a = ap.parse_args()
    a.reference_map = {"G0.5": "Gmix0.5", "HG0.5a3": "HGmix0.5a3"}
    for v in a.variants:
        parse(v)
    cases = sorted(q.stem for q in a.dump.glob("*.npy"))
    if a.case_index is not None:
        cases = [cases[a.case_index]]
    for case in cases:
        rep = run_case(a, case)
        rdir = a.out_root / "_reports"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"{case}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
