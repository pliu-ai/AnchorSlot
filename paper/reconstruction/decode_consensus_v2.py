#!/usr/bin/env python3
"""Step 1: Zero-retrain upgraded decoders for AnchorSlot 63-channel probability dumps.

Implements and exports:
  Gmix0.5                  Exact paper Eq. (7) baseline (sanity reproduction)
  TS_w<w>                  Two-Stage Hierarchical Bayes Decoding (eliminates vote-splitting)
  TS_w<wc>_g<wg>           Dual-Weight Two-Stage Hysteresis: core from TS_w<wc>, 6-connected growth from TS_w<wg>
  TS_LA<tau>_w<wc>_g<wg>   Same + leakage-free training-prior Logit Adjustment on group_head probabilities
  TS_w<wc>_g<wg>_veto      Same + component-level own_other & head/slot disagreement FP veto

All exports use nnU-Net's exact `convert_predicted_logits_to_segmentation_with_correct_shape`
and preserve original native NIfTI geometry.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
import torch

LOW = np.float32(-1e4)
EPS = np.float32(1e-8)

VARIANT_RE = re.compile(
    r"^(?:"
    r"(?P<gmix>Gmix(?P<gw>[0-9.]+))"
    r"|TS(?:_LA(?P<tau>-?[0-9.]+))?_w(?P<wc>[0-9.]+)(?:_g(?P<wg>[0-9.]+))?(?P<veto>_veto)?"
    r")$"
)


def parse_meta(meta: Sequence[dict]):
    compact_meta = [c for c in meta if c["kind"] != "group_head"]
    head_idx = [int(c["index"]) for c in meta if c["kind"] == "group_head"]
    anchors = sorted((c for c in compact_meta if c["kind"] == "anchor_mean"), key=lambda c: int(c["index"]))
    slots = [c for c in compact_meta if c["kind"] == "slot"]
    others = sorted((c for c in compact_meta if c["kind"] == "other"), key=lambda c: int(c["group"]))
    group_to_slots: Dict[int, List[Tuple[int, int]]] = {}
    for c in slots:
        group_to_slots.setdefault(int(c["group"]), []).append((int(c["label"]), int(c["index"])))
    for g in group_to_slots:
        group_to_slots[g].sort(key=lambda t: t[1])
    return compact_meta, head_idx, anchors, slots, others, group_to_slots


def compute_group_priors_from_json(priors_json: Path, anchors: Sequence[dict], group_to_slots: Dict[int, List[Tuple[int, int]]]) -> np.ndarray:
    """Compute leakage-free 18-class coarse voxel priors strictly from training-split counts."""
    data = json.loads(priors_json.read_text())
    lv = {int(k): float(v) for k, v in data["label_voxels"].items()}
    counts = np.zeros(18, dtype=np.float64)
    counts[0] = max(lv.get(0, 1.0), 1.0)
    for i, c in enumerate(anchors):
        counts[1 + i] = max(lv.get(int(c["label"]), 1.0), 1.0)
    for g in range(13):
        counts[5 + g] = max(sum(lv.get(lab, 0.0) for lab, _ in group_to_slots[g]), 1.0)
    return (counts / counts.sum()).astype(np.float32)


def adjust_head_with_priors(head: np.ndarray, priors18: np.ndarray, tau: float) -> np.ndarray:
    """Post-hoc Logit Adjustment on 18-class group_head probabilities: P_adj(k) ∝ P(k) * pi_k^(-tau).
    Background channel 0 is left unadjusted relative to geometric mean so background prior is never inflated.
    """
    if abs(tau) < 1e-12:
        return head
    # Adjust channels 1..17 (4 anchors + 13 dynamic groups) using training priors; keep bg at mean scale
    fg_priors = np.maximum(priors18[1:], np.float32(1e-12))
    mult = np.ones(18, dtype=np.float32)
    fg_mult = np.power(fg_priors, -np.float32(tau))
    # Normalize geometric mean of foreground multipliers to 1 so overall scale vs bg stays calibrated
    fg_mult = fg_mult / np.exp(np.mean(np.log(fg_mult)))
    mult[1:] = fg_mult.astype(np.float32)
    adj = head * mult[:, None, None, None]
    denom = np.maximum(adj.sum(axis=0, keepdims=True), EPS)
    return (adj / denom).astype(np.float32)


def build_gmix_scores(p: np.ndarray, head: np.ndarray, anchors: Sequence[dict], slots: Sequence[dict], weight: float) -> np.ndarray:
    spatial = p.shape[1:]
    S = np.full((32, *spatial), LOW, dtype=np.float32)
    S[0] = p[0]
    for c in anchors:
        S[int(c["label"])] = p[int(c["index"])]
    gsum: Dict[int, np.ndarray] = {}
    for c in slots:
        lab, idx, g = int(c["label"]), int(c["index"]), int(c["group"])
        S[lab] = p[idx]
        gsum[g] = gsum[g] + p[idx] if g in gsum else p[idx].copy()

    H = np.full_like(S, LOW)
    H[0] = head[0]
    for i, c in enumerate(anchors):
        H[int(c["label"])] = head[1 + i]
    for c in slots:
        lab, idx, g = int(c["label"]), int(c["index"]), int(c["group"])
        H[lab] = head[5 + g] * p[idx] / np.maximum(gsum[g], EPS)

    used = [0, *[int(c["label"]) for c in anchors], *[int(c["label"]) for c in slots]]
    out = S.copy()
    out[used] = np.exp((1.0 - weight) * np.log(np.maximum(S[used], EPS)) + weight * np.log(np.maximum(H[used], EPS)))
    return out


def build_twostage_scores(
    p: np.ndarray,
    head: np.ndarray,
    anchors: Sequence[dict],
    group_to_slots: Dict[int, List[Tuple[int, int]]],
    weight: float,
) -> np.ndarray:
    """Build 32-channel score map whose argmax is MATHEMATICALLY EQUIVALENT to Two-Stage Hierarchical Bayes Decoding!

    Proof:
      Stage 1 compares 18 coarse scores Q_fused[k] = Q_slot[k]^(1-w) * head[k]^w (no vote-splitting!).
      For each dynamic group g, we assign Q_fused[5+g] to its winning within-group slot (and Q_fused[5+g] * ratio < Q_fused[5+g]
      to losing slots of g, i.e., `Q_fused[5+g] * (p_s / max_s' p_s')`).
      Then:
        max_{c in G_g} score[c] == Q_fused[5+g]  (EXACTLY equal to the unsplit group mass!)
        argmax_{c in G_g} score[c] == argmax_{s in G_g} p_s  (EXACT within-group slot winner!)
      And because scores remain continuous and proportional, nnU-Net's linear resampling back to native 2nm/4nm
      spacing via `convert_predicted_logits_to_segmentation_with_correct_shape` works seamlessly and smoothly!
    """
    spatial = p.shape[1:]
    Q_slot = np.zeros((18, *spatial), dtype=np.float32)
    Q_slot[0] = p[0]
    for i, c in enumerate(anchors):
        Q_slot[1 + i] = p[int(c["index"])]

    g_max = {}
    for g in range(13):
        slist = group_to_slots[g]
        s_sum = np.zeros(spatial, dtype=np.float32)
        s_mx = np.full(spatial, EPS, dtype=np.float32)
        for _, idx in slist:
            s_sum += p[idx]
            np.maximum(s_mx, p[idx], out=s_mx)
        Q_slot[5 + g] = s_sum
        g_max[g] = s_mx

    if weight <= 0.0:
        Q_fused = Q_slot
    elif weight >= 1.0:
        Q_fused = head
    else:
        Q_fused = np.exp(
            (1.0 - weight) * np.log(np.maximum(Q_slot, EPS)) + weight * np.log(np.maximum(head, EPS))
        ).astype(np.float32)

    S32 = np.full((32, *spatial), LOW, dtype=np.float32)
    S32[0] = Q_fused[0]
    for i, c in enumerate(anchors):
        S32[int(c["label"])] = Q_fused[1 + i]
    for g in range(13):
        qg = Q_fused[5 + g]
        inv_max = 1.0 / np.maximum(g_max[g], EPS)
        for lab, idx in group_to_slots[g]:
            # Winning slot gets exact qg (ratio=1.0); losing sibling(s) get qg * (p_s / p_max) < qg
            S32[lab] = qg * (p[idx] * inv_max)
    return S32


def hysteresis_grow_groups(
    core: np.ndarray,
    relaxed: np.ndarray,
    group_to_slots: Dict[int, List[Tuple[int, int]]],
    dyn_labels: Sequence[int],
) -> np.ndarray:
    """Grow dynamic predictions from `core` into `relaxed` ONLY where 6-connected to an existing core component
    of the SAME organelle group (so membranes can grow onto cores containing lumen, and ZERO new components are created).
    """
    out = core.copy()
    dyn_lut = np.zeros(32, dtype=bool)
    dyn_lut[list(dyn_labels)] = True
    grow_any = (relaxed != core) & (~dyn_lut[np.clip(core, 0, 31)]) & dyn_lut[np.clip(relaxed, 0, 31)]
    if not grow_any.any():
        return out

    for g, slist in group_to_slots.items():
        glabs = [lab for lab, _ in slist]
        g_lut = np.zeros(32, dtype=bool)
        g_lut[glabs] = True
        core_g = g_lut[np.clip(core, 0, 31)]
        if not core_g.any():
            continue
        grow_g = grow_any & g_lut[np.clip(relaxed, 0, 31)]
        if not grow_g.any():
            continue
        comp, n = ndimage.label(core_g | grow_g)  # 6-connectivity
        keep = np.zeros(n + 1, dtype=bool)
        keep[np.unique(comp[core_g])] = True
        keep[0] = False
        accept = grow_g & keep[comp]
        if accept.any():
            out[accept] = relaxed[accept]
    return out


def compute_veto_mask_4nm(
    p: np.ndarray,
    head: np.ndarray,
    anchors: Sequence[dict],
    others: Sequence[dict],
    group_to_slots: Dict[int, List[Tuple[int, int]]],
    seg_4nm: np.ndarray,
    own_other_thresh: float = 0.12,
    disagree_thresh: float = 0.55,
    max_comp_voxels_4nm: int = 2000,
) -> np.ndarray:
    """Identify high-conflict false-positive dynamic components on the 4 nm grid using currently discarded signals:
      - mean own_other P_other^(g) > own_other_thresh (6.3x higher on FPs than TPs), OR
      - mean head!=slot coarse disagreement > disagree_thresh (5.4x higher on FPs than TPs).
    Only applies to small/medium components (<= max_comp_voxels_4nm) so large real organelles are never removed.
    Returns boolean mask of voxels to revert to their best background/anchor label.
    """
    spatial = p.shape[1:]
    Q_slot = np.zeros((18, *spatial), dtype=np.float32)
    Q_slot[0] = p[0]
    for i, c in enumerate(anchors):
        Q_slot[1 + i] = p[int(c["index"])]
    for g in range(13):
        s_sum = np.zeros(spatial, dtype=np.float32)
        for _, idx in group_to_slots[g]:
            s_sum += p[idx]
        Q_slot[5 + g] = s_sum

    disagree = (np.argmax(Q_slot, axis=0) != np.argmax(head, axis=0)).astype(np.float32)
    veto = np.zeros(spatial, dtype=bool)

    for g in range(13):
        glabs = [lab for lab, _ in group_to_slots[g]]
        g_lut = np.zeros(32, dtype=bool)
        g_lut[glabs] = True
        mask_g = g_lut[np.clip(seg_4nm, 0, 31)]
        if not mask_g.any():
            continue
        comp, n = ndimage.label(mask_g)
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        sizes = ndimage.sum(np.ones(spatial, dtype=np.float32), comp, ids)
        own_oth_map = p[int(others[g]["index"])]
        mean_oo = ndimage.mean(own_oth_map, comp, ids)
        mean_dis = ndimage.mean(disagree, comp, ids)
        bad = (sizes <= max_comp_voxels_4nm) & ((mean_oo > own_other_thresh) | (mean_dis > disagree_thresh))
        if bad.any():
            bad_lut = np.zeros(n + 1, dtype=bool)
            bad_lut[ids[bad]] = True
            veto |= bad_lut[comp]
    return veto


def apply_veto_on_scores(scores32: np.ndarray, veto_mask_4nm: np.ndarray, dyn_labels: Sequence[int]) -> np.ndarray:
    """Suppress vetoed dynamic components cleanly on the 4 nm score map before native-grid export."""
    if not veto_mask_4nm.any():
        return scores32
    out = scores32.copy()
    for lab in dyn_labels:
        out[lab][veto_mask_4nm] = LOW
    return out


class Exporter:
    def __init__(self, model_folder: Path, props: dict):
        CODE = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0])
        if str(CODE / "scripts") not in sys.path:
            sys.path.insert(0, str(CODE / "scripts"))
        import predict_structured_conditional as psc
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        self._export_fn = convert_predicted_logits_to_segmentation_with_correct_shape
        plans = PlansManager(json.loads((model_folder / "plans.json").read_text()))
        self.plans = plans
        self.cm = plans.get_configuration("3d_lowres_large_patch")
        self.lm = psc._build_original_32_identity_label_manager()
        self.props = props

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        seg = self._export_fn(
            torch.from_numpy(scores), self.plans, self.cm, self.lm, self.props,
            return_probabilities=False, num_threads_torch=4,
        )
        return np.asarray(seg, dtype=np.uint8)


def run_case(
    case: str,
    dump: Path,
    model_folder: Path,
    out_root: Path,
    variants: Sequence[str],
    priors_json: Optional[Path] = None,
) -> dict:
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    meta = json.loads((dump / "channels.json").read_text())
    compact_meta, head_idx, anchors, slots, others, group_to_slots = parse_meta(meta)
    dyn_labels = [int(c["label"]) for c in slots]

    p_all = np.load(dump / f"{case}.npy")
    props = pickle.load(open(dump / f"{case}.pkl", "rb"))
    p = p_all[: len(compact_meta)].astype(np.float32)
    head_raw = p_all[head_idx].astype(np.float32)
    del p_all

    priors18 = compute_group_priors_from_json(priors_json, anchors, group_to_slots) if priors_json else None
    export = Exporter(model_folder, props)
    rw = SimpleITKIO()

    head_cache: Dict[float, np.ndarray] = {0.0: head_raw}
    seg_cache: Dict[Tuple[float, float, bool], np.ndarray] = {}

    def get_head(tau: float) -> np.ndarray:
        if tau not in head_cache:
            if priors18 is None:
                raise ValueError(f"Variant requires --priors-json for tau={tau}")
            head_cache[tau] = adjust_head_with_priors(head_raw, priors18, tau)
        return head_cache[tau]

    def get_ts_seg(tau: float, w: float, veto: bool = False) -> np.ndarray:
        key = (tau, w, veto)
        if key not in seg_cache:
            h = get_head(tau)
            S = build_twostage_scores(p, h, anchors, group_to_slots, weight=w)
            if veto:
                seg_4nm = np.argmax(S, axis=0).astype(np.uint8)
                vmask = compute_veto_mask_4nm(p, h, anchors, others, group_to_slots, seg_4nm)
                S = apply_veto_on_scores(S, vmask, dyn_labels)
            seg_cache[key] = export(S)
            del S
        return seg_cache[key]

    report = {"case": case, "variants": {}}
    for v in variants:
        m = VARIANT_RE.fullmatch(v)
        if not m:
            raise ValueError(f"Unknown variant {v!r}")
        out = out_root / v / f"{case}.nii.gz"
        if out.exists():
            print(f"{case} {v} exists, skipping", flush=True)
            continue
        t0 = time.time()
        if m.group("gmix"):
            gw = float(m.group("gw"))
            S = build_gmix_scores(p, head_raw, anchors, slots, weight=gw)
            seg = export(S)
            del S
        else:
            tau = float(m.group("tau")) if m.group("tau") else 0.0
            wc = float(m.group("wc"))
            wg = float(m.group("wg")) if m.group("wg") else None
            veto = bool(m.group("veto"))
            core = get_ts_seg(tau, wc, veto=veto)
            if wg is None:
                seg = core
            else:
                relaxed = get_ts_seg(tau, wg, veto=False)
                seg = hysteresis_grow_groups(core, relaxed, group_to_slots, dyn_labels)
        out.parent.mkdir(parents=True, exist_ok=True)
        rw.write_seg(seg, str(out), props)
        elapsed = round(time.time() - t0, 1)
        report["variants"][v] = {"seconds": elapsed, "shape": list(seg.shape)}
        print(f"{case} {v} done ({elapsed}s)", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--priors-json", type=Path, default=None)
    ap.add_argument("--case-index", type=int, default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    cases = a.cases or sorted(q.stem for q in a.dump.glob("*.npy"))
    if a.case_index is not None:
        cases = [cases[a.case_index]]
    for case in cases:
        rep = run_case(case, a.dump, a.model_folder, a.out_root, a.variants, priors_json=a.priors_json)
        rdir = a.out_root / "_reports"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"{case}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
