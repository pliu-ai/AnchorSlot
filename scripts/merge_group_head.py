#!/usr/bin/env python3
"""Offline merges for a group-head AnchorSlot dump (predict_dump_gh.py): legacy, Garb<w>, Gmix<w>.

legacy   the original AnchorSlot merge (structured channels only)
Garb<w>  legacy background/anchor/dynamic decision; winning dynamic label re-ranked by slot * P_head(group)^w
Gmix<w>  geometric mix of legacy and the hierarchical merge P_head(group) * within-group slot share
Scores are built at 4 nm by `compact_scores` (tested in nnunetv2/tests/test_anchorslot_group_head.py) and exported with
nnU-Net's convert_predicted_logits_to_segmentation_with_correct_shape, exactly like the online merge.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0])
sys.path.insert(0, str(CODE / "scripts"))
import predict_structured_conditional as psc  # noqa: E402
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO  # noqa: E402
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape  # noqa: E402
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_er_dynamic_group_head import (  # noqa: E402
    compact_scores)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402


def parse(name: str):
    if name == "legacy":
        return "legacy", 0.0
    m = re.fullmatch(r"G(arb|mix)([0-9.]+)", name)
    if not m:
        raise ValueError(name)
    return {"arb": "arbiter", "mix": "mix"}[m.group(1)], float(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    meta = json.loads((a.dump / "channels.json").read_text())
    compact_meta = [c for c in meta if c["kind"] != "group_head"]
    head_idx = [c["index"] for c in meta if c["kind"] == "group_head"]
    plans = PlansManager(json.loads((a.model_folder / "plans.json").read_text()))
    cm = plans.get_configuration("3d_lowres_large_patch")
    lm = psc._build_original_32_identity_label_manager()
    rw = SimpleITKIO()
    specs = {v: parse(v) for v in a.variants}
    if any(mode != "legacy" for mode, _ in specs.values()) and not head_idx:
        raise SystemExit("dump has no group-head channels")
    for case in a.cases or sorted(p.stem for p in a.dump.glob("*.npy")):
        p = np.load(a.dump / f"{case}.npy").astype(np.float32)
        props = pickle.load(open(a.dump / f"{case}.pkl", "rb"))
        head = p[head_idx] if head_idx else None
        compact = p[: len(compact_meta)]
        for v, (mode, w) in specs.items():
            out = a.out_root / v / f"{case}.nii.gz"
            if out.exists():
                continue
            S = compact_scores(compact, compact_meta, head, mode=mode, weight=w)
            seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                torch.from_numpy(S), plans, cm, lm, props, return_probabilities=False, num_threads_torch=4)
            out.parent.mkdir(parents=True, exist_ok=True)
            rw.write_seg(np.asarray(seg, dtype=np.uint8), str(out), props)
            print(f"{case} {v} done", flush=True)
            del S, seg
        del p, head, compact


if __name__ == "__main__":
    main()
