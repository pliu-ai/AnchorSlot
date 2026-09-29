"""Reproduce the official CellMap instance matching for one crop/class and return per-component status.

Uses the pinned evaluator (commit 0300239): cc3d relabeling of the scored binary prediction mask,
then match_instances(truth, pred) with the default EvaluationConfig. Counts are checked against the
evaluator's stored per-crop tp/fp/fn so the figure shows exactly what was scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import zarr

CSC = Path("/projects/weilab/liupeng/code/projects/cellmap-segmentation-challenge-upstream-0300239/src")
sys.path.insert(0, str(CSC))
import cc3d  # noqa: E402
from cellmap_segmentation_challenge.utils.eval_utils import EvaluationConfig, match_instances  # noqa: E402

EVAL = Path("/projects/weilab/liupeng/runs/Anchorslot/evaluation_matrix")
LOCK = Path("/projects/weilab/liupeng/runs/Anchorslot/evaluation_protocol/high_4nm/truth_lock.json")
MODELS = {"cb_flat": "flat_nnunet_class_balanced", "anchorslot": "anchorslot_primary"}


def truth_store():
    return zarr.open_group(json.loads(LOCK.read_text())["truth_store"], mode="r")


def match(model: str, crop: str, cls: str):
    """Return (truth_ids, pred_cc, matched_pred_ids, counts, voxel_size)."""
    root = EVAL / MODELS[model] / "full_annotated/4nm"
    truth_arr = truth_store()[crop][cls]
    truth = np.asarray(truth_arr[:])
    pred_mask = np.asarray(zarr.open_group(root / "prediction.zarr", mode="r")[crop][cls][:]) > 0
    pred_cc, n_pred = cc3d.connected_components(pred_mask, return_N=True)
    mapping = match_instances(truth, pred_cc, EvaluationConfig.from_env())
    matched = {int(p) for p, t in mapping.items() if p != 0 and t != 0}
    truth_ids = np.unique(truth)
    truth_ids = truth_ids[truth_ids != 0]
    tp = len(set(mapping.values()) - {0})
    counts = dict(tp=tp, fp=int(n_pred) - len(set(mapping.keys()) - {0}), fn=int(truth_ids.size) - tp)
    stored = json.loads((root / "latest_official_metrics_extended.json").read_text())[crop][cls]
    if any(counts[k] != stored[k] for k in ("tp", "fp", "fn")):
        raise RuntimeError(f"{model} {crop}/{cls}: reproduced {counts} != evaluator {stored}")
    return truth, pred_cc, matched, counts, tuple(truth_arr.attrs["voxel_size"])


if __name__ == "__main__":
    for crop, cls in [("crop219", "lyso"), ("crop219", "mito"), ("crop219", "ld"), ("crop190", "ld")]:
        for model in MODELS:
            *_, counts, vs = match(model, crop, cls)
            print(crop, cls, model, counts, vs)
