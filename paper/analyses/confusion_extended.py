#!/usr/bin/env python3
"""Reconstructed-label confusion for flat and class-balanced flat (plus AnchorSlot as a check).

Same procedure as cellmap_paper_refresh_20260921/raw_diagnostics.py (27 dynamic labels, annotated
positive voxels only, validity masks honoured, raw hard labels on the manifest geometry), extended to the
flat baselines. Also reports the background (label 0) fraction of the 17 validation label maps and each
method's predicted-background fraction, so the 'to background' column can be read in context.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import zarr

HERE = Path(__file__).resolve().parent
RUNS = Path("/projects/weilab/liupeng/runs/Anchorslot")
REPO = Path("/projects/weilab/liupeng/code/projects/AnchorSlot")
LOCK = RUNS / "evaluation_protocol/high_4nm/truth_lock.json"
# Validation label maps (labelsTs *.nii.gz), used for the ground-truth background fraction.
LABELS_TS = Path(os.environ.get("CELLMAP_LABELS_TS", ""))
sys.path.insert(0, str(REPO))
from nnunetv2.evaluation.cellmap_challenge import (  # noqa: E402
    ATOMIC_TO_LABEL_ID, _load_hard_segmentation, entries_by_crop, index_predictions, read_manifest)

FIXED = {1, 2, 3, 27}
GROUPS = [{4, 5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}, {19, 20},
          {25, 26}, {21, 22}, {23, 24}, {28, 29}, {30, 31}, {17, 18}]
LUT = {label: g for g, labels in enumerate(GROUPS) for label in labels}
SOURCES = {
    "flat": RUNS / "evaluation_matrix/table_vi_current/flat_nnunet/full_annotated/4nm",
    "cb_flat": RUNS / "evaluation_matrix/flat_nnunet_class_balanced/full_annotated/4nm",
    "anchorslot": RUNS / "evaluation_matrix/anchorslot_primary/full_annotated/4nm",
}


def main() -> None:
    if not os.environ.get("CELLMAP_LABELS_TS") or not LABELS_TS.is_dir():
        raise SystemExit("set CELLMAP_LABELS_TS to the folder with the validation label maps (labelsTs)")
    lock = json.loads(LOCK.read_text())
    methods = {k: json.loads((v / "evaluation_provenance.json").read_text())["predictions"] for k, v in SOURCES.items()}
    preds = {k: index_predictions(v) for k, v in methods.items()}
    truth = zarr.open_group(lock["truth_store"], mode="r")
    counts = {k: np.zeros((32, 32), dtype=np.int64) for k in methods}
    pred_bg = {k: [0, 0] for k in methods}
    for crop, entries in sorted(entries_by_crop(read_manifest(lock["manifest"])).items()):
        shape = entries[0].shape
        arrays = {k: _load_hard_segmentation(files[crop], shape) for k, files in preds.items()}
        for k, a in arrays.items():
            pred_bg[k][0] += int((a == 0).sum()); pred_bg[k][1] += int(a.size)
        for entry in entries:
            label = ATOMIC_TO_LABEL_ID.get(entry.class_label)
            if label is None or label in FIXED:
                continue
            gt = truth[crop][entry.class_label]
            for start in range(0, shape[0], 32):
                region = (slice(start, min(start + 32, shape[0])), slice(None), slice(None))
                mask = gt[region] > 0
                if entry.class_label + "_mask" in truth[crop]:
                    mask &= truth[crop][entry.class_label + "_mask"][region] > 0
                for k, a in arrays.items():
                    counts[k][label] += np.bincount(a[region][mask], minlength=32)
        print("crop done:", crop, flush=True)
    gt_bg = [0, 0]
    for p in sorted(LABELS_TS.glob("*.nii.gz")):
        a = np.asarray(nib.load(str(p)).dataobj)
        gt_bg[0] += int((a == 0).sum()); gt_bg[1] += int(a.size)
    result = {"methods": methods, "gt_background_percent": 100 * gt_bg[0] / gt_bg[1],
              "predicted_background_percent": {k: 100 * v[0] / v[1] for k, v in pred_bg.items()}, "models": {}}
    for k, m in counts.items():
        total = int(m.sum())
        cross = sum(int(m[a, b]) for a in LUT for b in LUT if LUT[a] != LUT[b])
        intra = sum(int(m[a, b]) for a in LUT for b in LUT if a != b and LUT[a] == LUT[b])
        fixed = int(m[:, sorted(FIXED)].sum())
        bg = int(m[:, 0].sum())
        correct = sum(int(m[a, a]) for a in LUT)
        assert cross + intra + fixed + bg + correct == total
        result["models"][k] = dict(observations=total, cross_group_percent=100 * cross / total,
                                   within_group_percent=100 * intra / total, to_fixed_percent=100 * fixed / total,
                                   to_background_percent=100 * bg / total, correct_percent=100 * correct / total)
    assert len({v["observations"] for v in result["models"].values()}) == 1
    (HERE / "confusion_extended.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("gt_background_percent", "predicted_background_percent", "models")}, indent=1))


if __name__ == "__main__":
    main()
