#!/usr/bin/env python3
"""Uniform small-component filter applied identically to every method (instance score only).

thresholds: per instance class, the 1st percentile of ground-truth object volume (nm^3) over the
            202 training crops (Dataset200 imagesTr), from Dataset101 labelsTr_split instance files.
evaluate:   for one method, remove predicted connected components below the class threshold from the
            scored binary masks (evaluation_matrix/.../prediction.zarr), re-score every instance pair with
            the pinned evaluator's score_instance, and pool per class exactly as the official aggregation.
            --no-filter re-scores the unfiltered masks as a pipeline check against the official counts.
The semantic score is unchanged by construction and taken from the official metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/projects/weilab/liupeng/code/projects/cellmap-segmentation-challenge-upstream-0300239/src")
RAW = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_raw")
RUNS = Path("/projects/weilab/liupeng/runs/Anchorslot")
LOCK = RUNS / "evaluation_protocol/high_4nm/truth_lock.json"
INSTANCE = ["mito", "nuc", "cell", "perox", "ves", "np", "endo", "ld", "mt", "lyso"]
METHODS = {
    "flat": RUNS / "evaluation_matrix/table_vi_current/flat_nnunet/full_annotated/4nm",
    "cb_flat": RUNS / "evaluation_matrix/flat_nnunet_class_balanced/full_annotated/4nm",
    "anchorslot": RUNS / "evaluation_matrix/anchorslot_primary/full_annotated/4nm",
}
PERCENTILE = 1.0  # default rule; --percentile adds sensitivity variants with their own files


def _tag(percentile: float) -> str:
    return "" if percentile == PERCENTILE else f"_p{percentile:g}"


def _object_volumes(path: str):
    import nibabel as nib

    img = nib.load(path)
    voxel_nm3 = float(np.prod(img.header.get_zooms()[:3]))
    labels, counts = np.unique(np.asarray(img.dataobj), return_counts=True)
    return [float(c) * voxel_nm3 for lab, c in zip(labels, counts) if lab != 0]


def thresholds(workers: int, percentile: float = PERCENTILE) -> None:
    train_cases = sorted(p.name.replace("_0000.nii.gz", "")
                         for p in (RAW / "Dataset200_Dataset101_CellMap_model_high_res/imagesTr").glob("*_0000.nii.gz"))
    jobs = {cls: [] for cls in INSTANCE}
    for case in train_cases:
        dataset, crop = case.rsplit("_crop", 1)
        folder = RAW / "Dataset101_CellMap/labelsTr_split" / dataset / f"crop{crop}" / "labels"
        for cls in INSTANCE:
            jobs[cls].extend(str(p) for p in folder.glob(f"{case}_{cls}_*nm.nii.gz"))
    result = {"train_crops": len(train_cases), "percentile": percentile, "classes": {}}
    with ProcessPoolExecutor(workers) as pool:
        for cls, paths in jobs.items():
            vols = [v for vs in pool.map(_object_volumes, paths) for v in vs]
            result["classes"][cls] = dict(files=len(paths), objects=len(vols),
                                          threshold_nm3=float(np.percentile(vols, percentile)) if vols else 0.0,
                                          min_nm3=float(min(vols)) if vols else 0.0,
                                          median_nm3=float(np.median(vols)) if vols else 0.0)
            print(cls, result["classes"][cls], flush=True)
    (HERE / f"size_thresholds{_tag(percentile)}.json").write_text(json.dumps(result, indent=2) + "\n")


def _score_pair(args):
    import cc3d
    import zarr
    from cellmap_segmentation_challenge.utils.eval_utils import score_instance

    method, crop, cls, threshold_nm3 = args
    truth_arr = zarr.open_group(json.loads(LOCK.read_text())["truth_store"], mode="r")[crop][cls]
    voxel = tuple(float(v) for v in truth_arr.attrs["voxel_size"])
    truth = np.asarray(truth_arr[:])
    pred = np.asarray(zarr.open_group(METHODS[method] / "prediction.zarr", mode="r")[crop][cls][:]) > 0
    removed = 0
    if threshold_nm3 > 0:
        min_voxels = int(math.ceil(threshold_nm3 / float(np.prod(voxel))))
        before = int(cc3d.connected_components(pred.astype(np.uint8), return_N=True)[1])
        pred = cc3d.dust(pred.astype(np.uint8), threshold=min_voxels, connectivity=26, in_place=False) > 0
        removed = before - int(cc3d.connected_components(pred.astype(np.uint8), return_N=True)[1])
    s = score_instance(pred.astype(np.uint8), truth, voxel)
    return dict(crop=crop, cls=cls, removed_components=removed, **{k: s[k] for k in ("tp", "fp", "fn", "hausdorff_norm_sum", "n_hausdorff", "status")})


def evaluate(method: str, no_filter: bool, workers: int, percentile: float = PERCENTILE) -> None:
    official = json.loads((METHODS[method] / "latest_official_metrics_extended.json").read_text())
    pairs = [(c, k) for c, v in official.items() if c.startswith("crop") for k, r in v.items() if "hausdorff_norm_sum" in r]
    thr = {} if no_filter else {c: v["threshold_nm3"] for c, v in json.loads((HERE / f"size_thresholds{_tag(percentile)}.json").read_text())["classes"].items()}
    tasks = [(method, c, k, 0.0 if no_filter else thr[k]) for c, k in pairs]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_score_pair, tasks))
    pooled = {}
    for r in rows:
        a = pooled.setdefault(r["cls"], [0, 0, 0, 0.0, 0])
        for i, key in enumerate(("tp", "fp", "fn", "hausdorff_norm_sum", "n_hausdorff")):
            a[i] += r[key]
    per_class = {}
    for cls, (tp, fp, fn, hs, nh) in pooled.items():
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        per_class[cls] = dict(tp=tp, fp=fp, fn=fn, score=math.sqrt(f1 * (hs / nh if nh else 1.0)))
    inst = float(np.mean([v["score"] for v in per_class.values()]))
    sem = official["overall_semantic_score"]
    out = dict(method=method, filtered=not no_filter, instance=inst, semantic=sem, overall=math.sqrt(sem * inst),
               official_instance=official["overall_instance_score"],
               pathological=sum(r["status"] != "scored" for r in rows),
               removed_components=sum(r["removed_components"] for r in rows), per_class=per_class, pairs=rows)
    tag = f"{method}_{'nofilter' if no_filter else 'filtered'}{'' if no_filter else _tag(percentile)}"
    (HERE / f"uniform_filter_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(tag, {k: out[k] for k in ("instance", "overall", "official_instance", "pathological", "removed_components")}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["thresholds", "evaluate"])
    ap.add_argument("--method", choices=sorted(METHODS))
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    a = ap.parse_args()
    if a.action == "thresholds":
        thresholds(a.workers, a.percentile)
    else:
        evaluate(a.method, a.no_filter, a.workers, a.percentile)
