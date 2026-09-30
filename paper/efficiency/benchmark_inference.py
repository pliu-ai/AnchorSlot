"""Inference time per CellMap validation volume: flat, class-balanced flat, AnchorSlot (per-group and encoder-cached).

All methods run in one job on one GPU with the paper settings: 160^3 tiles, step 0.5, Gaussian weighting, eight
mirrored evaluations, autocast. Per volume we time
  network: sliding-window logits (for AnchorSlot all 13 groups), moved to the CPU;
  export:  reconstruction to one original-label map at the native geometry (in memory, no disk write).
Preprocessing and file writing are identical across methods and excluded. The released AnchorSlot script also writes
13 per-group label maps; they are not needed for the merged prediction and are skipped for both AnchorSlot modes.
Each method is warmed up once on the smallest volume before timing.

Checks: the cached and per-group AnchorSlot label maps are compared voxel by voxel on every volume, and optionally
against earlier predictions of scripts/predict_structured_conditional.py (--reference_predictions); on the two smallest
volumes the per-group logits are compared directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from release_predictor import StructuredConditionalPredictor
from methods import METHODS

MODES = {"flat": "flat", "cb_flat": "cb_flat", "anchorslot_per_group": "anchorslot", "anchorslot_cached": "anchorslot"}
EXPORT_THREADS = default_num_processes


def load_predictor(method: dict):
    kwargs = dict(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True,
                  device=torch.device("cuda"), allow_tqdm=False)
    if method["structured"]:
        p = StructuredConditionalPredictor(group_id=0, run_all_groups=True, output_mode="original",
                                           fixed_merge_mode="mean", cache_encoder=True, **kwargs)
    else:
        p = nnUNetPredictor(**kwargs)
    p.initialize_from_trained_model_folder(method["model_dir"], (method["fold"],), method["checkpoint"])
    return p


def preprocess(p, image_file: str):
    preprocessor = p.configuration_manager.preprocessor_class(verbose=False)
    data, _, props = preprocessor.run_case([image_file], None, p.plans_manager, p.configuration_manager,
                                           p.dataset_json)
    return torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format), props


def per_group_logits(p, data) -> dict:
    """The released script's loop: one full sliding-window pass per group."""
    logits = {}
    for g in range(p.num_dynamic_groups):
        p.group_id = g
        logits[g] = p.predict_logits_from_preprocessed_data(data).cpu()
    return logits


def run_network(mode: str, p, data):
    if mode in ("flat", "cb_flat"):
        return p.predict_logits_from_preprocessed_data(data).cpu()
    if mode == "anchorslot_per_group":
        return per_group_logits(p, data)
    return p.predict_logits_all_groups_from_preprocessed_data(data)


def run_export(mode: str, p, logits, props) -> np.ndarray:
    if mode in ("flat", "cb_flat"):
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, p.plans_manager, p.configuration_manager, p.label_manager, props,
            return_probabilities=False, num_threads_torch=EXPORT_THREADS)
    else:
        seg = p._decode_all_groups_merged_original({g: v.unsqueeze(0) for g, v in logits.items()}, props,
                                                   EXPORT_THREADS)
    return np.asarray(seg, dtype=np.uint8)


def timed_case(mode: str, p, data, props) -> tuple:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = run_network(mode, p, data)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    seg = run_export(mode, p, logits, props)
    t2 = time.perf_counter()
    return seg, {"network_s": t1 - t0, "export_s": t2 - t1, "total_s": t2 - t0}


def read_seg(p, path: Path) -> np.ndarray:
    seg, _ = p.plans_manager.image_reader_writer_class().read_seg(str(path))
    return np.asarray(seg[0], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", required=True, help="folder with the validation *_0000.nii.gz images")
    parser.add_argument("--reference_predictions", default=None,
                        help="optional folder of earlier per-group predictions (<case>.nii.gz) to compare with")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--cases", nargs="*", default=None, help="case ids (default: all 17)")
    args = parser.parse_args()

    if "anchorslot_cached" in args.modes and ("anchorslot_per_group" not in args.modes or
                                              args.modes.index("anchorslot_per_group") > args.modes.index("anchorslot_cached")):
        raise SystemExit("anchorslot_cached needs anchorslot_per_group earlier in --modes for the equality check")
    files = sorted(glob.glob(str(Path(args.images) / "*_0000.nii.gz")))
    cases = {Path(f).name[:-len("_0000.nii.gz")]: f for f in files}
    if args.cases:
        cases = {c: cases[c] for c in args.cases}
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
              "cpus": len(os.sched_getaffinity(0)), "export_threads": EXPORT_THREADS, "cases": list(cases),
              "modes": {}, "checks": {}}
    out = Path(args.out)
    per_group_segs: dict = {}

    for mode in args.modes:
        p = load_predictor(METHODS[MODES[mode]])
        inputs = {c: preprocess(p, f) for c, f in cases.items()}
        smallest = min(inputs, key=lambda c: inputs[c][0].numel())
        timed_case(mode, p, *inputs[smallest])  # warm-up, not recorded
        torch.cuda.reset_peak_memory_stats()
        timings = {}
        for c, (data, props) in inputs.items():
            seg, t = timed_case(mode, p, data, props)
            t["voxels"] = int(data[0].numel())
            if mode == "anchorslot_cached":
                t["group_chunks"] = [len(ch) for ch in p.last_group_chunks]
            timings[c] = t
            print(f"{mode} {c} {json.dumps(t)}", flush=True)
            if mode == "anchorslot_per_group":
                per_group_segs[c] = seg
                ref = Path(args.reference_predictions) / f"{c}.nii.gz" if args.reference_predictions else None
                if ref is not None and ref.is_file():
                    report["checks"].setdefault("per_group_vs_reference_files", {})[c] = \
                        int((read_seg(p, ref) != seg).sum())
            if mode == "anchorslot_cached" and c in per_group_segs:
                report["checks"].setdefault("cached_vs_per_group_label_diff_voxels", {})[c] = \
                    int((per_group_segs[c] != seg).sum())
        report["modes"][mode] = {"timings": timings, "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated())}

        if mode == "anchorslot_cached":
            for c in sorted(inputs, key=lambda c: inputs[c][0].numel())[:2]:
                data, _ = inputs[c]
                ref = per_group_logits(p, data)
                cached = p.predict_logits_all_groups_from_preprocessed_data(data)
                report["checks"].setdefault("cached_vs_per_group_max_abs_logit_diff", {})[c] = \
                    max(float((ref[g] - cached[g]).abs().max()) for g in ref)
        del p, inputs
        torch.cuda.empty_cache()
        out.write_text(json.dumps(report, indent=2))

    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
