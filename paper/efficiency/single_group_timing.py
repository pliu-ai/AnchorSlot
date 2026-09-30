"""Network time of ONE AnchorSlot query group per volume (same settings as benchmark_inference.py).

This is the per-volume network wall-clock time when the 13 groups run in parallel on separate GPUs (the merge and
export still run once per volume; see benchmark_inference.py for their cost).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import torch

from benchmark_inference import load_predictor, preprocess
from methods import METHODS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", required=True, help="folder with the validation *_0000.nii.gz images")
    parser.add_argument("--groups", nargs="+", type=int, default=[0, 6, 12])
    args = parser.parse_args()

    p = load_predictor(METHODS["anchorslot"])
    files = sorted(glob.glob(str(Path(args.images) / "*_0000.nii.gz")))
    inputs = {Path(f).name[:-len("_0000.nii.gz")]: preprocess(p, f) for f in files}
    smallest = min(inputs, key=lambda c: inputs[c][0].numel())
    p.group_id = 0
    p.predict_logits_from_preprocessed_data(inputs[smallest][0])  # warm-up

    report = {"gpu": torch.cuda.get_device_name(0), "groups": {}}
    for g in args.groups:
        p.group_id = g
        times = {}
        for c, (data, _) in inputs.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p.predict_logits_from_preprocessed_data(data).cpu()
            torch.cuda.synchronize()
            times[c] = time.perf_counter() - t0
        report["groups"][g] = times
        print(g, f"mean {sum(times.values()) / len(times):.2f} s", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
