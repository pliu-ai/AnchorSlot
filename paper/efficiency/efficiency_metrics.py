"""Parameters, forward FLOPs and training memory of flat, class-balanced flat and AnchorSlot (CellMap, 160^3 patches).

- Parameters and FLOPs are measured on the inference networks as built by the predictors from the paper checkpoints.
  FLOPs come from torch.utils.flop_counter (convolutions and linear layers, one multiply-add = 2 FLOPs) for one
  1x1x160^3 patch without mirroring. For AnchorSlot the encoder and the query-conditioned decoder are also counted
  separately, which gives the per-tile cost with and without encoder caching.
- Training memory is the peak allocated CUDA memory of the trainers' own train_step (autocast, grad scaler, SGD,
  deep supervision, the trainers' losses) on synthetic 160^3 batches, single GPU, no torch.compile, after one
  warm-up step. Measured at batch size 1 and 2 per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from release_predictor import StructuredConditionalPredictor
from methods import CONFIGURATION, METHODS, PATCH

NUM_GROUPS = 13
NUM_MIRRORS = 8


@contextmanager
def temporary_env(overrides: dict):
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update({k: str(v) for k, v in overrides.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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


def count_flops(fn) -> int:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


def inference_complexity(method: dict) -> dict:
    p = load_predictor(method)
    # The per-tile scaling below assumes the paper settings; fail rather than report wrong multipliers.
    assert 2 ** len(p.allowed_mirroring_axes) == NUM_MIRRORS, p.allowed_mirroring_axes
    if method["structured"]:
        assert p.num_dynamic_groups == NUM_GROUPS, p.num_dynamic_groups
    net = p.network.to("cuda").eval()
    x = torch.randn(1, 1, *PATCH, device="cuda")
    out = {"params": int(sum(t.numel() for t in net.parameters()))}
    if method["structured"]:
        gid = torch.zeros(1, dtype=torch.long, device="cuda")
        out["flops_forward"] = count_flops(lambda: net(x, gid))
        out["flops_encoder"] = count_flops(lambda: net.encode(x))
        with torch.no_grad():
            skips = net.encode(x)
        out["flops_decoder"] = count_flops(lambda: net.decode_from_skips(skips, gid))
        out["params_conditioning"] = int(sum(t.numel() for n, t in net.named_parameters()
                                             if n.startswith(("condition_encoder.", "decoder_film."))))
        e, d = out["flops_encoder"], out["flops_decoder"]
        out["flops_per_tile_uncached"] = NUM_MIRRORS * NUM_GROUPS * (e + d)
        out["flops_per_tile_cached"] = NUM_MIRRORS * (e + NUM_GROUPS * d)
    else:
        out["flops_forward"] = count_flops(lambda: net(x))
        out["flops_per_tile_uncached"] = NUM_MIRRORS * out["flops_forward"]
    del p, net
    torch.cuda.empty_cache()
    return out


def build_trainer(method: dict, results_dir: str):
    model_dir = Path(method["model_dir"])
    plans = load_json(str(model_dir / "plans.json"))
    dataset_json = load_json(str(model_dir / "dataset.json"))
    trainer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), method["trainer"],
        "nnunetv2.training.nnUNetTrainer")
    if trainer_class is None:
        raise RuntimeError(f"trainer {method['trainer']} not found")
    env = dict(method.get("train_env", {}), nnUNet_results=results_dir, nnUNet_compile="f")
    with temporary_env(env):
        trainer = trainer_class(plans=plans, configuration=CONFIGURATION, fold=0, dataset_json=dataset_json,
                                device=torch.device("cuda"))
        trainer.initialize()
    return trainer


def synthetic_batch(trainer, batch_size: int, num_labels: int) -> dict:
    g = torch.Generator().manual_seed(0)
    data = torch.randn(batch_size, 1, *PATCH, generator=g)
    target = []
    for scale in trainer._get_deep_supervision_scales():
        shape = [max(1, round(s * f)) for s, f in zip(PATCH, scale)]
        target.append(torch.randint(0, num_labels, (batch_size, 1, *shape), generator=g).float())
    return {"data": data, "target": target, "keys": [f"synthetic_{i}" for i in range(batch_size)]}


def training_memory(method: dict, batch_size: int, results_dir: str) -> dict:
    torch.cuda.empty_cache()
    trainer = build_trainer(method, results_dir)
    batch = synthetic_batch(trainer, batch_size, num_labels=32)
    trainer.network.train()
    trainer.train_step(batch)  # warm-up (cuDNN autotuning, optimizer state)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        trainer.train_step(batch)
    torch.cuda.synchronize()
    out = {"batch_size": batch_size,
           "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
           "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    del trainer, batch
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "patch": PATCH, "methods": {}}
    with tempfile.TemporaryDirectory() as results_dir:
        for name, method in METHODS.items():
            entry = {"inference": inference_complexity(method), "training": []}
            for bs in (1, 2):
                entry["training"].append(training_memory(method, bs, results_dir))
            report["methods"][name] = entry
            print(name, json.dumps(entry), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
