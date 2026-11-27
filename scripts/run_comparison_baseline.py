#!/usr/bin/env python3
"""Inspect or train a CellMap comparison. Defaults to a read-only recipe preview.

Use --run to train; this script does not submit Slurm jobs. Multi-GPU training
uses one local DDP process per GPU. Full-volume prediction is a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("flat", "salt", "blob", "dodnet", "dodnet_all", "hssn"), required=True)
    parser.add_argument("--dataset", default="Dataset205_Dataset101_CellMap_model_high_res_truegeom")
    parser.add_argument("--configuration", default="3d_lowres_large_patch")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--preprocessed-root", type=Path,
                        default=Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_preprocessed"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--train-iters", type=int, default=250)
    parser.add_argument("--val-iters", type=int, default=25)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset != Path(args.dataset).name or not args.dataset.startswith("Dataset"):
        parser.error("--dataset must be a dataset directory name, not a path")
    if min(args.batch_size, args.epochs, args.train_iters, args.val_iters, args.num_gpus) < 1 or args.lr <= 0:
        parser.error("Batch size, epochs, iterations, GPU count and learning rate must be positive")
    if args.device == "cpu" and args.num_gpus != 1:
        parser.error("Use --num-gpus 1 for CPU smoke training")
    if args.batch_size < args.num_gpus:
        parser.error("Global batch size must be at least the GPU count")
    return args


def recipe_preview(args):
    folder = args.preprocessed_root / args.dataset
    files = {name: folder / name for name in ("nnUNetPlans.json", "dataset.json", "splits_final.json")}
    payload = {name: json.loads(path.read_text()) for name, path in files.items()}
    if args.configuration not in payload["nnUNetPlans.json"]["configurations"]:
        raise ValueError(f"Missing configuration {args.configuration}")
    splits = payload["splits_final.json"]
    if not 0 <= args.fold < len(splits):
        raise ValueError("Requested fold does not exist; refusing to generate a replacement split")
    split = splits[args.fold]
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Training/validation split overlaps")
    return {"method": args.method, "dataset": args.dataset, "configuration": args.configuration,
            "fold": args.fold, "train_cases": len(split["train"]), "val_cases": len(split["val"]),
            "results_root": str(args.results_root.resolve()), "num_gpus": args.num_gpus,
            "device": args.device, "batch_size": args.batch_size, "epochs": args.epochs,
            "train_iters": args.train_iters, "val_iters": args.val_iters, "lr": args.lr,
            "resume": args.resume, "run": args.run,
            "metadata_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in files.items()}}


def worker(rank, world_size, args):
    import torch
    import torch.distributed as dist
    from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.trainers import TRAINERS
    folder = args.preprocessed_root / args.dataset
    plans = json.loads((folder / "nnUNetPlans.json").read_text())
    dataset = json.loads((folder / "dataset.json").read_text())
    if args.device == "cuda":
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        trainer = TRAINERS[args.method](plans, args.configuration, args.fold, dataset,
                                      device=torch.device(args.device))
        output = Path(trainer.output_folder)
        checkpoints = list(output.glob("checkpoint*.pth"))
        if args.resume:
            if (output / "checkpoint_final.pth").is_file():
                metadata = torch.load(output / "checkpoint_final.pth", map_location="cpu", weights_only=False)
                if metadata.get("comparison_recipe") != trainer.comparison_recipe():
                    raise ValueError("Completed checkpoint recipe differs from this launch")
                print("Training already completed:", output, flush=True)
                return
            checkpoint = next((output / name for name in ("checkpoint_latest.pth", "checkpoint_best.pth")
                               if (output / name).is_file()), None)
            if checkpoint is None:
                raise FileNotFoundError("--resume requires a checkpoint from this experiment")
            if checkpoint.resolve().parent != output.resolve():
                raise ValueError("Refusing a continuation checkpoint outside this experiment")
            print(f"Resuming {args.method} from {checkpoint}", flush=True)
            trainer.load_checkpoint(str(checkpoint))
        elif checkpoints:
            raise FileExistsError(f"Existing checkpoint(s) in {output}; choose another results root or --resume")
        trainer.run_training()
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    preview = recipe_preview(args)
    print(json.dumps(preview, indent=2), flush=True)
    if not args.run:
        return
    # Set paths before importing nnunetv2.paths (which caches environment values).
    for key in list(os.environ):
        if key.startswith(("NNUNET_STRUCTCOND_", "NNUNET_COMPARISON_")):
            del os.environ[key]
    os.environ.update({"nnUNet_preprocessed": str(args.preprocessed_root.resolve()),
                       "nnUNet_results": str(args.results_root.resolve()),
                       "nnUNet_compile": "false", "NNUNET_PROBABILISTIC_OVERSAMPLING": "0",
                       "NNUNET_COMPARISON_BATCH_SIZE": str(args.batch_size),
                       "NNUNET_COMPARISON_EPOCHS": str(args.epochs),
                       "NNUNET_COMPARISON_TRAIN_ITERS": str(args.train_iters),
                       "NNUNET_COMPARISON_VAL_ITERS": str(args.val_iters),
                       "NNUNET_COMPARISON_LR": str(args.lr)})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    if args.device == "cuda" and torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError("Requested GPUs are unavailable; launch inside an allocated GPU job")
    import fcntl
    args.results_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.results_root / f".{args.dataset}_{args.method}_fold{args.fold}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.num_gpus == 1:
            worker(0, 1, args)
        else:
            from nnunetv2.run.run_training import find_free_network_port
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(find_free_network_port())
            torch.multiprocessing.spawn(worker, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)


if __name__ == "__main__":
    main()
