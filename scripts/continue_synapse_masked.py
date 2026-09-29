#!/usr/bin/env python3
"""Smoke-gated, separate low-LR continuation of the corrected Synapse model.

The parent experiment and its immutable nnU-Net source are never modified.
Global epochs, optimizer momentum, logging and best-EMA selection are restored;
only the explicitly recorded tail LR schedule and stopping epoch change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


TRAINER = "nnUNetTrainerSynapseMasked"


def model_subdir(fold):
    if type(fold) is not int or fold < 0:
        raise ValueError("Fold must be a nonnegative integer")
    return f"Dataset210_Synapse13/{TRAINER}__nnUNetPlans__3d_fullres/fold_{fold}"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class TailLR:
    """Stateless-in-global-epoch polynomial tail; no negative fractional powers."""

    def __init__(self, optimizer, start_epoch, end_epoch, start_lr, end_lr, exponent=0.9):
        if not 0 <= start_epoch < end_epoch:
            raise ValueError("Continuation must add epochs")
        if not all(math.isfinite(x) for x in (start_lr, end_lr, exponent)):
            raise ValueError("Schedule parameters must be finite")
        if not 0 < end_lr <= start_lr or exponent <= 0:
            raise ValueError("Expected positive nonincreasing LR and positive exponent")
        self.optimizer = optimizer
        self.start_epoch, self.end_epoch = start_epoch, end_epoch
        self.start_lr, self.end_lr, self.exponent = start_lr, end_lr, exponent

    def value(self, epoch):
        if not math.isfinite(epoch) or epoch < self.start_epoch:
            raise ValueError("Epoch precedes this continuation")
        fraction = min(1.0, (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch))
        return self.end_lr + (self.start_lr - self.end_lr) * (1.0 - fraction) ** self.exponent

    def step(self, epoch):
        value = self.value(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value


def verify(manifest):
    for path, expected in manifest["locked_files"].items():
        if digest(path) != expected:
            raise ValueError(f"Locked input changed: {path}")
    source = Path(manifest["source_root"])
    for name, expected in read(manifest["source_hash_manifest"]).items():
        if name.endswith(".py") and digest(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def prepare(args):
    root, parent = args.run_root.resolve(), args.parent_run.resolve()
    if root.exists():
        raise FileExistsError("Refusing to overwrite or duplicate a continuation")
    fold = args.fold
    model = model_subdir(fold)
    contract_path = parent / "models" / model / "experiment_contract.json"
    reference = read(contract_path)
    if (reference["arm"], reference["fold"], reference["epochs"], reference["trainer"]) != (
            "masked", fold, 1000, TRAINER):
        raise ValueError(f"Expected the completed corrected-masking fold-{fold} experiment")
    schedule = dict(start_epoch=1000, end_epoch=args.epochs, start_lr=args.tail_lr,
                    end_lr=args.end_lr, exponent=0.9)
    TailLR(None, **schedule)
    source = Path(reference["code_root"])
    locked = [contract_path, parent / "source_sha256.json"]
    checkpoints, reports = {}, {}
    evaluation_root = parent / ("evaluation" if fold == 1 else f"evaluation_fold{fold}")
    for name in ("final", "best"):
        checkpoint = parent / "models" / model / f"checkpoint_{name}.pth"
        report_path = evaluation_root / f"masked/{name}/summary.json"
        report = read(report_path)
        meta = report["manifest"]
        if (meta["fold"], meta["mode"], meta["tta"], meta["native_export"]) != (fold, "masked", False, True):
            raise ValueError("Unexpected parent evaluation protocol")
        if meta["trainer_name"] != TRAINER or meta["cases"] != reference["validation_cases"]:
            raise ValueError("Parent evaluation trainer or validation cases differ")
        for key in ("splits_sha256", "plans_sha256"):
            if meta[key] != reference[key]:
                raise ValueError(f"Parent evaluation dataset configuration differs: {key}")
        if meta["evaluator_sha256"] != digest(source / "scripts/evaluate_synapse_consistency.py"):
            raise ValueError("Parent evaluation used a different evaluator")
        if name == "final" and meta["epoch"] != schedule["start_epoch"]:
            raise ValueError("Wrong parent endpoint")
        if digest(checkpoint) != meta["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint no longer matches its evaluation")
        checkpoints[name], reports[name] = str(checkpoint), str(report_path)
        locked.extend((checkpoint, report_path, report_path.parent / "masked/summary_native.json"))
    pp = args.data_root / "nnUNet_preprocessed/Dataset210_Synapse13"
    for name, key in (("splits_final.json", "splits_sha256"), ("nnUNetPlans.json", "plans_sha256")):
        if digest(pp / name) != reference[key]:
            raise ValueError(f"Parent dataset configuration changed: {name}")
        locked.append(pp / name)
    split = read(pp / "splits_final.json")[fold]
    if split["train"] != reference["train_cases"] or split["val"] != reference["validation_cases"]:
        raise ValueError("Training or validation cases changed")
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Training/validation overlap")
    for case in split["train"] + split["val"]:
        for suffix in (".npy", "_seg.npy", ".pkl"):
            if not (pp / "nnUNetPlans_3d_fullres" / (case + suffix)).is_file():
                raise FileNotFoundError(case + suffix)
    locked.append(args.data_root / "nnUNet_raw/Dataset210_Synapse13/dataset.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(root),
        "parent_run": str(parent), "fold": fold, "source_root": str(source), "data_root": str(args.data_root.resolve()),
        "source_hash_manifest": str(parent / "source_sha256.json"), "reference_contract": reference,
        "parent_checkpoints": checkpoints, "parent_evaluations": reports, "schedule": schedule,
        "additional_epochs": args.epochs - 1000, "additional_updates": (args.epochs - 1000) * reference["iterations_per_epoch"],
        "initialization": "resume final network, optimizer momentum, logger, epoch and best EMA",
        "checkpoint_policy": "final primary; best patch-EMA across parent and tail secondary (parent best copied)",
        "snapshots": "latest every 25 epochs; retained snapshots every 250 completed tail epochs",
        "caveat": "Low-LR continuation, not a fresh 2000-epoch polynomial run. Augmentation RNG is reseeded, "
                  "so continuation is not bitwise replay. Same six development cases; no equal-budget baseline claim.",
        "locked_files": {str(path): digest(path) for path in locked}, "jobs": {},
    }
    verify(manifest)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    repo = Path(__file__).resolve().parents[1]
    for src, name in ((Path(__file__), "runner.py"),
                      (repo / "scripts/slurm/synapse_masked_continuation.sbatch", "job.sbatch"),
                      (repo / "scripts/tests/test_synapse_continuation.py", "test_continuation.py")):
        shutil.copy2(src, root / name)
        manifest["locked_files"][str(root / name)] = digest(root / name)
    save(root / "submission.json", manifest)
    if args.submit:
        def submit(phase, extra):
            command = ["sbatch", "--parsable", "--export=ALL", f"--job-name=as_mask_f{fold}_{args.epochs}ep_{phase}",
                       f"--output={root}/logs/%x_%j.out", f"--error={root}/logs/%x_%j.err",
                       *extra, str(root / "job.sbatch")]
            job = subprocess.check_output(command, text=True,
                                          env=dict(os.environ, RUN_ROOT=str(root), PHASE=phase)).strip().split(";")[0]
            manifest["jobs"][phase] = {"job_id": job, "command": command}
            save(root / "submission.json", manifest)
            return job
        smoke = submit("smoke", ["--partition=short", "--time=01:00:00"])
        train = submit("train", ["--partition=medium", "--time=2-00:00:00", f"--dependency=afterok:{smoke}"])
        submit("evaluate", ["--partition=short", "--time=06:00:00", f"--dependency=afterok:{train}"])
    print(json.dumps({"run_root": str(root), "schedule": schedule, "jobs": manifest["jobs"]}, indent=2))


def configure_environment(manifest, smoke=False):
    for key in list(os.environ):
        if key.startswith("NNUNET_STRUCTCOND_"):
            del os.environ[key]
    os.environ.update(manifest["reference_contract"]["runtime_environment"])
    source, root, data = (Path(manifest[key]) for key in ("source_root", "run_root", "data_root"))
    os.environ.update({"PYTHONPATH": str(source), "nnUNet_raw": str(data / "nnUNet_raw"),
                       "nnUNet_preprocessed": str(data / "nnUNet_preprocessed"),
                       "nnUNet_results": str(root / ("smoke_models" if smoke else "models")),
                       "nnUNet_compile": "f", "NNUNET_USE_WANDB": "0", "nnUNet_n_proc_DA": "0" if smoke else "4",
                       "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
                       "NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS": "0" if smoke else "1",
                       "NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_FROM_EPOCH": "1250",
                       "NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_EVERY": "250"})
    sys.path[:0] = [str(source), str(source / "scripts")]


def assert_state_equal(actual, expected):
    import numpy as np
    import torch
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("State keys differ")
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError("State lengths differ")
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    elif isinstance(expected, (float, np.floating)) and math.isnan(expected):
        if not math.isnan(actual):
            raise AssertionError("Restored NaN differs")
    elif actual != expected:
        raise AssertionError("Restored state differs")


def make_restored_trainer(manifest, resume=False):
    import torch
    from run_synapse_consistency import make_trainer
    ref, schedule = manifest["reference_contract"], manifest["schedule"]
    args = argparse.Namespace(seed=ref["seed"], epochs=schedule["end_epoch"], anchor_lr=ref["initial_lr"],
                              val_iterations=ref["val_iterations"], iterations=ref["iterations_per_epoch"],
                              data_root=Path(manifest["data_root"]), fold=ref["fold"])
    trainer = make_trainer(args, "masked")
    if trainer.reset_optimizer_on_load or trainer.freeze_conditioning_on_load:
        raise ValueError("Continuation must restore the optimizer without freezing")
    checkpoint = (str(Path(trainer.output_folder) / "checkpoint_latest.pth") if resume
                  else manifest["parent_checkpoints"]["final"])
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob["trainer_name"] != TRAINER or int(blob["init_args"]["fold"]) != ref["fold"]:
        raise ValueError("Wrong source trainer or fold")
    epoch = int(blob["current_epoch"])
    if not schedule["start_epoch"] <= epoch < schedule["end_epoch"]:
        raise ValueError("Checkpoint epoch outside continuation interval")
    if not resume and epoch != schedule["start_epoch"]:
        raise ValueError("Fresh continuation must start at the parent final epoch")
    if not blob["optimizer_state"]["state"]:
        raise ValueError("Parent optimizer state is empty")
    trainer.load_checkpoint(blob)
    assert_state_equal(trainer.network.state_dict(), blob["network_weights"])
    assert_state_equal(trainer.optimizer.state_dict(), blob["optimizer_state"])
    assert_state_equal(trainer.logger.get_checkpoint(), blob["logging"])
    assert trainer.current_epoch == epoch and trainer._best_ema == blob["_best_ema"]
    if len(trainer.logger.my_fantastic_logging["train_losses"]) != epoch:
        raise ValueError("Checkpoint logger is not aligned with its epoch")
    previous_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.lr_scheduler = TailLR(trainer.optimizer, **schedule)
    current_lr = trainer.lr_scheduler.step(epoch)
    report = {"checkpoint": checkpoint, "checkpoint_sha256": digest(checkpoint), "restored_epoch": epoch,
              "target_epoch": trainer.num_epochs, "network_exact": True, "optimizer_exact_before_lr_change": True,
              "logger_exact": True, "inherited_best_ema": trainer._best_ema,
              "checkpoint_lr": previous_lr, "continuation_lr": current_lr, "schedule": schedule}
    assert trainer.num_epochs == schedule["end_epoch"]
    print(json.dumps(report, indent=2), flush=True)
    return trainer, report


def smoke(manifest):
    import torch
    from run_synapse_consistency import finish_loaders
    trainer, restored = make_restored_trainer(manifest)
    try:
        trainer.dataloader_train, trainer.dataloader_val = trainer.get_dataloaders()
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()
        before = next(trainer.network.parameters()).detach().clone()
        outputs = [trainer.train_step(next(trainer.dataloader_train)) for _ in range(2)]
        losses = [float(value["loss"]) for value in outputs]
        if not all(math.isfinite(x) for x in losses) or torch.equal(before, next(trainer.network.parameters()).detach()):
            raise AssertionError("Smoke training is nonfinite or did not update weights")
        trainer.on_train_epoch_end(outputs)
        trainer.on_validation_epoch_start()
        with torch.no_grad():
            validation = trainer.validation_step(next(trainer.dataloader_val))
            val = float(validation["loss"])
            trainer.on_validation_epoch_end([validation])
        if not math.isfinite(val):
            raise AssertionError("Nonfinite smoke validation")
        trainer.on_epoch_end()
        path = Path(trainer.output_folder) / "checkpoint_smoke.pth"
        # save_checkpoint stores current_epoch + 1; mirror on_train_end's convention.
        trainer.current_epoch -= 1
        trainer.save_checkpoint(str(path))
        trainer.current_epoch += 1
        saved = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            next(trainer.network.parameters()).add_(1)
        trainer.load_checkpoint(saved)
        assert_state_equal(trainer.network.state_dict(), saved["network_weights"])
        assert_state_equal(trainer.optimizer.state_dict(), saved["optimizer_state"])
        assert_state_equal(trainer.logger.get_checkpoint(), saved["logging"])
        assert trainer.current_epoch == manifest["schedule"]["start_epoch"] + 1
        assert len(trainer.logger.my_fantastic_logging["train_losses"]) == trainer.current_epoch
        trainer.lr_scheduler = TailLR(trainer.optimizer, **manifest["schedule"])
        resumed_lr = trainer.lr_scheduler.step(trainer.current_epoch)
        report = {"passed": True, "restoration": restored, "train_losses": losses, "val_loss": val,
                  "smoke_checkpoint_reload_exact": True, "reloaded_global_epoch": trainer.current_epoch,
                  "reloaded_schedule_lr": resumed_lr, "runner_sha256": digest(__file__)}
        save(Path(manifest["run_root"]) / "smoke_passed.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        finish_loaders(trainer)


def train(manifest, resume):
    from run_synapse_consistency import contract, finish_loaders
    root = Path(manifest["run_root"])
    gate = read(root / "smoke_passed.json")
    if not gate["passed"] or gate["runner_sha256"] != digest(__file__):
        raise ValueError("This continuation runner has not passed the GPU smoke test")
    folder = root / "models" / model_subdir(manifest["reference_contract"]["fold"])
    if (folder / "checkpoint_final.pth").exists():
        raise FileExistsError("Continuation is already complete")
    if folder.exists() and not resume:
        raise FileExistsError("Continuation already started; explicitly --resume latest")
    if resume and not (folder / "experiment_contract.json").is_file():
        raise FileNotFoundError("Cannot resume without a continuation contract")
    trainer, restored = make_restored_trainer(manifest, resume)
    ref = manifest["reference_contract"]
    args = argparse.Namespace(fold=ref["fold"], seed=ref["seed"], epochs=manifest["schedule"]["end_epoch"],
                              iterations=ref["iterations_per_epoch"], val_iterations=ref["val_iterations"],
                              data_root=Path(manifest["data_root"]))
    expected = contract(args, "masked", trainer)
    expected["continuation"] = {key: manifest[key] for key in ("parent_checkpoints", "schedule", "checkpoint_policy", "caveat")}
    path = folder / "experiment_contract.json"
    if resume:
        if read(path) != expected:
            raise ValueError("Continuation contract changed")
    else:
        save(path, expected)
        shutil.copy2(manifest["parent_checkpoints"]["best"], folder / "checkpoint_best.pth")
    save(folder / f"restore_epoch_{restored['restored_epoch']}.json", restored)
    try:
        trainer.run_training()
    finally:
        finish_loaders(trainer)


def evaluate(manifest):
    root, source = Path(manifest["run_root"]), Path(manifest["source_root"])
    fold = manifest["reference_contract"]["fold"]
    rows = []
    for name in ("final", "best"):
        checkpoint = root / "models" / model_subdir(fold) / f"checkpoint_{name}.pth"
        output = root / f"evaluation/masked/{name}"
        subprocess.run([sys.executable, str(source / "scripts/evaluate_synapse_consistency.py"),
                        "--checkpoint", str(checkpoint), "--fold", str(fold), "--data-root", manifest["data_root"],
                        "--output", str(output), "--mode", "masked", "--native"], check=True)
        report = read(output / "summary.json")
        parent = read(manifest["parent_evaluations"][name])
        keys = ("cases", "organs", "fold", "patch", "step_fraction", "tta", "stitching",
                "native_export", "splits_sha256", "plans_sha256", "evaluator_sha256")
        if any(report["manifest"][k] != parent["manifest"][k] for k in keys):
            raise ValueError("Parent/continuation evaluation protocols differ")
        if report["manifest"]["checkpoint_sha256"] != digest(checkpoint):
            raise ValueError("Evaluation/checkpoint mismatch")
        if name == "final" and report["manifest"]["epoch"] != manifest["schedule"]["end_epoch"]:
            raise ValueError("Final evaluation is not at the target epoch")
        scores, original = report["metrics"]["masked"], parent["metrics"]["masked"]
        rows.append({"checkpoint_policy": name, "epoch": report["manifest"]["epoch"], "metrics": scores,
                     "parent_epoch": parent["manifest"]["epoch"], "parent_metrics": original,
                     "delta_native_dice": scores["native_mean_dice"] - original["native_mean_dice"],
                     "summary": str(output / "summary.json")})
    save(root / "comparison_no_tta.json", {"complete": True, "rows": rows, "caveat": manifest["caveat"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "smoke", "train", "evaluate"), default="prepare")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, default=Path("/projects/weilab/liupeng/runs/Anchorslot/synapse_consistency_20260906_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("/projects/weilab/liupeng/data/nnUNet"))
    parser.add_argument("--fold", type=int, default=1, help="Parent fold for prepare; later phases use the locked manifest")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--tail-lr", type=float, default=2e-5)
    parser.add_argument("--end-lr", type=float, default=1e-6)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
        return
    manifest = read(args.run_root / "submission.json")
    verify(manifest)
    configure_environment(manifest, args.phase == "smoke")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Submit this phase to a GPU node")
    torch.set_num_threads(4)
    if args.phase == "smoke":
        subprocess.run([sys.executable, "-m", "pytest", str(args.run_root / "test_continuation.py"), "-q"], check=True)
        smoke(manifest)
    elif args.phase == "train":
        train(manifest, args.resume)
    else:
        evaluate(manifest)


if __name__ == "__main__":
    main()
