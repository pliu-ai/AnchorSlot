#!/usr/bin/env python3
"""Snapshot live Synapse best/latest files and submit fixed no-TTA evaluation.

Copies, not hard links, isolate evaluations from ongoing training writes.
The live checkpoints are deliberately not rechecked after snapshot creation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


TRAINER = "nnUNetTrainerSynapseMasked"
MODEL = f"Dataset210_Synapse13/{TRAINER}__nnUNetPlans__3d_fullres"
PROTOCOL_KEYS = ("cases", "organs", "patch", "step_fraction", "tta", "stitching", "native_export",
                 "splits_sha256", "plans_sha256", "evaluator_sha256")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def stable_copy(source, destination, attempts=3):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    for _ in range(attempts):
        before = stamp(source)
        shutil.copyfile(source, temporary)
        sha = digest(temporary)
        source_sha = digest(source)
        if before == stamp(source) and sha == source_sha:
            os.replace(temporary, destination)
            return {"sha256": sha, "source_stat_at_snapshot": list(before),
                    "snapshot_utc": datetime.now(timezone.utc).isoformat()}
        temporary.unlink()
    raise RuntimeError(f"Source changed while snapshotting: {source}")


def verify(manifest):
    for path, expected in manifest["locked_files"].items():
        if digest(path) != expected:
            raise ValueError(f"Locked input changed: {path}")
    source = Path(manifest["source_root"])
    for name, expected in read(manifest["source_hash_manifest"]).items():
        if name.endswith(".py") and digest(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def validate_report(report, item, manifest):
    meta = report["manifest"]
    if (meta["fold"], meta["epoch"], meta["checkpoint_sha256"], meta["mode"]) != (
            item["fold"], item["epoch"], item["checkpoint_sha256"], "masked"):
        raise ValueError("Checkpoint identity or fold mismatch")
    if {key: meta[key] for key in PROTOCOL_KEYS} != manifest["protocol"]:
        raise ValueError("Evaluation protocol mismatch")


def requested_runs(args):
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(fold not in (0, 1) for fold in args.folds):
        raise ValueError("Select unique folds from 0 and 1")
    if not args.policies or len(set(args.policies)) != len(args.policies) or any(p not in ("best", "latest") for p in args.policies):
        raise ValueError("Select unique checkpoint policies from best and latest")
    runs = {0: args.fold0_run or args.base_run, 1: args.continuation_run}
    return [(fold, runs[fold].resolve()) for fold in args.folds]


def evaluation_array(count):
    if type(count) is not int or count < 1:
        raise ValueError("At least one checkpoint must be selected")
    return "--array=0" if count == 1 else f"--array=0-{count - 1}%2"


def prepare(args):
    root, base = args.output.resolve(), args.base_run.resolve()
    if root.exists():
        raise FileExistsError("Refusing to overwrite or duplicate a snapshot evaluation")
    reference_path = base / "evaluation/masked/final/summary.json"
    reference = read(reference_path)
    protocol = {key: reference["manifest"][key] for key in PROTOCOL_KEYS}
    if protocol["tta"] or not protocol["native_export"]:
        raise ValueError("Expected no-TTA preprocessed and original-space evaluation")
    source = Path(reference["manifest"]["code_root"])
    locked = [reference_path, base / "source_sha256.json"]
    contracts = []
    for fold, run in requested_runs(args):
        folder = run / "models" / MODEL / f"fold_{fold}"
        path = folder / "experiment_contract.json"
        contract = read(path)
        if (contract["fold"], contract["trainer"], contract["arm"], contract["code_root"]) != (
                fold, TRAINER, "masked", str(source)):
            raise ValueError("Unexpected training contract")
        if contract["validation_cases"] != protocol["cases"] or contract["patch"] != protocol["patch"]:
            raise ValueError("Training/evaluation protocol mismatch")
        contracts.append((fold, folder, contract))
        locked.append(path)
    pp = args.data_root / "nnUNet_preprocessed/Dataset210_Synapse13"
    for name, key in (("splits_final.json", "splits_sha256"), ("nnUNetPlans.json", "plans_sha256")):
        if digest(pp / name) != protocol[key]:
            raise ValueError(f"Dataset configuration changed: {name}")
        locked.append(pp / name)
    for case in protocol["cases"]:
        for suffix in (".npy", "_seg.npy", ".pkl"):
            if not (pp / "nnUNetPlans_3d_fullres" / (case + suffix)).is_file():
                raise FileNotFoundError(case + suffix)
    locked.append(args.data_root / "nnUNet_raw/Dataset210_Synapse13/dataset.json")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(source),
                "source_hash_manifest": str(base / "source_sha256.json"),
                "data_root": str(args.data_root.resolve()), "protocol": protocol,
                "fold1_epoch1000_reference": str(reference_path), "sources": [], "jobs": {},
                "locked_files": {str(path): digest(path) for path in locked},
                "caveat": "Same six development validation cases, not independent test data. Folds 0 and 1 "
                          "use 24 and 8 training scans respectively; do not pool them as independent CV folds. "
                          "Latest means the latest saved periodic checkpoint at snapshot time, not in-memory weights."}
    verify(manifest)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "checkpoints").mkdir()
    import torch
    for fold, folder, contract in contracts:
        for policy in args.policies:
            name = f"fold{fold}_{policy}"
            origin = folder / f"checkpoint_{policy}.pth"
            checkpoint = root / "checkpoints" / f"{name}.pth"
            copied = stable_copy(origin, checkpoint)
            blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (blob["trainer_name"], int(blob["init_args"]["fold"]), blob["init_args"]["configuration"]) != (
                    TRAINER, fold, "3d_fullres"):
                raise ValueError(f"Wrong checkpoint trainer/fold/configuration: {origin}")
            epoch = int(blob["current_epoch"])
            if not 0 < epoch <= contract["epochs"] or not blob["network_weights"]:
                raise ValueError("Invalid checkpoint epoch or missing network")
            del blob
            item = {"name": name, "fold": fold, "policy": policy, "epoch": epoch,
                    "training_scans": len(contract["train_cases"]), "source_checkpoint": str(origin),
                    "checkpoint": str(checkpoint), "checkpoint_sha256": copied["sha256"], **copied}
            manifest["sources"].append(item)
            manifest["locked_files"][str(checkpoint)] = copied["sha256"]
    repo = Path(__file__).resolve().parents[1]
    for origin, name in ((Path(__file__), "runner.py"),
                         (repo / "scripts/slurm/synapse_checkpoint_snapshot.sbatch", "job.sbatch")):
        shutil.copy2(origin, root / name)
        manifest["locked_files"][str(root / name)] = digest(root / name)
    save(root / "submission.json", manifest)
    if args.submit:
        def submit(phase, extra):
            command = ["sbatch", "--parsable", "--export=ALL", f"--job-name=as_syn_snap_{phase}",
                       f"--output={root}/logs/%x_%A_%a.out", f"--error={root}/logs/%x_%A_%a.err",
                       *extra, str(root / "job.sbatch")]
            job = subprocess.check_output(command, text=True,
                                          env=dict(os.environ, SNAPSHOT_DIR=str(root), PHASE=phase)).strip().split(";")[0]
            manifest["jobs"][phase] = {"job_id": job, "command": command}
            save(root / "submission.json", manifest)
            return job
        job = submit("evaluate", [evaluation_array(len(manifest["sources"]))])
        submit("collect", ["--gres=none", "--mem=4G", "--cpus-per-task=1", "--time=00:10:00",
                           f"--dependency=afterany:{job}"])
    print(json.dumps({"output": str(root), "checkpoints": manifest["sources"], "jobs": manifest["jobs"]}, indent=2))


def collect(args, manifest):
    rows, missing = [], []
    for item in manifest["sources"]:
        folder = args.output / "evaluation" / item["name"]
        path = folder / "summary.json"
        if not path.exists():
            missing.append(item["name"])
            continue
        report = read(path)
        validate_report(report, item, manifest)
        scores = report["metrics"]["masked"]
        native = read(folder / "masked/summary_native.json")["foreground_mean"]
        rows.append({key: item[key] for key in ("name", "fold", "policy", "epoch", "training_scans")} |
                    {"preprocessed_dice": scores["mean_dice"], "preprocessed_iou": scores["mean_iou"],
                     "native_dice": native["Dice"], "native_iou": native["IoU"], "summary": str(path)})
    parent = read(manifest["fold1_epoch1000_reference"])
    reference = parent["metrics"]["masked"]
    for row in rows:
        if row["fold"] == 1:
            row["native_dice_minus_fold1_epoch1000_final"] = row["native_dice"] - reference["native_mean_dice"]
    result = {"complete": not missing, "missing": missing, "rows": rows, "protocol": manifest["protocol"],
              "fold1_epoch1000_reference": reference, "caveat": manifest["caveat"]}
    save(args.output / "comparison_no_tta.json", result)
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "evaluate", "collect"), default="prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, default=Path("/projects/weilab/liupeng/runs/Anchorslot/synapse_consistency_20260906_v1"))
    parser.add_argument("--continuation-run", type=Path, default=Path("/projects/weilab/liupeng/runs/Anchorslot/synapse_masked_fold1_2000ep_tail_20260908"))
    parser.add_argument("--fold0-run", type=Path, help="Fold-0 model run; defaults to --base-run (which supplies the frozen evaluator)")
    parser.add_argument("--folds", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--policies", nargs="+", choices=("best", "latest"), default=["best", "latest"])
    parser.add_argument("--data-root", type=Path, default=Path("/projects/weilab/liupeng/data/nnUNet"))
    parser.add_argument("--index", type=int, choices=range(4), default=0)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
        return
    manifest = read(args.output / "submission.json")
    verify(manifest)
    if args.phase == "collect":
        collect(args, manifest)
        return
    source, data = Path(manifest["source_root"]), Path(manifest["data_root"])
    env = {key: value for key, value in os.environ.items() if not key.startswith("NNUNET_STRUCTCOND_")}
    env.update({"PYTHONPATH": str(source), "nnUNet_raw": str(data / "nnUNet_raw"),
                "nnUNet_preprocessed": str(data / "nnUNet_preprocessed"),
                "nnUNet_results": str(args.output / "temporary_trainer_metadata"),
                "nnUNet_compile": "f", "NNUNET_USE_WANDB": "0", "NNUNET_STRUCTCOND_COMPILE": "0",
                "NNUNET_STRUCTCOND_AMP_DTYPE": "bf16", "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"})
    if not 0 <= args.index < len(manifest["sources"]):
        raise ValueError("Evaluation index is outside the selected checkpoint list")
    item = manifest["sources"][args.index]
    output = args.output / "evaluation" / item["name"]
    subprocess.run([sys.executable, str(source / "scripts/evaluate_synapse_consistency.py"),
                    "--checkpoint", item["checkpoint"], "--fold", str(item["fold"]), "--mode", "masked",
                    "--data-root", str(data), "--output", str(output), "--native"], env=env, check=True)
    validate_report(read(output / "summary.json"), item, manifest)


if __name__ == "__main__":
    main()
