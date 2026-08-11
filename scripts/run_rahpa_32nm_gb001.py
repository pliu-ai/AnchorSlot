#!/usr/bin/env python3
"""Run resumable RA-HPA 32-nm inference and official evaluation on gb001.

The low-resolution holdout volumes are too large for the short Slurm jobs used
by ``predict_and_evaluate_resolution_adaptive_hpa.sbatch``. This launcher runs
each case in an isolated directory, optionally runs several cases on separate
GPUs, promotes completed predictions into the canonical prediction directory,
and evaluates only after all expected predictions are present.

Typical usage on gb001::

    /projects/weilab/liupeng/conda/envs/ssl_seg/bin/python \
        scripts/run_rahpa_32nm_gb001.py run --gpus 0,1

The command is safe to restart. Existing non-empty predictions are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path("/projects/weilab/liupeng/code/projects/AnchorSlot")
NNUNET_DATA_ROOT = Path("/projects/weilab/liupeng/data/nnUNet")
RUN_ROOT = Path(
    "/projects/weilab/liupeng/runs/Anchorslot/"
    "resolution_adaptive_hierarchical_parallel_anchorslot_exp01"
)
TRUTH_ROOT = Path("/projects/weilab/liupeng/runs/Anchorslot/evaluation_protocol")
CSC_REPO = Path(
    "/projects/weilab/liupeng/code/projects/"
    "cellmap-segmentation-challenge-upstream-0300239"
)
CSC_PYTHON = Path("/projects/weilab/liupeng/conda/envs/csc/bin/python")
EXPECTED_EVALUATOR_COMMIT = "0300239cd0b4867d4bab008aa9e95161b2442d93"

DATASET = "Dataset201_Dataset101_CellMap_model_low_res"
PREPROCESSING_CONFIGURATION = "3d_fullres"
MODEL_FOLDER = NNUNET_DATA_ROOT / (
    "nnUNet_results/Dataset200_Dataset101_CellMap_model_high_res/"
    "nnUNetTrainerResolutionAdaptiveHierarchicalParallelAnchorSlot__"
    "nnUNetPlans__3d_lowres_large_patch"
)


@dataclass(frozen=True)
class Case:
    source: Path
    output_name: str

    @property
    def case_id(self) -> str:
        return self.output_name.removesuffix(".nii.gz")


def _input_to_output_name(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected an nnUNet input ending in {suffix!r}: {path}")
    return path.name.removesuffix(suffix) + ".nii.gz"


def _discover_cases(input_dir: Path) -> list[Case]:
    cases = [
        Case(source=path.resolve(), output_name=_input_to_output_name(path))
        for path in input_dir.glob("*_0000.nii.gz")
    ]
    if not cases:
        raise RuntimeError(f"No *_0000.nii.gz inputs found in {input_dir}")
    # Start larger compressed inputs first to reduce the total parallel runtime.
    return sorted(cases, key=lambda case: (-case.source.stat().st_size, case.source.name))


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU index")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus contains duplicate GPU indices")
    return gpus


def _base_environment(repo_root: Path, data_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}:{current_pythonpath}" if current_pythonpath else str(repo_root)
    )
    env["nnUNet_raw"] = str(data_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(data_root / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(data_root / "nnUNet_results")
    return env


def _prepare_isolated_input(case: Case, work_root: Path) -> Path:
    input_dir = work_root / "inputs" / case.case_id
    input_dir.mkdir(parents=True, exist_ok=True)
    link = input_dir / case.source.name
    if link.exists() or link.is_symlink():
        if link.resolve() != case.source:
            raise RuntimeError(f"Unexpected existing input link: {link}")
    else:
        link.symlink_to(case.source)
    return input_dir


def _promote_prediction(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Prediction was not created or is empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == 0:
            raise RuntimeError(f"Refusing to replace empty existing output: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prediction_command(
    *,
    python: Path,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    model_folder: Path,
    checkpoint: str,
    preprocessing_dataset_dir: Path,
    preprocessing_configuration: str,
) -> list[str]:
    return [
        str(python),
        str(repo_root / "scripts/predict_structured_conditional.py"),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-m",
        str(model_folder),
        "-f",
        "all",
        "-chk",
        checkpoint,
        "-device",
        "cuda",
        "--group_id",
        "0",
        "--output_mode",
        "original",
        "--voxel_size_nm",
        "32",
        "--preprocessing_dataset_dir",
        str(preprocessing_dataset_dir),
        "--preprocessing_configuration",
        preprocessing_configuration,
        "--continue_prediction",
        "--disable_progress_bar",
        "-npp",
        "1",
        "-nps",
        "1",
    ]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _completed(case: Case, predictions: Path) -> bool:
    output = predictions / case.output_name
    return output.is_file() and output.stat().st_size > 0


def predict(args: argparse.Namespace, cases: Sequence[Case]) -> None:
    predictions = args.run_root / "32nm/predictions"
    work_root = args.run_root / "32nm/gb001_work"
    log_root = work_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    preprocessing_dataset_dir = (
        args.nnunet_data_root / f"nnUNet_preprocessed/{DATASET}"
    )

    pending = []
    for case in cases:
        if _completed(case, predictions):
            print(f"[skip] {case.case_id}: canonical prediction already exists", flush=True)
            continue
        isolated_output = work_root / "outputs" / case.case_id
        isolated_prediction = isolated_output / case.output_name
        if isolated_prediction.is_file() and isolated_prediction.stat().st_size > 0:
            _promote_prediction(isolated_prediction, predictions / case.output_name)
            print(f"[recover] {case.case_id}: promoted completed isolated output", flush=True)
            continue
        pending.append(case)

    if not pending:
        print("All 32-nm predictions are already complete.", flush=True)
        return

    if args.dry_run:
        for index, case in enumerate(pending):
            gpu = args.gpus[index % len(args.gpus)]
            input_dir = work_root / "inputs" / case.case_id
            output_dir = work_root / "outputs" / case.case_id
            command = _prediction_command(
                python=args.inference_python,
                repo_root=args.repo_root,
                input_dir=input_dir,
                output_dir=output_dir,
                model_folder=args.model_folder,
                checkpoint=args.checkpoint,
                preprocessing_dataset_dir=preprocessing_dataset_dir,
                preprocessing_configuration=PREPROCESSING_CONFIGURATION,
            )
            print(f"[dry-run][gpu {gpu}] {' '.join(command)}")
        return

    env_base = _base_environment(args.repo_root, args.nnunet_data_root)
    active: dict[str, tuple[subprocess.Popen[bytes], Case, object, object]] = {}
    queue = list(pending)

    try:
        while queue or active:
            for gpu in args.gpus:
                if not queue or gpu in active:
                    continue
                case = queue.pop(0)
                input_dir = _prepare_isolated_input(case, work_root)
                output_dir = work_root / "outputs" / case.case_id
                output_dir.mkdir(parents=True, exist_ok=True)
                command = _prediction_command(
                    python=args.inference_python,
                    repo_root=args.repo_root,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    model_folder=args.model_folder,
                    checkpoint=args.checkpoint,
                    preprocessing_dataset_dir=preprocessing_dataset_dir,
                    preprocessing_configuration=PREPROCESSING_CONFIGURATION,
                )
                stdout_handle = (log_root / f"{case.case_id}.out").open("ab")
                stderr_handle = (log_root / f"{case.case_id}.err").open("ab")
                env = env_base.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                print(f"[start][gpu {gpu}] {case.case_id}", flush=True)
                process = subprocess.Popen(
                    command,
                    cwd=args.repo_root,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
                active[gpu] = (process, case, stdout_handle, stderr_handle)

            if not active:
                break

            time.sleep(args.poll_seconds)
            failures: list[str] = []
            for gpu, (process, case, stdout_handle, stderr_handle) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                stdout_handle.close()
                stderr_handle.close()
                del active[gpu]
                if returncode != 0:
                    failures.append(
                        f"{case.case_id} failed on GPU {gpu} with exit code {returncode}; "
                        f"see {log_root / (case.case_id + '.err')}"
                    )
                    continue
                isolated_prediction = work_root / "outputs" / case.case_id / case.output_name
                _promote_prediction(isolated_prediction, predictions / case.output_name)
                print(f"[done][gpu {gpu}] {case.case_id}", flush=True)

            if failures:
                raise RuntimeError("\n".join(failures))
    finally:
        for process, _, stdout_handle, stderr_handle in active.values():
            _terminate_process_group(process)
            stdout_handle.close()
            stderr_handle.close()

    missing = [case.output_name for case in cases if not _completed(case, predictions)]
    if missing:
        raise RuntimeError(f"Prediction run ended with missing outputs: {missing}")


def evaluate(args: argparse.Namespace, cases: Sequence[Case]) -> None:
    predictions = args.run_root / "32nm/predictions"
    missing = [case.output_name for case in cases if not _completed(case, predictions)]
    if missing:
        raise RuntimeError(
            "Official evaluation requires every prediction. Missing: " + ", ".join(missing)
        )

    output_dir = args.run_root / "32nm/official_evaluation"
    truth_lock = args.truth_root / "low_32nm/truth_lock.json"
    command = [
        str(args.csc_python),
        str(args.repo_root / "scripts/cellmap_evaluation_pipeline.py"),
        "score",
        "--predictions",
        str(predictions),
        "--truth-lock",
        str(truth_lock),
        "--output-dir",
        str(output_dir),
        "--csc-repo",
        str(args.csc_repo),
        "--evaluator-python",
        str(args.csc_python),
        "--expected-evaluator-commit",
        args.expected_evaluator_commit,
        "--model",
        "RA-HPA",
        "--resolution",
        "32nm",
        "--anchorslot-repo",
        str(args.repo_root),
        "--max-workers",
        "2",
        "--per-instance-threads",
        "1",
        "--resume",
    ]
    print("[evaluate] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(
            command,
            cwd=args.repo_root,
            env=_base_environment(args.repo_root, args.nnunet_data_root),
            check=True,
        )


def summarize(args: argparse.Namespace) -> None:
    command = [
        str(args.csc_python),
        str(args.repo_root / "scripts/cellmap_evaluation_pipeline.py"),
        "summarize",
        "--registry",
        str(args.repo_root / "configs/cellmap/evaluation_registry.tsv"),
        "--output-dir",
        str(args.run_root.parent / "evaluation_matrix/summary"),
    ]
    print("[summarize] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(
            command,
            cwd=args.repo_root,
            env=_base_environment(args.repo_root, args.nnunet_data_root),
            check=True,
        )


def status(args: argparse.Namespace, cases: Sequence[Case]) -> None:
    predictions = args.run_root / "32nm/predictions"
    print("32-nm RA-HPA prediction status:")
    for case in cases:
        output = predictions / case.output_name
        state = f"complete ({output.stat().st_size} bytes)" if _completed(case, predictions) else "missing"
        print(f"  {case.case_id}: {state}")

    metrics = args.run_root / "32nm/official_evaluation/latest_official_metrics.json"
    if not metrics.is_file():
        print(f"Official metrics: missing ({metrics})")
        return
    payload = json.loads(metrics.read_text())
    print(f"Official metrics: {metrics}")
    print(f"  semantic: {payload.get('overall_semantic_score')}")
    print(f"  instance: {payload.get('overall_instance_score')}")
    print(f"  overall: {payload.get('overall_score')}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "evaluate", "summarize", "run", "status"))
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("0,1"))
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--nnunet-data-root", type=Path, default=NNUNET_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--truth-root", type=Path, default=TRUTH_ROOT)
    parser.add_argument("--model-folder", type=Path, default=MODEL_FOLDER)
    parser.add_argument("--inference-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--csc-python", type=Path, default=CSC_PYTHON)
    parser.add_argument("--csc-repo", type=Path, default=CSC_REPO)
    parser.add_argument("--expected-evaluator-commit", default=EXPECTED_EVALUATOR_COMMIT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = args.nnunet_data_root / f"nnUNet_raw/{DATASET}/imagesTs"

    required = [
        args.repo_root / "scripts/predict_structured_conditional.py",
        args.repo_root / "scripts/cellmap_evaluation_pipeline.py",
        args.model_folder / "fold_all" / args.checkpoint,
        args.truth_root / "low_32nm/truth_lock.json",
        args.csc_repo,
        args.csc_python,
        args.inference_python,
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing_required))

    cases = _discover_cases(input_dir)
    if args.command == "status":
        status(args, cases)
    elif args.command == "predict":
        predict(args, cases)
    elif args.command == "evaluate":
        evaluate(args, cases)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "run":
        predict(args, cases)
        evaluate(args, cases)
        summarize(args)
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
