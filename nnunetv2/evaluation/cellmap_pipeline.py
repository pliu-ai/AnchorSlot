"""Reproducible, manifest-locked CellMap validation orchestration.

The submission exporter in :mod:`cellmap_challenge` owns the data boundary.
This module adds the experiment boundary around it: immutable truth locks,
the pinned official scorer, provenance records, and cross-resolution tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from nnunetv2.evaluation.cellmap_challenge import (
    INSTANCE_CLASSES,
    entries_by_crop,
    export_ground_truth,
    export_submission,
    manifest_sha256,
    read_manifest,
    validate_ground_truth_inputs,
    validate_prediction_inputs,
    validate_submission,
    write_annotated_validation_manifest,
    write_validation_manifest,
)


LOCK_SCHEMA = "anchorslot-cellmap-truth-lock-v1"
PROVENANCE_SCHEMA = "anchorslot-cellmap-evaluation-v1"
METRICS_NAME = "latest_official_metrics.json"
PROVENANCE_NAME = "evaluation_provenance.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: str | Path, value: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)
    return path


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _git_commit(repo: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_valid(report: Mapping[str, object], description: str) -> None:
    if report.get("status") != "valid":
        errors = report.get("errors", [])
        raise ValueError(f"{description} is invalid: {errors}")


def create_truth_lock(
    manifest: str | Path,
    truth_store: str | Path,
    validation_report: str | Path,
    output: str | Path,
    *,
    cohort: str,
    resolution_nm: float,
    full_geometry_check: bool = True,
) -> dict:
    """Lock a previously value-validated native-ID ground-truth store."""
    manifest = Path(manifest).resolve()
    truth_store = Path(truth_store).resolve()
    validation_report = Path(validation_report).resolve()
    report = _load_json(validation_report)
    _require_valid(report, "Ground-truth validation report")
    manifest_hash = manifest_sha256(manifest)
    if report.get("manifest_sha256") != manifest_hash:
        raise ValueError(
            "Ground-truth report was produced from a different manifest: "
            f"report={report.get('manifest_sha256')}, actual={manifest_hash}"
        )
    if _resolved(report.get("submission_path", "")) != str(truth_store):
        raise ValueError(
            "Ground-truth report points to a different store: "
            f"report={report.get('submission_path')}, requested={truth_store}"
        )

    if full_geometry_check:
        current = validate_submission(
            truth_store,
            manifest,
            role="ground_truth",
            check_values=False,
            check_hierarchy=False,
        )
        _require_valid(current, "Current ground-truth store geometry")

    entries = read_manifest(manifest)
    instance_arrays = sum(entry.class_label in INSTANCE_CLASSES for entry in entries)
    semantic_arrays = len(entries) - instance_arrays
    lock = {
        "schema": LOCK_SCHEMA,
        "created_utc": _now_utc(),
        "cohort": str(cohort),
        "resolution_nm": float(resolution_nm),
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "truth_store": str(truth_store),
        "truth_validation_report": str(validation_report),
        "truth_validation_report_sha256": _sha256(validation_report),
        "native_instance_ids_validated": True,
        "num_crops": len(entries_by_crop(entries)),
        "num_arrays": len(entries),
        "num_instance_arrays": instance_arrays,
        "num_semantic_arrays": semantic_arrays,
        "classes": sorted({entry.class_label for entry in entries}),
    }
    _write_json(output, lock)
    return lock


def verify_truth_lock(lock_path: str | Path) -> dict:
    """Verify that every file referenced by a truth lock is unchanged."""
    lock_path = Path(lock_path).resolve()
    lock = _load_json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"Unsupported truth-lock schema in {lock_path}")
    if lock.get("native_instance_ids_validated") is not True:
        raise ValueError("Truth lock does not certify native instance IDs")

    manifest = Path(str(lock["manifest"])).resolve()
    truth_store = Path(str(lock["truth_store"])).resolve()
    report_path = Path(str(lock["truth_validation_report"])).resolve()
    if manifest_sha256(manifest) != lock.get("manifest_sha256"):
        raise ValueError(f"Manifest changed after locking: {manifest}")
    if _sha256(report_path) != lock.get("truth_validation_report_sha256"):
        raise ValueError(f"Truth validation report changed after locking: {report_path}")

    report = _load_json(report_path)
    _require_valid(report, "Locked ground-truth validation report")
    if _resolved(report.get("submission_path", "")) != str(truth_store):
        raise ValueError("Locked validation report no longer points to the truth store")
    if report.get("manifest_sha256") != lock.get("manifest_sha256"):
        raise ValueError("Locked report and manifest hashes differ")

    # This is deliberately a geometry/hash check. The expensive value and
    # native-instance validation is certified by the hashed report above.
    current = validate_submission(
        truth_store,
        manifest,
        role="ground_truth",
        check_values=False,
        check_hierarchy=False,
    )
    _require_valid(current, "Locked ground-truth store geometry")
    if int(current["num_actual_arrays"]) != int(lock["num_arrays"]):
        raise ValueError("Truth store array count changed after locking")
    return lock


def prepare_truth(
    reference_volumes: str | Path,
    native_labels_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
    resolution_nm: float,
    classes: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """Build a filtered manifest, native-ID truth store, and immutable lock."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest = output_dir / "manifest_all48.csv"
    manifest = output_dir / "manifest_annotated.csv"
    truth_store = output_dir / "ground_truth.zarr"
    input_report_path = output_dir / "ground_truth_input_validation.json"
    validation_path = output_dir / "ground_truth_store_validation.json"
    lock_path = output_dir / "truth_lock.json"
    protected = (all_manifest, manifest, truth_store, validation_path, lock_path)
    if not overwrite and any(path.exists() for path in protected):
        raise FileExistsError(
            f"Truth output already exists under {output_dir}; pass --overwrite to rebuild"
        )

    write_validation_manifest(reference_volumes, all_manifest)
    write_annotated_validation_manifest(all_manifest, native_labels_root, manifest)
    if classes:
        requested = {str(value).strip() for value in classes}
        entries = [entry for entry in read_manifest(manifest) if entry.class_label in requested]
        missing = requested - {entry.class_label for entry in entries}
        if missing:
            raise ValueError(f"Requested classes are absent from annotated cohort: {sorted(missing)}")
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=("crop_name", "dataset", "class_label", "voxel_size", "translation", "shape"),
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "crop_name": entry.crop_name.removeprefix("crop"),
                        "dataset": entry.dataset,
                        "class_label": entry.class_label,
                        "voxel_size": "[" + ";".join(str(value) for value in entry.voxel_size) + "]",
                        "translation": "[" + ";".join(str(value) for value in entry.translation) + "]",
                        "shape": "[" + ";".join(str(value) for value in entry.shape) + "]",
                    }
                )
    input_report = validate_ground_truth_inputs(native_labels_root, manifest)
    _write_json(input_report_path, input_report)
    _require_valid(input_report, "Native ground-truth inputs")
    export_ground_truth(
        native_labels_root,
        manifest,
        truth_store,
        overwrite=overwrite,
    )
    validation = validate_submission(truth_store, manifest, role="ground_truth")
    _write_json(validation_path, validation)
    _require_valid(validation, "Exported ground-truth store")
    return create_truth_lock(
        manifest,
        truth_store,
        validation_path,
        lock_path,
        cohort=cohort,
        resolution_nm=resolution_nm,
    )


def _prediction_inventory(predictions: str | Path) -> list[dict]:
    root = Path(predictions).resolve()
    inventory = []
    for path in sorted(root.glob("*.nii.gz")):
        stat = path.stat()
        inventory.append(
            {
                "name": path.name,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return inventory


def _validate_metrics(metrics: Mapping[str, object], expected_evals: int, commit: str) -> None:
    if int(metrics.get("num_evals_done", -1)) != int(metrics.get("total_evals", -2)):
        raise ValueError("Official evaluator did not complete every requested array")
    if int(metrics.get("total_evals", -1)) != int(expected_evals):
        raise ValueError(
            f"Evaluator scored {metrics.get('total_evals')} arrays; expected {expected_evals}"
        )
    if metrics.get("git_version") != commit:
        raise ValueError(
            f"Evaluator reported git_version={metrics.get('git_version')}, expected {commit}"
        )
    for key in ("overall_semantic_score", "overall_instance_score", "overall_score"):
        value = metrics.get(key)
        # A cohort may contain only semantic or only instance annotations. The
        # absent side is allowed to be null/NaN, but the overall score must exist.
        if value is not None and not math.isfinite(float(value)):
            if key == "overall_score":
                raise ValueError("Official overall score is not finite")


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def score_predictions(
    predictions: str | Path,
    truth_lock: str | Path,
    output_dir: str | Path,
    csc_repo: str | Path,
    *,
    evaluator_python: str | Path,
    expected_evaluator_commit: str,
    model: str,
    resolution: str,
    anchorslot_repo: str | Path,
    max_workers: int = 4,
    per_instance_threads: int = 1,
    overwrite: bool = False,
    resume: bool = False,
) -> dict:
    """Export and score one prediction folder with the official evaluator."""
    predictions = Path(predictions).resolve()
    output_dir = Path(output_dir).resolve()
    csc_repo = Path(csc_repo).resolve()
    anchorslot_repo = Path(anchorslot_repo).resolve()
    evaluator_python = Path(evaluator_python).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_NAME
    provenance_path = output_dir / PROVENANCE_NAME
    if resume and metrics_path.is_file() and provenance_path.is_file():
        provenance = _load_json(provenance_path)
        lock = verify_truth_lock(truth_lock)
        metrics = _load_json(metrics_path)
        _validate_metrics(metrics, int(lock["num_arrays"]), expected_evaluator_commit)
        if provenance.get("schema") != PROVENANCE_SCHEMA or provenance.get("status") != "complete":
            raise ValueError(f"Cannot resume invalid provenance: {provenance_path}")
        if provenance.get("truth_lock_sha256") != _sha256(truth_lock):
            raise ValueError("Cannot resume because the truth lock changed")
        if provenance.get("prediction_inventory") != _prediction_inventory(predictions):
            raise ValueError("Cannot resume because the prediction folder changed")
        if provenance.get("evaluator_commit") != expected_evaluator_commit:
            raise ValueError("Cannot resume metrics produced by a different evaluator commit")
        return provenance
    if not overwrite and (metrics_path.exists() or (output_dir / "prediction.zarr").exists()):
        raise FileExistsError(
            f"Evaluation output exists under {output_dir}; use --overwrite or --resume"
        )

    lock = verify_truth_lock(truth_lock)
    manifest = Path(str(lock["manifest"]))
    truth_store = Path(str(lock["truth_store"]))
    evaluator_commit = _git_commit(csc_repo)
    if evaluator_commit != expected_evaluator_commit:
        raise ValueError(
            f"Evaluator checkout is {evaluator_commit}; expected {expected_evaluator_commit}"
        )

    input_report = validate_prediction_inputs(predictions, manifest)
    _write_json(output_dir / "prediction_input_validation.json", input_report)
    _require_valid(input_report, "Prediction inputs")
    export_report = export_submission(
        predictions,
        manifest,
        output_dir / "prediction.zarr",
        overwrite=overwrite,
    )
    _write_json(output_dir / "prediction_export.json", export_report)
    validation = validate_submission(output_dir / "prediction.zarr", manifest)
    _write_json(output_dir / "prediction_store_validation.json", validation)
    _require_valid(validation, "Prediction store")

    with tempfile.TemporaryDirectory(prefix="anchorslot-csc-") as temp_dir:
        evaluator_copy = Path(temp_dir) / "cellmap-segmentation-challenge"
        shutil.copytree(csc_repo, evaluator_copy, symlinks=True)
        environment = os.environ.copy()
        python_paths = [str(evaluator_copy / "src"), str(anchorslot_repo)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(python_paths),
                "CSC_TEST_CROP_MANIFEST_URL": str(manifest),
                "MAX_WORKERS": str(int(max_workers)),
                "PER_INSTANCE_THREADS": str(int(per_instance_threads)),
            }
        )
        subprocess.run(
            [
                str(evaluator_python),
                "-m",
                "cellmap_segmentation_challenge.evaluate",
                str(output_dir / "prediction.zarr"),
                str(metrics_path),
                "--truth-path",
                str(truth_store),
            ],
            check=True,
            cwd=evaluator_copy,
            env=environment,
        )

    metrics = _load_json(metrics_path)
    _validate_metrics(metrics, int(lock["num_arrays"]), evaluator_commit)
    repository_commit = _git_commit(anchorslot_repo)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "completed_utc": _now_utc(),
        "status": "complete",
        "model": str(model),
        "resolution": str(resolution),
        "resolution_nm": lock["resolution_nm"],
        "cohort": lock["cohort"],
        "predictions": str(predictions),
        "prediction_inventory": _prediction_inventory(predictions),
        "prediction_store": str(output_dir / "prediction.zarr"),
        "truth_lock": _resolved(truth_lock),
        "truth_lock_sha256": _sha256(truth_lock),
        "manifest": str(manifest),
        "manifest_sha256": lock["manifest_sha256"],
        "truth_store": str(truth_store),
        "native_instance_ids_validated": True,
        "evaluator_repo": str(csc_repo),
        "evaluator_commit": evaluator_commit,
        "anchorslot_repo": str(anchorslot_repo),
        "anchorslot_commit": repository_commit,
        "metrics": str(metrics_path),
        "num_evals": metrics["total_evals"],
        "scores": {
            "semantic": _finite_or_none(metrics.get("overall_semantic_score")),
            "instance": _finite_or_none(metrics.get("overall_instance_score")),
            "overall": _finite_or_none(metrics.get("overall_score")),
        },
        "score_coverage": {
            "semantic_arrays": lock["num_semantic_arrays"],
            "instance_arrays": lock["num_instance_arrays"],
            "overall_is_applicable": bool(lock["num_semantic_arrays"] and lock["num_instance_arrays"]),
        },
    }
    _write_json(provenance_path, provenance)
    with open(output_dir / "evaluator_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{evaluator_commit}\n")
    return provenance


def _score_cell(value: object) -> str:
    if value is None:
        return ""
    number = float(value)
    return "" if not math.isfinite(number) else f"{number:.9f}"


def read_registry(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"enabled", "model", "resolution", "predictions", "output_dir", "truth_lock"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Registry {path} is missing columns: {sorted(missing)}")
    return rows


def summarize_registry(registry: str | Path, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in read_registry(registry):
        if str(row["enabled"]).strip().lower() not in {"1", "true", "yes"}:
            continue
        metrics_path = Path(row["output_dir"]) / METRICS_NAME
        provenance_path = Path(row["output_dir"]) / PROVENANCE_NAME
        record: Dict[str, object] = {
            "model": row["model"],
            "resolution": row["resolution"],
            "protocol": row.get("protocol", "full_annotated") or "full_annotated",
            "status": "missing",
            "semantic": None,
            "instance": None,
            "overall": None,
            "num_evals": None,
            "manifest_sha256": None,
            "evaluator_commit": None,
            "primary_metric": None,
            "primary_score": None,
            "metrics": str(metrics_path),
            "notes": row.get("notes", ""),
        }
        if metrics_path.is_file() and provenance_path.is_file():
            metrics = _load_json(metrics_path)
            provenance = _load_json(provenance_path)
            coverage = provenance.get("score_coverage", {})
            has_semantic = bool(coverage.get("semantic_arrays", 0))
            has_instance = bool(coverage.get("instance_arrays", 0))
            if has_semantic and has_instance:
                primary_metric = "overall"
                primary_score = metrics.get("overall_score")
            elif has_instance:
                primary_metric = "instance"
                primary_score = metrics.get("overall_instance_score")
            else:
                primary_metric = "semantic"
                primary_score = metrics.get("overall_semantic_score")
            record.update(
                {
                    "status": "complete",
                    "semantic": metrics.get("overall_semantic_score") if has_semantic else None,
                    "instance": metrics.get("overall_instance_score") if has_instance else None,
                    "overall": metrics.get("overall_score") if has_semantic and has_instance else None,
                    "num_evals": metrics.get("total_evals"),
                    "manifest_sha256": provenance.get("manifest_sha256"),
                    "evaluator_commit": provenance.get("evaluator_commit"),
                    "primary_metric": primary_metric,
                    "primary_score": primary_score,
                }
            )
        records.append(record)

    fieldnames = list(records[0]) if records else ["model", "resolution", "status"]
    with open(output_dir / "per_resolution_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    by_model: Dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["status"] == "complete":
            key = (str(record["model"]), str(record["protocol"]))
            by_model.setdefault(key, []).append(record)
    aggregates = []
    for (model, protocol), model_records in sorted(by_model.items()):
        if len({str(item["resolution"]) for item in model_records}) < 2:
            continue
        complete_overall = [float(item["overall"]) for item in model_records if item["overall"] is not None]
        complete_semantic = [float(item["semantic"]) for item in model_records if item["semantic"] is not None]
        complete_instance = [float(item["instance"]) for item in model_records if item["instance"] is not None]
        aggregates.append(
            {
                "model": model,
                "protocol": protocol,
                "resolutions": ",".join(sorted(str(item["resolution"]) for item in model_records)),
                "num_resolutions": len(model_records),
                "macro_semantic": sum(complete_semantic) / len(complete_semantic) if complete_semantic else None,
                "macro_instance": sum(complete_instance) / len(complete_instance) if complete_instance else None,
                "macro_overall": sum(complete_overall) / len(complete_overall) if complete_overall else None,
                "worst_resolution_instance": min(complete_instance) if complete_instance else None,
                "worst_resolution_overall": min(complete_overall) if complete_overall else None,
            }
        )
    aggregate_fields = list(aggregates[0]) if aggregates else ["model", "resolutions", "num_resolutions"]
    with open(output_dir / "cross_resolution_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)

    lines = [
        "# CellMap official evaluation summary",
        "",
        "Scores are produced by the pinned official evaluator. Cross-resolution values are unweighted macro means; `worst` is the minimum resolution-level overall score.",
        "",
        "| Model | Protocol | Resolution | Status | Semantic | Instance | Overall | Primary | Evals |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        primary_text = (
            f"{_score_cell(record.get('primary_score'))} ({record.get('primary_metric')})"
            if record.get("primary_metric")
            else ""
        )
        lines.append(
            "| {model} | {protocol} | {resolution} | {status} | {semantic} | {instance} | {overall} | {primary} | {num_evals} |".format(
                **{**record, "semantic": _score_cell(record["semantic"]), "instance": _score_cell(record["instance"]), "overall": _score_cell(record["overall"]), "primary": primary_text, "num_evals": record["num_evals"] or ""}
            )
        )
    lines.extend(
        [
            "",
            "## Cross-resolution robustness",
            "",
            "| Model | Protocol | Resolutions | Macro semantic | Macro instance | Macro overall | Worst instance | Worst overall |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in aggregates:
        lines.append(
            f"| {item['model']} | {item['protocol']} | {item['resolutions']} | {_score_cell(item['macro_semantic'])} | {_score_cell(item['macro_instance'])} | {_score_cell(item['macro_overall'])} | {_score_cell(item['worst_resolution_instance'])} | {_score_cell(item['worst_resolution_overall'])} |"
        )
    (output_dir / "cellmap_evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"schema": PROVENANCE_SCHEMA, "created_utc": _now_utc(), "records": records, "cross_resolution": aggregates}
    _write_json(output_dir / "cellmap_evaluation_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-truth")
    prepare.add_argument("--reference-volumes", required=True)
    prepare.add_argument("--native-labels-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--cohort", required=True)
    prepare.add_argument("--resolution-nm", required=True, type=float)
    prepare.add_argument("--class-label", action="append", dest="classes")
    prepare.add_argument("--overwrite", action="store_true")

    lock = subparsers.add_parser("lock-truth")
    lock.add_argument("--manifest", required=True)
    lock.add_argument("--truth-store", required=True)
    lock.add_argument("--validation-report", required=True)
    lock.add_argument("--output", required=True)
    lock.add_argument("--cohort", required=True)
    lock.add_argument("--resolution-nm", required=True, type=float)

    verify = subparsers.add_parser("verify-truth")
    verify.add_argument("--truth-lock", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--truth-lock", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--csc-repo", required=True)
    score.add_argument("--evaluator-python", required=True)
    score.add_argument("--expected-evaluator-commit", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--resolution", required=True)
    score.add_argument("--anchorslot-repo", required=True)
    score.add_argument("--max-workers", type=int, default=4)
    score.add_argument("--per-instance-threads", type=int, default=1)
    score.add_argument("--overwrite", action="store_true")
    score.add_argument("--resume", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--registry", required=True)
    summarize.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare-truth":
        result = prepare_truth(
            args.reference_volumes,
            args.native_labels_root,
            args.output_dir,
            cohort=args.cohort,
            resolution_nm=args.resolution_nm,
            classes=args.classes,
            overwrite=args.overwrite,
        )
    elif args.command == "lock-truth":
        result = create_truth_lock(
            args.manifest,
            args.truth_store,
            args.validation_report,
            args.output,
            cohort=args.cohort,
            resolution_nm=args.resolution_nm,
        )
    elif args.command == "verify-truth":
        result = verify_truth_lock(args.truth_lock)
    elif args.command == "score":
        result = score_predictions(
            args.predictions,
            args.truth_lock,
            args.output_dir,
            args.csc_repo,
            evaluator_python=args.evaluator_python,
            expected_evaluator_commit=args.expected_evaluator_commit,
            model=args.model,
            resolution=args.resolution,
            anchorslot_repo=args.anchorslot_repo,
            max_workers=args.max_workers,
            per_instance_threads=args.per_instance_threads,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    else:
        result = summarize_registry(args.registry, args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
