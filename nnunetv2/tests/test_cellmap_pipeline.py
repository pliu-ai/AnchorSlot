from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("zarr")

from nnunetv2.evaluation.cellmap_challenge import export_ground_truth, validate_submission
from nnunetv2.evaluation.cellmap_pipeline import (
    create_truth_lock,
    summarize_registry,
    verify_truth_lock,
)


def _manifest(path: Path, labels=("mito", "cyto")) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("crop_name", "dataset", "class_label", "voxel_size", "translation", "shape"),
        )
        writer.writeheader()
        for label in labels:
            writer.writerow(
                {
                    "crop_name": "42",
                    "dataset": "synthetic",
                    "class_label": label,
                    "voxel_size": "[4;4;4]",
                    "translation": "[0;0;0]",
                    "shape": "[4;5;6]",
                }
            )


def test_truth_lock_detects_manifest_or_report_changes(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    labels = tmp_path / "labels" / "synthetic" / "crop42" / "labels"
    labels.mkdir(parents=True)
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    mito = np.zeros((4, 5, 6), dtype=np.uint16)
    mito[:2, :2, :2] = 7
    mito[2:, 3:, 4:] = 19
    cyto = np.zeros_like(mito, dtype=np.uint8)
    cyto[:, 2:] = 1
    nib.save(nib.Nifti1Image(mito, affine), labels / "synthetic_crop42_mito_4.0nm.nii.gz")
    nib.save(nib.Nifti1Image(cyto, affine), labels / "synthetic_crop42_cyto_4.0nm.nii.gz")

    truth = tmp_path / "truth.zarr"
    export_ground_truth(tmp_path / "labels", manifest, truth)
    report = validate_submission(truth, manifest, role="ground_truth")
    report_path = tmp_path / "truth_validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    lock_path = tmp_path / "truth_lock.json"
    lock = create_truth_lock(
        manifest,
        truth,
        report_path,
        lock_path,
        cohort="synthetic_4nm",
        resolution_nm=4,
    )
    assert lock["num_instance_arrays"] == 1
    assert lock["num_semantic_arrays"] == 1
    assert verify_truth_lock(lock_path)["manifest_sha256"] == lock["manifest_sha256"]

    report_path.write_text(report_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report changed"):
        verify_truth_lock(lock_path)


def test_summary_marks_instance_only_overall_not_applicable(tmp_path: Path):
    evaluation = tmp_path / "eval"
    evaluation.mkdir()
    metrics = {
        "overall_semantic_score": 0.0,
        "overall_instance_score": 0.625,
        "overall_score": 0.0,
        "total_evals": 6,
    }
    provenance = {
        "manifest_sha256": "abc",
        "evaluator_commit": "0300239",
        "score_coverage": {"semantic_arrays": 0, "instance_arrays": 6},
    }
    (evaluation / "latest_official_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (evaluation / "evaluation_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    registry = tmp_path / "registry.tsv"
    registry.write_text(
        "enabled\tmodel\tprotocol\tresolution\tpredictions\toutput_dir\ttruth_lock\tnotes\n"
        f"1\tRA-HPA\tcommon_parent_instance\t32nm\t/tmp/pred\t{evaluation}\t/tmp/lock\t\n",
        encoding="utf-8",
    )

    result = summarize_registry(registry, tmp_path / "summary")
    record = result["records"][0]
    assert record["semantic"] is None
    assert record["overall"] is None
    assert record["primary_metric"] == "instance"
    assert record["primary_score"] == pytest.approx(0.625)
    markdown = (tmp_path / "summary" / "cellmap_evaluation_summary.md").read_text()
    assert "0.625000000 (instance)" in markdown
