from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from nnunetv2.evaluation.cellmap_challenge import (  # noqa: E402
    ALL_CHALLENGE_CLASSES,
    ATOMIC_CLASSES,
    INSTANCE_CLASSES,
    PARENT_CLASSES,
    export_submission,
    hard_mask_for_class,
    read_manifest,
    validate_submission,
    validate_prediction_inputs,
    write_annotated_validation_manifest,
    write_validation_manifest,
)


def _write_prediction(path: Path) -> np.ndarray:
    segmentation = np.zeros((5, 6, 7), dtype=np.uint8)
    segmentation[0:2] = 4  # mito_mem
    segmentation[2:4] = 5  # mito_lum
    segmentation[4] = 6  # mito_ribo
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    nib.save(nib.Nifti1Image(segmentation, affine), path)
    return segmentation


def _write_manifest(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "crop_name",
                "dataset",
                "class_label",
                "voxel_size",
                "translation",
                "shape",
            ),
        )
        writer.writeheader()
        for label in ("mito_mem", "mito_lum", "mito_ribo", "mito", "cell"):
            writer.writerow(
                {
                    "crop_name": "42",
                    "dataset": "synthetic",
                    "class_label": label,
                    "voxel_size": "[2;3;4]",
                    "translation": "[10;20;30]",
                    "shape": "[5;6;7]",
                }
            )


def test_challenge_taxonomy_is_complete():
    assert len(ATOMIC_CLASSES) == 31
    assert len(PARENT_CLASSES) == 17
    assert len(ALL_CHALLENGE_CLASSES) == 48
    assert len(set(ALL_CHALLENGE_CLASSES)) == 48
    assert set(INSTANCE_CLASSES) <= set(PARENT_CLASSES)


def test_hard_parent_masks_are_atomic_unions():
    segmentation = np.asarray([0, 4, 5, 6, 7, 24, 31], dtype=np.uint8)
    assert hard_mask_for_class(segmentation, "mito_mem").tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert hard_mask_for_class(segmentation, "mito").tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert hard_mask_for_class(segmentation, "cell").tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_export_and_validate_submission(tmp_path: Path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    segmentation = _write_prediction(predictions / "synthetic_crop42.nii.gz")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    output = tmp_path / "submission.zarr"

    export_report = export_submission(predictions, manifest, output, chunks=(2, 3, 4))
    assert export_report["num_crops"] == 1
    assert export_report["num_arrays"] == 5

    root = zarr.open_group(str(output), mode="r")
    assert np.array_equal(
        root["crop42"]["mito"][:].astype(bool), np.isin(segmentation, [4, 5, 6])
    )
    assert root["crop42"]["mito"].attrs["translation"] == [10.0, 20.0, 30.0]

    validation = validate_submission(output, manifest)
    assert validation["status"] == "valid", validation["errors"]
    assert validation["num_expected_arrays"] == 5

    inputs = validate_prediction_inputs(predictions, manifest)
    assert inputs["status"] == "valid"
    assert inputs["crops"][0]["present_label_ids"] == [4, 5, 6]


def test_validation_manifest_uses_only_root_predictions(tmp_path: Path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_prediction(predictions / "synthetic_crop42.nii.gz")
    group = predictions / "group_00"
    group.mkdir()
    _write_prediction(group / "synthetic_crop42.nii.gz")
    manifest = tmp_path / "derived.csv"

    write_validation_manifest(predictions, manifest)
    entries = read_manifest(manifest)
    assert len(entries) == 48
    assert {entry.crop_name for entry in entries} == {"crop42"}
    assert {entry.class_label for entry in entries} == set(ALL_CHALLENGE_CLASSES)


def test_recursive_input_finds_resolution_dirs_but_skips_groups(tmp_path: Path):
    predictions = tmp_path / "predictions"
    nested = predictions / "2-8nm"
    nested.mkdir(parents=True)
    _write_prediction(nested / "synthetic_crop42.nii.gz")
    group = predictions / "group_00"
    group.mkdir()
    _write_prediction(group / "synthetic_crop999.nii.gz")
    manifest = tmp_path / "derived.csv"

    write_validation_manifest(predictions, manifest, recursive=True)
    entries = read_manifest(manifest)
    assert len(entries) == 48
    assert {entry.crop_name for entry in entries} == {"crop42"}


def test_annotated_manifest_keeps_only_available_ground_truth(tmp_path: Path):
    manifest = tmp_path / "all.csv"
    _write_manifest(manifest)
    labels = tmp_path / "labels" / "synthetic" / "crop42" / "labels"
    labels.mkdir(parents=True)
    (labels / "synthetic_crop42_mito_mem_2.0nm.nii.gz").touch()
    (labels / "synthetic_crop42_mito_2.0nm.nii.gz").touch()
    output = tmp_path / "annotated.csv"

    write_annotated_validation_manifest(manifest, tmp_path / "labels", output)

    entries = read_manifest(output)
    assert [(entry.crop_name, entry.class_label) for entry in entries] == [
        ("crop42", "mito_mem"),
        ("crop42", "mito"),
    ]


def test_validator_detects_missing_array(tmp_path: Path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_prediction(predictions / "synthetic_crop42.nii.gz")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    output = tmp_path / "submission.zarr"
    export_submission(predictions, manifest, output)
    root = zarr.open_group(str(output), mode="a")
    del root["crop42"]["mito_lum"]

    report = validate_submission(output, manifest, check_hierarchy=False)
    assert report["status"] == "invalid"
    assert any("Missing 1 crop-class arrays" in error for error in report["errors"])
