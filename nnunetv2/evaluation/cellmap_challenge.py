"""CellMap Challenge submission export and validation for AnchorSlot.

This module deliberately does not import the official challenge package.  It
implements only the stable submission boundary: convert an nnU-Net 0..31 hard
label map into the overlapping atomic/parent arrays requested by a pinned
CellMap test manifest, write a Zarr-2 store, and validate the result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import nibabel as nib
import numpy as np


LABEL_ID_TO_ATOMIC: Dict[int, str] = {
    1: "ecs",
    2: "pm",
    3: "cyto",
    4: "mito_mem",
    5: "mito_lum",
    6: "mito_ribo",
    7: "golgi_mem",
    8: "golgi_lum",
    9: "ves_mem",
    10: "ves_lum",
    11: "endo_mem",
    12: "endo_lum",
    13: "lyso_mem",
    14: "lyso_lum",
    15: "ld_mem",
    16: "ld_lum",
    17: "er_mem",
    18: "er_lum",
    19: "eres_mem",
    20: "eres_lum",
    21: "ne_mem",
    22: "ne_lum",
    23: "np_out",
    24: "np_in",
    25: "hchrom",
    26: "echrom",
    27: "nucpl",
    28: "mt_out",
    29: "mt_in",
    30: "perox_mem",
    31: "perox_lum",
}
ATOMIC_TO_LABEL_ID: Dict[str, int] = {
    name: idx for idx, name in LABEL_ID_TO_ATOMIC.items()
}
ATOMIC_CLASSES: Tuple[str, ...] = tuple(LABEL_ID_TO_ATOMIC.values())

# The 17 overlapping internal nodes in the CellMap challenge label DAG.
PARENT_FROM_ATOMIC: Dict[str, Tuple[str, ...]] = {
    "mito": ("mito_mem", "mito_lum", "mito_ribo"),
    "golgi": ("golgi_mem", "golgi_lum"),
    "ves": ("ves_mem", "ves_lum"),
    "endo": ("endo_mem", "endo_lum"),
    "lyso": ("lyso_mem", "lyso_lum"),
    "ld": ("ld_mem", "ld_lum"),
    "perox": ("perox_mem", "perox_lum"),
    "eres": ("eres_mem", "eres_lum"),
    "mt": ("mt_in", "mt_out"),
    "np": ("np_in", "np_out"),
    "chrom": ("hchrom", "echrom"),
    "ne": ("ne_mem", "ne_lum", "np_in", "np_out"),
    "ne_mem_all": ("ne_mem", "np_in", "np_out"),
    "nuc": ("nucpl", "hchrom", "echrom", "ne_mem", "ne_lum", "np_in", "np_out"),
    "er_mem_all": ("er_mem", "ne_mem", "eres_mem"),
    "er": (
        "er_mem",
        "er_lum",
        "ne_mem",
        "ne_lum",
        "np_in",
        "np_out",
        "eres_mem",
        "eres_lum",
    ),
    "cell": tuple(name for name in ATOMIC_CLASSES if name != "ecs"),
}
PARENT_CLASSES: Tuple[str, ...] = tuple(PARENT_FROM_ATOMIC)
ALL_CHALLENGE_CLASSES: Tuple[str, ...] = ATOMIC_CLASSES + PARENT_CLASSES

# The current official config also lists ``vim`` and ``instance``; neither is
# present in the pinned 0300239 test manifest.  Keeping them here makes a future
# manifest fail explicitly as unsupported instead of silently mis-scoring it.
INSTANCE_CLASSES: Tuple[str, ...] = (
    "nuc",
    "ves",
    "endo",
    "lyso",
    "ld",
    "perox",
    "mito",
    "np",
    "mt",
    "cell",
)

FORMAT_VERSION = "anchorslot-cellmap-zarr2-v1"
CROP_RE = re.compile(r"(?:^|_)crop(?P<crop>\d+)(?:_|\.|$)")
VECTOR_RE = re.compile(r"^\[(.*)\]$")


@dataclass(frozen=True)
class ManifestEntry:
    crop_name: str
    dataset: str
    class_label: str
    voxel_size: Tuple[float, float, float]
    translation: Tuple[float, float, float]
    shape: Tuple[int, int, int]


def _normalize_crop_name(value: str | int) -> str:
    text = str(value).strip()
    if text.startswith("crop"):
        text = text[4:]
    if not text.isdigit():
        raise ValueError(
            f"Invalid crop name {value!r}; expected an integer or 'crop<integer>'."
        )
    return f"crop{int(text)}"


def _parse_vector(value: str, cast, field_name: str) -> tuple:
    text = str(value).strip()
    match = VECTOR_RE.match(text)
    if match is not None:
        text = match.group(1)
    parts = [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"{field_name} must contain exactly three values, got {value!r}"
        )
    return tuple(cast(part) for part in parts)


def _format_vector(values: Sequence[float | int]) -> str:
    return "[" + ";".join(str(value) for value in values) + "]"


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> List[ManifestEntry]:
    path = Path(path)
    required = {
        "crop_name",
        "dataset",
        "class_label",
        "voxel_size",
        "translation",
        "shape",
    }
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Manifest {path} is missing columns: {sorted(missing_columns)}"
            )
        entries = [
            ManifestEntry(
                crop_name=_normalize_crop_name(row["crop_name"]),
                dataset=str(row["dataset"]).strip(),
                class_label=str(row["class_label"]).strip(),
                voxel_size=_parse_vector(row["voxel_size"], float, "voxel_size"),
                translation=_parse_vector(row["translation"], float, "translation"),
                shape=_parse_vector(row["shape"], int, "shape"),
            )
            for row in reader
        ]

    if not entries:
        raise ValueError(f"Manifest {path} has no entries.")
    seen = set()
    for entry in entries:
        pair = (entry.crop_name, entry.class_label)
        if pair in seen:
            raise ValueError(f"Duplicate manifest entry for {pair}.")
        seen.add(pair)
        if entry.class_label not in ALL_CHALLENGE_CLASSES:
            raise ValueError(
                f"Unsupported class {entry.class_label!r} in {path}. "
                f"AnchorSlot supports the 48 atomic/parent classes only."
            )
        if any(size <= 0 for size in entry.shape):
            raise ValueError(f"Invalid shape for {pair}: {entry.shape}")
        if any(size <= 0 for size in entry.voxel_size):
            raise ValueError(f"Invalid voxel size for {pair}: {entry.voxel_size}")

    # All labels within one crop must occupy exactly the same target space.
    crop_geometry: Dict[str, tuple] = {}
    for entry in entries:
        geometry = (entry.shape, entry.voxel_size, entry.translation)
        previous = crop_geometry.setdefault(entry.crop_name, geometry)
        if previous != geometry:
            raise ValueError(
                f"Inconsistent geometry within {entry.crop_name}: {previous} vs {geometry}"
            )
    return entries


def entries_by_crop(entries: Iterable[ManifestEntry]) -> Dict[str, List[ManifestEntry]]:
    grouped: Dict[str, List[ManifestEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.crop_name, []).append(entry)
    return grouped


def crop_name_from_filename(path: str | Path) -> str:
    name = Path(path).name
    match = CROP_RE.search(name)
    if match is None:
        raise ValueError(
            f"Cannot extract a crop number from prediction filename {name!r}."
        )
    return _normalize_crop_name(match.group("crop"))


def dataset_name_from_filename(path: str | Path) -> str:
    stem = Path(path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    match = CROP_RE.search(stem)
    return stem[: match.start()].rstrip("_") if match is not None else stem


def index_predictions(
    predictions_dir: str | Path, *, recursive: bool = False
) -> Dict[str, Path]:
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {predictions_dir}"
        )
    indexed: Dict[str, Path] = {}
    candidates = (
        predictions_dir.rglob("*.nii.gz")
        if recursive
        else predictions_dir.glob("*.nii.gz")
    )
    for path in sorted(candidates):
        relative_parts = path.relative_to(predictions_dir).parts[:-1]
        # group_*/ and split_labels/ contain intermediate or derived outputs,
        # never the one merged prediction expected for a crop.
        if any(
            part.startswith("group_") or part == "split_labels"
            for part in relative_parts
        ):
            continue
        crop_name = crop_name_from_filename(path)
        if crop_name in indexed:
            raise ValueError(
                f"Multiple root predictions map to {crop_name}: {indexed[crop_name]} and {path}"
            )
        indexed[crop_name] = path
    if not indexed:
        scope = "recursive" if recursive else "root-level"
        raise FileNotFoundError(
            f"No {scope} *.nii.gz predictions found in {predictions_dir}"
        )
    return indexed


def write_validation_manifest(
    predictions_dir: str | Path,
    output_csv: str | Path,
    crops: Iterable[str | int] | None = None,
    recursive: bool = False,
) -> Path:
    """Write an all-48-class manifest from legacy validation NIfTI headers."""
    predictions = index_predictions(predictions_dir, recursive=recursive)
    if crops is not None:
        requested = {_normalize_crop_name(crop) for crop in crops}
        missing = sorted(requested - set(predictions))
        if missing:
            raise ValueError(
                f"Requested validation crops have no root prediction: {missing}"
            )
        predictions = {
            crop: path for crop, path in predictions.items() if crop in requested
        }
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
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
        for crop_name, path in sorted(
            predictions.items(), key=lambda item: int(item[0][4:])
        ):
            image = nib.load(str(path))
            if len(image.shape) != 3:
                raise ValueError(
                    f"Expected a 3-D prediction, got {image.shape} in {path}"
                )
            voxel_size = tuple(float(v) for v in image.header.get_zooms()[:3])
            translation = tuple(float(v) for v in image.affine[:3, 3])
            shape = tuple(int(v) for v in image.shape)
            dataset = dataset_name_from_filename(path)
            for class_label in ALL_CHALLENGE_CLASSES:
                writer.writerow(
                    {
                        "crop_name": crop_name.removeprefix("crop"),
                        "dataset": dataset,
                        "class_label": class_label,
                        "voxel_size": _format_vector(voxel_size),
                        "translation": _format_vector(translation),
                        "shape": _format_vector(shape),
                    }
                )
    return output_csv


def write_annotated_validation_manifest(
    source_manifest: str | Path,
    labels_root: str | Path,
    output_csv: str | Path,
) -> Path:
    """Filter a validation manifest to crop/class pairs with ground truth.

    CellMap validation data is partially annotated: an absent class file means
    "not annotated", not an empty ground-truth mask.  This helper scans the
    conventional ``<dataset>/crop<ID>/labels/*.nii.gz`` tree and retains only
    the corresponding entries from an all-class manifest.
    """
    source_manifest = Path(source_manifest)
    labels_root = Path(labels_root)
    output_csv = Path(output_csv)
    entries = read_manifest(source_manifest)
    expected_pairs = {(entry.crop_name, entry.class_label) for entry in entries}
    expected_crops = {entry.crop_name for entry in entries}

    annotated_pairs: set[Tuple[str, str]] = set()
    classes_longest_first = sorted(ALL_CHALLENGE_CLASSES, key=len, reverse=True)
    for path in labels_root.rglob("*.nii.gz"):
        try:
            crop_name = crop_name_from_filename(path)
        except ValueError:
            continue
        if crop_name not in expected_crops:
            continue
        marker = f"_crop{int(crop_name[4:])}_"
        marker_start = path.name.find(marker)
        if marker_start < 0:
            continue
        suffix = path.name[marker_start + len(marker) :]
        class_label = next(
            (name for name in classes_longest_first if suffix.startswith(f"{name}_")),
            None,
        )
        if class_label is None:
            raise ValueError(f"Cannot identify a CellMap class from {path}")
        pair = (crop_name, class_label)
        if pair not in expected_pairs:
            raise ValueError(
                f"Ground-truth annotation {path} maps to {pair}, which is absent "
                f"from {source_manifest}."
            )
        if pair in annotated_pairs:
            raise ValueError(f"Duplicate ground-truth annotation for {pair}: {path}")
        annotated_pairs.add(pair)

    filtered = [
        entry
        for entry in entries
        if (entry.crop_name, entry.class_label) in annotated_pairs
    ]
    if not filtered:
        raise ValueError(f"No annotated crop/class pairs found under {labels_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
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
        for entry in filtered:
            writer.writerow(
                {
                    "crop_name": entry.crop_name.removeprefix("crop"),
                    "dataset": entry.dataset,
                    "class_label": entry.class_label,
                    "voxel_size": _format_vector(entry.voxel_size),
                    "translation": _format_vector(entry.translation),
                    "shape": _format_vector(entry.shape),
                }
            )
    return output_csv


def _load_hard_segmentation(
    path: Path, expected_shape: Tuple[int, int, int]
) -> np.ndarray:
    image = nib.load(str(path))
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(
            f"Shape mismatch for {path}: prediction={image.shape}, manifest={expected_shape}"
        )
    segmentation = np.asanyarray(image.dataobj)
    if segmentation.ndim != 3:
        raise ValueError(
            f"Expected a 3-D prediction in {path}, got shape {segmentation.shape}"
        )
    if not np.issubdtype(segmentation.dtype, np.integer):
        if not np.all(np.isfinite(segmentation)) or not np.array_equal(
            segmentation, np.rint(segmentation)
        ):
            raise ValueError(f"Prediction {path} is not an integer hard-label map.")
    minimum = int(np.min(segmentation))
    maximum = int(np.max(segmentation))
    if minimum < 0 or maximum > 31:
        raise ValueError(
            f"Prediction {path} contains labels outside 0..31: min={minimum}, max={maximum}"
        )
    return np.asarray(segmentation, dtype=np.uint8)


def validate_prediction_inputs(
    predictions_dir: str | Path,
    manifest_path: str | Path,
    *,
    recursive: bool = False,
) -> Dict[str, object]:
    """Validate all merged hard-label inputs without creating a Zarr store."""
    predictions_dir = Path(predictions_dir)
    manifest_path = Path(manifest_path)
    entries = read_manifest(manifest_path)
    grouped = entries_by_crop(entries)
    predictions = index_predictions(predictions_dir, recursive=recursive)
    errors: List[str] = []
    crop_reports = []
    missing_crops = sorted(set(grouped) - set(predictions))
    if missing_crops:
        errors.append(f"Missing merged predictions for manifest crops: {missing_crops}")
    for crop_name in sorted(
        set(grouped) & set(predictions), key=lambda value: int(value[4:])
    ):
        geometry = grouped[crop_name][0]
        try:
            segmentation = _load_hard_segmentation(
                predictions[crop_name], geometry.shape
            )
            labels, counts = np.unique(segmentation, return_counts=True)
            crop_reports.append(
                {
                    "crop_name": crop_name,
                    "prediction": str(predictions[crop_name]),
                    "shape": list(geometry.shape),
                    "present_label_ids": [int(value) for value in labels],
                    "label_voxel_counts": {
                        str(int(label)): int(count)
                        for label, count in zip(labels, counts)
                    },
                }
            )
            del segmentation
        except Exception as exc:
            errors.append(f"{crop_name}: {type(exc).__name__}: {exc}")
    return {
        "status": "valid" if not errors else "invalid",
        "predictions_dir": str(predictions_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256(manifest_path),
        "num_expected_crops": len(grouped),
        "num_checked_crops": len(crop_reports),
        "ignored_prediction_crops": sorted(set(predictions) - set(grouped)),
        "errors": errors,
        "crops": crop_reports,
    }


def _label_ids_for_class(class_label: str) -> Tuple[int, ...]:
    if class_label in ATOMIC_TO_LABEL_ID:
        return (ATOMIC_TO_LABEL_ID[class_label],)
    return tuple(ATOMIC_TO_LABEL_ID[child] for child in PARENT_FROM_ATOMIC[class_label])


def hard_mask_for_class(segmentation: np.ndarray, class_label: str) -> np.ndarray:
    label_ids = _label_ids_for_class(class_label)
    if len(label_ids) == 1:
        return segmentation == label_ids[0]
    return np.isin(segmentation, label_ids)


def _import_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "CellMap export requires Zarr. Install the optional dependency with "
            "`pip install -e '.[cellmap]'`; either API version 2 or 3 is supported, "
            "and the on-disk store is always format 2."
        ) from exc
    return zarr


def _open_zarr2_group(zarr, path: str | Path, mode: str):
    """Open a format-2 group with either the Zarr 2 or Zarr 3 Python API."""
    major = int(str(zarr.__version__).split(".")[0])
    if major >= 3:
        return zarr.open_group(str(path), mode=mode, zarr_format=2)
    return zarr.open_group(str(path), mode=mode)


def _create_zarr_array(group, name: str, *, shape, chunks, dtype):
    """Create an array across the Zarr 2.x and 3.x Python APIs."""
    if hasattr(group, "create_array"):
        return group.create_array(
            name, shape=shape, chunks=chunks, dtype=dtype, overwrite=False
        )
    return group.create_dataset(
        name, shape=shape, chunks=chunks, dtype=dtype, overwrite=False
    )


def _chunk_shape(
    shape: Sequence[int], requested: Sequence[int]
) -> Tuple[int, int, int]:
    if len(requested) != 3 or any(int(v) <= 0 for v in requested):
        raise ValueError(
            f"Chunk shape must have three positive integers, got {requested}"
        )
    return tuple(min(int(size), int(chunk)) for size, chunk in zip(shape, requested))


def _iter_axis0(
    shape: Sequence[int], depth: int
) -> Iterator[Tuple[slice, slice, slice]]:
    for start in range(0, int(shape[0]), int(depth)):
        yield (
            slice(start, min(start + int(depth), int(shape[0]))),
            slice(None),
            slice(None),
        )


def export_submission(
    predictions_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    chunks: Sequence[int] = (64, 64, 64),
    recursive: bool = False,
) -> Dict[str, object]:
    zarr = _import_zarr()
    predictions_dir = Path(predictions_dir)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if output_path.suffix != ".zarr":
        output_path = output_path.with_suffix(".zarr")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    entries = read_manifest(manifest_path)
    grouped = entries_by_crop(entries)
    predictions = index_predictions(predictions_dir, recursive=recursive)
    missing_crops = sorted(set(grouped) - set(predictions))
    if missing_crops:
        raise FileNotFoundError(
            f"Missing merged predictions for manifest crops: {missing_crops}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = _open_zarr2_group(zarr, output_path, mode="w")
    root.attrs.update(
        {
            "anchorslot_format_version": FORMAT_VERSION,
            "hierarchy_mode": "derived_hard_union",
            "manifest_sha256": manifest_sha256(manifest_path),
            "manifest_name": manifest_path.name,
            "source_predictions": str(predictions_dir.resolve()),
            "source_predictions_recursive": bool(recursive),
            "label_space": "background_0_atomic_1_to_31",
        }
    )

    arrays_written = 0
    crop_reports = []
    for crop_name in sorted(grouped, key=lambda value: int(value[4:])):
        crop_entries = grouped[crop_name]
        geometry = crop_entries[0]
        segmentation = _load_hard_segmentation(predictions[crop_name], geometry.shape)
        crop_group = root.create_group(crop_name)
        this_chunks = _chunk_shape(geometry.shape, chunks)
        for entry in sorted(crop_entries, key=lambda item: item.class_label):
            dataset = _create_zarr_array(
                crop_group,
                entry.class_label,
                shape=entry.shape,
                chunks=this_chunks,
                dtype=np.uint8,
            )
            label_ids = np.asarray(
                _label_ids_for_class(entry.class_label), dtype=np.uint8
            )
            for slicer in _iter_axis0(entry.shape, this_chunks[0]):
                block = segmentation[slicer]
                mask = (
                    block == label_ids[0]
                    if len(label_ids) == 1
                    else np.isin(block, label_ids)
                )
                dataset[slicer] = mask.astype(np.uint8, copy=False)
            dataset.attrs.update(
                {
                    "voxel_size": list(entry.voxel_size),
                    "translation": list(entry.translation),
                    "shape": list(entry.shape),
                    "source_label_ids": [int(value) for value in label_ids],
                }
            )
            arrays_written += 1
        crop_reports.append(
            {
                "crop_name": crop_name,
                "prediction": str(predictions[crop_name]),
                "shape": list(geometry.shape),
                "arrays_written": len(crop_entries),
            }
        )
        del segmentation

    return {
        "status": "exported",
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256(manifest_path),
        "num_crops": len(grouped),
        "num_arrays": arrays_written,
        "ignored_prediction_crops": sorted(set(predictions) - set(grouped)),
        "crops": crop_reports,
    }


def _all_array_pairs(root) -> set[Tuple[str, str]]:
    pairs = set()
    for crop_name in root.group_keys():
        crop_group = root[crop_name]
        for class_label in crop_group.array_keys():
            pairs.add((str(crop_name), str(class_label)))
    return pairs


def _vectors_close(actual, expected, tolerance: float = 1e-6) -> bool:
    try:
        return len(actual) == len(expected) and all(
            math.isclose(float(a), float(e), rel_tol=tolerance, abs_tol=tolerance)
            for a, e in zip(actual, expected)
        )
    except (TypeError, ValueError):
        return False


def validate_submission(
    submission_path: str | Path,
    manifest_path: str | Path,
    *,
    check_values: bool = True,
    check_hierarchy: bool = True,
    chunks_axis0: int = 64,
) -> Dict[str, object]:
    zarr = _import_zarr()
    submission_path = Path(submission_path)
    manifest_path = Path(manifest_path)
    entries = read_manifest(manifest_path)
    grouped = entries_by_crop(entries)
    expected_pairs = {(entry.crop_name, entry.class_label) for entry in entries}
    errors: List[str] = []
    warnings: List[str] = []
    array_reports = []

    if not submission_path.is_dir():
        raise FileNotFoundError(f"Submission Zarr does not exist: {submission_path}")
    if not (submission_path / ".zgroup").is_file():
        errors.append("Root .zgroup is missing; the store is not a Zarr-2 group.")
    root = _open_zarr2_group(zarr, submission_path, mode="r")
    actual_pairs = _all_array_pairs(root)
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)
    if missing_pairs:
        errors.append(
            f"Missing {len(missing_pairs)} crop-class arrays: {missing_pairs[:20]}"
        )
    if extra_pairs:
        errors.append(
            f"Found {len(extra_pairs)} arrays not present in the manifest: {extra_pairs[:20]}"
        )

    for entry in entries:
        pair = (entry.crop_name, entry.class_label)
        if pair not in actual_pairs:
            continue
        array = root[entry.crop_name][entry.class_label]
        if tuple(array.shape) != entry.shape:
            errors.append(f"{pair}: shape={array.shape}, expected={entry.shape}")
            continue
        if not _vectors_close(array.attrs.get("voxel_size", ()), entry.voxel_size):
            errors.append(
                f"{pair}: invalid voxel_size={array.attrs.get('voxel_size')}, expected={entry.voxel_size}"
            )
        if not _vectors_close(array.attrs.get("translation", ()), entry.translation):
            errors.append(
                f"{pair}: invalid translation={array.attrs.get('translation')}, expected={entry.translation}"
            )
        if tuple(array.attrs.get("shape", ())) != entry.shape:
            errors.append(
                f"{pair}: invalid shape attribute={array.attrs.get('shape')}, expected={entry.shape}"
            )

        nonzero = 0
        if check_values:
            for slicer in _iter_axis0(entry.shape, min(chunks_axis0, entry.shape[0])):
                block = np.asarray(array[slicer])
                if not np.all(np.isfinite(block)):
                    errors.append(f"{pair}: contains non-finite values")
                    break
                if np.any((block != 0) & (block != 1)):
                    values = np.unique(block[(block != 0) & (block != 1)])[:10].tolist()
                    errors.append(
                        f"{pair}: hard-label export is not binary; examples={values}"
                    )
                    break
                nonzero += int(np.count_nonzero(block))
        array_reports.append(
            {
                "crop_name": entry.crop_name,
                "class_label": entry.class_label,
                "shape": list(entry.shape),
                "nonzero_voxels": nonzero if check_values else None,
            }
        )

    hierarchy_checks = 0
    hierarchy_skipped = 0
    if check_hierarchy and root.attrs.get("hierarchy_mode") == "derived_hard_union":
        for crop_name, crop_entries in grouped.items():
            available = {entry.class_label for entry in crop_entries} & {
                class_label for crop, class_label in actual_pairs if crop == crop_name
            }
            shape = crop_entries[0].shape
            for parent, children in PARENT_FROM_ATOMIC.items():
                if parent not in available:
                    continue
                if not set(children).issubset(available):
                    hierarchy_skipped += 1
                    continue
                parent_array = root[crop_name][parent]
                mismatch = 0
                for slicer in _iter_axis0(shape, min(chunks_axis0, shape[0])):
                    expected = None
                    for child in children:
                        child_block = np.asarray(
                            root[crop_name][child][slicer], dtype=bool
                        )
                        expected = (
                            child_block
                            if expected is None
                            else (expected | child_block)
                        )
                    actual = np.asarray(parent_array[slicer], dtype=bool)
                    mismatch += int(np.count_nonzero(actual != expected))
                if mismatch:
                    errors.append(
                        f"{crop_name}/{parent}: differs from child union at {mismatch} voxels"
                    )
                hierarchy_checks += 1
    elif check_hierarchy:
        warnings.append(
            "Hierarchy check skipped because hierarchy_mode is not 'derived_hard_union'."
        )

    expected_hash = manifest_sha256(manifest_path)
    stored_hash = root.attrs.get("manifest_sha256")
    if stored_hash != expected_hash:
        errors.append(
            f"Manifest hash mismatch: stored={stored_hash!r}, expected={expected_hash!r}"
        )

    report = {
        "status": "valid" if not errors else "invalid",
        "submission_path": str(submission_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_hash,
        "num_expected_crops": len(grouped),
        "num_expected_arrays": len(expected_pairs),
        "num_actual_arrays": len(actual_pairs),
        "num_hierarchy_checks": hierarchy_checks,
        "num_hierarchy_checks_skipped": hierarchy_skipped,
        "errors": errors,
        "warnings": warnings,
        "arrays": array_reports,
    }
    return report


def zip_submission(
    submission_path: str | Path, output_zip: str | Path | None = None
) -> Path:
    submission_path = Path(submission_path)
    if output_zip is None:
        output_zip = submission_path.with_suffix(".zip")
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for path in sorted(submission_path.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(submission_path))
    return output_zip


def _parse_chunks(value: str) -> Tuple[int, int, int]:
    return _parse_vector(value, int, "chunks")


def build_manifest_entry_point(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an all-class manifest from validation NIfTI predictions."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--crop", action="append", help="Optional crop ID to include; may be repeated."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search nested prediction directories."
    )
    args = parser.parse_args(argv)
    path = write_validation_manifest(
        args.predictions, args.output, crops=args.crop, recursive=args.recursive
    )
    entries = read_manifest(path)
    print(
        json.dumps(
            {"status": "written", "path": str(path), "num_entries": len(entries)},
            indent=2,
        )
    )


def build_annotated_manifest_entry_point(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an all-class validation manifest to pairs with annotation files."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = write_annotated_validation_manifest(
        args.manifest, args.labels_root, args.output
    )
    entries = read_manifest(path)
    print(
        json.dumps(
            {
                "status": "written",
                "path": str(path),
                "num_entries": len(entries),
                "num_crops": len(entries_by_crop(entries)),
            },
            indent=2,
        )
    )


def export_entry_point(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export AnchorSlot 0..31 NIfTI predictions as CellMap Zarr-2."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--chunks", type=_parse_chunks, default=(64, 64, 64))
    parser.add_argument(
        "--recursive", action="store_true", help="Search nested prediction directories."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every input without writing Zarr data.",
    )
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.make_zip:
            parser.error("--zip cannot be combined with --dry-run")
        report = validate_prediction_inputs(
            args.predictions, args.manifest, recursive=args.recursive
        )
    else:
        if not args.output:
            parser.error("--output is required unless --dry-run is used")
        report = export_submission(
            args.predictions,
            args.manifest,
            args.output,
            overwrite=args.overwrite,
            chunks=args.chunks,
            recursive=args.recursive,
        )
        if args.make_zip:
            report["zip_path"] = str(zip_submission(report["output_path"]))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if report["status"] == "invalid":
        raise SystemExit(1)


def validate_entry_point(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Strictly validate an AnchorSlot CellMap Zarr-2 submission."
    )
    parser.add_argument("--submission", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skip-values", action="store_true")
    parser.add_argument("--skip-hierarchy", action="store_true")
    parser.add_argument(
        "--zip",
        action="store_true",
        dest="make_zip",
        help="Zip the store only if validation passes.",
    )
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    report = validate_submission(
        args.submission,
        args.manifest,
        check_values=not args.skip_values,
        check_hierarchy=not args.skip_hierarchy,
    )
    if report["status"] == "valid" and args.make_zip:
        report["zip_path"] = str(zip_submission(args.submission))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if report["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    export_entry_point()
