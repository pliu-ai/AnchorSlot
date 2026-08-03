#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_cellmap_export
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --output=anchorslot_cellmap_export_%j.out

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the AnchorSlot repository}"
: "${PREDICTIONS:?Set PREDICTIONS to the merged official test prediction root}"
: "${MANIFEST:?Set MANIFEST to the pinned official test_crop_manifest.csv}"
: "${OUTPUT:?Set OUTPUT to the destination submission.zarr}"

PYTHON="${PYTHON:-python}"
RECURSIVE_FLAG=()
if [[ "${RECURSIVE_PREDICTIONS:-0}" == "1" ]]; then
  RECURSIVE_FLAG+=(--recursive)
fi

OUTPUT="${OUTPUT%.zarr}.zarr"
REPORT_STEM="${OUTPUT%.zarr}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" "${REPO_ROOT}/scripts/export_cellmap_submission.py" \
  --predictions "${PREDICTIONS}" \
  --manifest "${MANIFEST}" \
  --dry-run \
  --report "${REPORT_STEM}_input_validation.json" \
  "${RECURSIVE_FLAG[@]}"

"${PYTHON}" "${REPO_ROOT}/scripts/export_cellmap_submission.py" \
  --predictions "${PREDICTIONS}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --overwrite \
  --report "${REPORT_STEM}_export.json" \
  "${RECURSIVE_FLAG[@]}"

"${PYTHON}" "${REPO_ROOT}/scripts/validate_cellmap_submission.py" \
  --submission "${OUTPUT}" \
  --manifest "${MANIFEST}" \
  --report "${REPORT_STEM}_validation.json" \
  --zip
