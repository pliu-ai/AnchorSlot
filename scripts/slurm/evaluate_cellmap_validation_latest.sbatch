#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_csc_val
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --output=anchorslot_csc_val_%j.out

set -euo pipefail

: "${ANCHORSLOT_ROOT:?Set ANCHORSLOT_ROOT to the AnchorSlot repository}"
: "${PREDICTIONS:?Set PREDICTIONS to merged validation NIfTI files}"
: "${GROUND_TRUTH_LABELS_ROOT:?Set GROUND_TRUTH_LABELS_ROOT to the native per-class labels tree}"
: "${MANIFEST:?Set MANIFEST to the annotation-only validation manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for Zarr stores and metrics}"
: "${CSC_REPO:?Set CSC_REPO to a pinned official evaluator checkout}"

PYTHON="${PYTHON:-python}"
MAX_WORKERS="${MAX_WORKERS:-4}"
PER_INSTANCE_THREADS="${PER_INSTANCE_THREADS:-3}"
mkdir -p "${OUTPUT_DIR}"

# The evaluator refreshes its manifest inside its package. Use a node-local
# copy so the pinned source checkout remains pristine.
JOB_CSC="${TMPDIR:-/tmp}/anchorslot-csc-${SLURM_JOB_ID}"
rm -rf "${JOB_CSC}"
cp -a "${CSC_REPO}" "${JOB_CSC}"
trap 'rm -rf "${JOB_CSC}"' EXIT

export PYTHONPATH="${ANCHORSLOT_ROOT}:${JOB_CSC}/src:${PYTHONPATH:-}"

"${PYTHON}" "${ANCHORSLOT_ROOT}/scripts/export_cellmap_submission.py" \
  --predictions "${PREDICTIONS}" \
  --manifest "${MANIFEST}" \
  --dry-run \
  --report "${OUTPUT_DIR}/prediction_input_validation.json"

"${PYTHON}" "${ANCHORSLOT_ROOT}/scripts/export_cellmap_submission.py" \
  --truth-labels-root "${GROUND_TRUTH_LABELS_ROOT}" \
  --manifest "${MANIFEST}" \
  --dry-run \
  --report "${OUTPUT_DIR}/ground_truth_input_validation.json"

"${PYTHON}" "${ANCHORSLOT_ROOT}/scripts/export_cellmap_submission.py" \
  --predictions "${PREDICTIONS}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT_DIR}/prediction.zarr" \
  --overwrite \
  --report "${OUTPUT_DIR}/prediction_export.json"

"${PYTHON}" "${ANCHORSLOT_ROOT}/scripts/export_cellmap_submission.py" \
  --truth-labels-root "${GROUND_TRUTH_LABELS_ROOT}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT_DIR}/ground_truth.zarr" \
  --overwrite \
  --report "${OUTPUT_DIR}/ground_truth_export.json"

"${PYTHON}" "${ANCHORSLOT_ROOT}/scripts/validate_cellmap_submission.py" \
  --submission "${OUTPUT_DIR}/prediction.zarr" \
  --manifest "${MANIFEST}" \
  --report "${OUTPUT_DIR}/prediction_store_validation.json"

"${PYTHON}" "${ANCHORSLOT_ROOT}/scripts/validate_cellmap_submission.py" \
  --submission "${OUTPUT_DIR}/ground_truth.zarr" \
  --manifest "${MANIFEST}" \
  --role ground_truth \
  --report "${OUTPUT_DIR}/ground_truth_store_validation.json"

env \
  CSC_TEST_CROP_MANIFEST_URL="${MANIFEST}" \
  MAX_WORKERS="${MAX_WORKERS}" \
  PER_INSTANCE_THREADS="${PER_INSTANCE_THREADS}" \
  "${PYTHON}" -m cellmap_segmentation_challenge.evaluate \
    "${OUTPUT_DIR}/prediction.zarr" \
    "${OUTPUT_DIR}/latest_official_metrics.json" \
    --truth-path "${OUTPUT_DIR}/ground_truth.zarr"

git -C "${CSC_REPO}" rev-parse HEAD > "${OUTPUT_DIR}/evaluator_commit.txt"
