#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_csc_score
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --output=anchorslot_csc_score_%j.out

set -euo pipefail

: "${ANCHORSLOT_ROOT:?Set ANCHORSLOT_ROOT to the AnchorSlot repository}"
: "${MANIFEST:?Set MANIFEST to the annotation-only validation manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR containing prediction.zarr and ground_truth.zarr}"
: "${CSC_REPO:?Set CSC_REPO to a pinned official evaluator checkout}"

PYTHON="${PYTHON:-python}"
MAX_WORKERS="${MAX_WORKERS:-4}"
PER_INSTANCE_THREADS="${PER_INSTANCE_THREADS:-1}"
test -d "${OUTPUT_DIR}/prediction.zarr"
test -d "${OUTPUT_DIR}/ground_truth.zarr"

JOB_CSC="${TMPDIR:-/tmp}/anchorslot-csc-${SLURM_JOB_ID}"
rm -rf "${JOB_CSC}"
cp -a "${CSC_REPO}" "${JOB_CSC}"
trap 'rm -rf "${JOB_CSC}"' EXIT
export PYTHONPATH="${ANCHORSLOT_ROOT}:${JOB_CSC}/src:${PYTHONPATH:-}"

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
