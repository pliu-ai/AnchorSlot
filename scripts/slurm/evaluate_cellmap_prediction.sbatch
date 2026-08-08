#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_csc_eval
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/weilab/liupeng/code/projects/AnchorSlot}"
PYTHON="${PYTHON:-/projects/weilab/liupeng/conda/envs/csc/bin/python}"
CSC_REPO="${CSC_REPO:-/projects/weilab/liupeng/code/projects/cellmap-segmentation-challenge-upstream-0300239}"
EXPECTED_EVALUATOR_COMMIT="${EXPECTED_EVALUATOR_COMMIT:-0300239cd0b4867d4bab008aa9e95161b2442d93}"
: "${PREDICTIONS:?Set PREDICTIONS to root-level merged 0..31 NIfTI predictions}"
: "${TRUTH_LOCK:?Set TRUTH_LOCK to a validated cohort truth_lock.json}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for prediction Zarr, metrics, and provenance}"
: "${MODEL:?Set MODEL to the table/model identifier}"
: "${RESOLUTION:?Set RESOLUTION to the cohort resolution identifier}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

args=(
  score
  --predictions "${PREDICTIONS}"
  --truth-lock "${TRUTH_LOCK}"
  --output-dir "${OUTPUT_DIR}"
  --csc-repo "${CSC_REPO}"
  --evaluator-python "${PYTHON}"
  --expected-evaluator-commit "${EXPECTED_EVALUATOR_COMMIT}"
  --model "${MODEL}"
  --resolution "${RESOLUTION}"
  --anchorslot-repo "${REPO_ROOT}"
  --max-workers "${MAX_WORKERS:-4}"
  --per-instance-threads "${PER_INSTANCE_THREADS:-1}"
)
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)
[[ "${RESUME:-1}" == "1" ]] && args+=(--resume)
"${PYTHON}" scripts/cellmap_evaluation_pipeline.py "${args[@]}"
