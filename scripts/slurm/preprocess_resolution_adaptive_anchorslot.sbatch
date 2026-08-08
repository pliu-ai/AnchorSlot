#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_rahpa_prep
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/weilab/liupeng/code/projects/AnchorSlot}"
CONDA_ENV="${CONDA_ENV:-/projects/weilab/liupeng/conda/envs/ssl_seg}"
NNUNET_DATA_ROOT="${NNUNET_DATA_ROOT:-/projects/weilab/liupeng/data/nnUNet}"
DATASET_ID="${DATASET_ID:-200}"
CONFIGURATION="${CONFIGURATION:-3d_lowres_large_patch}"
PLANS="${PLANS:-nnUNetPlans}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export nnUNet_raw="${NNUNET_DATA_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_DATA_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_DATA_ROOT}/nnUNet_results"
cd "${REPO_ROOT}"

DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset$(printf '%03d' "${DATASET_ID}")_*" -print -quit)"
if [[ -z "${DATASET_DIR}" ]]; then
  echo "Cannot find preprocessed metadata directory for Dataset$(printf '%03d' "${DATASET_ID}")" >&2
  exit 1
fi
TARGET_DIR="${DATASET_DIR}/${PLANS}_${CONFIGURATION}"
if [[ -d "${TARGET_DIR}" && -n "$(find "${TARGET_DIR}" -maxdepth 1 -type f -print -quit)" ]]; then
  echo "[preprocess] already complete: ${TARGET_DIR}"
  exit 0
fi

echo "[preprocess] Dataset${DATASET_ID} ${CONFIGURATION} -> ${TARGET_DIR}"
nnUNetv2_preprocess -d "${DATASET_ID}" -plans_name "${PLANS}" \
  -c "${CONFIGURATION}" -np "${NNUNET_PREPROCESS_WORKERS:-8}" --verbose
