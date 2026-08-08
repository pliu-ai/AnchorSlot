#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_rahpa
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/weilab/liupeng/code/projects/AnchorSlot}"
CONDA_ENV="${CONDA_ENV:-/projects/weilab/liupeng/conda/envs/ssl_seg}"
NNUNET_DATA_ROOT="${NNUNET_DATA_ROOT:-/projects/weilab/liupeng/data/nnUNet}"
DATASET_ID="${DATASET_ID:-200}"
CONFIGURATION="${CONFIGURATION:-3d_lowres_large_patch}"
# Production uses every development crop and evaluates only on the disjoint
# high-/low-resolution imagesTs cohorts. Override FOLD for debugging only.
FOLD="${FOLD:-all}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER="nnUNetTrainerResolutionAdaptiveHierarchicalParallelAnchorSlot"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export nnUNet_raw="${NNUNET_DATA_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_DATA_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_DATA_ROOT}/nnUNet_results"
export NNUNET_RAHPA_NATIVE_LABELS_ROOT="${NNUNET_RAHPA_NATIVE_LABELS_ROOT:-${nnUNet_raw}/Dataset101_CellMap/labelsTr_split}"
# Two augmented streams run concurrently; cap each stream's worker count.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
source "${REPO_ROOT}/configs/resolution_adaptive_hierarchical_anchorslot.env"
cd "${REPO_ROOT}"

PRIMARY_DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset$(printf '%03d' "${DATASET_ID}")_*" -print -quit)"
AUX_DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset$(printf '%03d' "${NNUNET_RAHPA_AUX_DATASET_ID}")_*" -print -quit)"
if [[ -z "${PRIMARY_DATASET_DIR}" || -z "${AUX_DATASET_DIR}" ]]; then
  echo "Missing primary or auxiliary preprocessed dataset directory" >&2
  exit 1
fi
if [[ ! -d "${PRIMARY_DATASET_DIR}/${PLANS}_${CONFIGURATION}" ]]; then
  echo "Missing ${PRIMARY_DATASET_DIR}/${PLANS}_${CONFIGURATION}; preprocess it first" >&2
  exit 1
fi
if [[ ! -d "${AUX_DATASET_DIR}/${PLANS}_${NNUNET_RAHPA_AUX_CONFIGURATION}" ]]; then
  echo "Missing ${AUX_DATASET_DIR}/${PLANS}_${NNUNET_RAHPA_AUX_CONFIGURATION}; preprocess it first" >&2
  exit 1
fi

echo "[train] shared 4/32-nm RA-HPA; primary=${DATASET_ID}/${CONFIGURATION}; auxiliary=${NNUNET_RAHPA_AUX_DATASET_ID}/${NNUNET_RAHPA_AUX_CONFIGURATION}"
echo "[train] native annotation masks: ${NNUNET_RAHPA_NATIVE_LABELS_ROOT}"

CONTINUE_FLAG=()
[[ "${RESUME:-0}" == "1" ]] && CONTINUE_FLAG=(--c)
nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" -p "${PLANS}" "${CONTINUE_FLAG[@]}"
