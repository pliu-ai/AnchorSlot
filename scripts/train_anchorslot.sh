#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="${1:-200}"
CONFIGURATION="${2:-3d_lowres_large_patch}"
FOLD="${3:-0}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER="nnUNetTrainerHierarchicalParallelAnchorSlot"

: "${nnUNet_raw:?Set nnUNet_raw before training}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before training}"
: "${nnUNet_results:?Set nnUNet_results before training}"

if [[ "${DATASET_ID}" == "202" && "${ANCHORSLOT_ALLOW_LEAKY_REFERENCE:-0}" != "1" ]]; then
  echo "[ERROR] Dataset202 merged Dataset200 imagesTs/labelsTs into training." >&2
  echo "        Refusing a leaky paper run. Use Dataset200 (default)." >&2
  echo "        Set ANCHORSLOT_ALLOW_LEAKY_REFERENCE=1 only for historical reconstruction." >&2
  exit 2
fi

source "${REPO_ROOT}/configs/hierarchical_parallel_anchorslot.env"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[AnchorSlot] dataset=${DATASET_ID} configuration=${CONFIGURATION} fold=${FOLD}"
echo "[AnchorSlot] trainer=${TRAINER} plans=${PLANS}"
nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}"
