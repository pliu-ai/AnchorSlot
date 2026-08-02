#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_FOLDER:?Set MODEL_FOLDER to the nnU-Net trained-model directory}"
: "${INPUT_DIR:?Set INPUT_DIR to the folder containing *_0000 input files}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions}"
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

FOLD="${FOLD:-0}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
DEVICE="${DEVICE:-cuda}"
NPP="${NPP:-3}"
NPS="${NPS:-3}"
CONTINUE_FLAG=()
if [[ "${CONTINUE_PREDICTION:-0}" == "1" ]]; then
  CONTINUE_FLAG+=(--continue_prediction)
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python "${REPO_ROOT}/scripts/predict_structured_conditional.py" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -m "${MODEL_FOLDER}" \
  -f "${FOLD}" \
  -chk "${CHECKPOINT}" \
  -device "${DEVICE}" \
  --group_id all \
  --output_mode original \
  --fixed_merge_mode mean \
  -npp "${NPP}" \
  -nps "${NPS}" \
  --disable_progress_bar \
  "${CONTINUE_FLAG[@]}"
