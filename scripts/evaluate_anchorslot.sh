#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PREDICTIONS:?Set PREDICTIONS to the prediction directory}"
: "${GROUND_TRUTH:?Set GROUND_TRUTH to the labelsTs directory}"
: "${UNIQUE_LABELS_JSON:?Set UNIQUE_LABELS_JSON to the per-case present-label JSON}"

DATASET_JSON="${DATASET_JSON:-${REPO_ROOT}/configs/cellmap/paper_dataset200/dataset.json}"
PLANS_JSON="${PLANS_JSON:-${REPO_ROOT}/configs/cellmap/paper_dataset200/nnUNetPlans.json}"

nnUNetv2_evaluate_folder "${GROUND_TRUTH}" "${PREDICTIONS}" \
  -djfile "${DATASET_JSON}" \
  -pfile "${PLANS_JSON}"

python "${REPO_ROOT}/scripts/reevaluate_metrics.py" \
  --input_root "${PREDICTIONS}" \
  --unique_labels_path "${UNIQUE_LABELS_JSON}"
