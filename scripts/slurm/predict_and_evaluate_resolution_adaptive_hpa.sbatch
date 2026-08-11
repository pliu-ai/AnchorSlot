#!/usr/bin/env bash
#SBATCH --job-name=rahpa_holdout
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/weilab/liupeng/code/projects/AnchorSlot}"
NNUNET_DATA_ROOT="${NNUNET_DATA_ROOT:-/projects/weilab/liupeng/data/nnUNet}"
CONDA_ENV="${CONDA_ENV:-/projects/weilab/liupeng/conda/envs/ssl_seg}"
CSC_PYTHON="${CSC_PYTHON:-/projects/weilab/liupeng/conda/envs/csc/bin/python}"
CSC_REPO="${CSC_REPO:-/projects/weilab/liupeng/code/projects/cellmap-segmentation-challenge-upstream-0300239}"
EXPECTED_EVALUATOR_COMMIT="${EXPECTED_EVALUATOR_COMMIT:-0300239cd0b4867d4bab008aa9e95161b2442d93}"
MODEL_FOLDER="${MODEL_FOLDER:-${NNUNET_DATA_ROOT}/nnUNet_results/Dataset200_Dataset101_CellMap_model_high_res/nnUNetTrainerResolutionAdaptiveHierarchicalParallelAnchorSlot__nnUNetPlans__3d_lowres_large_patch}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
RUN_ROOT="${RUN_ROOT:-/projects/weilab/liupeng/runs/Anchorslot/resolution_adaptive_hierarchical_parallel_anchorslot_exp01}"
TRUTH_ROOT="${TRUTH_ROOT:-/projects/weilab/liupeng/runs/Anchorslot/evaluation_protocol}"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  RESOLUTION="4nm"
  VOXEL_SIZE_NM="4"
  DATASET="Dataset200_Dataset101_CellMap_model_high_res"
  TRUTH_LOCK="${TRUTH_ROOT}/high_4nm/truth_lock.json"
  COMMON_TRUTH_LOCK="${TRUTH_ROOT}/common_parent_instance_high_4nm/truth_lock.json"
  PREPROCESSING_ARGS=()
else
  RESOLUTION="32nm"
  VOXEL_SIZE_NM="32"
  DATASET="Dataset201_Dataset101_CellMap_model_low_res"
  TRUTH_LOCK="${TRUTH_ROOT}/low_32nm/truth_lock.json"
  COMMON_TRUTH_LOCK="${TRUTH_ROOT}/low_32nm/truth_lock.json"
  # The shared network is defined by the 4-nm model folder, but the 32-nm
  # stream was trained with Dataset201's native-resolution preprocessing plan.
  PREPROCESSING_ARGS=(
    --preprocessing_dataset_dir "${NNUNET_DATA_ROOT}/nnUNet_preprocessed/${DATASET}"
    --preprocessing_configuration "${NNUNET_RAHPA_AUX_CONFIGURATION:-3d_fullres}"
  )
fi

INPUT_DIR="${NNUNET_DATA_ROOT}/nnUNet_raw/${DATASET}/imagesTs"
OUTPUT_ROOT="${RUN_ROOT}/${RESOLUTION}"
PREDICTIONS="${OUTPUT_ROOT}/predictions"
EVALUATION="${OUTPUT_ROOT}/official_evaluation"
COMMON_EVALUATION="${OUTPUT_ROOT}/official_evaluation_common_parent_instance"
test -f "${MODEL_FOLDER}/fold_all/${CHECKPOINT}"
test -d "${INPUT_DIR}"
test -f "${TRUTH_LOCK}"
test -f "${COMMON_TRUTH_LOCK}"
mkdir -p "${PREDICTIONS}" "${EVALUATION}" "${COMMON_EVALUATION}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export nnUNet_raw="${NNUNET_DATA_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_DATA_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_DATA_ROOT}/nnUNet_results"
cd "${REPO_ROOT}"

python scripts/predict_structured_conditional.py \
  -i "${INPUT_DIR}" \
  -o "${PREDICTIONS}" \
  -m "${MODEL_FOLDER}" \
  -f all \
  -chk "${CHECKPOINT}" \
  -device cuda \
  --group_id 0 \
  --output_mode original \
  --voxel_size_nm "${VOXEL_SIZE_NM}" \
  "${PREPROCESSING_ARGS[@]}" \
  --continue_prediction \
  --disable_progress_bar \
  -npp "${NPP:-3}" \
  -nps "${NPS:-3}"

"${CSC_PYTHON}" scripts/cellmap_evaluation_pipeline.py score \
  --predictions "${PREDICTIONS}" \
  --truth-lock "${TRUTH_LOCK}" \
  --output-dir "${EVALUATION}" \
  --csc-repo "${CSC_REPO}" \
  --evaluator-python "${CSC_PYTHON}" \
  --expected-evaluator-commit "${EXPECTED_EVALUATOR_COMMIT}" \
  --model RA-HPA \
  --resolution "${RESOLUTION}" \
  --anchorslot-repo "${REPO_ROOT}" \
  --max-workers "${MAX_WORKERS:-4}" \
  --per-instance-threads "${PER_INSTANCE_THREADS:-1}" \
  --resume

if [[ "${COMMON_TRUTH_LOCK}" != "${TRUTH_LOCK}" ]]; then
  "${CSC_PYTHON}" scripts/cellmap_evaluation_pipeline.py score \
    --predictions "${PREDICTIONS}" \
    --truth-lock "${COMMON_TRUTH_LOCK}" \
    --output-dir "${COMMON_EVALUATION}" \
    --csc-repo "${CSC_REPO}" \
    --evaluator-python "${CSC_PYTHON}" \
    --expected-evaluator-commit "${EXPECTED_EVALUATOR_COMMIT}" \
    --model RA-HPA \
    --resolution "${RESOLUTION}" \
    --anchorslot-repo "${REPO_ROOT}" \
    --max-workers "${MAX_WORKERS:-4}" \
    --per-instance-threads "${PER_INSTANCE_THREADS:-1}" \
    --resume
fi
