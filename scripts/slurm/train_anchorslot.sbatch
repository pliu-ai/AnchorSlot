#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_train
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
: "${REPO_ROOT:=${SLURM_SUBMIT_DIR}}"
: "${CONDA_ENV:?Set CONDA_ENV to the AnchorSlot conda environment path}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
bash scripts/train_anchorslot.sh "${DATASET_ID:-200}" "${CONFIGURATION:-3d_lowres_large_patch}" "${FOLD:-0}"
