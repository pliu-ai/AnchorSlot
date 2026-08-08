#!/usr/bin/env bash
#SBATCH --job-name=anchorslot_csc_summary
#SBATCH --partition=interactive
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/projects/weilab/liupeng/code/projects/AnchorSlot}"
PYTHON="${PYTHON:-/projects/weilab/liupeng/conda/envs/csc/bin/python}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/configs/cellmap/evaluation_registry.tsv}"
SUMMARY_DIR="${SUMMARY_DIR:-/projects/weilab/liupeng/runs/Anchorslot/evaluation_matrix/summary}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
"${PYTHON}" scripts/cellmap_evaluation_pipeline.py summarize \
  --registry "${REGISTRY}" \
  --output-dir "${SUMMARY_DIR}"
