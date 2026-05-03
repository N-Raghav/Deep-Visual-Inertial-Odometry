#!/bin/bash
#SBATCH --job-name=cross_attn_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
mkdir -p "${PROJECT_DIR}/logs"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
python test_pipeline.py
