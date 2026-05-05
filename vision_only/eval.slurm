#!/bin/bash
#SBATCH --job-name=vision_only_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=academic
#SBATCH --account=rbe577
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Submit with:
#   sbatch --export=ALL,DATA_ROOT=/path/to/dataset,\
#                  CHECKPOINT=/path/to/best.pt \
#          eval.slurm

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ -z "${DATA_ROOT:-}" ]; then
    echo "ERROR: DATA_ROOT is not set."
    exit 1
fi
if [ -z "${CHECKPOINT:-}" ]; then
    echo "ERROR: CHECKPOINT is not set."
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

echo "[eval] data root   : ${DATA_ROOT}"
echo "[eval] checkpoint  : ${CHECKPOINT}"
echo "[eval] python      : $(python --version 2>&1)"
echo "[eval] gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'n/a')"

python evaluate.py \
    --data_root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --results_dir "${RESULTS_DIR:-results/vision_only}" \
    --img_height "${IMG_HEIGHT:-224}" \
    --img_width "${IMG_WIDTH:-224}"
