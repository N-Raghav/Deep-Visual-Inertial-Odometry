#!/bin/bash
#SBATCH --job-name=vision_only_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=academic
#SBATCH --account=rbe577
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================
# Train Branch A on a GPU node. Assumes setup_env.slurm has
# already been run and the venv lives at ${VENV_DIR}.
#
# Submit with the dataset path:
#     sbatch --export=ALL,DATA_ROOT=/path/to/dataset train.slurm
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ -z "${DATA_ROOT:-}" ]; then
    echo "ERROR: DATA_ROOT is not set. Submit with:"
    echo "    sbatch --export=ALL,DATA_ROOT=/path/to/dataset train.slurm"
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"

echo "[train] data root    : ${DATA_ROOT}"
echo "[train] python       : $(python --version 2>&1)"
echo "[train] gpu          : $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'n/a')"

python train.py \
    --data_root "${DATA_ROOT}" \
    --epochs "${EPOCHS:-100}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --lr "${LR:-1e-4}" \
    --lambda_rot "${LAMBDA_ROT:-100.0}" \
    --sequence_length "${SEQUENCE_LENGTH:-10}" \
    --img_height "${IMG_HEIGHT:-224}" \
    --img_width "${IMG_WIDTH:-224}" \
    --checkpoint_dir "${CHECKPOINT_DIR:-checkpoints/vision_only}" \
    --log_dir "${LOG_DIR:-logs/vision_only}" \
    --num_workers "${NUM_WORKERS:-4}"
