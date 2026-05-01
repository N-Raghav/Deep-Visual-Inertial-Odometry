#!/bin/bash
#SBATCH --job-name=airimu_train
#SBATCH --output=logs/airimu_train_%j.out
#SBATCH --error=logs/airimu_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================
# Pre-train AirIMU on a GPU node. Submit with:
#     sbatch --export=ALL,DATA_ROOT=/path/to/dataset train_airimu.slurm
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
    echo "ERROR: DATA_ROOT is not set."
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

python train_airimu.py \
    --data_root "${DATA_ROOT}" \
    --epochs "${EPOCHS:-60}" \
    --batch_size "${BATCH_SIZE:-128}" \
    --lr "${LR:-1e-3}" \
    --imu_rate "${IMU_RATE:-200.0}" \
    --window_size "${WINDOW_SIZE:-20}" \
    --step_size "${STEP_SIZE:-10}" \
    --checkpoint_dir "${CHECKPOINT_DIR:-checkpoints/airimu}" \
    --log_dir "${LOG_DIR:-logs/airimu}" \
    --num_workers "${NUM_WORKERS:-4}"
