#!/bin/bash
#SBATCH --job-name=airio_train
#SBATCH --output=logs/airio_train_%j.out
#SBATCH --error=logs/airio_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================
# Train AirIO on top of a frozen AirIMU. Submit with:
#     sbatch --export=ALL,DATA_ROOT=/path/to/dataset,\
#                    AIRIMU_CHECKPOINT=checkpoints/airimu/best.pt \
#            train_airio.slurm
#
# AIRIMU_CHECKPOINT is optional — if unset AirIO is trained on
# raw IMU (matches the paper's "AirIO Net" baseline column).
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

AIRIMU_ARG=""
if [ -n "${AIRIMU_CHECKPOINT:-}" ]; then
    AIRIMU_ARG="--airimu_checkpoint ${AIRIMU_CHECKPOINT}"
fi

python train_airio.py \
    --data_root "${DATA_ROOT}" \
    ${AIRIMU_ARG} \
    --epochs "${EPOCHS:-100}" \
    --batch_size "${BATCH_SIZE:-128}" \
    --lr "${LR:-1e-3}" \
    --imu_rate "${IMU_RATE:-200.0}" \
    --window_size "${WINDOW_SIZE:-1000}" \
    --step_size "${STEP_SIZE:-10}" \
    --lambda_c "${LAMBDA_C:-1e-4}" \
    --huber_delta "${HUBER_DELTA:-0.005}" \
    --checkpoint_dir "${CHECKPOINT_DIR:-checkpoints/airio}" \
    --log_dir "${LOG_DIR:-logs/airio}" \
    --num_workers "${NUM_WORKERS:-4}"
