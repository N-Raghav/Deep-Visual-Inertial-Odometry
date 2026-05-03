#!/bin/bash
#SBATCH --job-name=gated_fusion_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Submit:
#   sbatch --export=ALL,DATA_ROOT=/path/to/data,\
#                  VISION_CKPT=../vision_only/checkpoints/branch_a/best.pt,\
#                  AIRIO_CKPT=../imu_only/checkpoints/airio/best.pt \
#          train.slurm

set -euo pipefail
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ -z "${DATA_ROOT:-}" ]; then
    echo "ERROR: DATA_ROOT not set."; exit 1
fi
mkdir -p "${PROJECT_DIR}/logs"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

VIS_ARG=""; [ -n "${VISION_CKPT:-}" ] && VIS_ARG="--vision_checkpoint ${VISION_CKPT}"
AIO_ARG=""; [ -n "${AIRIO_CKPT:-}"  ] && AIO_ARG="--airio_checkpoint ${AIRIO_CKPT}"

python train.py \
    --data_root "${DATA_ROOT}" \
    ${VIS_ARG} ${AIO_ARG} \
    --epochs "${EPOCHS:-100}" \
    --warmup_epochs "${WARMUP_EPOCHS:-5}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --lr "${LR:-1e-4}" \
    --lr_finetune "${LR_FINETUNE:-2e-5}" \
    --sequence_length "${SEQ_LEN:-10}" \
    --imu_context "${IMU_CONTEXT:-32}" \
    --imu_rate "${IMU_RATE:-30.0}" \
    --num_workers "${NUM_WORKERS:-4}"
