#!/bin/bash
#SBATCH --job-name=imu_only_eval
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
#                  AIRIO_CHECKPOINT=/path/to/airio/best.pt,\
#                  AIRIMU_CHECKPOINT=/path/to/airimu/best.pt \
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
if [ -z "${AIRIO_CHECKPOINT:-}" ]; then
    echo "ERROR: AIRIO_CHECKPOINT is not set."
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

echo "[eval] data root      : ${DATA_ROOT}"
echo "[eval] airio ckpt     : ${AIRIO_CHECKPOINT}"
echo "[eval] airimu ckpt    : ${AIRIMU_CHECKPOINT:-none}"
echo "[eval] python         : $(python --version 2>&1)"
echo "[eval] gpu            : $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'n/a')"

python evaluate.py \
    --data_root "${DATA_ROOT}" \
    --airio_checkpoint "${AIRIO_CHECKPOINT}" \
    ${AIRIMU_ARG} \
    --results_dir "${RESULTS_DIR:-results/imu_only}" \
    --imu_rate "${IMU_RATE:-1000.0}" \
    --airio_chunk "${AIRIO_CHUNK:-1000}" \
    --rte_interval "${RTE_INTERVAL:-5.0}"
