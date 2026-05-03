#!/bin/bash
#SBATCH --job-name=gated_fusion_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ -z "${DATA_ROOT:-}" ] || [ -z "${CHECKPOINT:-}" ]; then
    echo "ERROR: DATA_ROOT and CHECKPOINT must be set."; exit 1
fi
mkdir -p "${PROJECT_DIR}/logs"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

python evaluate.py \
    --data_root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --imu_rate "${IMU_RATE:-30.0}" \
    --imu_context "${IMU_CONTEXT:-32}" \
    --results_dir "${RESULTS_DIR:-results/gated_fusion}"
