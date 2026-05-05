#!/bin/bash
#SBATCH --job-name=cross_attn_eval
#SBATCH --output=/home/rnallaperumal/deep-vio/cross_attention/logs/eval_%j.out
#SBATCH --error=/home/rnallaperumal/deep-vio/cross_attention/logs/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=academic
#SBATCH --account=rbe577
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="/home/rnallaperumal/deep-vio/cross_attention"
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
    --imu_rate "${IMU_RATE:-1000.0}" \
    --imu_context "${IMU_CONTEXT:-100}" \
    --results_dir "${RESULTS_DIR:-results/cross_attention}"
