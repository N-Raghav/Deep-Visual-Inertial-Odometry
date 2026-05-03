#!/bin/bash
#SBATCH --job-name=loose_fusion_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Run loose-fusion evaluation on test sequences.
# Submit with:
#     sbatch --export=ALL,DATA_ROOT=/path/to/dataset,\
#                    VISION_CKPT=/path/vision_only/checkpoints/best.pt,\
#                    AIRIO_CKPT=/path/imu_only/checkpoints/airio/best.pt,\
#                    AIRIMU_CKPT=/path/imu_only/checkpoints/airimu/best.pt \
#            eval.slurm
# AIRIMU_CKPT is optional.
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ -z "${DATA_ROOT:-}" ] || [ -z "${VISION_CKPT:-}" ] || [ -z "${AIRIO_CKPT:-}" ]; then
    echo "ERROR: DATA_ROOT, VISION_CKPT, AIRIO_CKPT must all be set."
    exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

AIRIMU_ARG=""
if [ -n "${AIRIMU_CKPT:-}" ]; then
    AIRIMU_ARG="--airimu_checkpoint ${AIRIMU_CKPT}"
fi

python evaluate.py \
    --data_root "${DATA_ROOT}" \
    --vision_checkpoint "${VISION_CKPT}" \
    --airio_checkpoint "${AIRIO_CKPT}" \
    ${AIRIMU_ARG} \
    --imu_rate "${IMU_RATE:-30.0}" \
    --vision_vel_sigma "${VIS_VEL_SIGMA:-0.05}" \
    --vision_rot_sigma_deg "${VIS_ROT_SIGMA_DEG:-0.5}" \
    --results_dir "${RESULTS_DIR:-results/loose_fusion}"
