#!/bin/bash
#SBATCH --job-name=imu_only_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# ============================================================
# Smoke test for the IMU-only branch. Run after setup_env.slurm.
#
#     sbatch test.slurm
#
# Verifies SO(3) ops, the IMU integrator, AirIMU + AirIO
# forward/backward, and the EKF on a synthetic helical
# trajectory generated in /tmp on the compute node.
# ============================================================

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
