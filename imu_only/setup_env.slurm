#!/bin/bash
#SBATCH --job-name=imu_only_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# IMU-only branch environment setup.
# Creates a Python virtualenv and installs everything required
# by AirIMU, AirIO, the EKF, training scripts, and the smoke
# test.
#
# Submit with:
#     sbatch setup_env.slurm
# Override the CUDA build via:
#     TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 sbatch setup_env.slurm
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================="
echo "IMU-only branch environment setup"
echo "Date          : $(date)"
echo "Host          : $(hostname)"
echo "Project dir   : ${PROJECT_DIR}"
echo "Venv dir      : ${VENV_DIR}"
echo "Torch index   : ${TORCH_INDEX_URL}"
echo "Python        : $(python3 --version 2>&1)"
echo "=========================================="

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
pip install -r "${REQ_FILE}"

python - <<'PY'
import importlib

mods = ["torch", "torchvision", "numpy", "matplotlib", "tensorboard", "tqdm", "PIL"]
for m in mods:
    importlib.import_module(m)
    print(f"  ok  {m}")

import torch
print(f"torch        : {torch.__version__}")
print(f"cuda build   : {torch.version.cuda}")
print(f"cuda runtime : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0     : {torch.cuda.get_device_name(0)}")
PY

echo "[setup] done. Activate with: source ${VENV_DIR}/bin/activate"
