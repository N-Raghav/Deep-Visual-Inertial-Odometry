#!/bin/bash
#SBATCH --job-name=vision_only_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Branch A environment setup.
#
# Creates a Python virtual environment and installs every package
# required by model.py / dataset.py / train.py / evaluate.py.
# Submit with:
#     sbatch setup_env.slurm
# Then activate the env in a training job with:
#     source $VENV_DIR/bin/activate
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Cluster module setup. Adjust the module names to match your
# cluster (run `module avail python` and `module avail cuda` to
# see what is available). The defaults below are conservative.
# ------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

# ------------------------------------------------------------
# Paths.
# ------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
REQ_FILE="${PROJECT_DIR}/requirements.txt"

# Pick a CUDA wheel index. Override TORCH_INDEX_URL when submitting
# if your cluster needs a different CUDA build, e.g.:
#     TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 sbatch setup_env.slurm
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================="
echo "Branch A environment setup"
echo "Date          : $(date)"
echo "Host          : $(hostname)"
echo "Project dir   : ${PROJECT_DIR}"
echo "Venv dir      : ${VENV_DIR}"
echo "Torch index   : ${TORCH_INDEX_URL}"
echo "Python        : $(python3 --version 2>&1)"
echo "=========================================="

# ------------------------------------------------------------
# Create / refresh the virtual environment.
# ------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
else
    echo "[setup] reusing existing virtualenv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# PyTorch + torchvision (CUDA build).
# ------------------------------------------------------------
echo "[setup] installing torch/torchvision from ${TORCH_INDEX_URL}"
pip install --index-url "${TORCH_INDEX_URL}" torch torchvision

# ------------------------------------------------------------
# Project dependencies.
# ------------------------------------------------------------
echo "[setup] installing project requirements"
pip install -r "${REQ_FILE}"

# ------------------------------------------------------------
# Sanity check — fail the job if anything is broken.
# ------------------------------------------------------------
echo "[setup] verifying install"
python - <<'PY'
import importlib, sys

mods = [
    "torch", "torchvision", "numpy", "PIL", "matplotlib",
    "tensorboard", "tqdm",
]
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

echo "[setup] done. Activate with:"
echo "    source ${VENV_DIR}/bin/activate"
