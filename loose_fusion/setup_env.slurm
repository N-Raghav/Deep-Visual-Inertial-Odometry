#!/bin/bash
#SBATCH --job-name=loose_fusion_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Loose-fusion environment setup. The branch needs only the same
# Python deps as vision_only / imu_only — there is no training,
# only evaluation, so no extra heavy libraries.
# ============================================================

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load python/3.10 || module load python/3.11 || module load python || true
    module load cuda/12.1   || module load cuda/11.8   || module load cuda     || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${PROJECT_DIR}/logs"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
pip install -r "${PROJECT_DIR}/requirements.txt"

python - <<'PY'
import torch
print(f"torch        : {torch.__version__}")
print(f"cuda runtime : {torch.cuda.is_available()}")
PY
