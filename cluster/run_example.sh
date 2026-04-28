#!/usr/bin/env bash
# Example runner: generate IMU + render with Blender GPU inside the container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIF="${SIF:-$SCRIPT_DIR/p4ph2.sif}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
TEXTURES="${TEXTURES:-$DATA_DIR/textures}"
TRAJ="${TRAJ:-figure8}"
SEED="${SEED:-0}"
OUT="$DATA_DIR/${TRAJ}_${SEED}"

mkdir -p "$OUT"

apptainer run --nv \
    --bind "$REPO_DIR:/workspace" \
    --bind "$DATA_DIR:/data" \
    --app imu "$SIF" \
        --out "/data/${TRAJ}_${SEED}" \
        --traj "$TRAJ" \
        --seed "$SEED"

apptainer run --nv \
    --bind "$REPO_DIR:/workspace" \
    --bind "$DATA_DIR:/data" \
    --app render "$SIF" \
        --data "/data/${TRAJ}_${SEED}" \
        --textures "/data/textures" \
        --seed "$SEED"

echo "done -> $OUT"
