#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/../data}"
TEXTURES="${2:-$SCRIPT_DIR/../data/textures}"
BLENDER="${3:-$SCRIPT_DIR/../blender-5.1.0-linux-x64/blender}"

TRAJS=(figure8 oval clover lemniscate figure8_3d)
N=5

for traj in "${TRAJS[@]}"; do
    for seed in $(seq 0 $((N - 1))); do
        out="$DATA_DIR/${traj}_${seed}"
        echo "==> $traj  seed=$seed  out=$out"

        python "$SCRIPT_DIR/generate_imu.py" \
            --out "$out" \
            --traj "$traj" \
            --seed "$seed"

        "$BLENDER" --background --gpu-backend vulkan --python "$SCRIPT_DIR/blender_render.py" -- \
            --data "$out" \
            --textures "$TEXTURES" \
            --seed "$seed"
    done
done

echo "done -- $(( ${#TRAJS[@]} * N )) sequences in $DATA_DIR"
