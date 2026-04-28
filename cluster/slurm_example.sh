#!/usr/bin/env bash
#SBATCH --job-name=p4ph2-render
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=p4ph2-%j.out

# Edit these for your cluster layout.
SIF=/path/to/p4ph2.sif
REPO=/path/to/P4Ph2
DATA=/path/to/data

# Many HPC sites require loading a module to expose `apptainer`/`singularity`.
# module load apptainer

# Run the full batch (5 trajectories x 5 seeds).
apptainer run --nv \
    --bind "$REPO:/workspace" \
    --bind "$DATA:/data" \
    --app batch "$SIF" \
        /data /data/textures /usr/local/bin/blender
