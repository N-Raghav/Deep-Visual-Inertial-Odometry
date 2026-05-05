#!/bin/bash
# Waits for epoch 040 in both training logs, cancels jobs, submits evals.

GATED_LOG="/home/rnallaperumal/deep-vio/gated_fusion/logs/train_1999539.out"
CROSS_LOG="/home/rnallaperumal/deep-vio/cross_attention/logs/train_1999540.out"
GATED_JOB=1999539
CROSS_JOB=1999540

echo "[watcher] Waiting for epoch 040 in both logs..."

while true; do
    GATED_DONE=0
    CROSS_DONE=0
    grep -q "epoch 040/100" "$GATED_LOG" 2>/dev/null && GATED_DONE=1
    grep -q "epoch 040/100" "$CROSS_LOG"  2>/dev/null && CROSS_DONE=1

    if [ "$GATED_DONE" -eq 1 ] && [ "$CROSS_DONE" -eq 1 ]; then
        echo "[watcher] Both jobs reached epoch 040. Cancelling..."
        scancel $GATED_JOB $CROSS_JOB
        sleep 5

        echo "[watcher] Submitting gated_fusion eval..."
        sbatch --export=ALL,\
DATA_ROOT=/home/rnallaperumal/vio_dataset_new,\
CHECKPOINT=/home/rnallaperumal/deep-vio/gated_fusion/checkpoints/gated_fusion/best.pt,\
VENV_DIR=/home/rnallaperumal/CVP1Ph2/cv_env \
/home/rnallaperumal/deep-vio/gated_fusion/eval.slurm

        echo "[watcher] Submitting cross_attention eval..."
        sbatch --export=ALL,\
DATA_ROOT=/home/rnallaperumal/vio_dataset_new,\
CHECKPOINT=/home/rnallaperumal/deep-vio/cross_attention/checkpoints/cross_attention/best.pt,\
VENV_DIR=/home/rnallaperumal/CVP1Ph2/cv_env \
/home/rnallaperumal/deep-vio/cross_attention/eval.slurm

        echo "[watcher] Done."
        exit 0
    fi

    echo "[watcher $(date +%H:%M:%S)] gated=${GATED_DONE}/1 cross=${CROSS_DONE}/1 — sleeping 60s..."
    sleep 60
done
