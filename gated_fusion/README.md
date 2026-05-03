# Gated Tight Fusion (Vision + IMU)

## Overview

Tight fusion of BranchA (vision) and AirIO (IMU) features through a
learned per-dimension gate. At each time step the network produces a
gate vector ``g ∈ [0, 1]^D`` that softly mixes the projected vision
feature ``f_v`` and projected IMU feature ``f_i``:

    fused = g ⊙ f_v + (1 - g) ⊙ f_i

When vision is informative the network drives ``g → 1`` and the IMU
branch is muted; when vision is degraded the gate moves toward ``0``
and the IMU carries the prediction. The gate is interpretable — you
can plot it over time and read off "where the network is trusting
each modality."

The two backbones ([vision_only.BranchA](../vision_only/model.py),
[imu_only.AirIONet](../imu_only/model.py)) are reused as-is. Their
pretrained checkpoints are loaded into the fusion model and frozen
for the first ``--warmup_epochs`` epochs; the gate + heads are trained
on top, then the whole stack is jointly fine-tuned with a smaller
learning rate.

## Pipeline

```
        frames_t, frames_t1                 imu_acc, imu_gyro, attitude
                │                                       │
                ▼                                       ▼
            BranchA                                 AirIONet
       (frozen pretrained)                       (frozen pretrained)
                │                                       │
            features [B,T,128]                     bi-GRU output [B,T*W,256]
                │                            take last sample of every window
                │                                       │
                ▼                                       ▼
           vis_proj                                  imu_proj
                │                                       │
                └────────────────┬──────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ gate = sigmoid(MLP([v ; u])) │
                  │ fused = g ⊙ v + (1-g) ⊙ u    │
                  └──────────────┬───────────────┘
                                 ▼
                          LayerNorm + heads
                       ┌─────────┴─────────┐
                       ▼                   ▼
                   trans (3)          rot_6d (6)
                                          │
                                    Gram-Schmidt
                                          │
                                       R (3x3)
```

## Training

```bash
python train.py \
    --data_root /path/to/dataset \
    --vision_checkpoint ../vision_only/checkpoints/branch_a/best.pt \
    --airio_checkpoint  ../imu_only/checkpoints/airio/best.pt
```

| Hyperparameter | Default | Notes |
|---|---|---|
| optimizer | Adam | weight decay 1e-4 |
| lr (warm-up)        | 1e-4    | gate + heads only |
| lr (joint fine-tune) | 2e-5   | unfreezes both backbones |
| warmup_epochs       | 5       | epochs of frozen-backbone training |
| epochs              | 100     | total |
| batch size          | 8       | smaller than each branch alone — vision images dominate memory |
| sequence_length     | 10      | frame pairs per LSTM unroll |
| imu_context         | 32      | IMU samples per pair (≈1 s at 30 Hz) |
| lambda_rot          | 100     | translation/rotation balance |
| gradient clip       | 1.0     | full network |

## Evaluation

```bash
python evaluate.py --data_root /path/to/dataset \
                   --checkpoint checkpoints/gated_fusion/best.pt
```

Reports per-sequence and mean **ATE**, **RTE-5s**, **mean rotation
deg**. Writes four plots per sequence:

- `<seq>_trajectory.png` — top-down trajectory
- `<seq>_trans_error.png` — per-pair translation error
- `<seq>_rot_error.png` — per-pair rotation error in degrees
- `<seq>_gate.png` — mean gate value over time. Read this as: 1 means
  "trusting vision," 0 means "trusting IMU." Useful to confirm the
  network learns sensible cross-modal behaviour.

## Files

```
gated_fusion/
├── model.py            GatedFusionNet wrapping BranchA + AirIO
├── train.py            warm-up + joint fine-tune
├── evaluate.py         ATE / RTE / rotation + plots + gate plot
├── test_pipeline.py    smoke test
├── setup_env.slurm
├── train.slurm
├── eval.slurm
├── test.slurm
├── requirements.txt
└── README.md
```

## References

- Clark, R., Wang, S., Wen, H., Markham, A., Trigoni, N., 2017.
  "VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning
  Problem." *AAAI*.
- Chen, C., Rosa, S., Miao, Y., Lu, C. X., Wu, W., Markham, A.,
  Trigoni, N., 2019. "Selective Sensor Fusion for Neural Visual-Inertial
  Odometry." *CVPR*.
- Wang, S., Clark, R., Wen, H., Trigoni, N., 2017. "DeepVO." *ICRA*.
- Qiu, Y. et al., 2025. "AirIO." *RA-L*.
