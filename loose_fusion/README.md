# Loose Fusion (Vision + IMU)

## Overview

Loose coupling: BranchA (vision) and AirIMU + AirIO (IMU) run
independently. Their outputs are fed as separate measurements into a
15-state Extended Kalman Filter that reconciles them. Neither network
sees the other's features or gradients — the fusion lives entirely in
the EKF.

This is the simplest of the three fusion approaches and **does not
require any training**. You run pre-trained backbones and only tune
the per-modality measurement covariances.

## Pipeline

```
            corrected IMU (AirIMU)
                   │
                   ▼
          ┌────────────────┐                     RGB frames
          │  EKF predict   │                          │
          └───────┬────────┘                          ▼
                  │                          ┌────────────────┐
                  ▼                          │   BranchA      │
          ┌────────────────┐                 │  (vision)      │
          │  EKF update v  │ ◄── ᴮv from     └───────┬────────┘
          │     (AirIO)    │                          │
          └───────┬────────┘                          ▼
                  │                          ┌────────────────┐
                  ▼                          │ ΔR_vis, Δt_vis │
          ┌────────────────┐                 └───────┬────────┘
          │ EKF update v_v │ ◄────────────────────── │
          │   (vision)     │                         │
          └───────┬────────┘                         │
                  ▼                                  │
          ┌────────────────┐                         │
          │ EKF update R   │ ◄───────────────────────┘
          │   (vision)     │
          └───────┬────────┘
                  ▼
              trajectory
```

## Files

```
loose_fusion/
├── fusion_ekf.py        FusionEKF: VelocityEKF + vision velocity + vision rotation update
├── evaluate.py          end-to-end runner with three pretrained backbones
├── test_pipeline.py     synthetic-data smoke test
├── setup_env.slurm
├── eval.slurm
├── test.slurm
├── requirements.txt
└── README.md
```

## Required pretrained checkpoints

| Backbone | Trained by | Default location |
|---|---|---|
| BranchA            | `vision_only/train.py`        | `vision_only/checkpoints/branch_a/best.pt` |
| AirIO              | `imu_only/train_airio.py`     | `imu_only/checkpoints/airio/best.pt` |
| AirIMU (optional)  | `imu_only/train_airimu.py`    | `imu_only/checkpoints/airimu/best.pt` |

If `AIRIMU_CKPT` is not provided, the EKF uses raw IMU measurements
with a constant default process-noise covariance — still works, just
less accurate.

## Running on the cluster

```bash
sbatch setup_env.slurm
sbatch test.slurm                                                            # smoke test
sbatch --export=ALL,DATA_ROOT=/path/to/dataset,\
              VISION_CKPT=../vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=../imu_only/checkpoints/airio/best.pt,\
              AIRIMU_CKPT=../imu_only/checkpoints/airimu/best.pt \
       eval.slurm
```

## Tuning knobs

The two values you'll likely want to sweep:

- `--vision_vel_sigma`     (default 0.05 m/s) — assumed std of vision
  velocity. Lower → trust vision more.
- `--vision_rot_sigma_deg` (default 0.5°)     — assumed std of vision
  rotation increments.

You can also disable either modality at evaluation time:

- `--no_vision`    — IMU-only (matches `imu_only/evaluate.py`).
- `--no_imu_update` — vision + IMU prediction but no AirIO update; useful
  to see how much AirIO contributes vs. the vision update.

## Why loose first

Loose fusion is a useful sanity check before training a tight model:
if the EKF can't beat either branch alone, something is wrong with the
backbones (or with the measurement covariances) and a tight fusion
network is unlikely to fix it.

## References

1. Mourikis, A. I., Roumeliotis, S. I., 2007. "A Multi-State Constraint
   Kalman Filter for Vision-aided Inertial Navigation." *ICRA*.
2. Forster, C., Carlone, L., Dellaert, F., Scaramuzza, D., 2015. "IMU
   Preintegration on Manifold for Efficient Visual-Inertial
   Maximum-A-Posteriori Estimation." *RSS*.
3. Qiu, Y. et al., 2025. "AirIO: Learning Inertial Odometry with
   Enhanced IMU Feature Observability." *RA-L*.
