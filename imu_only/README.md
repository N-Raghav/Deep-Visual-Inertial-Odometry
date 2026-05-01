# IMU-only Branch — AirIO + AirIMU + EKF

## Overview

This branch is a from-scratch PyTorch reimplementation of **AirIO**
(Qiu et al., RA-L 2025) plus its preceding IMU-correction network
**AirIMU** (Qiu et al. 2023) and the tightly coupled velocity-measurement
Extended Kalman Filter that fuses them.

The system takes a stream of body-frame IMU measurements and produces a
trajectory estimate without any visual sensor or control input. The
two design choices that make AirIO outperform earlier learning-based
inertial odometry on UAVs are:

1. **Body-frame representation.** The raw IMU is kept in the sensor's
   own body frame instead of being rotated to a global frame. This
   preserves the rotation-coupled gravity term `Rᵀg` which implicitly
   encodes attitude and is the dominant signal for highly dynamic UAV
   motion.
2. **Explicit attitude encoding.** The drone's attitude `ξ ∈ so(3)` is
   passed through a *separate* CNN encoder and fused with the IMU
   features before the recurrent backbone.

---

## Pipeline

```
                         raw IMU  (200 Hz, body frame)
                              │
                ┌─────────────┴──────────────┐
                ↓                            ↓
            AirIMU                      attitude ξ
   (correction + uncertainty)     (so(3) log of pose)
                │                            │
                ▼                            ▼
        â, ŵ, η̂_a, η̂_g                   ξ encoder
                │                            │
                └────────────┬───────────────┘
                             ↓
                    AirIO motion network
                  (CNN + bi-GRU + 2 heads)
                             │
                ┌────────────┴────────────┐
                ↓                         ↓
            ᴮv (body vel)           Σᵛ (diag cov)
                             ↓
                       Velocity EKF
                  state X = (R, ᴳv, ᴳp, b_a, b_g)
                   meas:  h(X) = Rᵀᴳv = ᴮv
                             │
                             ↓
                        Trajectory
```

---

## Files

```
imu_only/
├── model.py             AirIMUNet, AirIONet, CNNEncoder1D
├── loss.py              airio_loss (Huber + NLL), airimu_loss (integration)
├── dataset.py           IMUWindowDataset (sliding windows of IMU + GT)
├── ekf.py               15-state velocity-measurement EKF
├── utils.py             SO(3) ops, IMU integrator, GT velocity from poses
├── train_airimu.py      pre-train AirIMU
├── train_airio.py       train AirIO on top of frozen AirIMU
├── evaluate.py          full pipeline ATE/RTE/rotation + plots
├── test_pipeline.py     end-to-end smoke test on synthetic helical IMU
├── setup_env.slurm
├── train_airimu.slurm
├── train_airio.slurm
├── test.slurm
├── requirements.txt
└── README.md            this file
```

---

## Design Choices and Justification

### 1. Body-frame IMU input

**Choice:** Feed the raw `[acc, gyro]` directly, in the body frame.

**Justification:** The accelerometer measures specific force in the body
frame: `ᴮa = ᴮF/m + Rᵀᴳg` (Eq. 1 of the paper). This couples motion
force and attitude *linearly* (the gravity term is constant in the
world). After rotating to the global frame the coupling becomes
non-linear (`ᴳa = R̂ ᴮF/m + R̂ᵀg` interactions), which the network has
to disentangle. PCA on the latent features confirms body-frame inputs
need 3-8× fewer principal components to explain 95 % of the variance.
ATE drops by 66.7 % on the Blackbird dataset compared to global-frame
inputs.

**Reference:**
> Qiu, Y., Xu, C., Chen, Y., Zhao, S., Geng, J., Scherer, S., 2025.
> AirIO: Learning Inertial Odometry with Enhanced IMU Feature
> Observability. RA-L.

### 2. Explicit attitude encoding via `so(3)` log

**Choice:** Map the drone's orientation to its `so(3)` log
representation `ξ ∈ ℝ³` and feed it through a *separate* CNN encoder.

**Justification:** Quaternions have a double-cover discontinuity and
require unit-norm constraints; rotation matrices are 9-D and redundant.
The Lie-algebra log is a continuous, smooth, minimal representation
that is straightforward to differentiate. Encoding attitude
*alongside* (not concatenated to) the IMU stream gives the network a
dedicated pathway for orientation-related dynamics and yields an
additional 23.8 % accuracy improvement (Sec. V.B of the paper).

### 3. Bi-directional GRU backbone

**Choice:** Two-layer bidirectional GRU with hidden size 128, fed with
fused IMU + attitude features.

**Justification:** Body-frame velocity at sample `i` depends on the
local context of accelerations and angular rates around `i`. A
bidirectional model lets each prediction see both its past and future
within the 1000-sample (5 s) window. GRUs converge faster than LSTMs
on this task because the gating is simpler and the gradient path
shorter, which matters for the long sequence length.

### 4. Per-sample diagonal Gaussian uncertainty

**Choice:** Output a per-sample log-variance vector `log Σᵛ` and add a
Gaussian negative log-likelihood term to the velocity loss
(`λ = 1e-4`, Eq. 5 of the paper).

**Justification:** A constant measurement-noise EKF over-trusts
predictions at noisy time-steps and under-trusts them at clean ones.
A learned per-frame variance lets the EKF down-weight ambiguous
samples (e.g. heavy yaw maneuvers) and gives an ~17 % ATE improvement
on EuRoC after EKF integration. Diagonal covariance is sufficient at
the 200 Hz rate because the measurement is body-frame velocity which
has already been spatially decorrelated.

### 5. Huber loss on velocity

**Choice:** Huber (smooth-L1) loss with `δ = 0.005` instead of MSE
(Eq. 4).

**Justification:** Body-frame velocities at 200 Hz have occasional
outliers near aggressive maneuvers where the finite-difference label
`ᴮv = Rᵀ Δp/Δt` is itself noisy. Huber loss limits the gradient
magnitude on such outliers while remaining quadratic near zero,
preventing them from dominating the optimisation.

### 6. AirIMU pre-training with an integration loss

**Choice:** Pre-train AirIMU on short windows (default 20 IMU samples
≈ 100 ms) using a loss that integrates the corrected IMU forward in
time and supervises on the resulting end-of-window rotation, velocity,
and displacement against ground truth, plus a Gaussian NLL on the
predicted IMU uncertainty.

**Justification:** Forward-Euler integration error grows
super-linearly with window length, so windows are kept short. This
keeps the loss well-conditioned while still forcing AirIMU to learn
both the deterministic bias correction and an honest estimate of the
remaining noise.

### 7. Velocity-measurement EKF

**Choice:** 15-state error-state EKF (`δX = [δξ, δᴳv, δᴳp, δb_a, δb_g]`)
with the body-frame velocity from AirIO as the measurement and the
AirIMU log-variance as the per-sample process noise.

**Justification:** A pure feedforward AirIO has no notion of bias and
drifts when accel/gyro biases shift slowly. The EKF maintains explicit
bias states and uses the velocity measurement to observe them. The
measurement Jacobian (Eq. 11)
```
H_{δᴳv} = Rᵀ
H_{δξ } = Rᵀ [ᴳv]_×
```
makes both translational and rotational error states observable from a
single body-frame velocity measurement.

---

## Training

| Hyperparameter | AirIMU | AirIO | Source |
|----|----|----|----|
| Optimizer       | Adam        | Adam        | paper |
| LR              | 1e-3        | 1e-3        | paper |
| Scheduler       | ReduceLROnPlateau (factor 0.2, patience 5) | same | paper |
| Batch size      | 128         | 128         | paper |
| Window size     | 20 (≈100 ms) | 1000 (5 s) | paper for AirIO; AirIMU window is implementation choice |
| Step size       | 10          | 10          | paper |
| Dropout         | 0.5 in CNN  | 0.5 in CNN  | paper |
| `λ_c` (NLL)     | n/a         | 1e-4        | paper Eq. 3 |
| `δ` (Huber)     | n/a         | 0.005       | paper Eq. 4 |

Training order:

1. `python train_airimu.py --data_root <data>` — pre-trains AirIMU.
   Save the best checkpoint to `checkpoints/airimu/best.pt`.
2. `python train_airio.py --data_root <data> --airimu_checkpoint checkpoints/airimu/best.pt`
   — trains AirIO with the frozen AirIMU in the loop.

If you skip step 1, AirIO is trained directly on the raw IMU. This
matches the paper's "AirIO Net" baseline column.

---

## Evaluation

```bash
python evaluate.py \
    --data_root /path/to/dataset \
    --airio_checkpoint checkpoints/airio/best.pt \
    --airimu_checkpoint checkpoints/airimu/best.pt \
    --attitude_source ekf
```

Reports per-sequence and mean **ATE** (Eq. 13), **RTE** at the 5-second
interval used in the paper (Eq. 14), and the mean rotation error in
degrees. Writes top-down trajectory, per-sample translation error,
and per-sample rotation error plots to `results/imu_only/`.

The `--attitude_source` flag controls how the attitude is fed back into
AirIO during inference:

- `gt`  — uses ground truth (matches the paper's training-time
  protocol; gives an *upper bound* on AirIO accuracy).
- `ekf` — re-uses the EKF-estimated attitude in chunks of
  `--airio_chunk` samples, reflecting realistic real-time deployment.

---

## Cluster usage

```bash
sbatch setup_env.slurm                                              # one-time
sbatch test.slurm                                                   # smoke test
sbatch --export=ALL,DATA_ROOT=/path/to/dataset train_airimu.slurm
sbatch --export=ALL,DATA_ROOT=/path/to/dataset,\
              AIRIMU_CHECKPOINT=checkpoints/airimu/best.pt \
       train_airio.slurm
```

All SLURM scripts pick up overrides from environment variables
(`EPOCHS`, `BATCH_SIZE`, `LR`, `WINDOW_SIZE`, `STEP_SIZE`, …).

---

## References

1. Qiu, Y., Xu, C., Chen, Y., Zhao, S., Geng, J., Scherer, S., 2025.
   "AirIO: Learning Inertial Odometry with Enhanced IMU Feature
   Observability." *IEEE Robotics and Automation Letters*. arXiv:2501.15659.
2. Qiu, Y., Wang, C., Xu, Y., Chen, X., Zhou, Y., Xia, W., Scherer, S.,
   2023. "AirIMU: Learning Uncertainty Propagation for Inertial
   Odometry."
3. Liu, W., Caruso, F., Ilg, E., Dong, J., Mourikis, A. I., Daniilidis,
   K., Kumar, V., Engel, J., 2020. "TLIO: Tight Learned Inertial
   Odometry." *IEEE RA-L*.
4. Herath, S., Yan, H., Furukawa, Y., 2020. "RoNIN: Robust Neural
   Inertial Navigation in the Wild." *ICRA*.
5. Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H., 2019. "On the
   Continuity of Rotation Representations in Neural Networks." *CVPR*.
6. Forster, C., Carlone, L., Dellaert, F., Scaramuzza, D., 2015. "IMU
   Preintegration on Manifold for Efficient Visual-Inertial
   Maximum-A-Posteriori Estimation." *RSS*.
