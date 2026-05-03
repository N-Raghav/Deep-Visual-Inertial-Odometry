# Deep Visual-Inertial Odometry — Project Notes

This repository implements a complete deep-learning Visual-Inertial
Odometry stack for a downward-facing UAV camera observing a planar
surface, with five interlocking branches. **Read this file first** —
it explains how the pieces fit together, the dataset format the
branches assume, the order in which things are trained, and the
conventions used by every SLURM script.

---

## Repository layout

```
deep-vio/
├── data_generation/            Blender pipeline that produces synthetic
│                               sequences (frames + poses + IMU).
├── data/                       Generated datasets live here.
├── cluster/                    Top-level cluster utilities (preexisting).
├── vision_only/                Branch A — vision-only odometry (BranchA).
├── imu_only/                   AirIMU + AirIO + 15-state velocity EKF.
├── fusion_common/              Shared dataset / loss / metrics for the
│                               three fusion branches.
├── loose_fusion/               EKF-based loose coupling (no training).
├── gated_fusion/               Tight coupling via per-dimension gating.
├── cross_attention/            Tight coupling via cross-modal transformer.
└── CLAUDE.md                   This file.
```

Five trainable / runnable branches in total:

| Branch | What it does | Trained? | Depends on |
|---|---|---|---|
| `vision_only`     | RGB → relative pose                                   | yes | dataset only |
| `imu_only`        | IMU → body-frame velocity → trajectory via EKF        | yes | dataset only |
| `loose_fusion`    | EKF fuses BranchA pose + AirIO velocity + AirIMU      | **no** | pretrained vision_only + imu_only checkpoints |
| `gated_fusion`    | Tight fusion: gate softly mixes vision + IMU features | yes | pretrained backbones (recommended) |
| `cross_attention` | Tight fusion: transformer cross-modal attention       | yes | pretrained backbones (recommended) |

---

## Dataset format

Every branch reads from a top-level dataset directory containing one
sub-directory per trajectory. The actual format on the cluster (after
the dataset.py refactor in commit `6b2d17d`) is **multi-rate** with
separate CSVs for each modality:

```
dataset_root/
  clover_100/
    images/
      000000.png
      000001.png
      ...                 # camera-rate, ≈100 Hz
    frame_index.csv       # t, imu_index, image_path  (one row per frame)
    poses_body.csv        # t, x, y, z, qw, qx, qy, qz  (IMU-rate, ≈1000 Hz)
    imu_clean.csv         # t, gx, gy, gz, ax, ay, az   (IMU-rate, ≈1000 Hz)
  clover_101/
    ...
```

**Trajectory naming.** Sequences come in trajectory families
(`clover`, `figure8`, `figure8_3d`, `lemniscate`, `oval`, `spiral`)
with numerical suffixes that index difficulty:

- `_100` — baseline maneuver (smooth, well-represented in training)
- `_101` — moderate aggressiveness
- `_102` — most aggressive (fastest yaw rates, sharpest turns)

The `_102` instances cause the largest accuracy regressions across
every method we've tested — see [Empirical observations](#empirical-observations-from-baseline-runs).

**Conventions assumed by all branches**

- **Vision (camera) and IMU run at different rates** — typically 100 Hz
  vs 1000 Hz (10:1). `frame_index.csv`'s ``imu_index`` column maps each
  camera frame to its corresponding IMU sample.
- Poses are stored as **unit quaternions** ``(qw, qx, qy, qz)`` plus
  position, both at IMU rate. The shared helper
  [`imu_only.utils.quat_to_rotmat_np`](imu_only/utils.py) converts
  per-sequence quaternion arrays to ``(N, 3, 3)`` rotation matrices.
- IMU CSV column order is `t, gx, gy, gz, ax, ay, az`
  (gyro-then-accel). The dataset loaders extract `acc = data[:, 4:7]`
  and `gyro = data[:, 1:4]`. Watch for this if you write any external
  preprocessing — the convention differs from many ROS bag dumps.
- IMU specific-force convention: `ᴮa = ᴮF/m + Rᵀᴳg` (an IMU at rest
  level on the ground reads ``+9.81`` along its body-up axis).
- Gravity vector: `g = [0, 0, -9.81]` (Z-up world). Defined once in
  [imu_only/utils.py](imu_only/utils.py) (`GRAVITY`, `GRAVITY_VEC`).
- Image preprocessing: resize to `(img_height, img_width)` then
  ImageNet normalize (mean `[0.485, 0.456, 0.406]`, std
  `[0.229, 0.224, 0.225]`). Photometric augmentation in training is
  applied **identically** to the two frames of a pair so the relative
  pose label stays valid.

**Pose representation in `fusion_common/dataset.py`.** Primary buffers
(`acc`, `gyro`, `attitude`, `v_body`, `R`, `p`) are stored at IMU rate.
Camera-rate views (`R_cam`, `p_cam`) are derived via the
`frame_imu_indices` mapping. Loose fusion uses the IMU-rate buffers
for its sample-by-sample EKF; the tight-fusion branches use the
camera-rate views for trajectory comparison and IMU-rate context
windows for AirIO input.

---

## Branch A — `vision_only/`

**Reference:** Wang et al. 2017 (DeepVO) + Zhou et al. 2019 (6D
rotation) + PWC-Net-style correlation.

**Inputs:** two consecutive RGB frames `[B, T, 3, H, W]` plus their
sequence offset.

**Outputs:**
- `trans` `[B, T, 3]` — relative translation in metric units
- `rot_6d` `[B, T, 6]` — raw 6D rotation
- `R` `[B, T, 3, 3]` — Gram-Schmidt rotation matrix
- `hidden` — LSTM state `(h, c)` for trajectory continuation
- `features` `[B, T, 128]` — **shared with the fusion branches**

**Architecture (4 stages):**

1. Siamese MobileNetV2 (layers 0-13, no pretraining) → 96 ch → 1×1
   conv → 64 ch.
2. Correlation layer with `max_displacement=4` → 81 ch → two 3×3
   conv blocks → 64 ch.
3. Concat `[corr | feat_t | feat_t1]` (192 ch) → conv → AdaptiveAvgPool
   `(4, 4)` → flatten `[B, T, 1024]`.
4. 2-layer unidirectional LSTM (hidden=256, dropout=0.3) → FC(256,
   128) → two heads (`trans 3`, `rot 6`). 6D output recovered via
   `gram_schmidt`.

**Loss:** smooth-L1 on translation + geodesic SO(3) (`atan2`
formulation), `lambda_rot=100`.

**Files:**

```
vision_only/
├── model.py             BranchA, CNNEncoder, CorrelationLayer (vectorised via unfold)
├── loss.py              geodesic_loss_stable, branch_a_loss
├── dataset.py           UAVOdometryDataset
├── train.py             Adam 1e-4, weight_decay 1e-4, CosineAnnealingLR, AMP, grad-clip 1.0
├── evaluate.py          ATE (Umeyama-aligned) + RTE + rot deg + plots
├── utils.py             gram_schmidt, pose_to_matrix, umeyama_alignment, integrate_trajectory
├── test_pipeline.py     synthetic-data smoke test
├── setup_env.slurm
├── train.slurm
├── test.slurm
├── requirements.txt
└── README.md            full design choices write-up
```

**Run:**

```bash
sbatch vision_only/setup_env.slurm
sbatch vision_only/test.slurm
sbatch --export=ALL,DATA_ROOT=/path/to/data vision_only/train.slurm
```

---

## IMU — `imu_only/`

**References:** Qiu et al. RA-L 2025 ("AirIO") and Qiu et al. 2023
("AirIMU"). The full pipeline is **AirIMU → AirIO → EKF**, where:

- `AirIMUNet` corrects raw IMU and reports per-sample uncertainty.
- `AirIONet` predicts body-frame velocity + diagonal covariance.
- `VelocityEKF` (15-state) fuses both with IMU pre-integration.

### AirIMU

**Inputs:** raw `[acc, gyro] [B, W, 6]`.
**Outputs:**
- `correction [B, W, 6]` so `â = a + σ̂_a`, `ŵ = ω + σ̂_g`,
- `log_var [B, W, 6]` clamped to `[-10, 10]` (3 accel + 3 gyro).

**Architecture:** Conv1d → BN → GELU stack (ch 64→128→128) →
Dropout(0.5) → bi-GRU (hidden=128, 2 layers) → 2 MLP heads.

**Loss (training, [imu_only/loss.py](imu_only/loss.py))**: integrate
the corrected IMU forward over the window with forward-Euler
([`integrate_imu_window`](imu_only/utils.py)), supervise on:

- geodesic SO(3) of the integrated end-of-window rotation,
- MSE of the integrated end-of-window velocity (world frame),
- MSE of the integrated displacement,
- Gaussian NLL on per-sample uncertainty against a window-summary
  residual.

Default loss weights: `λ_rot=10, λ_vel=1, λ_pos=1, λ_nll=1e-4`.
**Window size is short by design** (default 20 = 100 ms at 200 Hz)
because forward-Euler error compounds super-linearly with window
length.

### AirIO

**Inputs:** corrected `[acc, gyro] [B, W, 6]` + per-sample attitude
`ξ ∈ ℝ³ [B, W, 3]` (so(3) log of `R`).
**Outputs:** `v_body [B, W, 3]`, `log_var [B, W, 3]` clamped.

**Architecture:** two parallel CNN1D encoders (IMU 6→64→128→128;
attitude 3→32→64), concat → bi-GRU (hidden=128, 2 layers) → 2 MLP
heads.

**Loss (paper Eq. 3-5):** Huber on velocity (`δ=0.005`) + `λ_c=1e-4`
× Gaussian NLL of residual under the predicted diagonal variance.

### EKF — [imu_only/ekf.py](imu_only/ekf.py)

15-state error-state EKF.

State (nominal): `X = (R, ᴳv, ᴳp, b_a, b_g)`.
Error state: `δX = [δξ, δᴳv, δᴳp, δb_a, δb_g] ∈ ℝ¹⁵`.

**Predict (Eq. 7):**

```
R_{i+1} = R_i Exp((ω - b_g) Δt)
v_{i+1} = v_i + (R_i (a - b_a) + g) Δt
p_{i+1} = p_i + v_i Δt + 0.5 Δt² (R_i (a - b_a) + g)
b_a, b_g unchanged
```

Process noise = AirIMU `log_var` (rotated through `R`) + bias random
walk (defaults `σ_ba_rw=1e-3`, `σ_bg_rw=1e-4`).

**Update (Eq. 10-12):**
- Measurement: `h(X) = Rᵀᴳv` = body-frame velocity from AirIO.
- Jacobians: `H_{δᴳv} = Rᵀ`, `H_{δξ} = Rᵀ [ᴳv]_×`.
- Joseph form covariance update.

### Files

```
imu_only/
├── model.py             AirIMUNet, AirIONet, CNNEncoder1D
├── loss.py              airio_loss, airimu_loss, geodesic_loss_stable, gaussian_nll
├── dataset.py           IMUWindowDataset (sliding windows)
├── ekf.py               VelocityEKF (numpy)
├── utils.py             SO(3) ops (torch + numpy), GT-velocity-from-poses, integrator
├── train_airimu.py      Adam 1e-3, ReduceLROnPlateau, batch=128, window=20
├── train_airio.py       same, window=1000 (5 s at 200 Hz); loads frozen AirIMU
├── evaluate.py          per-IMU-sample EKF, ATE/RTE-5s/rot-deg, plots
├── test_pipeline.py     synthetic helical-trajectory smoke test
├── setup_env.slurm
├── train_airimu.slurm
├── train_airio.slurm
├── test.slurm
├── requirements.txt
└── README.md            full design choices write-up
```

**Run order:**

```bash
sbatch imu_only/setup_env.slurm
sbatch imu_only/test.slurm
sbatch --export=ALL,DATA_ROOT=/path imu_only/train_airimu.slurm
sbatch --export=ALL,DATA_ROOT=/path,\
              AIRIMU_CHECKPOINT=imu_only/checkpoints/airimu/best.pt \
       imu_only/train_airio.slurm
```

If you skip AirIMU, AirIO trains on raw IMU directly — matches the
"AirIO Net" baseline column in the paper.

---

## Fusion shared library — `fusion_common/`

Imported by `loose_fusion`, `gated_fusion`, and `cross_attention` via
the standard sys-path injection pattern at the top of each entry
script:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion_common.dataset import PairedDataset
from fusion_common.loss import pose_loss
from fusion_common.metrics import trajectory_metrics
```

| File | Provides |
|---|---|
| [dataset.py](fusion_common/dataset.py) | `PairedDataset` — sliding window of `T` frame pairs with synchronized IMU context (`imu_context` samples ending at the next-frame timestamp). Returns `frames_t`, `frames_t1`, `imu_acc`, `imu_gyro`, `attitude` (per-sample so(3)), `v_body_gt`, `trans_gt`, `R_gt`. Also exposes `_PairedSequence` for evaluation scripts that want raw arrays. |
| [loss.py](fusion_common/loss.py) | `geodesic_loss_stable`, `pose_loss` (smooth-L1 + λ × geodesic). |
| [metrics.py](fusion_common/metrics.py) | `umeyama_alignment`, `trajectory_metrics` (ATE, RTE-`step`, mean rotation deg, per-frame error series). |

`PairedDataset` requires `imu_context` IMU samples of history before
the first valid frame pair, so the first `imu_context` pairs of each
sequence are skipped at evaluation time and filled with identity.

---

## Loose fusion — `loose_fusion/`

**No training.** Loads three pretrained backbones, runs them
independently, and steps a fusion EKF that consumes their outputs.

`FusionEKF` (extends `imu_only.VelocityEKF`):
- inherits `predict()` (corrected IMU + AirIMU log-var) and
  `update_velocity()` (AirIO body-frame velocity);
- adds `update_vision_velocity(delta_t_vis, dt_frame, sigma)` —
  vision Δt/Δt is treated as a body-frame velocity measurement and
  reuses the same Eq. 10-12 update model;
- adds `update_vision_rotation(delta_R_vis, sigma_deg)` — innovation
  in `so(3)`, `H` is the identity on `δξ` to first order.

Files:

```
loose_fusion/
├── fusion_ekf.py        FusionEKF subclass
├── evaluate.py          end-to-end evaluator with --no_vision / --no_imu_update
├── test_pipeline.py
├── setup_env.slurm
├── eval.slurm
├── test.slurm
├── requirements.txt
└── README.md
```

**Run:**

```bash
sbatch loose_fusion/setup_env.slurm
sbatch loose_fusion/test.slurm
sbatch --export=ALL,DATA_ROOT=/path,\
              VISION_CKPT=../vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=../imu_only/checkpoints/airio/best.pt,\
              AIRIMU_CKPT=../imu_only/checkpoints/airimu/best.pt \
       loose_fusion/eval.slurm
```

Two tuning knobs worth sweeping: `--vision_vel_sigma` (default
0.05 m/s) and `--vision_rot_sigma_deg` (default 0.5°). Lower → trust
vision more.

---

## Gated tight fusion — `gated_fusion/`

**Idea:** per-dimension soft switch between projected vision and IMU
features.

```
fused_d = g_d ⊙ vis_proj_d + (1 - g_d) ⊙ imu_proj_d
g       = sigmoid(MLP([vis_proj ; imu_proj]))    # ∈ [0, 1]^D
```

Both backbones are imported, **not** copied. `GatedFusionNet`:

1. `BranchA(frames_t, frames_t1)` → exposes its post-FC `features
   [B, T, 128]`.
2. `_ImuFeatureExtractor` runs `AirIONet`'s CNN encoders + bi-GRU and
   takes the **last** sample of every window as a per-pair feature
   `[B, T, 256]`.
3. `vis_proj` (128 → 128) and `imu_proj` (256 → 128) project both to
   the shared dim `feat_dim=128`.
4. Gate MLP `[2D → head_hidden=256 → D, sigmoid]` outputs the
   per-dimension gate.
5. `LayerNorm(fused)` → `trans_head` (3) and `rot_head` (6) →
   `gram_schmidt(rot_6d)`.

**Training:** Adam, weight_decay 1e-4, CosineAnnealingLR.
- Phase 1 (`--warmup_epochs=5`): both backbones frozen, only gate +
  projections + heads update at lr=1e-4.
- Phase 2: full network unfrozen, lr drops to lr_finetune=2e-5.

**Loss:** `pose_loss` (`λ_rot=100`).

**Diagnostic plot:** `<seq>_gate.png` — mean gate value over time
(1 = trusting vision, 0 = trusting IMU).

Files:

```
gated_fusion/
├── model.py             GatedFusionNet, _ImuFeatureExtractor
├── train.py             warm-up + joint fine-tune
├── evaluate.py          metrics + trajectory + gate plots
├── test_pipeline.py
├── setup_env.slurm, train.slurm, eval.slurm, test.slurm
├── requirements.txt
└── README.md
```

**Run:**

```bash
sbatch gated_fusion/setup_env.slurm
sbatch gated_fusion/test.slurm
sbatch --export=ALL,DATA_ROOT=/path,\
              VISION_CKPT=../vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=../imu_only/checkpoints/airio/best.pt \
       gated_fusion/train.slurm
sbatch --export=ALL,DATA_ROOT=/path,\
              CHECKPOINT=checkpoints/gated_fusion/best.pt \
       gated_fusion/eval.slurm
```

---

## Cross-attention tight fusion — `cross_attention/`

**Idea:** treat each modality as one token per timestep, concatenate
the two streams, and let a transformer encoder do self-attention over
the combined `2T` tokens — equivalent to alternating within-modality
self-attention and cross-modality attention but parameter-shared.

`CrossAttentionFusionNet`:

1. Same `BranchA` features and `AirIO` last-sample-per-window
   features as the gated branch (projections to `feat_dim=128`).
2. Add learned modality embedding (`(2, D)` table) + learned temporal
   positional embedding (`(max_len, D)`) — same temporal embedding
   shared across modalities, modality embedding distinguishes them.
3. Concat as `[B, 2T, D]`, run through `nn.TransformerEncoder` (2
   layers by default, 4 heads, FFN width 256, GELU, `norm_first=True`,
   `batch_first=True`).
4. LayerNorm; readout = mean of vision-half and IMU-half output
   tokens at each `t`. Same `trans_head` and `rot_head` as gated.

**Training:** identical schedule to gated (Adam, weight_decay,
CosineAnnealingLR, warm-up + joint fine-tune). Extra args:
`--feat_dim`, `--num_heads`, `--num_layers`, `--ffn_hidden`,
`--dropout`.

**Loss:** `pose_loss`.

**Diagnostic plot:** `<seq>_vis_imu_cos.png` — per-timestep cosine
similarity between the two output tokens (vision vs IMU). Closest
analogue to gated's gate plot.

Files:

```
cross_attention/
├── model.py             CrossAttentionFusionNet, _ImuFeatureExtractor
├── train.py             same shape as gated_fusion/train.py + transformer args
├── evaluate.py
├── test_pipeline.py
├── setup_env.slurm, train.slurm, eval.slurm, test.slurm
├── requirements.txt
└── README.md
```

**Run:**

```bash
sbatch cross_attention/setup_env.slurm
sbatch cross_attention/test.slurm
sbatch --export=ALL,DATA_ROOT=/path,\
              VISION_CKPT=../vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=../imu_only/checkpoints/airio/best.pt \
       cross_attention/train.slurm
```

---

## Recommended end-to-end pipeline

To go from a fresh dataset to evaluation of every method:

```
                   data_generation/                    (Blender)
                          │
                          ▼
                       data/<run>/
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
   vision_only/train.py        imu_only/train_airimu.py
            │                            │
            │                            ▼
            │                  imu_only/train_airio.py    (uses frozen AirIMU)
            │                            │
            └────────────┬───────────────┘
                         ▼
       ┌─────────────────┼─────────────────────┐
       ▼                 ▼                     ▼
 loose_fusion/eval  gated_fusion/train   cross_attention/train
                         │                     │
                         ▼                     ▼
                   gated/eval            cross_attention/eval
```

Or in shell:

```bash
# 1. Pretrain backbones
sbatch --export=ALL,DATA_ROOT=$D vision_only/train.slurm
sbatch --export=ALL,DATA_ROOT=$D imu_only/train_airimu.slurm
# (wait, then…)
sbatch --export=ALL,DATA_ROOT=$D,\
              AIRIMU_CHECKPOINT=imu_only/checkpoints/airimu/best.pt \
       imu_only/train_airio.slurm

# 2. Loose fusion (no training)
sbatch --export=ALL,DATA_ROOT=$D,\
              VISION_CKPT=vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=imu_only/checkpoints/airio/best.pt,\
              AIRIMU_CKPT=imu_only/checkpoints/airimu/best.pt \
       loose_fusion/eval.slurm

# 3. Tight fusion (independent training runs)
sbatch --export=ALL,DATA_ROOT=$D,\
              VISION_CKPT=vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=imu_only/checkpoints/airio/best.pt \
       gated_fusion/train.slurm
sbatch --export=ALL,DATA_ROOT=$D,\
              VISION_CKPT=vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=imu_only/checkpoints/airio/best.pt \
       cross_attention/train.slurm
```

Each branch has its own venv (`<branch>/.venv/`) created by
`setup_env.slurm` — they're not shared. The fusion branches still
**import** sibling code via the `sys.path` injection trick, so the
imported modules see whatever venv the importing branch is using.

---

## SLURM script conventions

Every branch has a near-identical set:

| File | Job | Notes |
|---|---|---|
| `setup_env.slurm` | one-time venv create + torch + reqs | overridable via `TORCH_INDEX_URL` (default `cu121`) |
| `test.slurm`      | `python test_pipeline.py`            | ~1 min, runs the smoke test on synthetic data |
| `train*.slurm`    | `python train*.py`                   | reads everything from `${VAR:-default}` env vars |
| `eval.slurm`      | `python evaluate.py`                 | only loose / gated / cross_attention; `imu_only` and `vision_only` use train.slurm + a manual eval |

All scripts:
- `set -euo pipefail`,
- `module load python/3.10|3.11|python` and `cuda/12.1|11.8|cuda`
  (with `|| true` so unknown clusters fall back gracefully),
- `mkdir -p logs/`,
- read `SLURM_SUBMIT_DIR` for the project dir,
- assume the venv lives at `${VENV_DIR:-${PROJECT_DIR}/.venv}`.

To override anything pass it via `--export=ALL,VAR=value` at submit
time. Example: `sbatch --export=ALL,EPOCHS=200,LR=5e-5
gated_fusion/train.slurm`.

---

## Code conventions

- **No pypose dependency.** SO(3) ops are implemented in
  [imu_only/utils.py](imu_only/utils.py) (torch + numpy variants).
- **Numerical stability rules:** geodesic loss uses `atan2`, never
  `arccos`. SO(3) `exp` / `log` use small-angle Taylor expansions
  near `θ = 0`. AirIMU / AirIO `log_var` are clamped to `[-10, 10]`.
- **Mixed precision is opt-in** via the absence of `--no_amp`. All
  training scripts use `torch.cuda.amp` when on a CUDA device.
- **Gradient clipping** at max-norm `1.0` everywhere there's an LSTM
  or GRU.
- **Hidden state hygiene:** LSTM / GRU hidden states are detached
  between batches; they are reset (`None`) at trajectory boundaries
  during training.
- **Checkpoint format:** every `train*.py` saves
  `{epoch, model, optimizer, scheduler, args, val_*}` so reloading
  via `state.get("model", state)` always works.
- **Augmentation rules for vision:** photometric only (brightness +
  contrast, factor 0.2). Same jitter applied to both frames of a pair.
  No geometric augmentation (would invalidate the relative-pose
  label).
- **All paths in scripts are absolute or relative to
  `SLURM_SUBMIT_DIR`** — never to the user's `pwd`.

---

## Smoke tests

Every branch ships a `test_pipeline.py` that:

1. Generates a tiny synthetic dataset (helical trajectory with
   analytical IMU readings, ~60-600 samples, ~3 sequences) in `/tmp`.
2. Builds the network with random weights.
3. Runs forward + backward + an optimizer step.
4. Asserts shape correctness, finite gradients, and pipeline-specific
   invariants (e.g. EKF covariance stays PSD, SO(3) round-trips,
   integrator vs analytic helix matches).

`sbatch <branch>/test.slurm` is the fastest way to confirm a clone
works on a new cluster after `setup_env.slurm`. Total wall-clock <1
minute on a single GPU. Output ends with `All checks passed.`

---

## Common gotchas

- **Validation Huber loss is a poor proxy for trajectory ATE.**
  Empirically a 2.3× increase in val Huber on AirIO produced a 16×
  increase in ATE on the same evaluation sequence. Do **not**
  early-stop on Huber alone — track ATE on a held-out sequence during
  training, even if it's expensive. See observation #3 below.
- **Per-axis velocity errors are wildly asymmetric** in AirIO outputs.
  On `clover` trajectories the Z-axis RMSE is 5× the X-axis RMSE; on
  planar `oval` / `lemniscate` trajectories Y and Z errors are 2-3
  orders of magnitude smaller than X. This isn't a bug — it's the
  network failing to generalise across motion axes that aren't
  well-represented in training. Symptom: huge ATE on the first
  out-of-distribution trajectory family you evaluate.
- **`acc_pct` / confidence outputs are modality-specific.** If a model
  reports a translation-classification confidence, that signal can
  be 100% on a sequence whose rotation error is 100°. Don't use a
  single confidence channel as a global "did it work" signal.
- **Adding a new prediction head without commensurate data hurts.**
  PRGFlow's "improved" model with an added yaw head regressed 3-5×
  on translation and 10-25× on rotation versus the no-yaw baseline
  on the same evaluation set. Treat extra heads as a hypothesis to
  test, not a free improvement.
- **Watch for degenerate constants in metric outputs.** The PRGFlow
  baseline's scale prediction is *literally constant* (0.00824) on
  4 of 5 evaluation sequences — mean = p95 = max identically. The
  network learned to output a default rather than predict scale.
  If you see suspiciously low variance in any metric, plot the
  per-frame distribution before trusting the mean.
- **`loose_fusion._R_prev_vis` initialises to identity on `reset()`.**
  The first vision rotation update is therefore silently skipped
  (innovation against itself); intentional, but worth knowing.
- **Cross-attention's `pos_emb` has length `max_len=256` by default.**
  Training with `--sequence_length > 256` triggers the runtime check
  at the top of `forward`.
- **`AirIONet` clamps its `log_var` to `[-10, 10]`.** If your loss
  starts oscillating it is *not* because the variance is exploding —
  investigate the Huber term first.

---

## Empirical observations from baseline runs

The `results/` directory contains evaluations from before the fusion
branches existed: BranchA (vision_only), AirIO+EKF (imu_only), and
PRGFlow VIO (an external visual-inertial baseline not in this repo,
but a useful comparator). Eight non-obvious findings worth remembering
when interpreting future results:

### 1. Better per-frame rotation can produce worse trajectories.
BranchA's per-pair rotation error is **0.5°**; PRGFlow VIO's is
**6.9°** — vision is 14× more accurate per frame. Yet PRGFlow's ATE
is **0.72 m** vs BranchA's **2.87 m** — VIO is 4× better globally.
Bias dominates trajectory error, not variance. Vision dead-reckons;
VIO's IMU+EKF stops the bleed.

### 2. AirIO doesn't generalise across motion axes despite physics being symmetric.

| Family | rmse_vx | rmse_vy | rmse_vz |
|---|---:|---:|---:|
| oval (planar) | 0.17 | 0.004 | 0.04 |
| clover (3D loops) | 0.52 | 0.34 | **2.79** |
| spiral | 0.35 | 0.08 | 1.28 |

The Z-axis error is 30× the X-axis error on `clover` even though
accelerometer/gyro physics are identical for both axes. The network
learned a motion-distribution prior, not an axis-symmetric inverse
model. Strong argument for diversifying training trajectories or for
data-augmentation that randomly rotates the body frame.

### 3. Validation loss massively understates trajectory error.

Same `clover_100` sequence on two AirIO checkpoints:

| Checkpoint | val Huber | ATE | rotation |
|---|---:|---:|---:|
| run0 | 0.000139 | **0.78 m** | 14° |
| new_data_run | 0.000538 | **12.73 m** | 104° |

A 2.3× rise in Huber → 16× rise in ATE. Per-window velocity residuals
can be sub-millimeter while the integrated trajectory drifts meters.
Validation should include trajectory-level metrics.

### 4. PRGFlow's confidence detector is rotation-blind.
On `oval_102` the model self-reports `acc_pct = 100%` (perfectly
confident) yet rotation error is **92.7°**. On `clover_102` it
self-reports `acc_pct = 15.6%` with rotation error of 105° — knows it
failed. The confidence head was trained on translation classification
and never saw rotation residuals during supervision. Lesson: any
confidence signal is partial, scoped to whatever it was supervised on.

### 5. PRGFlow's "improved" model is worse than the baseline.
Adding a 167K-parameter yaw head produced 3-5× worse translation and
10-25× worse rotation on the same evaluation sequences. Capacity
without commensurate data caused specialisation in yaw at the expense
of translation generalisation.

### 6. Scale prediction is a frozen constant on most sequences.
PRGFlow VIO's `e_scale_mean`, `e_scale_p95`, and `e_scale_max` are all
**identically `0.00824`** on 4 of 5 evaluation sequences. The network
learned a degenerate solution where it outputs a constant rather than
predicting scale. Worth adopting a habit: **always plot the per-frame
distribution before trusting the mean** of a metric.

### 7. AirIO's *capability* is excellent — generalisation is broken.
- Best result: `oval_100`, ATE **0.24 m** in 20 s = 1.2 cm/s drift.
- Worst result: `clover_102`, same network, same training: ATE
  **26.9 m**.
- Same parameters → 112× error gap from a 5% change in trajectory
  shape. Frames the fusion work as solving generalisation, not
  capability.

### 8. AirIO is 30× faster than PRGFlow at inference.
Wall-clock for evaluating 15 sequences: **AirIO+EKF = 71 s**,
**PRGFlow_imp = 2279 s** (≈70% of which is preprocessing). The
deployment story for onboard UAV use isn't accuracy alone — at the
latency budget that matters for closed-loop control AirIO's
accuracy gap is much smaller than the table suggests.

### Implications for the fusion branches

These observations directly motivate specific fusion design choices:

- **Loose fusion's vision-rotation update is the highest-leverage
  measurement.** AirIO rotations drift 8-119° while BranchA rotations
  are sub-degree per frame — feeding vision rotation into the EKF
  patches AirIO's biggest weakness. Tune
  `--vision_rot_sigma_deg` aggressively low (0.1-0.3°) before
  `--vision_vel_sigma`.
- **Gated fusion should learn `g → 1` (trust vision) on `_102`
  instances.** Plot `<seq>_gate.png` for `clover_102` after training:
  if the gate isn't strongly biased toward vision there, the network
  hasn't learned the right behaviour.
- **Cross-attention's interpretability is via `vis_imu_cos`.** Low
  cosine similarity on `clover_102` means the modalities disagree
  (which is correct — IMU is failing) and the transformer is
  reconciling. High cosine similarity on planar trajectories means
  the modalities concur.
- **Validation strategy for the fusion branches.** Don't just track
  the pose loss. Pick 1-2 held-out sequences with different
  trajectory families and report ATE per epoch. The Huber/pose-loss
  validation curve will look smooth while ATE silently regresses.
- **The bar to beat (small set, 5 trajectories).** PRGFlow VIO scores
  ATE 0.72 m, RTE 0.005 m, rotation 6.9°. Loose fusion is unlikely to
  beat this outright (PRGFlow is already tightly-coupled VIO); the
  interesting comparison is `gated_fusion` and `cross_attention`
  versus PRGFlow on identical sequences.

---

## References

The full bibliography lives in each branch's `README.md`. The most
load-bearing papers:

- Wang, S. et al., 2017. "DeepVO." *ICRA*. (vision_only backbone)
- Sandler, M. et al., 2018. "MobileNetV2." *CVPR*. (vision_only backbone)
- Sun, D. et al., 2018. "PWC-Net." *CVPR*. (correlation layer)
- Zhou, Y. et al., 2019. "On the Continuity of Rotation
  Representations in Neural Networks." *CVPR*. (6D rotation)
- Qiu, Y. et al., 2025. "AirIO." *RA-L*. (imu_only)
- Qiu, Y. et al., 2023. "AirIMU." (imu_only AirIMU sub-network)
- Forster, C. et al., 2015. "IMU Preintegration on Manifold." *RSS*.
  (EKF kinematic model)
- Clark, R. et al., 2017. "VINet." *AAAI*. (gated_fusion / fusion design)
- Chen, C. et al., 2019. "Selective Sensor Fusion for Neural
  Visual-Inertial Odometry." *CVPR*. (gated_fusion)
- Vaswani, A. et al., 2017. "Attention Is All You Need." *NeurIPS*.
  (cross_attention)
