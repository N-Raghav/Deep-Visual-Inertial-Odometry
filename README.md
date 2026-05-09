# Deep Visual-Inertial Odometry — RBE/CS549 Project 4 (Group 7)

Submission for [Project 4](https://rbe549.github.io/spring2026/proj/p4/),
Phase 2: a deep-learning visual-inertial odometry stack built around the
**AirIO** RA-L 2025 paper plus PRGFlow baselines and vision-IMU fusion
strategies. The Phase 1 Stereo MSCKF implementation lives in a separate
repository.

All branches run on quadrotor data — a custom Blender-rendered dataset of
agile UAV trajectories generated for this project.

## Repository contents

```
deep-vio/
├── data_generation/        Blender pipeline (frames + IMU + poses)
├── data/                   Generated datasets and Blender textures
├── cluster/                Apptainer/SLURM helpers shared across branches
├── vision_only/            Branch A — RGB → relative pose (DeepVO-style)
├── imu_only/               AirIMU + AirIO + 15-state velocity EKF
├── prgflow/                PRGFlow visual-inertial baseline
├── prgflow_mod/            PRGFlow + yaw head (improved variant)
├── loose_fusion/           EKF loose coupling (vision pose + AirIO velocity)
├── cross_attention/        Tight coupling (cross-modal transformer)
├── fusion_common/          Shared dataset / loss / metrics for fusion branches
├── CLAUDE.md               Detailed engineering notes
├── Project4_Group7.pdf     Final slides
├── Report.pdf              Final report
└── README.md               This file
```

---

## Branches

Six trainable / runnable branches that progress from single-modality
baselines through loose coupling to tight (feature-level) fusion. All
share a single Blender-rendered dataset of UAV trajectories that span
six manoeuvre families (`clover`, `figure8`, `figure8_3d`,
`lemniscate`, `oval`, `spiral`) with three difficulty levels each
(`_100` baseline → `_102` most aggressive).

| Branch | What it does | Trained? | Reference |
|---|---|---|---|
| `vision_only`     | RGB → relative pose (Siamese MobileNetV2 + correlation + LSTM) | yes | DeepVO (Wang 2017) + 6D rot (Zhou 2019) |
| `imu_only`        | AirIMU corrects raw IMU; AirIO predicts body-frame velocity; 15-state EKF integrates | yes | AirIO RA-L 2025 + AirIMU 2023 |
| `prgflow`         | PRGFlow VIO — pixel-flow + IMU baseline | yes | course-provided baseline |
| `prgflow_mod`     | PRGFlow with an added yaw head | yes | course-provided baseline |
| `loose_fusion`    | EKF fuses BranchA pose + AirIO velocity + AirIMU | **no** | Mourikis 2007 (loose coupling) |
| `cross_attention` | Tight fusion via cross-modal transformer over vision + IMU tokens | yes | VINet (Clark 2017), Vaswani 2017 |

---

## Requirements

PyTorch + standard scientific stack. Each branch carries its own
`requirements.txt` and `setup_env.slurm` (creates a per-branch
`<branch>/.venv/`):

```bash
cd <branch>
sbatch setup_env.slurm
sbatch test.slurm
```

---

## Dataset

Self-generated with [data_generation/](data_generation/) (Blender +
analytic IMU). Each sequence directory contains:

```
sequence_name/                       # e.g. clover_102
├── images/                          # camera-rate frames (≈100 Hz)
│   ├── 000000.png
│   └── ...
├── frame_index.csv                  # t, imu_index, image_path
├── poses_body.csv                   # t, x, y, z, qw, qx, qy, qz   (≈1000 Hz)
└── imu_clean.csv                    # t, gx, gy, gz, ax, ay, az    (≈1000 Hz)
```

The fusion branches use `frame_imu_indices` to align the multi-rate
streams.

---

## Run order

```bash
export D=/path/to/dataset

# 1. Pretrain backbones
sbatch --export=ALL,DATA_ROOT=$D vision_only/train.slurm
sbatch --export=ALL,DATA_ROOT=$D imu_only/train_airimu.slurm
sbatch --export=ALL,DATA_ROOT=$D,\
              AIRIMU_CHECKPOINT=imu_only/checkpoints/airimu/best.pt \
       imu_only/train_airio.slurm

# 2. Loose fusion (no training, evaluation only)
sbatch --export=ALL,DATA_ROOT=$D,\
              VISION_CKPT=vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=imu_only/checkpoints/airio/best.pt,\
              AIRIMU_CKPT=imu_only/checkpoints/airimu/best.pt \
       loose_fusion/eval.slurm

# 3. Tight fusion (independent training run)
sbatch --export=ALL,DATA_ROOT=$D,\
              VISION_CKPT=vision_only/checkpoints/branch_a/best.pt,\
              AIRIO_CKPT=imu_only/checkpoints/airio/best.pt \
       cross_attention/train.slurm
```

For per-branch design notes, hyperparameters, and gotchas, see
[CLAUDE.md](CLAUDE.md) and the individual `README.md` files inside
[vision_only/](vision_only/), [imu_only/](imu_only/),
[loose_fusion/](loose_fusion/), and [cross_attention/](cross_attention/).

## Authors

Group 7, RBE/CS549, Spring 2026, WPI.

## License

Code is original work unless otherwise indicated; AirIO and AirIMU
references and PRGFlow baselines are credited in the report.
