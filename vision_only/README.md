# Branch A — Vision Only Odometry

## Overview

Branch A is the vision-only baseline of a three-branch deep
Visual-Inertial Odometry system designed for a downward-facing
monocular camera on a UAV observing a planar surface. It takes
two consecutive RGB frames as input and predicts the relative
6-DoF pose between them.

This branch serves two purposes:
1. A standalone vision-only odometry system
2. The visual encoder whose features are reused in Branch C
   (Visual-Inertial fusion)

---

## Network Architecture

```
[Frame t]   ──┐
              ├──→ Siamese MobileNetV2 ──→ feat_t   [B, 64, H/16, W/16]
[Frame t+1] ──┘    (shared weights)    ──→ feat_t1  [B, 64, H/16, W/16]
                                │
                ┌───────────────┤
                ↓               │
        Correlation Layer       │
       [B, 81, H/16, W/16]      │
                │               │
        Compress convs          │
       [B, 64, H/16, W/16]      │
                └───────────────┘
                        │
              Concatenate [192ch]
                        │
              Compress + Pool
                        │
                Flatten [B, 1024]
                        │
                ┌───────┘  (per timestep)
                ↓
            2-layer LSTM
             hidden=256
                │
            FC(256→128)
            ┌────┴────┐
            ↓         ↓
        trans_head   rot_head
       Linear(128,3) Linear(128,6)
            ↓         ↓
         [B,T,3]    [B,T,6]
                        │
                 Gram-Schmidt
                        │
                    [B,T,3,3]
```

---

## Design Choices and Justification

### 1. Siamese CNN Encoder

**Choice:** Two separate CNN encoders with shared weights,
one per frame.

**Justification:** By encoding both frames with the same
weights, both feature maps live in the same embedding space.
This makes cross-frame comparison in the correlation layer
geometrically meaningful — a feature responding to a corner
in frame t will produce the same activation pattern for a
corner in frame t+1. In contrast, the naive 6-channel stacked
input used in DeepVO has no such symmetry guarantee.

**Reference:**
> Bertinetto, L., Valmadre, J., Henriques, J.F., Vedaldi, A.,
> Torr, P.H., 2016. Fully-convolutional siamese networks for
> object tracking. ECCV Workshop on Visual Object Challenge.

---

### 2. MobileNetV2 Backbone

**Choice:** MobileNetV2 layers 0–13, trained from scratch.

**Justification:** Training data is synthetic (Blender-generated).
MobileNetV2's depthwise separable convolutions reduce parameter
count significantly compared to ResNet or VGG, reducing the risk
of overfitting on limited synthetic data. Spatial stride of 16
(after layer 13) gives a 14×14 feature map for 224×224 input —
sufficient resolution for texture matching on a planar scene
while keeping computation tractable.

**Reference:**
> Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.,
> 2018. MobileNetV2: Inverted residuals and linear bottlenecks.
> CVPR 2018.

---

### 3. Correlation Layer

**Choice:** Explicit cross-frame feature correlation with
max_displacement=4.

**Justification:** The correlation layer computes dot-product
similarity between every location in feat_t and all displaced
locations in feat_t1 within a 9×9 neighborhood. This gives the
network an explicit motion signal before any learned processing —
the inductive bias that features move smoothly between frames.
Without this, the network must learn to compare features
implicitly through subsequent convolutions, requiring more data
and more parameters. This design was validated for optical flow
(pixel-level motion) and the same argument applies to odometry
(camera-level motion).

**References:**
> Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas,
> C., Golkov, V., Van Der Smagt, P., Cremers, D., Brox, T., 2015.
> FlowNet: Learning optical flow with convolutional networks.
> ICCV 2015.

> Ilg, E., Mayer, N., Saikia, T., Keuper, M., Dosovitskiy, A.,
> Brox, T., 2017. FlowNet 2.0: Evolution of optical flow
> estimation with deep networks. CVPR 2017.

> Sun, D., Yang, X., Liu, M.Y., Kautz, J., 2018. PWC-Net: CNNs
> for optical flow using pyramid, warping, and cost volume.
> CVPR 2018.

---

### 4. Retaining feat_t and feat_t1 Alongside Correlation

**Choice:** Concatenate correlation map with both individual
frame feature maps before spatial compression.

**Justification:** The correlation map encodes where things
moved but discards appearance information. Individual frame
features carry texture density, scene scale, and lighting
context. For a downward-facing UAV over a planar scene,
apparent texture scale directly encodes camera height, which
is essential for accurate translation estimation in Z. PWC-Net
demonstrated that combining the cost volume with per-frame
features consistently outperforms using either alone.

**Reference:**
> Sun, D., Yang, X., Liu, M.Y., Kautz, J., 2018. PWC-Net: CNNs
> for optical flow using pyramid, warping, and cost volume.
> CVPR 2018.

---

### 5. LSTM for Temporal Consistency

**Choice:** 2-layer unidirectional LSTM, hidden size 256,
hidden state carried across the full trajectory during inference.

**Justification:** Frame-to-frame pose predictions are
independent without temporal modeling — each prediction has
no knowledge of previous errors, causing rapid trajectory
drift during dead-reckoning. The LSTM hidden state accumulates
trajectory context, effectively learning to be consistent with
its own history. DeepVO demonstrated that removing the LSTM
from an otherwise identical architecture approximately doubles
ATE on standard benchmarks. Hidden state is reset at trajectory
boundaries and detached between batches to prevent gradient
flow across trajectories.

**References:**
> Wang, S., Clark, R., Wen, H., Trigoni, N., 2017. DeepVO:
> Towards end-to-end visual odometry with deep recurrent
> convolutional networks. ICRA 2017.

> Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory.
> Neural Computation 9(8), 1735–1780.

---

### 6. Separate Translation and Rotation Heads

**Choice:** Two independent linear layers after the shared FC,
one for translation (→3) and one for rotation (→6).

**Justification:** Translation and rotation have different
physical units (meters vs radians), different magnitudes, and
different geometric properties. A shared head produces
conflicting gradient signals because improving translation
prediction may worsen rotation prediction and vice versa.
Separate heads allow each to develop independently and are
standard practice in pose regression literature.

**Reference:**
> Kendall, A., Grimes, M., Cipolla, R., 2017. Geometric loss
> functions for camera pose regression with deep learning.
> CVPR 2017.

---

### 7. 6D Rotation Representation

**Choice:** Output 6 unconstrained numbers representing the
first two columns of a 3×3 rotation matrix. Third column
recovered via Gram-Schmidt orthogonalization.

**Justification:** All compact rotation representations with
fewer than 5 dimensions have discontinuities on SO(3).
Specifically:
- Euler angles: gimbal lock at ±90° pitch, wrapping
  discontinuity at ±180°
- Quaternions: double cover (q and -q represent the same
  rotation, creating two valid targets for the same pose),
  unit norm constraint breaks gradient flow
- Axis-angle: discontinuity at 0° and 360°

The 6D representation is continuous everywhere on SO(3).
Small changes in network output always produce small,
predictable changes in the recovered rotation — which is
exactly the smoothness assumption that gradient descent
requires. Zhou et al. proved this is the minimum overhead
representation satisfying this requirement.

**Reference:**
> Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H., 2019. On the
> continuity of rotation representations in neural networks.
> CVPR 2019.

---

### 8. Geodesic Loss for Rotation

**Choice:** Angular distance on SO(3) computed via the atan2
formulation for numerical stability.

**Justification:** MSE on rotation parameters does not
correspond to actual angular error. The geodesic distance θ
is the true shortest path between two rotations on the SO(3)
manifold:

    θ = atan2(||skew(R_diff)||/2, (trace(R_diff)-1)/2)

where R_diff = R_pred^T @ R_gt. This θ is directly
interpretable as angular error in radians and is exactly what
you report in the evaluation table. Optimizing a loss function
that matches the evaluation metric is a fundamental principle
of supervised learning. The atan2 formulation is used instead
of arccos because arccos has infinite gradient at 0 and π,
causing NaN during training.

**References:**
> Mahendran, S., Ali, H., Vidal, R., 2017. 3D pose regression
> using convolutional neural networks. CVPR Workshops 2017.

> Hartley, R., Trumpf, J., Dai, Y., Li, H., 2013. Rotation
> averaging. IJCV 101(1), 86–128.

---

## Training Details

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Optimizer | Adam | Adaptive LR, standard for deep odometry |
| Learning rate | 1e-4 | Standard starting LR for Adam on vision tasks |
| Weight decay | 1e-4 | L2 regularization, prevents overfitting on synthetic data |
| Scheduler | CosineAnnealingLR | Smooth LR decay, avoids sharp drops |
| Batch size | 16 | Balance between gradient quality and memory |
| Sequence length | 10 | 10 frame pairs per LSTM unroll during training |
| λ_rot | 100 | Balances meter-scale translation with radian-scale rotation |
| Gradient clip | 1.0 | Prevents LSTM gradient explosion |
| LSTM dropout | 0.3 | Regularization between LSTM layers |

---

## Evaluation Metrics

**ATE — Absolute Trajectory Error**
RMSE of per-frame translation error after Umeyama alignment
of the full predicted trajectory to ground truth. Measures
global consistency of the trajectory.

**RTE — Relative Trajectory Error**
Mean per-frame translation error without global alignment.
Measures local accuracy of individual pose predictions.

**Mean Rotation Error**
Mean geodesic distance in degrees between predicted and ground
truth rotation matrices across all frames.

---

## Data Format

```
dataset/
  sequence_001/
    frames/
      frame_000000.png
      frame_000001.png
      ...
    poses.txt    ← 4x4 homogeneous transforms, 16 floats per line
    imu.txt      ← IMU readings: [ax, ay, az, gx, gy, gz] per line
```

Relative pose computed as:
```python
T_rel = inv(T_world_t) @ T_world_t1
trans = T_rel[:3, 3]
R     = T_rel[:3, :3]
```

---

## File Structure

```
branch_a/
├── model.py      BranchA, CNNEncoder, CorrelationLayer
├── loss.py       geodesic_loss_stable, branch_a_loss
├── dataset.py    UAVOdometryDataset
├── train.py      Training loop with TensorBoard logging
├── evaluate.py   Evaluation metrics + trajectory plots
├── utils.py      gram_schmidt, pose_to_matrix, Umeyama alignment
└── README.md     This file
```

---

## References

1. Bertinetto et al., "Fully-Convolutional Siamese Networks for
   Object Tracking," ECCV 2016
2. Sandler et al., "MobileNetV2: Inverted Residuals and Linear
   Bottlenecks," CVPR 2018
3. Dosovitskiy et al., "FlowNet: Learning Optical Flow with
   Convolutional Networks," ICCV 2015
4. Ilg et al., "FlowNet 2.0: Evolution of Optical Flow
   Estimation with Deep Networks," CVPR 2017
5. Sun et al., "PWC-Net: CNNs for Optical Flow Using Pyramid,
   Warping, and Cost Volume," CVPR 2018
6. Wang et al., "DeepVO: Towards End-to-End Visual Odometry
   with Deep Recurrent Convolutional Networks," ICRA 2017
7. Hochreiter and Schmidhuber, "Long Short-Term Memory,"
   Neural Computation 1997
8. Kendall et al., "Geometric Loss Functions for Camera Pose
   Regression with Deep Learning," CVPR 2017
9. Zhou et al., "On the Continuity of Rotation Representations
   in Neural Networks," CVPR 2019
10. Mahendran et al., "3D Pose Regression Using Convolutional
    Neural Networks," CVPR Workshops 2017
11. Hartley et al., "Rotation Averaging," IJCV 2013
