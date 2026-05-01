"""
End-to-end smoke test for the Branch A pipeline.

Run this after `setup_env.slurm` succeeds (or any local env that has
torch + torchvision + Pillow + numpy) to verify the implementation
without needing a real Blender dataset:

    python test_pipeline.py

It:

    1. Generates a tiny synthetic dataset on disk (3 sequences, 12 frames
       each, 64x64 pixels) with realistic per-frame relative poses.
    2. Checks ``gram_schmidt`` produces valid rotations.
    3. Checks ``geodesic_loss_stable`` is zero at identity and matches an
       analytic reference for a known rotation.
    4. Checks ``CorrelationLayer`` peaks at zero displacement when both
       feature maps are identical.
    5. Loads the synthetic dataset and runs a forward + backward pass
       through ``BranchA``, then a single optimizer step.
    6. Checks ``umeyama_alignment`` recovers a known similarity transform.

Exits non-zero if any check fails. All work happens in a temporary
directory, so nothing is left behind.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Local imports — this script is intended to be run from within the
# ``vision_only/`` directory so that the sibling modules resolve.
from dataset import UAVOdometryDataset
from loss import branch_a_loss, geodesic_loss_stable
from model import BranchA, CorrelationLayer
from utils import gram_schmidt, umeyama_alignment


# Pretty-print helper -------------------------------------------------------
_OK = "\033[32mOK\033[0m"
_FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, condition: bool, detail: str = "") -> None:
    """Print a check result and abort the script on failure."""
    if condition:
        print(f"  [{_OK}] {name}{(' — ' + detail) if detail else ''}")
    else:
        print(f"  [{_FAIL}] {name}{(' — ' + detail) if detail else ''}")
        sys.exit(1)


# Synthetic dataset --------------------------------------------------------
def _make_synthetic_dataset(
    root: Path, n_sequences: int, n_frames: int, h: int, w: int
) -> None:
    """Create a tiny dataset on disk with structured RGB textures + poses."""
    rng = np.random.default_rng(0)
    for s in range(n_sequences):
        seq_dir = root / f"sequence_{s:03d}"
        (seq_dir / "frames").mkdir(parents=True, exist_ok=True)

        # Frames: a static random texture per sequence, shifted a few
        # pixels per frame so there is real motion in the sequence.
        base = rng.integers(0, 255, size=(h + 32, w + 32, 3), dtype=np.uint8)
        T_world = np.eye(4, dtype=np.float64)
        poses = [T_world.copy()]
        imu_rows = []
        for f in range(n_frames):
            shift_x = (f * 2) % 16
            shift_y = (f * 1) % 16
            patch = base[shift_y : shift_y + h, shift_x : shift_x + w, :].copy()
            Image.fromarray(patch).save(seq_dir / "frames" / f"frame_{f:06d}.png")

            if f > 0:
                # Small random rigid step.
                t_step = rng.normal(scale=0.05, size=3)
                ang = rng.normal(scale=0.02, size=3)  # small-angle Euler
                cx, cy, cz = np.cos(ang)
                sx, sy, sz = np.sin(ang)
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                R_step = Rz @ Ry @ Rx
                T_step = np.eye(4)
                T_step[:3, :3] = R_step
                T_step[:3, 3] = t_step
                T_world = T_world @ T_step
                poses.append(T_world.copy())
                imu_rows.append(np.concatenate([t_step / 0.05, ang / 0.05]))
            else:
                imu_rows.append(np.zeros(6))

        flat = np.stack([p.reshape(-1) for p in poses], axis=0)
        np.savetxt(seq_dir / "poses.txt", flat)
        np.savetxt(seq_dir / "imu.txt", np.stack(imu_rows, axis=0))


# Individual checks --------------------------------------------------------
def check_gram_schmidt() -> None:
    print("\n[1/6] gram_schmidt")
    torch.manual_seed(0)
    r6 = torch.randn(4, 5, 6)
    R = gram_schmidt(r6)
    _check("output shape", R.shape == (4, 5, 3, 3), str(tuple(R.shape)))

    eye = torch.eye(3).expand_as(R)
    err = (R.transpose(-1, -2) @ R - eye).abs().max().item()
    _check("orthonormal columns (R^T R = I)", err < 1e-5, f"max|err|={err:.2e}")

    det = torch.linalg.det(R)
    det_err = (det - 1.0).abs().max().item()
    _check("right-handed (det R = 1)", det_err < 1e-5, f"max|det-1|={det_err:.2e}")


def check_geodesic_loss() -> None:
    print("\n[2/6] geodesic_loss_stable")
    R_id = torch.eye(3).expand(2, 4, 3, 3).contiguous()
    loss_zero = geodesic_loss_stable(R_id, R_id)
    _check("zero at identity", loss_zero.item() < 1e-7, f"loss={loss_zero.item():.2e}")

    # Known rotation: 30 degrees around Z.
    theta = math.radians(30.0)
    Rz = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta),  math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R_pred = torch.eye(3).expand(1, 1, 3, 3)
    R_gt = Rz.expand(1, 1, 3, 3)
    angle = geodesic_loss_stable(R_pred, R_gt).item()
    _check(
        "matches analytic 30 deg",
        abs(math.degrees(angle) - 30.0) < 1e-3,
        f"got {math.degrees(angle):.4f} deg",
    )


def check_correlation_layer() -> None:
    print("\n[3/6] CorrelationLayer")
    layer = CorrelationLayer(max_displacement=4)
    feat = torch.randn(2, 8, 14, 14)
    out = layer(feat, feat)
    _check("output shape", out.shape == (2, 81, 14, 14), str(tuple(out.shape)))

    # When both features are identical, zero displacement (centre channel,
    # index 4*9 + 4 = 40) should dominate at every spatial location.
    centre_argmax = out[0, :, 5, 5].argmax().item()
    _check("centre offset wins on identical inputs", centre_argmax == 40,
           f"argmax={centre_argmax}")


def check_branch_a_forward_backward(tmp_root: Path) -> None:
    print("\n[4/6] BranchA forward + backward")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"     device={device}")

    dataset = UAVOdometryDataset(
        root_dir=str(tmp_root),
        sequence_length=4,
        img_height=64,
        img_width=64,
        augment=True,
    )
    _check("dataset is non-empty", len(dataset) > 0, f"len={len(dataset)}")

    sample = dataset[0]
    _check("sample shapes (frames_t)", sample["frames_t"].shape == (4, 3, 64, 64))
    _check("sample shapes (R_gt)",     sample["R_gt"].shape == (4, 3, 3))
    _check("sample shapes (rot_6d)",   sample["rot_6d_gt"].shape == (4, 6))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))

    model = BranchA().to(device)
    n_params = model.num_trainable_parameters()
    _check("model has trainable params", n_params > 0, f"{n_params:,}")

    frames_t = batch["frames_t"].to(device)
    frames_t1 = batch["frames_t1"].to(device)
    trans_gt = batch["trans_gt"].to(device)
    R_gt = batch["R_gt"].to(device)

    trans, rot_6d, R, hidden, features = model(frames_t, frames_t1, hidden=None)
    _check("trans shape",    trans.shape == (2, 4, 3),    str(tuple(trans.shape)))
    _check("rot_6d shape",   rot_6d.shape == (2, 4, 6),   str(tuple(rot_6d.shape)))
    _check("R shape",        R.shape == (2, 4, 3, 3),     str(tuple(R.shape)))
    _check("features shape", features.shape == (2, 4, 128), str(tuple(features.shape)))
    _check("LSTM h shape",   hidden[0].shape == (2, 2, 256), str(tuple(hidden[0].shape)))

    # Hidden-state passthrough.
    _, _, _, hidden2, _ = model(frames_t, frames_t1, hidden=hidden)
    _check("hidden carries forward without crash", hidden2[0].shape == hidden[0].shape)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    optim.zero_grad()
    total, t_loss, r_loss = branch_a_loss(trans, trans_gt, R, R_gt, lambda_rot=100.0)
    _check("loss is finite",
           torch.isfinite(total).item(),
           f"total={total.item():.4f} trans={t_loss.item():.4f} rot={r_loss.item():.4f}")

    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    _check("gradients are finite", math.isfinite(float(grad_norm)),
           f"grad_norm={float(grad_norm):.3f}")
    optim.step()
    _check("optimizer step ran", True)

    # Single-pair helper.
    flat = model.encode_frame_pair(frames_t[:, 0], frames_t1[:, 0])
    _check("encode_frame_pair shape", flat.shape == (2, 1024), str(tuple(flat.shape)))


def check_umeyama() -> None:
    print("\n[5/6] umeyama_alignment")
    rng = np.random.default_rng(7)
    P = rng.normal(size=(50, 3))
    # Known similarity transform.
    angle = math.radians(20.0)
    R_true = np.array(
        [[math.cos(angle), -math.sin(angle), 0.0],
         [math.sin(angle),  math.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    s_true = 1.7
    t_true = np.array([0.5, -0.2, 1.1])
    Q = (s_true * (R_true @ P.T)).T + t_true

    # Recover P->Q.
    R_hat, t_hat, s_hat, aligned = umeyama_alignment(P, Q, with_scale=True)
    _check("scale recovered",        abs(s_hat - s_true) < 1e-6, f"s={s_hat:.6f}")
    _check("rotation recovered",     np.allclose(R_hat, R_true, atol=1e-6))
    _check("translation recovered",  np.allclose(t_hat, t_true, atol=1e-6))
    _check("aligned points match Q", np.allclose(aligned, Q, atol=1e-6))


def check_loss_is_metric_on_full_pipeline(tmp_root: Path) -> None:
    """Sanity: identical predictions and ground truth -> zero loss components."""
    print("\n[6/6] loss === 0 when prediction equals ground truth")
    dataset = UAVOdometryDataset(
        root_dir=str(tmp_root),
        sequence_length=4,
        img_height=64,
        img_width=64,
        augment=False,
    )
    sample = dataset[0]
    trans_gt = sample["trans_gt"].unsqueeze(0)  # [1, T, 3]
    R_gt = sample["R_gt"].unsqueeze(0)          # [1, T, 3, 3]
    total, t_loss, r_loss = branch_a_loss(trans_gt, trans_gt, R_gt, R_gt, lambda_rot=100.0)
    _check("trans_loss == 0", t_loss.item() < 1e-6, f"{t_loss.item():.2e}")
    _check("rot_loss == 0",   r_loss.item() < 1e-6, f"{r_loss.item():.2e}")
    _check("total == 0",      total.item()  < 1e-6, f"{total.item():.2e}")


# Entry point --------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print(" Branch A pipeline smoke test")
    print("=" * 60)
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0    : {torch.cuda.get_device_name(0)}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="vision_only_smoke_"))
    print(f"tmp dataset : {tmp_dir}")
    try:
        _make_synthetic_dataset(tmp_dir, n_sequences=3, n_frames=12, h=64, w=64)
        check_gram_schmidt()
        check_geodesic_loss()
        check_correlation_layer()
        check_branch_a_forward_backward(tmp_dir)
        check_umeyama()
        check_loss_is_metric_on_full_pipeline(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(" All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
