"""
Train AirIO on top of a frozen, pre-trained AirIMU.

Usage::

    python train_airio.py \\
        --data_root /path/to/dataset \\
        --airimu_checkpoint checkpoints/airimu/best.pt

The frozen AirIMU is run inside ``torch.no_grad()`` to produce the
corrected IMU stream and is *not* updated during AirIO training. AirIO
takes the corrected IMU plus the ground-truth attitude (per the paper's
training protocol) and is supervised on per-sample body-frame velocity
with the Huber + Gaussian-NLL loss (Eq. 3 of the paper).

If ``--airimu_checkpoint`` is omitted, AirIO is trained directly on the
raw IMU (this matches the "AirIO Net" column of the paper, without the
EKF / AirIMU benefit).
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import IMUWindowDataset
from loss import airio_loss
from model import AirIMUNet, AirIONet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AirIO motion network.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--airimu_checkpoint", type=str, default=None,
                   help="Optional path to a frozen AirIMU checkpoint.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--imu_rate", type=float, default=1000.0)
    p.add_argument("--window_size", type=int, default=5000)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--lambda_c", type=float, default=1e-4)
    p.add_argument("--huber_delta", type=float, default=0.005)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/airio")
    p.add_argument("--log_dir", type=str, default="logs/airio")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sequences(root: str, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    seq_names = IMUWindowDataset.list_sequences(root)
    rng = random.Random(seed)
    rng.shuffle(seq_names)
    n_val = max(1, int(round(val_fraction * len(seq_names))))
    val = sorted(seq_names[:n_val])
    train = sorted(seq_names[n_val:])
    if not train:
        train = val[:1]
        val = val[1:]
    return train, val


def load_airimu(path: str | None, device: torch.device) -> AirIMUNet | None:
    """Load a frozen AirIMU. Returns ``None`` when ``path`` is empty."""
    if not path:
        return None
    print(f"[airio] loading frozen AirIMU from {path}")
    model = AirIMUNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_epoch(
    airio: AirIONet,
    airimu: AirIMUNet | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    scaler: torch.cuda.amp.GradScaler | None,
    train: bool,
) -> dict[str, float]:
    airio.train(train)
    sums = {"total": 0.0, "huber": 0.0, "nll": 0.0}
    n = 0

    for batch in loader:
        acc = batch["acc"].to(device, non_blocking=True)
        gyro = batch["gyro"].to(device, non_blocking=True)
        attitude = batch["attitude"].to(device, non_blocking=True)
        v_body_gt = batch["v_body"].to(device, non_blocking=True).float()

        # Run AirIMU correction without gradients.
        if airimu is not None:
            with torch.no_grad():
                acc_in, gyro_in, _ = airimu.correct(acc, gyro)
        else:
            acc_in, gyro_in = acc, gyro

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            v_pred, log_var = airio(acc_in, gyro_in, attitude)
            total, huber, nll = airio_loss(
                v_pred, v_body_gt, log_var,
                huber_delta=args.huber_delta,
                lambda_c=args.lambda_c,
            )

        if train:
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(airio.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(airio.parameters(), 1.0)
                optimizer.step()

        sums["total"] += float(total.detach())
        sums["huber"] += float(huber.detach())
        sums["nll"] += float(nll.detach())
        n += 1

    n = max(1, n)
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_seqs, val_seqs = split_sequences(args.data_root, args.val_fraction, args.seed)
    print(f"AirIO train seqs ({len(train_seqs)}): {train_seqs}")
    print(f"AirIO val   seqs ({len(val_seqs)}): {val_seqs}")

    common = dict(
        root_dir=args.data_root,
        window_size=args.window_size,
        step_size=args.step_size,
        imu_rate=args.imu_rate,
    )
    train_set = IMUWindowDataset(sequences=train_seqs, **common)
    val_set = IMUWindowDataset(sequences=val_seqs, **common)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    airimu = load_airimu(args.airimu_checkpoint, device)
    airio = AirIONet().to(device)
    print(f"AirIO trainable params: {airio.num_trainable_parameters():,}")

    optimizer = torch.optim.Adam(airio.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    last_path = os.path.join(args.checkpoint_dir, "last.pt")

    for epoch in range(args.epochs):
        tr = run_epoch(airio, airimu, train_loader, optimizer, device, args, scaler, True)
        with torch.no_grad():
            va = run_epoch(airio, airimu, val_loader, None, device, args, None, False)
        scheduler.step(va["total"])
        lr = optimizer.param_groups[0]["lr"]

        for k in ("total", "huber", "nll"):
            writer.add_scalar(f"train/{k}", tr[k], epoch)
            writer.add_scalar(f"val/{k}", va[k], epoch)
        writer.add_scalar("lr", lr, epoch)

        print(
            f"[airio epoch {epoch + 1:03d}/{args.epochs}] "
            f"train total={tr['total']:.4f} huber={tr['huber']:.4f} | "
            f"val total={va['total']:.4f} huber={va['huber']:.4f} | "
            f"lr={lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": airio.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "val_huber": va["huber"],
            "airimu_checkpoint": args.airimu_checkpoint,
        }
        torch.save(ckpt, last_path)
        if va["huber"] < best_val:
            best_val = va["huber"]
            torch.save(ckpt, best_path)
            print(f"  -> saved new best (val huber={best_val:.6f}) to {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
