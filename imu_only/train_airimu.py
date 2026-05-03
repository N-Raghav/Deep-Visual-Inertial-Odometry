"""
Pre-train AirIMU on short integration windows.

Usage::

    python train_airimu.py --data_root /path/to/dataset --epochs 60

Trains the IMU-correction network described in Qiu et al. 2023 (ref [30]
of the AirIO paper). The corrected IMU is integrated through the
forward-Euler kinematic model in :func:`utils.integrate_imu_window` and
the network is supervised on the resulting end-of-window rotation,
velocity and position residuals plus a Gaussian NLL on the predicted
per-sample uncertainty.

The default window length is short (20 IMU samples ≈ 100 ms at 200 Hz)
because integration error in a forward-Euler scheme blows up with
window length. Longer windows can be used with more careful integrators.
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
from loss import airimu_loss
from model import AirIMUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train AirIMU.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--imu_rate", type=float, default=1000.0)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/airimu")
    p.add_argument("--log_dir", type=str, default="logs/airimu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--lambda_rot", type=float, default=10.0)
    p.add_argument("--lambda_vel", type=float, default=1.0)
    p.add_argument("--lambda_pos", type=float, default=1.0)
    p.add_argument("--lambda_nll", type=float, default=1e-4)
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


def run_epoch(
    model: AirIMUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    scaler: torch.cuda.amp.GradScaler | None,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    sums = {"total": 0.0, "rot": 0.0, "vel": 0.0, "pos": 0.0, "nll": 0.0}
    n = 0

    for batch in loader:
        acc = batch["acc"].to(device, non_blocking=True)
        gyro = batch["gyro"].to(device, non_blocking=True)
        R_start = batch["R_start"].to(device, non_blocking=True).float()
        R_end = batch["R_end"].to(device, non_blocking=True).float()
        v_world_start = batch["v_world_start"].to(device, non_blocking=True).float()
        v_world_end = batch["v_world_end"].to(device, non_blocking=True).float()
        p_start = batch["p_start"].to(device, non_blocking=True).float()
        p_end = batch["p_end"].to(device, non_blocking=True).float()
        dt = batch["dt"][0].item()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            acc_hat, gyro_hat, log_var = model.correct(acc, gyro)
            total, terms = airimu_loss(
                acc_hat, gyro_hat, log_var,
                R_start, R_end,
                v_world_start, v_world_end,
                p_start, p_end,
                dt=dt,
                lambda_rot=args.lambda_rot,
                lambda_vel=args.lambda_vel,
                lambda_pos=args.lambda_pos,
                lambda_nll=args.lambda_nll,
            )

        if train:
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        sums["total"] += float(total.detach())
        for k in ("rot", "vel", "pos", "nll"):
            sums[k] += float(terms[k])
        n += 1

    n = max(1, n)
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_seqs, val_seqs = split_sequences(args.data_root, args.val_fraction, args.seed)
    print(f"AirIMU train seqs ({len(train_seqs)}): {train_seqs}")
    print(f"AirIMU val   seqs ({len(val_seqs)}): {val_seqs}")

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

    model = AirIMUNet().to(device)
    print(f"AirIMU trainable params: {model.num_trainable_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        tr = run_epoch(model, train_loader, optimizer, device, args, scaler, train=True)
        with torch.no_grad():
            va = run_epoch(model, val_loader, None, device, args, None, train=False)
        scheduler.step(va["total"])
        lr = optimizer.param_groups[0]["lr"]

        for k in ("total", "rot", "vel", "pos", "nll"):
            writer.add_scalar(f"train/{k}", tr[k], epoch)
            writer.add_scalar(f"val/{k}", va[k], epoch)
        writer.add_scalar("lr", lr, epoch)

        rot_deg_train = math.degrees(math.sqrt(max(tr["rot"], 0.0))) if tr["rot"] > 0 else 0.0
        rot_deg_val = math.degrees(math.sqrt(max(va["rot"], 0.0))) if va["rot"] > 0 else 0.0
        print(
            f"[airimu epoch {epoch + 1:03d}/{args.epochs}] "
            f"train total={tr['total']:.4f} rot={tr['rot']:.4f} "
            f"vel={tr['vel']:.4f} pos={tr['pos']:.4f} | "
            f"val total={va['total']:.4f} rot={va['rot']:.4f} "
            f"vel={va['vel']:.4f} pos={va['pos']:.4f} | lr={lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "val_total": va["total"],
        }
        torch.save(ckpt, last_path)
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save(ckpt, best_path)
            print(f"  -> saved new best (val total={best_val:.4f}) to {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
