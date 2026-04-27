"""
Training script for Branch A.

Usage::

    python train.py --data_root /path/to/dataset --epochs 100

Logs per-epoch losses (total, translation, rotation in degrees, learning
rate) to TensorBoard and saves the best validation checkpoint to
``--checkpoint_dir``.
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

from dataset import UAVOdometryDataset
from loss import branch_a_loss
from model import BranchA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Branch A vision-only odometry.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_rot", type=float, default=100.0)
    p.add_argument("--sequence_length", type=int, default=10)
    p.add_argument("--img_height", type=int, default=224)
    p.add_argument("--img_width", type=int, default=224)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/vision_only")
    p.add_argument("--log_dir", type=str, default="logs/vision_only")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sequences(
    root: str, val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split sequences into train/val groups by sequence (not by frame)."""
    seq_names = UAVOdometryDataset.list_sequences(root)
    rng = random.Random(seed)
    rng.shuffle(seq_names)
    n_val = max(1, int(round(val_fraction * len(seq_names))))
    val = sorted(seq_names[:n_val])
    train = sorted(seq_names[n_val:])
    if not train:
        # Tiny datasets: keep at least one sequence in train.
        train = val[:1]
        val = val[1:]
    return train, val


def run_epoch(
    model: BranchA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_rot: float,
    scaler: torch.cuda.amp.GradScaler | None,
    train: bool,
) -> dict[str, float]:
    """Run one training or validation epoch.

    Returns a dictionary with mean total / translation / rotation losses
    over the epoch. Rotation loss is reported in degrees for readability.
    """
    model.train(train)

    totals = {"total": 0.0, "trans": 0.0, "rot_rad": 0.0}
    n_batches = 0

    for batch in loader:
        frames_t = batch["frames_t"].to(device, non_blocking=True)
        frames_t1 = batch["frames_t1"].to(device, non_blocking=True)
        trans_gt = batch["trans_gt"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        # Each batch starts a fresh trajectory window (sequences from the
        # same trajectory are not shuffled across batches), so the LSTM
        # hidden state is reset at every step.
        autocast_enabled = scaler is not None
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            trans_pred, _, R_pred, _, _ = model(frames_t, frames_t1, hidden=None)
            total, trans_loss, rot_loss = branch_a_loss(
                trans_pred, trans_gt, R_pred, R_gt, lambda_rot=lambda_rot
            )

        if train:
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        totals["total"] += float(total.detach())
        totals["trans"] += float(trans_loss.detach())
        totals["rot_rad"] += float(rot_loss.detach())
        n_batches += 1

    n = max(1, n_batches)
    return {
        "total": totals["total"] / n,
        "trans": totals["trans"] / n,
        "rot_rad": totals["rot_rad"] / n,
        "rot_deg": math.degrees(totals["rot_rad"] / n),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_seqs, val_seqs = split_sequences(args.data_root, args.val_fraction, args.seed)
    print(f"Train sequences ({len(train_seqs)}): {train_seqs}")
    print(f"Val sequences   ({len(val_seqs)}): {val_seqs}")

    train_set = UAVOdometryDataset(
        root_dir=args.data_root,
        sequence_length=args.sequence_length,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=True,
        sequences=train_seqs,
    )
    val_set = UAVOdometryDataset(
        root_dir=args.data_root,
        sequence_length=args.sequence_length,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=False,
        sequences=val_seqs,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = BranchA().to(device)
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_trans = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    last_path = os.path.join(args.checkpoint_dir, "last.pt")

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, args.lambda_rot, scaler, train=True
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, None, device, args.lambda_rot, None, train=False
            )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("train/total", train_metrics["total"], epoch)
        writer.add_scalar("train/trans", train_metrics["trans"], epoch)
        writer.add_scalar("train/rot_deg", train_metrics["rot_deg"], epoch)
        writer.add_scalar("val/total", val_metrics["total"], epoch)
        writer.add_scalar("val/trans", val_metrics["trans"], epoch)
        writer.add_scalar("val/rot_deg", val_metrics["rot_deg"], epoch)
        writer.add_scalar("lr", lr, epoch)

        print(
            f"[epoch {epoch + 1:03d}/{args.epochs}] "
            f"train total={train_metrics['total']:.4f} "
            f"trans={train_metrics['trans']:.4f} "
            f"rot={train_metrics['rot_deg']:.3f}deg | "
            f"val total={val_metrics['total']:.4f} "
            f"trans={val_metrics['trans']:.4f} "
            f"rot={val_metrics['rot_deg']:.3f}deg | "
            f"lr={lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "val_trans": val_metrics["trans"],
        }
        torch.save(ckpt, last_path)

        if val_metrics["trans"] < best_val_trans:
            best_val_trans = val_metrics["trans"]
            torch.save(ckpt, best_path)
            print(f"  -> saved new best (val trans={best_val_trans:.4f}) to {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
