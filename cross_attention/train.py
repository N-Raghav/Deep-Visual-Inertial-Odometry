"""
Train the cross-attention tight-fusion network.

Mirrors ``gated_fusion/train.py`` but replaces ``GatedFusionNet`` with
``CrossAttentionFusionNet``. All other training mechanics are identical
(Adam, warm-up with frozen backbones, joint fine-tune with smaller lr,
cosine LR schedule, smooth-L1 + geodesic pose loss).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion_common.dataset import PairedDataset  # noqa: E402
from fusion_common.loss import pose_loss  # noqa: E402
from model import CrossAttentionFusionNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cross-attention fusion.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--vision_checkpoint", type=str, default=None)
    p.add_argument("--airio_checkpoint", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_finetune", type=float, default=2e-5)
    p.add_argument("--lambda_rot", type=float, default=100.0)
    p.add_argument("--sequence_length", type=int, default=10)
    p.add_argument("--imu_context", type=int, default=32)
    p.add_argument("--imu_rate", type=float, default=30.0)
    p.add_argument("--img_height", type=int, default=224)
    p.add_argument("--img_width", type=int, default=224)
    # Transformer-specific:
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--ffn_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/cross_attention")
    p.add_argument("--log_dir", type=str, default="logs/cross_attention")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sequences(root: str, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    seq_names = PairedDataset.list_sequences(root)
    rng = random.Random(seed); rng.shuffle(seq_names)
    n_val = max(1, int(round(val_fraction * len(seq_names))))
    val = sorted(seq_names[:n_val]); train = sorted(seq_names[n_val:])
    if not train:
        train = val[:1]; val = val[1:]
    return train, val


def run_epoch(model, loader, optimizer, device, args, scaler, train):
    model.train(train)
    sums = {"total": 0.0, "trans": 0.0, "rot_rad": 0.0, "vis_imu_cos": 0.0}
    n = 0
    for batch in loader:
        frames_t = batch["frames_t"].to(device, non_blocking=True)
        frames_t1 = batch["frames_t1"].to(device, non_blocking=True)
        imu_acc = batch["imu_acc"].to(device, non_blocking=True)
        imu_gyro = batch["imu_gyro"].to(device, non_blocking=True)
        attitude = batch["attitude"].to(device, non_blocking=True)
        trans_gt = batch["trans_gt"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            out = model(frames_t, frames_t1, imu_acc, imu_gyro, attitude)
            total, trans_l, rot_l = pose_loss(
                out["trans"], trans_gt, out["R"], R_gt, lambda_rot=args.lambda_rot
            )
        if train:
            if scaler is not None:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        sums["total"] += float(total.detach())
        sums["trans"] += float(trans_l.detach())
        sums["rot_rad"] += float(rot_l.detach())
        sums["vis_imu_cos"] += float(out["vis_imu_cos"].detach().mean())
        n += 1

    n = max(1, n)
    return {
        "total": sums["total"] / n,
        "trans": sums["trans"] / n,
        "rot_rad": sums["rot_rad"] / n,
        "rot_deg": math.degrees(sums["rot_rad"] / n),
        "vis_imu_cos": sums["vis_imu_cos"] / n,
    }


def _set_backbone_freeze(model, frozen):
    for p in model.branch_a.parameters():
        p.requires_grad_(not frozen)
    for p in model.airio.parameters():
        p.requires_grad_(not frozen)


def _build_optimizer(model, lr):
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_seqs, val_seqs = split_sequences(args.data_root, args.val_fraction, args.seed)
    print(f"CrossAttn train seqs ({len(train_seqs)}): {train_seqs}")
    print(f"CrossAttn val   seqs ({len(val_seqs)}): {val_seqs}")

    common = dict(
        root_dir=args.data_root,
        sequence_length=args.sequence_length,
        imu_context=args.imu_context,
        img_height=args.img_height,
        img_width=args.img_width,
        imu_rate=args.imu_rate,
    )
    train_set = PairedDataset(sequences=train_seqs, augment=True, **common)
    val_set = PairedDataset(sequences=val_seqs, augment=False, **common)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = CrossAttentionFusionNet.load_pretrained(
        vision_checkpoint=args.vision_checkpoint,
        airio_checkpoint=args.airio_checkpoint,
        device=device,
        freeze_backbones=args.warmup_epochs > 0,
        feat_dim=args.feat_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_hidden=args.ffn_hidden,
        dropout=args.dropout,
    )
    print(f"CrossAttn trainable params: {model.num_trainable_parameters():,}")

    optimizer = _build_optimizer(model, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_trans = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    last_path = os.path.join(args.checkpoint_dir, "last.pt")

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            print(f"[cross_attention] unfreezing backbones at epoch {epoch + 1}; "
                  f"lr -> {args.lr_finetune:.2e}")
            _set_backbone_freeze(model, frozen=False)
            optimizer = _build_optimizer(model, lr=args.lr_finetune)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch)
            )

        tr = run_epoch(model, train_loader, optimizer, device, args, scaler, True)
        with torch.no_grad():
            va = run_epoch(model, val_loader, None, device, args, None, False)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        for k in ("total", "trans", "rot_deg", "vis_imu_cos"):
            writer.add_scalar(f"train/{k}", tr[k], epoch)
            writer.add_scalar(f"val/{k}", va[k], epoch)
        writer.add_scalar("lr", lr, epoch)

        print(
            f"[cross_attention epoch {epoch + 1:03d}/{args.epochs}] "
            f"train total={tr['total']:.4f} trans={tr['trans']:.4f} "
            f"rot={tr['rot_deg']:.3f}deg cos={tr['vis_imu_cos']:.3f} | "
            f"val total={va['total']:.4f} trans={va['trans']:.4f} "
            f"rot={va['rot_deg']:.3f}deg cos={va['vis_imu_cos']:.3f} | "
            f"lr={lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "val_trans": va["trans"],
        }
        torch.save(ckpt, last_path)
        if va["trans"] < best_val_trans:
            best_val_trans = va["trans"]
            torch.save(ckpt, best_path)
            print(f"  -> saved new best (val trans={best_val_trans:.4f}) to {best_path}")

    writer.close()


if __name__ == "__main__":
    main()
