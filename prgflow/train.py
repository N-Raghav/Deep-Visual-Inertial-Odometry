import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from prgflow.dataset import VIOPairsDataset, split_trajectories
from prgflow.loss import accuracy_percent, prgflow_loss, scale_error_px, translation_error_px
from prgflow.model import PRGFlow


def parse_args():
    p = argparse.ArgumentParser(description="Train PRGFlow VIO.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--fire_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/vio")
    p.add_argument("--log_dir", type=str, default="logs/vio")
    p.add_argument("--run_name", type=str, default="",
                   help="Tag appended to checkpoint_dir and log_dir to identify the run.")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_trans = 0.0
    total_scale = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        patch_t = batch["patch_t"].to(device, non_blocking=True)
        patch_t1 = batch["patch_t1"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(patch_t, patch_t1)
        loss = prgflow_loss(pred, label)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = patch_t.shape[0]
        total_loss += float(loss.detach()) * bs
        total_trans += float(translation_error_px(pred, label).detach()) * bs
        total_scale += float(scale_error_px(pred, label, patch_size=patch_t.shape[-1]).detach()) * bs
        total_acc += float(accuracy_percent(pred, label, patch_size=patch_t.shape[-1]).detach()) * bs
        n += bs

    if n == 0:
        return {"l2": 0.0, "e_trans": 0.0, "e_scale": 0.0, "acc": 0.0}

    return {
        "l2": total_loss / n,
        "e_trans": total_trans / n,
        "e_scale": total_scale / n,
        "acc": total_acc / n,
    }


def main():
    args = parse_args()
    if args.run_name:
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
        args.log_dir = os.path.join(args.log_dir, args.run_name)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = split_trajectories(args.data_root, seed=args.seed)
    print("train:", splits["train"])
    print("val  :", splits["val"])
    print("test :", splits["test"])

    train_set = VIOPairsDataset(args.data_root, trajectories=splits["train"])
    val_set = VIOPairsDataset(args.data_root, trajectories=splits["val"] or splits["train"][:1])
    test_set = VIOPairsDataset(args.data_root, trajectories=splits["test"] or splits["val"] or splits["train"][:1])

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
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PRGFlow(
        base_channels=args.base_channels,
        fire_blocks=args.fire_blocks,
        dropout=args.dropout,
    ).to(device)
    print(f"params: {model.num_trainable_parameters():,}")
    print(f"size_mb: {model.size_mb():.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device)

        print(
            f"[{epoch + 1:03d}/{args.epochs}] "
            f"train l2={train_metrics['l2']:.4f} et={train_metrics['e_trans']:.3f}px es={train_metrics['e_scale']:.3f}px acc={train_metrics['acc']:.2f}% | "
            f"val l2={val_metrics['l2']:.4f} et={val_metrics['e_trans']:.3f}px es={val_metrics['e_scale']:.3f}px acc={val_metrics['acc']:.2f}%"
        )

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        if val_metrics["l2"] < best_val:
            best_val = val_metrics["l2"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "splits": splits,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"saved {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stop")
                break

    writer.close()

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, None, device)
    print(
        f"test l2={test_metrics['l2']:.4f} "
        f"et={test_metrics['e_trans']:.3f}px "
        f"es={test_metrics['e_scale']:.3f}px "
        f"acc={test_metrics['acc']:.2f}%"
    )

    writer.add_hparams(
        {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "base_channels": args.base_channels,
            "fire_blocks": args.fire_blocks,
            "dropout": args.dropout,
            "patience": args.patience,
        },
        {
            "hparam/test_l2": test_metrics["l2"],
            "hparam/test_e_trans": test_metrics["e_trans"],
            "hparam/test_e_scale": test_metrics["e_scale"],
            "hparam/test_acc": test_metrics["acc"],
            "hparam/best_val_l2": best_val,
        },
    )

    if test_metrics["e_trans"] > 2.0 or test_metrics["e_scale"] > 4.0:
        print("warning: target thresholds not reached yet")


if __name__ == "__main__":
    main()
