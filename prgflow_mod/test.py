import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prgflow_mod.dataset import VIOPairsDataset
from prgflow_mod.evaluate import run_sequence
from prgflow_mod.loss import accuracy_percent, composite_loss, scale_error_px, translation_error_px, yaw_error_rad
from prgflow_mod.model import PRGFlow
from prgflow_mod.preprocess import list_trajectories, process_one


def parse_args():
    p = argparse.ArgumentParser(
        description="Test PRGFlow VIO checkpoint on raw (unpreprocessed) test data."
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to raw test data (e.g. data_test).")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (e.g. checkpoints/vio_imp/best.pt).")
    p.add_argument("--pairs_root", type=str, default=None,
                   help="Path to write/read preprocessed pairs. "
                        "Defaults to <data_root>_preprocessed alongside data_root.")
    p.add_argument("--results_dir", type=str, default="results/vio_imp_test",
                   help="Directory to write logs, JSON, CSV, and plots.")
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.08,
                   help="Visual/IMU complementary fusion weight.")
    p.add_argument("--overwrite_pairs", action="store_true",
                   help="Re-run preprocessing even if pairs already exist.")
    p.add_argument("--skip_trajectory_eval", action="store_true",
                   help="Skip the full ATE/RTE trajectory evaluation.")
    p.add_argument("--skip_patch_eval", action="store_true",
                   help="Skip the patch-level (loss/e_trans/e_scale/e_yaw/acc) evaluation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default="",
                   help="Tag appended to results_dir to identify the run.")
    return p.parse_args()


def log_print(msg, log_file=None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def preprocess_test(raw_root, pairs_root, crop_size, pad, overwrite, log_file):
    raw_root = Path(raw_root)
    pairs_root = Path(pairs_root)
    pairs_root.mkdir(parents=True, exist_ok=True)
    trajs = list_trajectories(raw_root)
    log_print(f"preprocessing {len(trajs)} trajectories from {raw_root} -> {pairs_root}", log_file)
    for traj_dir in trajs:
        log_print(f"  preprocess: {traj_dir.name}", log_file)
        process_one(
            traj_dir,
            pairs_root / traj_dir.name,
            crop_size=crop_size,
            pad=pad,
            overwrite=overwrite,
        )


def patch_eval_trajectory(model, traj, pairs_root, device, batch_size, num_workers, crop_size):
    dataset = VIOPairsDataset(pairs_root, trajectories=[traj])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_loss = 0.0
    total_trans = 0.0
    total_scale = 0.0
    total_yaw = 0.0
    total_acc = 0.0
    n = 0
    trans_per_sample = []
    scale_per_sample = []
    yaw_per_sample = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            patch_t = batch["patch_t"].to(device, non_blocking=True)
            patch_t1 = batch["patch_t1"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            pred = model(patch_t, patch_t1)

            loss = composite_loss(pred, label)
            et = translation_error_px(pred, label)
            es = scale_error_px(pred, label, patch_size=crop_size)
            ey = yaw_error_rad(pred, label)
            ac = accuracy_percent(pred, label, patch_size=crop_size)

            bs = patch_t.shape[0]
            total_loss += float(loss.detach()) * bs
            total_trans += float(et.detach()) * bs
            total_scale += float(es.detach()) * bs
            total_yaw += float(ey.detach()) * bs
            total_acc += float(ac.detach()) * bs
            n += bs

            per_trans = torch.linalg.norm(pred[:, 2:] - label[:, 2:], dim=-1)
            per_scale = (pred[:, 0] - label[:, 0]).abs() * (crop_size * 0.5)
            per_yaw = torch.atan2(torch.sin(pred[:, 1] - label[:, 1]), torch.cos(pred[:, 1] - label[:, 1])).abs()
            trans_per_sample.extend(per_trans.detach().cpu().tolist())
            scale_per_sample.extend(per_scale.detach().cpu().tolist())
            yaw_per_sample.extend(per_yaw.detach().cpu().tolist())

    if n == 0:
        return None

    return {
        "n_pairs": n,
        "loss_mean": total_loss / n,
        "e_trans_mean_px": total_trans / n,
        "e_scale_mean_px": total_scale / n,
        "e_yaw_mean_rad": total_yaw / n,
        "acc_pct": total_acc / n,
        "e_trans_median_px": float(statistics.median(trans_per_sample)) if trans_per_sample else 0.0,
        "e_trans_p95_px": float(np.percentile(trans_per_sample, 95)) if trans_per_sample else 0.0,
        "e_trans_max_px": float(max(trans_per_sample)) if trans_per_sample else 0.0,
        "e_scale_median_px": float(statistics.median(scale_per_sample)) if scale_per_sample else 0.0,
        "e_scale_p95_px": float(np.percentile(scale_per_sample, 95)) if scale_per_sample else 0.0,
        "e_scale_max_px": float(max(scale_per_sample)) if scale_per_sample else 0.0,
        "e_yaw_median_rad": float(statistics.median(yaw_per_sample)) if yaw_per_sample else 0.0,
        "e_yaw_p95_rad": float(np.percentile(yaw_per_sample, 95)) if yaw_per_sample else 0.0,
        "e_yaw_max_rad": float(max(yaw_per_sample)) if yaw_per_sample else 0.0,
    }


def aggregate(per_traj, key):
    vals = [m[key] for m in per_traj.values() if m is not None and key in m]
    if not vals:
        return None
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
    }


def main():
    args = parse_args()
    if args.run_name:
        args.results_dir = str(Path(args.results_dir) / args.run_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root).resolve()
    if args.pairs_root is None:
        pairs_root = data_root.parent / f"{data_root.name}_preprocessed"
    else:
        pairs_root = Path(args.pairs_root).resolve()

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "test.log"
    log_file = open(log_path, "w")

    log_print("=" * 70, log_file)
    log_print("PRGFlow VIO test", log_file)
    log_print(f"  data_root:   {data_root}", log_file)
    log_print(f"  pairs_root:  {pairs_root}", log_file)
    log_print(f"  checkpoint:  {args.checkpoint}", log_file)
    log_print(f"  results_dir: {results_dir}", log_file)
    log_print(f"  device:      {device}", log_file)
    log_print(f"  crop_size:   {args.crop_size}  pad: {args.pad}  beta: {args.beta}", log_file)
    log_print("=" * 70, log_file)

    if not data_root.exists():
        log_print(f"ERROR: data_root {data_root} does not exist", log_file)
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        log_print(f"ERROR: checkpoint {args.checkpoint} does not exist", log_file)
        sys.exit(1)

    t0 = time.time()
    preprocess_test(
        data_root, pairs_root,
        crop_size=args.crop_size, pad=args.pad,
        overwrite=args.overwrite_pairs, log_file=log_file,
    )
    log_print(f"preprocessing took {time.time() - t0:.1f}s", log_file)

    log_print(f"loading checkpoint {args.checkpoint}", log_file)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = PRGFlow(
        base_channels=ckpt_args.get("base_channels", 32),
        fire_blocks=ckpt_args.get("fire_blocks", 4),
        dropout=ckpt_args.get("dropout", 0.7),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log_print(f"  params:   {model.num_trainable_parameters():,}", log_file)
    log_print(f"  size_mb:  {model.size_mb():.2f}", log_file)
    if "val_metrics" in ckpt:
        log_print(f"  ckpt val_metrics: {ckpt['val_metrics']}", log_file)
    if "splits" in ckpt:
        log_print(f"  ckpt splits.train: {ckpt['splits'].get('train')}", log_file)
        log_print(f"  ckpt splits.val:   {ckpt['splits'].get('val')}", log_file)
        log_print(f"  ckpt splits.test:  {ckpt['splits'].get('test')}", log_file)

    trajectories = sorted(p.name for p in pairs_root.iterdir()
                          if p.is_dir() and (p / "pairs").exists())
    log_print(f"found {len(trajectories)} preprocessed trajectories", log_file)

    patch_per_traj = {}
    if not args.skip_patch_eval:
        log_print("", log_file)
        log_print("--- patch-level evaluation ---", log_file)
        header = (
            f"{'sequence':<22} {'n':>6} {'loss':>10} {'eT_mean':>10} {'eT_med':>10} "
            f"{'eT_p95':>10} {'eS_mean':>10} {'eY_mean':>10} {'acc%':>8}"
        )
        log_print(header, log_file)
        log_print("-" * len(header), log_file)
        for traj in trajectories:
            try:
                m = patch_eval_trajectory(
                    model, traj, pairs_root, device,
                    args.batch_size, args.num_workers, args.crop_size,
                )
            except Exception as e:
                log_print(f"  {traj}: patch eval failed: {e}", log_file)
                m = None
            patch_per_traj[traj] = m
            if m is None:
                log_print(f"{traj:<22} {'-':>6}", log_file)
                continue
            log_print(
                f"{traj:<22} {m['n_pairs']:>6d} {m['loss_mean']:>10.4f} "
                f"{m['e_trans_mean_px']:>10.3f} {m['e_trans_median_px']:>10.3f} "
                f"{m['e_trans_p95_px']:>10.3f} {m['e_scale_mean_px']:>10.3f} "
                f"{m['e_yaw_mean_rad']:>10.4f} {m['acc_pct']:>8.2f}",
                log_file,
            )

        log_print("-" * len(header), log_file)
        for key in ["loss_mean", "e_trans_mean_px", "e_scale_mean_px", "e_yaw_mean_rad", "acc_pct"]:
            agg = aggregate(patch_per_traj, key)
            if agg is not None:
                log_print(
                    f"  {key:<22} mean={agg['mean']:.4f} median={agg['median']:.4f} "
                    f"min={agg['min']:.4f} max={agg['max']:.4f} std={agg['std']:.4f}",
                    log_file,
                )

    traj_per_seq = {}
    if not args.skip_trajectory_eval:
        log_print("", log_file)
        log_print("--- trajectory-level evaluation (ATE/RTE/rot) ---", log_file)
        header = f"{'sequence':<22} {'ATE [m]':>12} {'RTE [m]':>12} {'rot [deg]':>14}"
        log_print(header, log_file)
        log_print("-" * len(header), log_file)
        plot_dir = results_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for traj in trajectories:
            seq_dir = data_root / traj
            if not seq_dir.exists():
                log_print(f"  {traj}: raw seq dir missing, skipping trajectory eval", log_file)
                continue
            try:
                result = run_sequence(
                    seq_dir, model, device,
                    beta=args.beta, crop_size=args.crop_size,
                )
            except Exception as e:
                log_print(f"  {traj}: trajectory eval failed: {e}", log_file)
                continue
            traj_per_seq[traj] = {
                "ate": float(result["ate"]),
                "rte": float(result["rte"]),
                "rot_deg": float(result["rot_deg"]),
            }
            log_print(
                f"{traj:<22} {result['ate']:>12.4f} {result['rte']:>12.4f} "
                f"{result['rot_deg']:>14.4f}",
                log_file,
            )
            try:
                from prgflow_mod.evaluate import plot_sequence
                plot_sequence(traj, result, plot_dir)
            except Exception as e:
                log_print(f"  {traj}: plot failed: {e}", log_file)

        log_print("-" * len(header), log_file)
        for key in ["ate", "rte", "rot_deg"]:
            agg = aggregate(traj_per_seq, key)
            if agg is not None:
                log_print(
                    f"  {key:<10} mean={agg['mean']:.4f} median={agg['median']:.4f} "
                    f"min={agg['min']:.4f} max={agg['max']:.4f} std={agg['std']:.4f}",
                    log_file,
                )

    summary = {
        "data_root": str(data_root),
        "pairs_root": str(pairs_root),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "crop_size": args.crop_size,
        "pad": args.pad,
        "beta": args.beta,
        "trajectories": trajectories,
        "patch_per_trajectory": patch_per_traj,
        "trajectory_per_sequence": traj_per_seq,
        "patch_aggregate": {
            k: aggregate(patch_per_traj, k)
            for k in ["loss_mean", "e_trans_mean_px", "e_scale_mean_px", "e_yaw_mean_rad", "acc_pct"]
        } if patch_per_traj else None,
        "trajectory_aggregate": {
            k: aggregate(traj_per_seq, k) for k in ["ate", "rte", "rot_deg"]
        } if traj_per_seq else None,
        "elapsed_sec": time.time() - t0,
    }

    json_path = results_dir / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_print("", log_file)
    log_print(f"wrote {json_path}", log_file)

    csv_path = results_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence", "n_pairs", "loss_mean",
            "e_trans_mean_px", "e_trans_median_px", "e_trans_p95_px", "e_trans_max_px",
            "e_scale_mean_px", "e_scale_median_px", "e_scale_p95_px", "e_scale_max_px",
            "e_yaw_mean_rad", "e_yaw_median_rad", "e_yaw_p95_rad", "e_yaw_max_rad",
            "acc_pct", "ate_m", "rte_m", "rot_deg",
        ])
        for traj in trajectories:
            pm = patch_per_traj.get(traj) or {}
            tm = traj_per_seq.get(traj) or {}
            writer.writerow([
                traj,
                pm.get("n_pairs", ""),
                pm.get("loss_mean", ""),
                pm.get("e_trans_mean_px", ""),
                pm.get("e_trans_median_px", ""),
                pm.get("e_trans_p95_px", ""),
                pm.get("e_trans_max_px", ""),
                pm.get("e_scale_mean_px", ""),
                pm.get("e_scale_median_px", ""),
                pm.get("e_scale_p95_px", ""),
                pm.get("e_scale_max_px", ""),
                pm.get("e_yaw_mean_rad", ""),
                pm.get("e_yaw_median_rad", ""),
                pm.get("e_yaw_p95_rad", ""),
                pm.get("e_yaw_max_rad", ""),
                pm.get("acc_pct", ""),
                tm.get("ate", ""),
                tm.get("rte", ""),
                tm.get("rot_deg", ""),
            ])
    log_print(f"wrote {csv_path}", log_file)
    log_print(f"total elapsed: {time.time() - t0:.1f}s", log_file)
    log_file.close()


if __name__ == "__main__":
    main()
