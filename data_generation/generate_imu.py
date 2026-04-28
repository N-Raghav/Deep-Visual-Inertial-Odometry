"""
Generate Blackbird-style trajectories with 1000 Hz IMU data.
Usage: python generate_imu.py --out ../data/run_001 --traj figure8 --duration 20

World: ENU. Body: FLU (yaw aligned with velocity). Camera looks straight down.
Does NOT call Blender -- run blender_render.py after this.
"""

import argparse
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation

from gnss_ins_sim.pathgen.pathgen import acc_gen, gyro_gen
from gnss_ins_sim.sim.imu_model import IMU

GRAVITY = 9.81


def traj_figure8(t, A=4.0, B=2.5, f=0.08, h=4.0):
    w = 2 * np.pi * f
    x = A * np.sin(w * t)
    y = B * np.sin(2 * w * t)
    z = np.full_like(t, h)
    return np.stack([x, y, z], axis=1)


def traj_oval(t, A=4.0, B=2.5, f=0.1, h=4.0):
    w = 2 * np.pi * f
    x = A * np.cos(w * t)
    y = B * np.sin(w * t)
    z = np.full_like(t, h)
    return np.stack([x, y, z], axis=1)


def traj_clover(t, A=3.5, k=3, f=0.07, h=4.0):
    w = 2 * np.pi * f
    phi = w * t
    r = A * np.cos(k * phi)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.full_like(t, h)
    return np.stack([x, y, z], axis=1)


def traj_lemniscate(t, A=4.0, f=0.08, h=4.0):
    w = 2 * np.pi * f
    s = np.sin(w * t)
    c = np.cos(w * t)
    denom = 1 + s * s
    x = A * c / denom
    y = A * s * c / denom
    z = np.full_like(t, h)
    return np.stack([x, y, z], axis=1)


def traj_3d_figure8(t, A=4.0, B=2.5, C=0.6, f=0.08, h=4.0):
    w = 2 * np.pi * f
    x = A * np.sin(w * t)
    y = B * np.sin(2 * w * t)
    z = h + C * np.sin(w * t)
    return np.stack([x, y, z], axis=1)


TRAJECTORIES = {
    "figure8": traj_figure8,
    "oval": traj_oval,
    "clover": traj_clover,
    "lemniscate": traj_lemniscate,
    "figure8_3d": traj_3d_figure8,
}


def random_traj_params(name, rng):
    h = rng.uniform(3.0, 6.5)
    if name == "figure8":
        return dict(A=rng.uniform(3.0, 5.0), B=rng.uniform(1.5, 3.0), f=rng.uniform(0.06, 0.12), h=h)
    if name == "oval":
        return dict(A=rng.uniform(3.0, 5.0), B=rng.uniform(2.0, 3.5), f=rng.uniform(0.07, 0.13), h=h)
    if name == "clover":
        return dict(A=rng.uniform(2.5, 4.0), k=rng.choice([3, 4, 5]), f=rng.uniform(0.04, 0.08), h=h)
    if name == "lemniscate":
        return dict(A=rng.uniform(3.0, 5.0), f=rng.uniform(0.06, 0.12), h=h)
    if name == "figure8_3d":
        return dict(A=rng.uniform(3.0, 5.0), B=rng.uniform(1.5, 3.0), C=rng.uniform(0.3, 1.0), f=rng.uniform(0.06, 0.12), h=h)
    raise ValueError(name)


def compute_ideal_imu(p, fs):
    dt = 1.0 / fs
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)

    yaw = np.arctan2(v[:, 1], v[:, 0])
    # forward-fill yaw at near-zero speed (trajectory reversals)
    bad = np.hypot(v[:, 0], v[:, 1]) < 1e-4
    ffill = np.maximum.accumulate(np.where(~bad, np.arange(len(bad)), 0))
    yaw = np.unwrap(yaw[ffill])

    # quadrotor diff-flatness: body_z aligned with required thrust = a - g_w
    g_w = np.array([0.0, 0.0, -GRAVITY])
    thrust = a - g_w
    body_z = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)
    fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    body_x = fwd - np.einsum('ni,ni->n', fwd, body_z)[:, None] * body_z
    body_x /= np.linalg.norm(body_x, axis=1, keepdims=True)
    body_y = np.cross(body_z, body_x)
    R_wb = Rotation.from_matrix(np.stack([body_x, body_y, body_z], axis=2))

    # angular velocity from attitude derivative: omega_b = log(R_t^-1 * R_t+dt) / dt
    R_rel = R_wb[:-1].inv() * R_wb[1:]
    gyro_b = np.zeros_like(p)
    gyro_b[:-1] = R_rel.as_rotvec() / dt
    gyro_b[-1] = gyro_b[-2]

    f_b = R_wb.inv().apply(a - g_w)
    return gyro_b, f_b, R_wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", default="figure8", choices=list(TRAJECTORIES.keys()))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--fs", type=int, default=1000)
    ap.add_argument("--camera-fs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accuracy", default="mid-accuracy", choices=["low-accuracy", "mid-accuracy", "high-accuracy"])
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--img-h", type=int, default=480)
    ap.add_argument("--fx", type=float, default=525.0)
    ap.add_argument("--fy", type=float, default=525.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    n = int(args.duration * args.fs)
    t = np.arange(n) / args.fs
    params = random_traj_params(args.traj, rng)
    p = TRAJECTORIES[args.traj](t, **params)

    speed = np.linalg.norm(np.gradient(p, 1.0 / args.fs, axis=0), axis=1)
    print(f"[traj] {args.traj}  params={params}  "
          f"max={speed.max():.2f} m/s  mean={speed.mean():.2f} m/s")

    gyro_clean, accel_clean, R_wb = compute_ideal_imu(p, args.fs)

    imu = IMU(accuracy=args.accuracy, axis=6, gps=False)
    accel_noisy = acc_gen(args.fs, accel_clean, imu.accel_err)
    gyro_noisy  = gyro_gen(args.fs, gyro_clean,  imu.gyro_err)

    # camera rigidly mounted on body (R_bc = I), like Camera_D in Blackbird
    quat_wxyz = R_wb.as_quat()[:, [3, 0, 1, 2]]
    body_quat = quat_wxyz
    cam_pos, cam_quat = p.copy(), quat_wxyz

    def save(name, arr, header):
        np.savetxt(os.path.join(args.out, name), arr,
                   delimiter=",", header=header, comments="")

    save("poses_body.csv", np.c_[t, p, body_quat],         "t,x,y,z,qw,qx,qy,qz")
    save("poses_cam.csv",  np.c_[t, cam_pos, cam_quat],    "t,x,y,z,qw,qx,qy,qz")
    save("imu.csv",        np.c_[t, gyro_noisy, accel_noisy],  "t,gx,gy,gz,ax,ay,az")
    save("imu_clean.csv",  np.c_[t, gyro_clean, accel_clean],  "t,gx,gy,gz,ax,ay,az")

    stride = args.fs // args.camera_fs
    cx, cy = (args.img_w - 1) / 2.0, (args.img_h - 1) / 2.0

    meta = {
        "trajectory": args.traj,
        "trajectory_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
        "duration_s": args.duration,
        "fs_imu": args.fs,
        "fs_camera": args.camera_fs,
        "camera_stride": stride,
        "n_imu_samples": int(n),
        "n_camera_frames": int(n // stride),
        "image_w": args.img_w,
        "image_h": args.img_h,
        "K": [[args.fx, 0, cx], [0, args.fy, cy], [0, 0, 1]],
        "gravity_world": [0.0, 0.0, -GRAVITY],
        "camera_height_nominal": float(params["h"]),
        "imu_accuracy": args.accuracy,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {n} IMU samples, {n // stride} camera frames -> {args.out}")


if __name__ == "__main__":
    main()
