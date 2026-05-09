import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from prgflow.attitude import Madgwick
from prgflow.warp import warp_image


class VIOFusion:
    def __init__(self, model, K, crop_size=128, beta=0.08, device=None):
        self.model = model
        self.K = np.asarray(K, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.crop_size = int(crop_size)
        self.focal = 0.5 * (self.K[0, 0] + self.K[1, 1])
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.madgwick = Madgwick(beta=beta)
        self.reset()

    def reset(self, q0=None):
        self.madgwick.reset(q0)
        self.position = np.zeros(3, dtype=np.float64)
        self.prev_frame = None
        self.prev_patch = None
        self.prev_R = None

    def _to_tensor(self, frame):
        if torch.is_tensor(frame):
            x = frame.detach().float().cpu()
            if x.ndim == 2:
                x = x.unsqueeze(0)
            elif x.ndim == 3 and x.shape[0] not in (1, 3):
                x = x.permute(2, 0, 1)
        else:
            x = np.asarray(frame)
            if x.ndim == 2:
                x = torch.from_numpy(x).float().unsqueeze(0)
            else:
                x = torch.from_numpy(x).float().permute(2, 0, 1)
        if x.max() > 1.0:
            x = x / 255.0
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x.unsqueeze(0).to(self.device)

    def _make_patch(self, frame):
        gray = TF.rgb_to_grayscale(frame)
        H = gray.shape[-2]
        W = gray.shape[-1]
        top = max(0, (H - self.crop_size) // 2)
        left = max(0, (W - self.crop_size) // 2)
        patch = gray[:, :, top : top + self.crop_size, left : left + self.crop_size]
        if patch.shape[-1] != self.crop_size or patch.shape[-2] != self.crop_size:
            patch = F.interpolate(patch, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        return patch

    def step(self, frame, imu_samples_since_last_frame, altitude, dt):
        for gyro, accel, dt_imu in imu_samples_since_last_frame:
            self.madgwick.step(gyro, accel, dt_imu)
        R_now = self.madgwick.R()

        frame_tensor = self._to_tensor(frame)
        frame_patch = self._make_patch(frame_tensor)

        if self.prev_frame is None:
            self.prev_frame = frame_tensor
            self.prev_patch = frame_patch
            self.prev_R = R_now
            return None

        H_R = self.K @ (self.prev_R.T @ R_now) @ self.K_inv
        H_R = torch.from_numpy(H_R.astype(np.float32)).unsqueeze(0).to(self.device)
        frame_rotcomp = warp_image(frame_tensor, H_R, out_size=self.prev_frame.shape[-2:])
        frame_rotcomp_patch = self._make_patch(frame_rotcomp)

        with torch.no_grad():
            pred = self.model(self.prev_patch, frame_rotcomp_patch)[0].detach().cpu().numpy()
        s, tx, ty = float(pred[0]), float(pred[1]), float(pred[2])

        altitude = float(abs(altitude))
        dt = float(dt)
        if dt <= 0.0:
            dt = 1e-6

        vx_body = tx * altitude / (self.focal * dt)
        vy_body = ty * altitude / (self.focal * dt)
        vz_body = -s * altitude / dt

        v_world = self.prev_R @ np.array([vx_body, vy_body, vz_body], dtype=np.float64)
        self.position = self.position + v_world * dt

        self.prev_frame = frame_tensor
        self.prev_patch = frame_patch
        self.prev_R = R_now
        return self.position.copy(), R_now.copy(), np.array([s, tx, ty], dtype=np.float64)
