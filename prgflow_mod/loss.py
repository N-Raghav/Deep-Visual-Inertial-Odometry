import torch
import torch.nn.functional as F

_EPS = 1e-7
_LAMBDA_ROT = 500.0
_LAMBDA_TRANS = 1.0


def yaw_to_rotation_matrix(yaw):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    return torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def rotation_loss_geodesic(pred, target, eps=_EPS):
    # pred, target: (N, 4) as [s, yaw, tx, ty].
    R_pred = yaw_to_rotation_matrix(pred[:, 1])
    R_gt = yaw_to_rotation_matrix(target[:, 1])
    R_res = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_res.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_angle).mean()


def translation_loss_mse(pred, target):
    # MSE on [s, tx, ty] (indices 0, 2, 3)
    t_pred = pred[:, [0, 2, 3]]
    t_gt = target[:, [0, 2, 3]]
    return F.mse_loss(t_pred, t_gt)


def composite_loss(pred, target, lambda_rot=_LAMBDA_ROT, lambda_trans=_LAMBDA_TRANS):
    return lambda_rot * rotation_loss_geodesic(pred, target) + lambda_trans * translation_loss_mse(pred, target)


def prgflow_loss(pred, target):
    return torch.linalg.norm(pred - target, dim=-1).mean()


def translation_error_px(pred, target):
    diff = pred[:, 2:] - target[:, 2:]
    return torch.linalg.norm(diff, dim=-1).mean()


def scale_error_px(pred, target, patch_size=128):
    return (pred[:, 0] - target[:, 0]).abs().mean() * (patch_size * 0.5)


def yaw_error_rad(pred, target):
    diff = pred[:, 1] - target[:, 1]
    return torch.atan2(torch.sin(diff), torch.cos(diff)).abs().mean()


def accuracy_percent(pred, target, patch_size=128, trans_thresh=2.0, scale_thresh=4.0, yaw_thresh=0.1):
    trans = torch.linalg.norm(pred[:, 2:] - target[:, 2:], dim=-1)
    scale = (pred[:, 0] - target[:, 0]).abs() * (patch_size * 0.5)
    diff = pred[:, 1] - target[:, 1]
    yaw = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
    good = (trans <= trans_thresh) & (scale <= scale_thresh) & (yaw <= yaw_thresh)
    return good.float().mean() * 100.0


def metric_dict(pred, target, patch_size=128):
    return {
        "loss": float(composite_loss(pred, target).detach()),
        "l_rot": float(rotation_loss_geodesic(pred, target).detach()),
        "l_trans": float(translation_loss_mse(pred, target).detach()),
        "l2": float(prgflow_loss(pred, target).detach()),
        "e_trans": float(translation_error_px(pred, target).detach()),
        "e_scale": float(scale_error_px(pred, target, patch_size=patch_size).detach()),
        "e_yaw": float(yaw_error_rad(pred, target).detach()),
        "acc": float(accuracy_percent(pred, target, patch_size=patch_size).detach()),
    }
