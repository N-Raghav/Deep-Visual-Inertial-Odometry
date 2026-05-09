import torch


def prgflow_loss(pred, target):
    return torch.linalg.norm(pred - target, dim=-1).mean()


def translation_error_px(pred, target):
    diff = pred[:, 1:] - target[:, 1:]
    return torch.linalg.norm(diff, dim=-1).mean()


def scale_error_px(pred, target, patch_size=128):
    return (pred[:, 0] - target[:, 0]).abs().mean() * (patch_size * 0.5)


def accuracy_percent(pred, target, patch_size=128, trans_thresh=2.0, scale_thresh=4.0):
    trans = torch.linalg.norm(pred[:, 1:] - target[:, 1:], dim=-1)
    scale = (pred[:, 0] - target[:, 0]).abs() * (patch_size * 0.5)
    good = (trans <= trans_thresh) & (scale <= scale_thresh)
    return good.float().mean() * 100.0


def metric_dict(pred, target, patch_size=128):
    return {
        "l2": float(prgflow_loss(pred, target).detach()),
        "e_trans": float(translation_error_px(pred, target).detach()),
        "e_scale": float(scale_error_px(pred, target, patch_size=patch_size).detach()),
        "acc": float(accuracy_percent(pred, target, patch_size=patch_size).detach()),
    }
