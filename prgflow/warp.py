import torch
import torch.nn.functional as F


def _batchify_homography(H_mat, device=None, dtype=None):
    if not torch.is_tensor(H_mat):
        H_mat = torch.tensor(H_mat, device=device, dtype=dtype)
    if H_mat.ndim == 2:
        H_mat = H_mat.unsqueeze(0)
    return H_mat


def _batchify_param(x, batch, device, dtype):
    if torch.is_tensor(x):
        out = x.to(device=device, dtype=dtype).reshape(-1)
    else:
        out = torch.tensor(x, device=device, dtype=dtype).reshape(-1)
    if out.numel() == 1 and batch != 1:
        out = out.repeat(batch)
    if out.numel() != batch:
        raise ValueError(f"expected {batch} values, got {out.numel()}")
    return out


def pseudo_similarity_matrix(s, tx, ty, H, W):
    device = None
    dtype = None
    for x in (s, tx, ty):
        if torch.is_tensor(x):
            device = x.device
            dtype = x.dtype
            break
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    sizes = []
    for x in (s, tx, ty):
        if torch.is_tensor(x):
            sizes.append(x.numel())
        else:
            sizes.append(1)
    batch = max(sizes)
    s = _batchify_param(s, batch, device, dtype)
    tx = _batchify_param(tx, batch, device, dtype)
    ty = _batchify_param(ty, batch, device, dtype)

    a = 1.0 + s
    cx = (W - 1.0) * 0.5
    cy = (H - 1.0) * 0.5

    out = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
    out[:, 0, 0] = a
    out[:, 1, 1] = a
    out[:, 2, 2] = 1.0
    out[:, 0, 2] = tx + (1.0 - a) * cx
    out[:, 1, 2] = ty + (1.0 - a) * cy
    return out


def compose(H_a, H_b):
    H_a = _batchify_homography(H_a)
    H_b = _batchify_homography(H_b, device=H_a.device, dtype=H_a.dtype)
    if H_a.shape[0] == 1 and H_b.shape[0] > 1:
        H_a = H_a.expand(H_b.shape[0], -1, -1)
    if H_b.shape[0] == 1 and H_a.shape[0] > 1:
        H_b = H_b.expand(H_a.shape[0], -1, -1)
    return H_a @ H_b


def decompose_to_pseudosim(H_mat, H=None, W=None):
    H_mat = _batchify_homography(H_mat)
    a = 0.5 * (H_mat[:, 0, 0] + H_mat[:, 1, 1])
    s = a - 1.0
    tx = H_mat[:, 0, 2]
    ty = H_mat[:, 1, 2]
    if H is not None and W is not None:
        cx = (W - 1.0) * 0.5
        cy = (H - 1.0) * 0.5
        tx = tx - (1.0 - a) * cx
        ty = ty - (1.0 - a) * cy
    return s, tx, ty


def warp_image(img, H_mat, out_size=None, mode="bilinear", padding_mode="zeros"):
    squeeze = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        squeeze = True
    if img.ndim != 4:
        raise ValueError(f"expected image tensor with 3 or 4 dims, got {img.shape}")

    B, C, H_in, W_in = img.shape
    H_mat = _batchify_homography(H_mat, device=img.device, dtype=img.dtype)
    if H_mat.shape[0] == 1 and B > 1:
        H_mat = H_mat.expand(B, -1, -1)
    if H_mat.shape[0] != B:
        raise ValueError(f"batch mismatch: image batch={B}, homography batch={H_mat.shape[0]}")

    if out_size is None:
        H_out, W_out = H_in, W_in
    else:
        H_out, W_out = out_size

    ys, xs = torch.meshgrid(
        torch.arange(H_out, device=img.device, dtype=img.dtype),
        torch.arange(W_out, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    base = torch.stack([xs.reshape(-1), ys.reshape(-1), ones.reshape(-1)], dim=0)
    base = base.unsqueeze(0).expand(B, -1, -1)

    H_inv = torch.linalg.inv(H_mat)
    src = H_inv @ base
    z = src[:, 2:3, :].clamp_min(1e-8)
    x = src[:, 0, :] / z[:, 0, :]
    y = src[:, 1, :] / z[:, 0, :]

    grid_x = (2.0 * x + 1.0) / W_in - 1.0
    grid_y = (2.0 * y + 1.0) / H_in - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(B, H_out, W_out, 2)

    out = F.grid_sample(
        img,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    if squeeze:
        out = out[0]
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    H_img = 64
    W_img = 80

    s0 = torch.tensor([0.1, -0.05], dtype=torch.float32)
    tx0 = torch.tensor([2.0, -3.5], dtype=torch.float32)
    ty0 = torch.tensor([-1.0, 4.0], dtype=torch.float32)
    H0 = pseudo_similarity_matrix(s0, tx0, ty0, H_img, W_img)
    s1, tx1, ty1 = decompose_to_pseudosim(H0, H_img, W_img)
    err = max(
        (s0 - s1).abs().max().item(),
        (tx0 - tx1).abs().max().item(),
        (ty0 - ty1).abs().max().item(),
    )
    print("param round-trip:", err)

    img = torch.zeros(1, 1, H_img, W_img)
    img[:, :, 16:48, 20:60] = 1.0
    Hf = H0[:1]
    Hb = torch.linalg.inv(Hf)
    rec = warp_image(warp_image(img, Hf), Hb)
    img_mid = img[:, :, 20:44, 24:56]
    rec_mid = rec[:, :, 20:44, 24:56]
    print("warp/unwarp:", (img_mid - rec_mid).abs().mean().item())
