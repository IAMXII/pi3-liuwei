import os
import math
import random
import argparse
from typing import Tuple
from gsplat import rasterization
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from lpips import LPIPS

# ---------- 超参（可改为 argparse） ----------
STAGE_EPOCHS = 80
ITERS_PER_EPOCH = 800
STAGE1_RES = (224, 224)
STAGE2_PIX_MIN = 100_000
STAGE2_PIX_MAX = 255_000
STAGE2_AR_MIN = 0.5
STAGE2_AR_MAX = 2.0
BATCH_MIN = 2
BATCH_MAX = 24
SAMPLES_PER_GPU_STAGE1 = 64
SAMPLES_PER_GPU_STAGE2 = 48

# Loss weights
LAMBDA_NORMAL = 0.5
LAMBDA_LPIPS = 0.05
LAMBDA_CONF = 0.05
LAMBDA_CAM = 0.1
LAMBDA_TRANS = 100.0
LAMBDA_DEPTH = 1.0
LAMBDA_RGB = 1


def scale_invariant_depth_loss_batch(pred_depth, gt_depth, mask=None, eps=1e-6):
    """
    批量版本：每个batch样本都有自己的s*，在GPU上计算。

    Args:
        pred_depth: (B, H, W)
        gt_depth:   (B, H, W)
        mask:       (B, H, W) 有效像素掩码
    Returns:
        loss: 平均的scale-invariant L1损失
        s_star: (B,) 每个样本的最优尺度
    """
    B = pred_depth.shape[0]
    device = pred_depth.device
    if mask is None:
        mask = gt_depth > 0

    # 避免除0
    gt_safe = gt_depth.clamp(min=eps)

    # numerator 和 denominator 分别为 batch 内聚合结果
    num = torch.sum((pred_depth * gt_depth / (gt_safe ** 2)) * mask, dim=(1, 2))
    den = torch.sum((pred_depth ** 2 / (gt_safe ** 2)) * mask, dim=(1, 2))
    s_star = num / (den + eps)  # [B]

    # 需要对每个样本广播 s*
    s_star_broadcast = s_star.view(B, 1, 1)
    loss = torch.abs(s_star_broadcast * pred_depth - gt_depth) / gt_safe
    loss = (loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + eps)

    return loss.mean(), s_star


def normal_from_depth_batch(depth, mask=None):
    """
    从深度图估计法向（支持batch，GPU加速）
    depth: (B, H, W)
    mask:  (B, H, W) 或 None
    返回法向: (B, 3, H, W)
    """
    B, H, W = depth.shape
    device = depth.device

    # 计算梯度（中心差分）
    dzdx = F.pad(depth[:, :, :, 2:] - depth[:, :, :, :-2], (1, 1), mode="replicate") / 2
    dzdy = F.pad(depth[:, :, 2:, :] - depth[:, :, :-2, :], (0, 0, 1, 1), mode="replicate") / 2

    # 构造法向向量 (-dzdx, -dzdy, 1)
    n = torch.stack([-dzdx, -dzdy, torch.ones_like(depth)], dim=1)  # (B, 3, H, W)
    n = F.normalize(n, dim=1, eps=1e-6)

    # 对mask外区域归零（避免干扰）
    if mask is not None:
        n = n * mask.unsqueeze(1)
    return n


def normal_loss_batch(pred_depth, gt_depth, mask=None):
    """
    法线损失（batch + GPU 版本）
    使用角度误差 arccos(n_pred ⋅ n_gt)
    """
    n_pred = normal_from_depth_batch(pred_depth, mask)
    n_gt = normal_from_depth_batch(gt_depth, mask)

    # 余弦相似度：n_pred ⋅ n_gt
    cos_sim = torch.clamp((n_pred * n_gt).sum(1), -1, 1)  # (B, H, W)
    angle = torch.acos(cos_sim)

    if mask is not None:
        angle = angle * mask

    # 对batch取平均
    loss = (angle.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)).mean()
    return loss


def geometry_loss(pred_depth, gt_depth, mask=None, lambda_n=LAMBDA_NORMAL):
    loss_d, s_star = scale_invariant_depth_loss_batch(pred_depth, gt_depth, mask)
    loss_n = normal_loss_batch(pred_depth, gt_depth, mask)
    return loss_d + lambda_n * loss_n, {'L_depth': loss_d, 'L_normal': loss_n, 's*': s_star}


def compute_loss_sky(res, gt):
    """
    计算天空预测的 BCE 损失，带掩膜。
    Args:
        res: dict，包含预测结果，如 res['sky']，形状为 (B, N, H, W, 1)
        gt: dict，包含 gt['depth'] 或 gt['sky_mask']
    """
    if 'sky' not in res:
        return 0.0

    pred_sky = res['sky']  # (B, N, H, W, 1)
    gt_depth = gt['depth']  # (B, N, H, W)

    # === 1. 生成天空真值 mask ===
    # 这里假设深度 <= 0 为天空
    gt_sky = (gt_depth <= 0).float().unsqueeze(-1)  # (B, N, H, W, 1)

    # === 2. 有效掩膜（只在深度有效区域计算）===
    valid_mask = (gt_depth >= 0).float().unsqueeze(-1)  # (B, N, H, W, 1)

    # === 3. BCE 损失 ===
    bce = F.binary_cross_entropy_with_logits(
        pred_sky, gt_sky, reduction='none'
    )

    # === 4. 掩膜加权平均 ===
    bce = bce * valid_mask
    loss = bce.sum() / (valid_mask.sum() + 1e-6)

    return loss


# ---------- 损失函数 ----------
# def compute_losses(res, gt, device):
#     losses = {}
#     lpips_loss_fn = LPIPS(net='vgg').to(device)
#     if 'gaussians' in res:
#         height = gt['image'].shape[-2]
#         width = gt['image'].shape[-1]
#         B, C, _ = res['camera_poses'].shape
#         N = res['gaussians']['center'].shape[1]
#         viewmats = res['camera_poses']
#         Ks = torch.tensor([[[1., 0., 0.],
#                             [0., 1., 0.],
#                             [0., 0., 1.]]],
#                           device=device).repeat(B, C, 1, 1)
#         means = res['gaussians']['center']
#         quats = res['gaussians']['quat']
#         scales = res['gaussians']['scale']
#         colors = res['gaussians']['color']
#         opacities = res['gaussians']['opac']
#         render_mode = "RGB+ED"
#
#         render_colors, render_alphas, meta = rasterization(
#             means=means,
#             quats=quats,
#             scales=scales,
#             opacities=opacities,
#             colors=colors,
#             viewmats=viewmats,
#             Ks=Ks,
#             width=width,
#             height=height,
#             render_mode=render_mode
#         )
#
#         render_colors = render_colors[..., :3]
#         render_depths = render_colors[..., 3:]
#         losses['rgb_loss'] = F.l1_loss(render_colors, gt['image'])
#         render_norm = render_colors.permute(0, 3, 1, 2) * 2 - 1  # [B, 3, H, W]
#         gt_norm = render_colors.permute(0, 3, 1, 2) * 2 - 1  # [B, 3, H, W]
#
#         # 2. 计算 LPIPS（返回的是每个样本的平均值）
#         lpips_val = lpips_loss_fn(render_norm, gt_norm)
#         losses['lpips_loss'] = LAMBDA_LPIPS * lpips_val.mean()
#         mask = gt['depth'] > 0
#         losses['geometry_loss'] = geometry_loss(render_depths, gt['depth'])
#
#     total = sum(v for k, v in losses.items())
#     losses['total_loss'] = total
#     return losses

#################### liuwei
def compute_losses(
    res,
    gt,
    device,
    pose_only: bool = False,
    pose_weight: float = 1.0,
):
    losses = {}

    # ==========================================================
    #  Camera pose decomposition
    # ==========================================================
    viewmats = res['camera_poses']          # (B, N, 4, 4)
    B, N, _, _ = viewmats.shape

    R_pred = viewmats[..., :3, :3]
    t_pred = viewmats[..., :3, 3]

    t_pred_norm = torch.norm(t_pred, dim=-1, keepdim=True) + 1e-6
    t_pred_dir  = t_pred / t_pred_norm

    # --------------------------
    #  GT pose
    # --------------------------
    gt_pose = gt['camera_poses']
    R_gt = gt_pose[..., :3, :3]
    t_gt = gt_pose[..., :3, 3]

    t_gt_norm = torch.norm(t_gt, dim=-1, keepdim=True) + 1e-6
    t_gt_dir  = t_gt / t_gt_norm

    # ==========================================================
    #  SO(3) geodesic distance
    # ==========================================================
    def so3_geodesic(R1, R2):
        R = torch.matmul(R1.transpose(-1, -2), R2)
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
        cos = (trace - 1.0) / 2.0
        cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)
        return torch.acos(cos)

    # ==========================================================
    #  Pose losses (π³-style)
    # ==========================================================
    # --- rotation ---
    rot_loss = so3_geodesic(R_pred, R_gt).mean()
    losses['pose_rot_loss'] = LAMBDA_ROT * rot_loss * pose_weight

    # --- translation direction ---
    dir_loss = torch.norm(t_pred_dir - t_gt_dir, dim=-1).mean()
    losses['pose_dir_loss'] = LAMBDA_DIR * dir_loss * pose_weight

    # ----------------------------------------------------------
    #  π³-style O(N) global scale estimation
    # ----------------------------------------------------------
    # 去中心
    t_pred_c = t_pred - t_pred.mean(dim=1, keepdim=True)
    t_gt_c   = t_gt   - t_gt.mean(dim=1, keepdim=True)

    # 二阶矩（RMS radius）
    pred_var = (t_pred_c ** 2).sum(dim=-1).mean(dim=1)   # (B,)
    gt_var   = (t_gt_c   ** 2).sum(dim=-1).mean(dim=1)   # (B,)

    scale = torch.sqrt(gt_var / (pred_var + 1e-8))       # (B,)

    # --- absolute scale supervision (Huber, π³) ---
    scale_loss = F.huber_loss(
        scale,
        torch.ones_like(scale),
        delta=0.5
    )
    losses['pose_scale_loss'] = LAMBDA_SCALE * scale_loss.mean() * pose_weight

    # --- scale-aligned translation loss ---
    t_pred_scaled = t_pred * scale[:, None, None]

    trans_loss = F.smooth_l1_loss(
        t_pred_scaled,
        t_gt
    )
    losses['pose_trans_loss'] = LAMBDA_TRANS * trans_loss * pose_weight

    # ==========================================================
    #  Pose-only stage
    # ==========================================================
    if pose_only:
        losses['total_loss'] = sum(losses.values())
        return losses

    # ==========================================================
    #  Below: full Gaussian + rendering losses
    # ==========================================================
    lpips_loss_fn = LPIPS(net='vgg').to(device)

    height = gt['img'].shape[-2]
    width  = gt['img'].shape[-1]
    Ks = gt['camera_intrs']

    # ==========================================================
    #  Gaussian data
    # ==========================================================
    gauss = res["gaussians_all"]

    means     = gauss['center']
    quats     = gauss['quat']
    scales_g  = gauss['scale']
    colors    = gauss['color']
    opacities = gauss['opacity']

    s_fg  = gauss['s_fg']
    s_sky = gauss['s_sky']
    s_dyn = gauss['s_dyn']

    # ==========================================================
    #  Rendering
    # ==========================================================
    render_out, render_alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales_g,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB+ED"
    )

    render_rgb   = render_out[..., :3]
    render_depth = render_out[..., 3]

    # ==========================================================
    #  RGB + LPIPS
    # ==========================================================
    losses['rgb_loss'] = LAMBDA_RGB * F.l1_loss(render_rgb, gt['img'])

    render_norm = (
        render_rgb.permute(0, 1, 4, 2, 3)
        .reshape(-1, 3, height, width) * 2 - 1
    )
    gt_norm = (
        gt['img'].permute(0, 1, 4, 2, 3)
        .reshape(-1, 3, height, width) * 2 - 1
    )

    lpips_val = lpips_loss_fn(render_norm, gt_norm)
    losses['lpips_loss'] = LAMBDA_LPIPS * lpips_val.mean()

    # ==========================================================
    #  Geometry (depth)
    # ==========================================================
    fg_weight = torch.clamp(
        render_alpha * s_fg.mean(dim=1, keepdim=True),
        0.0, 1.0
    )

    geom_loss, _ = geometry_loss(
        render_depth.view(-1, height, width),
        gt['depthmap'].view(-1, height, width),
        mask=(fg_weight.view(-1, height, width) > 0.1)
    )
    losses['geometry_loss'] = LAMBDA_DEPTH * geom_loss

    # ==========================================================
    #  Sky + dynamic
    # ==========================================================
    scene_center = means.mean(dim=1, keepdim=True)
    scene_radius = torch.norm(
        means - scene_center, dim=-1
    ).max(dim=1, keepdim=True)[0]

    R_sky = scene_radius * 10.0
    v = means - scene_center
    dist = torch.norm(v, dim=-1, keepdim=True)
    v_dir = v / (dist + 1e-6)
    target_pos = scene_center + v_dir * R_sky

    losses['sky_loss'] = (
        (s_sky * F.relu(dist - R_sky)).mean() +
        (s_sky * F.relu(target_pos[..., 1:2])).mean()
    )

    sky_weight = torch.clamp(
        render_alpha * s_sky.mean(dim=1, keepdim=True),
        0.0, 1.0
    )

    sky_rgb = render_rgb * sky_weight.unsqueeze(-1)
    sky_rgb_mean = sky_rgb.mean(dim=1, keepdim=True)
    losses['sky_view_loss'] = ((sky_rgb - sky_rgb_mean) ** 2).mean()

    sky_depth = render_depth * sky_weight
    sky_depth_mean = sky_depth.mean(dim=1, keepdim=True)
    losses['sky_depth_loss'] = 0.1 * ((sky_depth - sky_depth_mean) ** 2).mean()

    losses['dynamic_loss'] = (s_dyn * opacities).mean()

    # ==========================================================
    #  Total
    # ==========================================================
    losses['total_loss'] = sum(losses.values())
    return losses