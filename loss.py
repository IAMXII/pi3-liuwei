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
def compute_losses(res, gt, device):
    losses = {}
    lpips_loss_fn = LPIPS(net='vgg').to(device)
    if 'gaussians' in res:
        height = gt['img'].shape[-2]
        width = gt['img'].shape[-1]
        B, C, _ = res['camera_poses'].shape
        N = res['gaussians']['center'].shape[1]
        viewmats = res['camera_poses']
        Ks = torch.tensor([[[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]]],
                          device=device).repeat(B, C, 1, 1)
        means = res['gaussians']['center']
        quats = res['gaussians']['quat']
        scales = res['gaussians']['scale']
        colors = res['gaussians']['color']
        opacities = res['gaussians']['opac']
        render_mode = "RGB+ED"

        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            render_mode=render_mode
        )

        render_colors = render_colors[..., :3]
        render_depths = render_colors[..., 3:]
        losses['rgb_loss'] = F.l1_loss(render_colors, gt['img'])
        render_norm = render_colors.permute(0, 3, 1, 2) * 2 - 1  # [B, 3, H, W]
        gt_norm = render_colors.permute(0, 3, 1, 2) * 2 - 1  # [B, 3, H, W]

        # 2. 计算 LPIPS（返回的是每个样本的平均值）
        lpips_val = lpips_loss_fn(render_norm, gt_norm)
        losses['lpips_loss'] = LAMBDA_LPIPS * lpips_val.mean()
        mask = gt['depth'] > 0
        losses['geometry_loss'] = geometry_loss(render_depths, gt['depthmap'])

    total = sum(v for k, v in losses.items())
    losses['total_loss'] = total
    return losses
