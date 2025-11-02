import os
import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR

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
SAMPLES_PER_GPU_STAGE1 = 64  # 描述性，仅用于参考/日志（具体batch由动态策略决定）
SAMPLES_PER_GPU_STAGE2 = 48

# Loss weights
LAMBDA_NORMAL = 1.0
LAMBDA_CONF = 0.05
LAMBDA_CAM = 0.1
LAMBDA_TRANS = 100.0

ENCODER_LR = 5e-6
REST_LR = 5e-5

GRAD_CLIP_NORM = 1.0

# ---------- 辅助函数 ----------
def sample_random_resolution(total_px_min: int, total_px_max: int,
                             ar_min: float, ar_max: float) -> Tuple[int,int]:
    """随机采样一个分辨率，使得像素数在 [min,max] 且长宽比在 [ar_min,ar_max]"""
    target_px = random.randint(total_px_min, total_px_max)
    ar = random.uniform(ar_min, ar_max)
    # 解方程: h * w = target_px, w/h = ar -> h = sqrt(target_px / ar), w = ar * h
    h = int(round(math.sqrt(target_px / ar)))
    w = int(round(ar * h))
    # 保证至少 16 的倍数（常见约束），并且不为 0
    h = max(16, (h // 16) * 16)
    w = max(16, (w // 16) * 16)
    if h == 0: h = 16
    if w == 0: w = 16
    return h, w

def choose_batch_size_for_resolution(h: int, w: int,
                                     batch_min: int = BATCH_MIN,
                                     batch_max: int = BATCH_MAX,
                                     target_pixels_per_batch: int = 200_000) -> int:
    """
    基于单张图像像素数，选择一个 batch_size，使得 total pixels ≈ target_pixels_per_batch
    但受限于 [batch_min, batch_max]。
    target_pixels_per_batch 是经验值，可调。
    """
    px = h * w
    if px == 0:
        return batch_min
    est = max(batch_min, min(batch_max, target_pixels_per_batch // px))
    # 若 est==0，则至少 1
    return max(batch_min, est)

# ---------- 损失函数 ----------
def compute_losses(res, gt, device):
    """
    计算多分量 loss：
    - points: L1
    - local_points: smooth_l1 (alignment to a fixed resolution)
    - normal: MoGe 风格的 normal loss（这里简化实现）
    - conf: BCE (sigmoid)
    - camera_poses: L2（trans + rot）
    - gaussian regularization for local_points (optional)
    返回字典包含各项和 total_loss（按权重合并）
    """
    losses = {}
    # points
    if 'points' in res and 'points' in gt:
        losses['points_loss'] = F.l1_loss(res['points'], gt['points'])
    else:
        losses['points_loss'] = torch.tensor(0.0, device=device)

    # local_points (对齐到 resolution 4096 的实现示例：这里假设 res['local_points'] shape 为 (B, N, 3))
    if 'local_points' in res and 'local_points' in gt:
        # 如果需要 downsample/align 到 4096，我们可以简单地随机采样或插值；这里使用 L1 / smooth_l1
        pred_lp = res['local_points']
        gt_lp = gt['local_points']
        # 对形状不同的情况，做随机采样/截断或投影匹配。下面是保守的方式：按第 1 维采样相同数量点
        if pred_lp.shape != gt_lp.shape:
            n = min(pred_lp.shape[1], gt_lp.shape[1], 4096)
            pred_lp_s = pred_lp[:, :n, ...]
            gt_lp_s = gt_lp[:, :n, ...]
        else:
            pred_lp_s = pred_lp
            gt_lp_s = gt_lp
        losses['local_points_loss'] = F.smooth_l1_loss(pred_lp_s, gt_lp_s)
    else:
        losses['local_points_loss'] = torch.tensor(0.0, device=device)

    # normal loss (按照 MoGe 风格：在对齐分辨率上计算法线并做 cos 相似度损失)
    # 这里假设 res['local_points'] 可以转换为 depth map 或法线图；简化为对 local_points 的方向差做 loss
    if 'local_points' in res and 'local_points' in gt and \
       res['local_points'].shape == gt['local_points'].shape:
        # 计算向量差作为近似 normal loss
        pred_vectors = F.normalize(res['local_points'] + 1e-6, dim=-1)
        gt_vectors = F.normalize(gt['local_points'] + 1e-6, dim=-1)
        cos = (pred_vectors * gt_vectors).sum(dim=-1)  # (B, N)
        normal_loss = (1.0 - cos).mean()
        losses['normal_loss'] = normal_loss * LAMBDA_NORMAL
    else:
        losses['normal_loss'] = torch.tensor(0.0, device=device)

    # conf loss
    if 'conf' in res and 'conf' in gt:
        pred_conf = torch.sigmoid(res['conf'])
        losses['conf_loss'] = F.binary_cross_entropy(pred_conf, gt['conf']) * LAMBDA_CONF
    else:
        losses['conf_loss'] = torch.tensor(0.0, device=device)

    # camera poses
    if 'camera_poses' in res and 'camera_poses' in gt:
        pose_pred = res['camera_poses']
        pose_gt = gt['camera_poses']
        trans_loss = F.mse_loss(pose_pred[..., :3], pose_gt[..., :3])
        rot_loss = F.mse_loss(pose_pred[..., 3:], pose_gt[..., 3:])
        losses['pose_loss'] = (trans_loss * LAMBDA_TRANS + rot_loss * LAMBDA_CAM)
    else:
        losses['pose_loss'] = torch.tensor(0.0, device=device)

    # gaussian regularization on local_points (optional small weight)
    if 'local_points' in res:
        mean = res['local_points'].mean(dim=1, keepdim=True)
        var = res['local_points'].var(dim=1, keepdim=True)
        gauss_loss = ((mean ** 2).mean() + ((var - 1.0) ** 2).mean())
        losses['gaussian_loss'] = gauss_loss * 0.1
    else:
        losses['gaussian_loss'] = torch.tensor(0.0, device=device)

    total = sum(v for k, v in losses.items())
    losses['total_loss'] = total
    return losses

# ---------- 模型参数分组和冻结逻辑 ----------
def prepare_model_for_training(model: torch.nn.Module, pretrained_vggt_checkpoint: str = None, freeze_encoder: bool = True):
    """
    - optional: load pretrained weights into encoder & alternating attention module
    - freeze encoder parameters if freeze_encoder=True
    - return model with attributes:
        encoder_params: list of params (low lr)
        rest_params: list of params (higher lr)
        conf_params: list of params for confidence head
    """
    # 如果有 checkpoint，尝试加载匹配的键（实际场景中需要更细致的映射）
    if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
        ckpt = torch.load(pretrained_vggt_checkpoint, map_location='cpu')
        # 尝试只加载 encoder 和 alt attention 的键（这一步可能需要你根据实际 ckpt key 调整）
        try:
            model_state = model.state_dict()
            load_state = {k: v for k, v in ckpt.items() if k in model_state}
            model_state.update(load_state)
            model.load_state_dict(model_state)
            print("Loaded partial VGGT weights into model.")
        except Exception as e:
            print("Warning: failed to load vgtt checkpoint automatically:", e)

    # freeze encoder if requested
    encoder_params = []
    rest_params = []
    conf_params = []
    for name, p in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(p)
            if freeze_encoder:
                p.requires_grad = False
        else:
            rest_params.append(p)
        # 依据名字判断 conf_head（视你的实现而定）
        if 'conf' in name or 'confidence' in name:
            conf_params.append(p)

    # ensure encoder_params not empty; if empty, just fallback
    if len(encoder_params) == 0:
        print("Warning: no parameter matched 'encoder' substring. Check model.named_parameters() naming.")

    return encoder_params, rest_params, conf_params

# ---------- 训练主循环（单进程/单 GPU） ----------
def train_worker(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 构建模型并移动到设备
    model = Pi3().to(device)

    # 初始化权重并冻结 encoder（假定 args.pretrained_vggt 传入）
    encoder_params, rest_params, conf_params = prepare_model_for_training(
        model, pretrained_vggt_checkpoint=getattr(args, 'pretrained_vggt', None), freeze_encoder=True
    )

    # DDP 包装（注意若冻结了 encoder，仍然要传入全部参数）
    model = DDP(model, device_ids=[rank])

    # 参数组：encoder（低 lr），rest（高 lr）
    param_groups = []
    if encoder_params:
        param_groups.append({'params': [p for p in encoder_params if p.requires_grad], 'lr': ENCODER_LR})
    param_groups.append({'params': [p for p in rest_params if p.requires_grad], 'lr': REST_LR})

    # 优化器
    optimizer = torch.optim.AdamW(param_groups, lr=REST_LR, weight_decay=1e-2)

    # total steps for scheduler (per stage later we will re-init scheduler with stage total steps)
    # scheduler will be created per-stage since OneCycleLR needs total_steps
    scaler = torch.cuda.amp.GradScaler(enabled=getattr(args, 'amp', True))

    # Dataset: 这里假设存在一个能按传入分辨率返回 batch 的 dataset/dataloader 实现（为了示例用简单 loader）
    # 你应提供一个能在 runtime 动态 resize images 的 dataset：load_images_as_tensor(..., size=(h,w))
    # 我在示例里使用一个非常简单的 dataloader 工厂：
    def make_dataloader_for_resolution(h, w, batch_size):
        # 伪实现：你的 dataset 需支持 size 参数或在 __getitem__ 中 resize
        dataset = SimpleImageDataset(args.data_path, interval=args.interval, out_size=(h, w))
        sampler = DistributedSampler(dataset)
        dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)
        return dl, sampler

    # ========= Stage loop =========
    for stage_idx in [1, 2]:
        if stage_idx == 1:
            stage_name = "stage1"
            # stage1 固定分辨率 224x224，按文档每 GPU sample 64 个图像（仅信息），最终我们会选择 batch_size 使得每迭代步含多张图片（2-24）
            h, w = STAGE1_RES
            # choose batch size target: 使用经验 target_pixels_per_batch，使得 batch_size 约为 200k / (224*224) ≈ 3
            batch_size = choose_batch_size_for_resolution(h, w, BATCH_MIN, BATCH_MAX, target_pixels_per_batch=200_000)
            epochs = STAGE_EPOCHS
        else:
            stage_name = "stage2"
            # 每个迭代随机采样分辨率；但是 OneCycleLR 需要 total steps 固定，所以我们以 ITERS_PER_EPOCH * epochs 为 scheduler steps
            # 在每个迭代内部随机采样分辨率并构造 dataloader（真实实现中建议用一个支持按-sample-size-变分的 dataset）
            h, w = sample_random_resolution(STAGE2_PIX_MIN, STAGE2_PIX_MAX, STAGE2_AR_MIN, STAGE2_AR_MAX)
            batch_size = choose_batch_size_for_resolution(h, w, BATCH_MIN, BATCH_MAX, target_pixels_per_batch=200_000)
            epochs = STAGE_EPOCHS

        total_steps = epochs * ITERS_PER_EPOCH
        # 为 OneCycleLR 设定 max_lr：分别为 encoder 和 rest 的 lr（取 param group lr）
        max_lrs = [ENCODER_LR if len(param_groups) > 1 else REST_LR, REST_LR]
        # 因为 torch.optim 已经用了 param_groups[...], 下面将用 OneCycleLR（每个 param_group 一个 lr）
        scheduler = OneCycleLR(optimizer, max_lr=[ENCODER_LR, REST_LR] if len(param_groups) > 1 else [REST_LR],
                              total_steps=total_steps, pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=1e4)

        if rank == 0:
            print(f"[Rank {rank}] Starting {stage_name}: resolution=({h},{w}), batch_size={batch_size}, epochs={epochs}, total_steps={total_steps}")

        # 重新构造 dataloader（示例逻辑，真实项目推荐 dataset 支持动态 resize 而无需重建 DataLoader 每次）
        dataloader, sampler = make_dataloader_for_resolution(h, w, batch_size)

        global_step = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            it = 0
            # NOTE: 为了确保每 epoch 恰好 ITERS_PER_EPOCH 次，我们用 enumerate(dataloader) 但也可能循环/截断以满足步数
            data_iter = iter(dataloader)
            while it < ITERS_PER_EPOCH:
                try:
                    imgs, gt = next(data_iter)
                except StopIteration:
                    # 重新开始 dataloader
                    data_iter = iter(dataloader)
                    imgs, gt = next(data_iter)
                imgs = imgs.to(device)
                # gt 应为字典，且 key 与 compute_losses 中要求一致；这里只是示例
                # 将 gt 中的张量也移动到 device
                for k, v in gt.items():
                    if torch.is_tensor(v):
                        gt[k] = v.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=getattr(args, 'amp', True)):
                    res = model(imgs)  # 前向（注意：Pi3 的 forward 是否需要 batch dim 或其他形态，按实际调整）
                    losses = compute_losses(res, gt, device)
                    loss = losses['total_loss']

                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

                # scheduler step
                scheduler.step()
                epoch_loss += loss.item()
                it += 1
                global_step += 1

                if (it % 50 == 0 or it == 1) and rank == 0:
                    print(f"[{stage_name}] Epoch {epoch+1}/{epochs} Iter {it}/{ITERS_PER_EPOCH} loss={loss.item():.6f}")

            # epoch 完成后保存 checkpoint（rank0）
            if rank == 0 and ((epoch + 1) % getattr(args, 'save_interval', 1) == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"pi3_{stage_name}_epoch{epoch+1}.pt")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"[Rank 0] Saved checkpoint: {ckpt_path}")

        # stage 结束
        if rank == 0:
            print(f"[Rank {rank}] Finished {stage_name}")

    # ======= confidence head 单独训练阶段 =======
    # 冻结除 conf head 外的参数
    # 假设 conf 参数名在 conf_params 中（prepare_model_for_training 已收集）
    if len(conf_params) == 0:
        if rank == 0:
            print("No conf params detected, skipping confidence head stage.")
    else:
        # unfreeze conf and freeze rest
        for name, p in model.module.named_parameters():
            p.requires_grad = False
        for p in conf_params:
            p.requires_grad = True

        # 单独优化器只包含 conf params（用较小 lr 或默认 REST_LR）
        opt_conf = torch.optim.AdamW([p for p in conf_params if p.requires_grad], lr=REST_LR)
        total_steps_conf = getattr(args, 'conf_epochs', 5) * ITERS_PER_EPOCH  # 通常只需数个 epoch
        scheduler_conf = OneCycleLR(opt_conf, max_lr=REST_LR, total_steps=total_steps_conf, pct_start=0.1, anneal_strategy='cos')

        # dataloader: 固定一个合理分辨率（选 stage2 的平均）
        h, w = sample_random_resolution(STAGE2_PIX_MIN, STAGE2_PIX_MAX, STAGE2_AR_MIN, STAGE2_AR_MAX)
        batch_size = choose_batch_size_for_resolution(h, w)
        dataloader_conf, sampler_conf = make_dataloader_for_resolution(h, w, batch_size)

        if rank == 0:
            print(f"[Rank {rank}] Starting confidence-head-only training: resolution=({h},{w}), batch_size={batch_size}")

        step = 0
        for epoch in range(getattr(args, 'conf_epochs', 5)):
            sampler_conf.set_epoch(epoch)
            model.train()
            it = 0
            data_iter = iter(dataloader_conf)
            while it < ITERS_PER_EPOCH:
                try:
                    imgs, gt = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader_conf)
                    imgs, gt = next(data_iter)
                imgs = imgs.to(device)
                for k, v in gt.items():
                    if torch.is_tensor(v):
                        gt[k] = v.to(device)

                opt_conf.zero_grad()
                with torch.cuda.amp.autocast(enabled=getattr(args, 'amp', True)):
                    res = model(imgs)
                    # conf loss 只用 conf 部分
                    if 'conf' in res and 'conf' in gt:
                        pred_conf = torch.sigmoid(res['conf'])
                        loss_conf = F.binary_cross_entropy(pred_conf, gt['conf']) * LAMBDA_CONF
                    else:
                        loss_conf = torch.tensor(0.0, device=device)
                loss_conf.backward()
                torch.nn.utils.clip_grad_norm_(conf_params, GRAD_CLIP_NORM)
                opt_conf.step()
                scheduler_conf.step()

                it += 1
                step += 1
                if (it % 50 == 0 or it == 1) and rank == 0:
                    print(f"[conf stage] Epoch {epoch+1} Iter {it}/{ITERS_PER_EPOCH} loss_conf={loss_conf.item():.6f}")

        if rank == 0:
            ckpt_path = os.path.join(args.save_dir, "pi3_conf_head_trained.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Rank 0] Saved conf-head checkpoint: {ckpt_path}")

    # 清理
    dist.destroy_process_group()
