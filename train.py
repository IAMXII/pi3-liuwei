import os
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge


# ----------------------------
#       Loss Function
# ----------------------------
def compute_losses(res, gt, stage=1):
    """
    Multi-component Pi3 loss (stage-aware).
    """
    losses = {}

    # 1. normal loss (MoGe-style)
    if 'local_points' in res and 'local_points' in gt:
        losses['normal_loss'] = torch.nn.functional.smooth_l1_loss(res['local_points'], gt['local_points'])

    # 2. confidence loss
    if 'conf' in res and 'conf' in gt:
        pred_conf = torch.sigmoid(res['conf'])
        losses['conf_loss'] = torch.nn.functional.binary_cross_entropy(pred_conf, gt['conf'])

    # 3. camera pose loss (translation + rotation)
    if 'camera_poses' in res and 'camera_poses' in gt:
        pose_pred, pose_gt = res['camera_poses'], gt['camera_poses']
        trans_loss = torch.nn.functional.mse_loss(pose_pred[..., :3], pose_gt[..., :3])
        rot_loss = torch.nn.functional.mse_loss(pose_pred[..., 3:], pose_gt[..., 3:])
        losses['cam_loss'] = trans_loss + 0.1 * rot_loss

    # 4. Gaussian / trans regularization
    if 'points' in res:
        mean = torch.mean(res['points'], dim=1, keepdim=True)
        var = torch.var(res['points'], dim=1, keepdim=True)
        gaussian_loss = torch.mean((mean ** 2) + (var - 1.0) ** 2)
        losses['trans_loss'] = gaussian_loss

    # Weighting scheme from paper
    λnormal = 1.0
    λconf = 0.05
    λcam = 0.1
    λtrans = 100.0

    total_loss = (
        λnormal * losses.get('normal_loss', 0) +
        λconf * losses.get('conf_loss', 0) +
        λcam * losses.get('cam_loss', 0) +
        λtrans * losses.get('trans_loss', 0)
    )
    losses['total_loss'] = total_loss
    return losses


# ----------------------------
#        Train Function
# ----------------------------
def train_one_stage(rank, args, stage):
    dist.init_process_group(backend='nccl', init_method='env://')

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"[Stage {stage}] Starting training on rank {rank} with {dist.get_world_size()} GPUs")

    # 1. Model init
    model = Pi3().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 2. Optimizer with differential LR
    enc_params = []
    other_params = []
    for name, p in model.named_parameters():
        if 'encoder' in name:
            enc_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": 5e-6},
        {"params": other_params, "lr": 5e-5},
    ], weight_decay=1e-4)

    # 3. Dummy dataset
    dataset = torch.utils.data.TensorDataset(torch.randn(800 * 2, 3, 224, 224))
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4)

    # 4. Scheduler: OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-6, 5e-5],
        total_steps=len(dataloader) * args.epochs,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 5. Train loop
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Stage {stage} Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader

        for imgs, in pbar:
            imgs = imgs.to(device)
            gt = {
                'local_points': torch.randn_like(imgs[:, :3, :, :]),
                'conf': torch.rand(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).to(device),
                'camera_poses': torch.zeros(imgs.shape[0], 6).to(device),
                'points': torch.randn(imgs.shape[0], 1024, 3).to(device)
            }

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                res = model(imgs)
                losses = compute_losses(res, gt, stage)
                loss = losses['total_loss']

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if rank == 0:
                pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items() if isinstance(v, torch.Tensor)})

    dist.barrier()
    if rank == 0:
        torch.save(model.module.state_dict(), f'pi3_stage{stage}.pth')
        print(f"[Stage {stage}] Model saved.")

    dist.destroy_process_group()


# ----------------------------
#     Confidence Head Train
# ----------------------------
def train_confidence_head(rank, args):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = Pi3().to(device)
    model.load_state_dict(torch.load('pi3_stage2.pth', map_location=device), strict=False)

    # Freeze everything except conf head
    for n, p in model.named_parameters():
        if 'conf_head' not in n:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    dataloader = DataLoader(torch.randn(1000, 3, 224, 224), batch_size=8)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    for epoch in range(3):
        for imgs in dataloader:
            imgs = imgs.to(device)
            gt = {'conf': torch.rand(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]).to(device)}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                res = model(imgs)
                pred_conf = torch.sigmoid(res['conf'])
                loss = torch.nn.functional.binary_cross_entropy(pred_conf, gt['conf'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if rank == 0:
            print(f"[Conf Head] Epoch {epoch+1}, Loss {loss.item():.4f}")
    dist.destroy_process_group()


# ----------------------------
#           MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--stage", type=int, default=1, help="Training stage (1 or 2)")
    args = parser.parse_args()

    # Launch training for your 8-GPU A6000 machine
    if args.stage == 1:
        torch.multiprocessing.spawn(train_one_stage, args=(args, 1), nprocs=8)
    elif args.stage == 2:
        torch.multiprocessing.spawn(train_one_stage, args=(args, 2), nprocs=8)
    else:
        torch.multiprocessing.spawn(train_confidence_head, args=(args,), nprocs=8)
