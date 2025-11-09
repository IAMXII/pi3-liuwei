#!/usr/bin/env python3
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
LAMBDA_NORMAL = 1.0
LAMBDA_LPIPS = 0.05
LAMBDA_CONF = 0.05
LAMBDA_CAM = 0.1
LAMBDA_TRANS = 100.0
LAMBDA_DEPTH = 1.0
LAMBDA_RGB = 1

ENCODER_LR = 5e-6
REST_LR = 5e-5

GRAD_CLIP_NORM = 1.0

# ---------------------------
# You must provide these in your project
# from pi3.models.pi3 import Pi3
# from your_dataset_module import SimpleImageDataset
# ---------------------------
try:
    from pi3.models.pi3 import Pi3
except Exception as e:
    print("Warning: failed to import Pi3. Make sure your PYTHONPATH includes the project. Error:", e)
    Pi3 = None

# Placeholder dataset class name - replace with your actual dataset import
try:
    from your_dataset_module import SimpleImageDataset  # <- replace with real import
except Exception:
    # if not available, define a minimal placeholder to avoid crash during static analysis
    class SimpleImageDataset:
        def __init__(self, data_path, interval=1, out_size=(224,224)):
            self.data_path = data_path
            self.interval = interval
            self.out_size = out_size
            self._len = 10000
        def __len__(self):
            return self._len
        def __getitem__(self, idx):
            # return fake data - in practice you must use real dataset
            img = torch.randn(3, self.out_size[0], self.out_size[1])
            # gt must be a dict with appropriate tensors used by compute_losses
            gt = {
                'points': torch.randn(1,3),
                'local_points': torch.randn(1,1024,3),
                'conf': torch.randint(0,2,(1,1024)).float(),
                'camera_poses': torch.randn(1,6)
            }
            return img, gt

# ---------- 辅助函数 ----------
def sample_random_resolution(total_px_min: int, total_px_max: int,
                             ar_min: float, ar_max: float) -> Tuple[int,int]:
    target_px = random.randint(total_px_min, total_px_max)
    ar = random.uniform(ar_min, ar_max)
    h = int(round(math.sqrt(target_px / ar)))
    w = int(round(ar * h))
    h = max(16, (h // 16) * 16)
    w = max(16, (w // 16) * 16)
    if h == 0: h = 16
    if w == 0: w = 16
    return h, w

def choose_batch_size_for_resolution(h: int, w: int,
                                     batch_min: int = BATCH_MIN,
                                     batch_max: int = BATCH_MAX,
                                     target_pixels_per_batch: int = 200_000) -> int:
    px = h * w
    if px == 0:
        return batch_min
    est = max(batch_min, min(batch_max, target_pixels_per_batch // px))
    return max(batch_min, est)

# ---------- 损失函数 ----------
def compute_losses(res, gt, device):
    losses = {}
    lpips_loss_fn = LPIPS(net='vgg').to(device)
    if 'gaussians' in res:
        height = gt['image'].shape[-2]
        width = gt['image'].shape[-1]
        B, C, _= res['camera_poses'].shape
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
        losses['rgb_loss'] = F.l1_loss(render_colors, gt['image'])
        render_norm = render_colors.permute(0, 3, 1, 2) * 2 - 1  # [B, 3, H, W]
        gt_norm = render_colors.permute(0, 3, 1, 2) * 2 - 1  # [B, 3, H, W]

        # 2. 计算 LPIPS（返回的是每个样本的平均值）
        lpips_val = lpips_loss_fn(render_norm, gt_norm)
        losses['lpips_loss'] = lpips_val.mean()

    if 'camera_poses' in res and 'camera_poses' in gt:
        pose_pred = res['camera_poses']
        pose_gt = gt['camera_poses']
        trans_loss = F.mse_loss(pose_pred[..., :3], pose_gt[..., :3])
        rot_loss = F.mse_loss(pose_pred[..., 3:], pose_gt[..., 3:])
        losses['pose_loss'] = (trans_loss * LAMBDA_TRANS + rot_loss * LAMBDA_CAM)
    else:
        losses['pose_loss'] = torch.tensor(0.0, device=device)

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
    if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
        ckpt = torch.load(pretrained_vggt_checkpoint, map_location='cpu')
        try:
            model_state = model.state_dict()
            load_state = {k: v for k, v in ckpt.items() if k in model_state}
            model_state.update(load_state)
            model.load_state_dict(model_state)
            print("Loaded partial VGGT weights into model.")
        except Exception as e:
            print("Warning: failed to load vgtt checkpoint automatically:", e)

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
        if 'conf' in name or 'confidence' in name:
            conf_params.append(p)

    if len(encoder_params) == 0:
        print("Warning: no parameter matched 'encoder' substring. Check model.named_parameters() naming.")

    return encoder_params, rest_params, conf_params

# ---------- 训练主循环（在每个进程内运行） ----------
def train_worker(local_rank, world_size, args):
    # local_rank: mp.spawn 给的索引 0..world_size-1
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(world_size))  # ensures visibility, optional
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # init distributed
    dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)

    # build model
    if Pi3 is None:
        raise RuntimeError("Pi3 model not imported. Please ensure pi3.models.pi3 is available in PYTHONPATH.")
    model = Pi3().to(device)

    encoder_params, rest_params, conf_params = prepare_model_for_training(
        model, pretrained_vggt_checkpoint=getattr(args, 'pretrained_vggt', None), freeze_encoder=args.freeze_encoder
    )

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    param_groups = []
    if encoder_params:
        enc_trainable = [p for p in encoder_params if p.requires_grad]
        if len(enc_trainable) > 0:
            param_groups.append({'params': enc_trainable, 'lr': ENCODER_LR})
    rest_trainable = [p for p in rest_params if p.requires_grad]
    if len(rest_trainable) > 0:
        param_groups.append({'params': rest_trainable, 'lr': REST_LR})

    optimizer = torch.optim.AdamW(param_groups, lr=REST_LR, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # dataloader factory
    def make_dataloader_for_resolution(h, w, batch_size):
        dataset = SimpleImageDataset(args.data_path, interval=args.interval, out_size=(h, w))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        return dl, sampler

    for stage_idx in [1, 2]:
        if stage_idx == 1:
            stage_name = "stage1"
            h, w = STAGE1_RES
            batch_size = choose_batch_size_for_resolution(h, w, BATCH_MIN, BATCH_MAX, target_pixels_per_batch=200_000)
            epochs = STAGE_EPOCHS
        else:
            stage_name = "stage2"
            h, w = sample_random_resolution(STAGE2_PIX_MIN, STAGE2_PIX_MAX, STAGE2_AR_MIN, STAGE2_AR_MAX)
            batch_size = choose_batch_size_for_resolution(h, w, BATCH_MIN, BATCH_MAX, target_pixels_per_batch=200_000)
            epochs = STAGE_EPOCHS

        total_steps = epochs * ITERS_PER_EPOCH
        max_lr_list = [ENCODER_LR, REST_LR] if len(param_groups) > 1 else [REST_LR]
        scheduler = OneCycleLR(optimizer, max_lr=max_lr_list, total_steps=total_steps, pct_start=0.1,
                              anneal_strategy='cos', div_factor=10.0, final_div_factor=1e4)

        if local_rank == 0:
            print(f"[Rank {local_rank}] Starting {stage_name}: resolution=({h},{w}), batch_size={batch_size}, epochs={epochs}, total_steps={total_steps}")

        dataloader, sampler = make_dataloader_for_resolution(h, w, batch_size)

        global_step = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            it = 0
            data_iter = iter(dataloader)
            while it < ITERS_PER_EPOCH:
                try:
                    imgs, gt = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    imgs, gt = next(data_iter)
                imgs = imgs.to(device)
                for k, v in gt.items():
                    if torch.is_tensor(v):
                        gt[k] = v.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=args.amp):
                    res = model(imgs)
                    losses = compute_losses(res, gt, device)
                    loss = losses['total_loss']

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                epoch_loss += loss.item()
                it += 1
                global_step += 1

                if (it % 50 == 0 or it == 1) and local_rank == 0:
                    print(f"[{stage_name}] Epoch {epoch+1}/{epochs} Iter {it}/{ITERS_PER_EPOCH} loss={loss.item():.6f}")

            if local_rank == 0 and ((epoch + 1) % args.save_interval == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"pi3_{stage_name}_epoch{epoch+1}.pt")
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"[Rank 0] Saved checkpoint: {ckpt_path}")

        if local_rank == 0:
            print(f"[Rank {local_rank}] Finished {stage_name}")

    # confidence head single-stage (same as your original logic)
    if len(conf_params) == 0:
        if local_rank == 0:
            print("No conf params detected, skipping confidence head stage.")
    else:
        # freeze all then unfreeze conf params
        for name, p in model.module.named_parameters():
            p.requires_grad = False
        for p in conf_params:
            p.requires_grad = True

        opt_conf = torch.optim.AdamW([p for p in conf_params if p.requires_grad], lr=REST_LR)
        total_steps_conf = args.conf_epochs * ITERS_PER_EPOCH
        scheduler_conf = OneCycleLR(opt_conf, max_lr=REST_LR, total_steps=total_steps_conf, pct_start=0.1, anneal_strategy='cos')

        h, w = sample_random_resolution(STAGE2_PIX_MIN, STAGE2_PIX_MAX, STAGE2_AR_MIN, STAGE2_AR_MAX)
        batch_size = choose_batch_size_for_resolution(h, w)
        dataloader_conf, sampler_conf = make_dataloader_for_resolution(h, w, batch_size)

        if local_rank == 0:
            print(f"[Rank {local_rank}] Starting confidence-head-only training: resolution=({h},{w}), batch_size={batch_size}")

        step = 0
        for epoch in range(args.conf_epochs):
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
                with torch.cuda.amp.autocast(enabled=args.amp):
                    res = model(imgs)
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
                if (it % 50 == 0 or it == 1) and local_rank == 0:
                    print(f"[conf stage] Epoch {epoch+1} Iter {it}/{ITERS_PER_EPOCH} loss_conf={loss_conf.item():.6f}")

        if local_rank == 0:
            ckpt_path = os.path.join(args.save_dir, "pi3_conf_head_trained.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"[Rank 0] Saved conf-head checkpoint: {ckpt_path}")

    dist.destroy_process_group()

# ---------- 启动 / CLI ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use on this machine. Default: auto-detect")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--save-dir", type=str, required=True, help="Where to save checkpoints")
    parser.add_argument("--interval", type=int, default=1, help="Dataset interval option")
    parser.add_argument("--amp", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use AMP (True/False)")
    parser.add_argument("--num-workers", type=int, default=4, help="num_workers for DataLoader")
    parser.add_argument("--save-interval", type=int, default=1, help="how many epochs between saves")
    parser.add_argument("--pretrained-vggt", type=str, default=None, help="path to pretrained vgtt checkpoint")
    parser.add_argument("--freeze-encoder", action='store_true', help="freeze encoder at start")
    parser.add_argument("--conf-epochs", type=int, default=5, help="epochs for confidence head training")
    return parser.parse_args()

def main():
    args = parse_args()
    # decide world size
    if args.gpus is None:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No GPUs found. Please set --gpus or run on a machine with GPUs.")
    else:
        world_size = args.gpus
        if torch.cuda.device_count() < world_size:
            print(f"Warning: torch.cuda.device_count()={torch.cuda.device_count()} < requested --gpus {world_size}. Using available device count instead.")
            world_size = torch.cuda.device_count()

    # default target: 10 GPUs if available and user asked for 10
    if world_size <= 0:
        raise RuntimeError("No GPUs available.")

    # set env for init_method='env://'
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    print(f"Spawning {world_size} processes for distributed training (single-node).")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
