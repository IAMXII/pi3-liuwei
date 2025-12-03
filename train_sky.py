# #!/usr/bin/env python3
# import os
# import math
# import random
# import argparse
# from typing import Tuple
# from gsplat import rasterization
# import torch
# import torch.nn.functional as F
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.utils.dataset import DataLoader, DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.optim.lr_scheduler import OneCycleLR
# from lpips import LPIPS
# from loss import *
# from dataset.sky import SkyDataset
# from safetensors.torch import save_file, load_file
# from torchmetrics import JaccardIndex  # ✅ IoU计算
# # ---------- 超参（可改为 argparse） ----------
# STAGE_EPOCHS = 80
# ITERS_PER_EPOCH = 800
# STAGE1_RES = (224, 224)
# STAGE2_PIX_MIN = 100_000
# STAGE2_PIX_MAX = 255_000
# STAGE2_AR_MIN = 0.5
# STAGE2_AR_MAX = 2.0
# BATCH_MIN = 2
# BATCH_MAX = 24
# SAMPLES_PER_GPU_STAGE1 = 64
# SAMPLES_PER_GPU_STAGE2 = 48
#
# # Loss weights
# LAMBDA_NORMAL = 1.0
# LAMBDA_LPIPS = 0.05
# LAMBDA_CONF = 0.05
# LAMBDA_CAM = 0.1
# LAMBDA_TRANS = 100.0
# LAMBDA_DEPTH = 1.0
# LAMBDA_RGB = 1
#
# ENCODER_LR = 5e-6
# REST_LR = 5e-5
#
# GRAD_CLIP_NORM = 1.0
#
# # ---------------------------
# try:
#     from pi3.models.pi3 import Pi3
# except Exception as e:
#     print("Warning: failed to import Pi3. Make sure your PYTHONPATH includes the project. Error:", e)
#     Pi3 = None
#
#
# # ---------- 辅助函数 ----------
# def sample_random_resolution(total_px_min: int, total_px_max: int,
#                              ar_min: float, ar_max: float) -> Tuple[int, int]:
#     target_px = random.randint(total_px_min, total_px_max)
#     ar = random.uniform(ar_min, ar_max)
#     h = int(round(math.sqrt(target_px / ar)))
#     w = int(round(ar * h))
#     h = max(16, (h // 16) * 16)
#     w = max(16, (w // 16) * 16)
#     if h == 0: h = 16
#     if w == 0: w = 16
#     return h, w
#
#
# def choose_batch_size_for_resolution(h: int, w: int,
#                                      batch_min: int = BATCH_MIN,
#                                      batch_max: int = BATCH_MAX,
#                                      target_pixels_per_batch: int = 200_000) -> int:
#     px = h * w
#     if px == 0:
#         return batch_min
#     est = max(batch_min, min(batch_max, target_pixels_per_batch // px))
#     return max(batch_min, est)
#
#
# # ---------- 模型参数分组和冻结逻辑 ----------
# def prepare_model_for_training(model: torch.nn.Module, pretrained_vggt_checkpoint: str = None,
#                                freeze_encoder: bool = True):
#     if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
#         # 自动根据文件后缀选择加载方式
#         if pretrained_vggt_checkpoint.endswith(".safetensors"):
#             print(f"Loading safetensors checkpoint from {pretrained_vggt_checkpoint}")
#             ckpt = load_file(pretrained_vggt_checkpoint)
#         else:
#             print(f"Loading torch checkpoint from {pretrained_vggt_checkpoint}")
#             ckpt = torch.load(pretrained_vggt_checkpoint, map_location='cpu', weights_only=True)
#
#         try:
#             model_state = model.state_dict()
#             # 只加载匹配的参数（忽略结构不同的部分）
#             load_state = {k: v for k, v in ckpt.items() if k in model_state and model_state[k].shape == v.shape}
#             model_state.update(load_state)
#             model.load_state_dict(model_state)
#             print(f"✅ Loaded {len(load_state)}/{len(model_state)} VGGT weights into model.")
#         except Exception as e:
#             print("⚠️ Warning: failed to load VGGT checkpoint automatically:", e)
#
#     encoder_params = []
#     rest_params = []
#     conf_params = []
#     for name, p in model.named_parameters():
#         if 'encoder' in name:
#             encoder_params.append(p)
#             if freeze_encoder:
#                 p.requires_grad = False
#         else:
#             rest_params.append(p)
#         if 'conf' in name or 'confidence' in name:
#             conf_params.append(p)
#
#     if len(encoder_params) == 0:
#         print("Warning: no parameter matched 'encoder' substring. Check model.named_parameters() naming.")
#
#     return encoder_params, rest_params, conf_params
#
#
# # ---------- 训练主循环（在每个进程内运行） ----------
# def train_worker(local_rank, world_size, args):
#     # local_rank: mp.spawn 给的索引 0..world_size-1
#     os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(world_size))  # ensures visibility, optional
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f"cuda:{local_rank}")
#
#     # init distributed
#     dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)
#
#     # build model
#     if Pi3 is None:
#         raise RuntimeError("Pi3 model not imported. Please ensure pi3.models.pi3 is available in PYTHONPATH.")
#     model = Pi3().to(device)
#
#     encoder_params, rest_params, conf_params = prepare_model_for_training(
#         model, pretrained_vggt_checkpoint=getattr(args, 'pretrained_vggt', None), freeze_encoder=args.freeze_encoder
#     )
#
#     model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
#
#     param_groups = []
#     if encoder_params:
#         enc_trainable = [p for p in encoder_params if p.requires_grad]
#         if len(enc_trainable) > 0:
#             param_groups.append({'params': enc_trainable, 'lr': ENCODER_LR})
#     rest_trainable = [p for p in rest_params if p.requires_grad]
#     if len(rest_trainable) > 0:
#         param_groups.append({'params': rest_trainable, 'lr': REST_LR})
#
#     optimizer = torch.optim.AdamW(param_groups, lr=REST_LR, weight_decay=1e-2)
#     scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
#
#     # dataloader factory
#     def make_dataloader_for_resolution(h, w, batch_size, train_sky=False):
#         if not train_sky:
#             # This script is focused on sky-only training. Guard against accidental non-sky mode.
#             raise ValueError("This training script is configured for sky-only training. Set --train_sky True.")
#         dataset = SkyDataset(args.data_path)
#         sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
#         dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
#                         num_workers=args.num_workers, pin_memory=True, drop_last=True)
#         return dl, sampler
#
#     ### gaussian head training
#     if args.train_sky == False:
#         # Keep original non-sky code path but do not run it in sky-only script.
#         if local_rank == 0:
#             print("Non-sky training path is disabled in this script. Use --train_sky True to train sky head.")
#     # confidence head single-stage (same as your original logic)
#     else:
#         # collect conf/sky params robustly
#         # freeze all params first
#         for name, p in model.module.named_parameters():
#             p.requires_grad = False
#
#         # try to find sky-related parameters (sky_head, conf, confidence)
#         sky_param_names = [n for n, _ in model.module.named_parameters() if ('sky' in n.lower() or 'conf' in n.lower())]
#         sky_params = [p for n, p in model.module.named_parameters() if ('sky' in n.lower() or 'conf' in n.lower())]
#
#         if len(sky_params) == 0:
#             if local_rank == 0:
#                 print("No sky-related parameters found in model. Check module names (expect 'sky' or 'confidence' substring). Skipping training.")
#         else:
#             # enable grads for sky params
#             for p in sky_params:
#                 p.requires_grad = True
#
#             # define optimizer only containing sky params
#             opt_sky = torch.optim.AdamW(sky_params, lr=REST_LR, weight_decay=1e-2)
#             total_steps_sky = args.sky_epochs * ITERS_PER_EPOCH
#             scheduler_sky = OneCycleLR(
#                 opt_sky,
#                 max_lr=REST_LR,
#                 total_steps=total_steps_sky,
#                 pct_start=0.1,
#                 anneal_strategy='cos'
#             )
#
#             h, w = sample_random_resolution(STAGE2_PIX_MIN, STAGE2_PIX_MAX, STAGE2_AR_MIN, STAGE2_AR_MAX)
#             batch_size = choose_batch_size_for_resolution(h, w)
#             dataloader_sky, sampler_sky = make_dataloader_for_resolution(h, w, batch_size, train_sky=True)
#
#             if local_rank == 0:
#                 print(
#                     f"[Rank {local_rank}] Starting confidence-head-only training: resolution=({h},{w}), batch_size={batch_size}")
#
#             step = 0
#             # training loop
#             for epoch in range(args.sky_epochs):
#                 sampler_sky.set_epoch(epoch)
#                 model.train()
#                 it = 0
#                 data_iter = iter(dataloader_sky)
#                 while it < ITERS_PER_EPOCH:
#                     try:
#                         gt = next(data_iter)
#                     except StopIteration:
#                         data_iter = iter(dataloader_sky)
#                         gt = next(data_iter)
#                     imgs = gt['image']
#                     imgs = imgs.to(device)
#                     # move tensors in gt to device
#                     for k, v in gt.items():
#                         if torch.is_tensor(v):
#                             gt[k] = v.to(device)
#
#                     opt_sky.zero_grad()
#                     with torch.cuda.amp.autocast(enabled=args.amp):
#                         imgs = imgs.unsqueeze(1)
#                         res = model(imgs, train_sky=True)
#
#                         # ensure 'sky' in res
#                         if 'sky' not in res:
#                             # skip this batch if model didn't return sky predictions
#                             if local_rank == 0:
#                                 print("Warning: model did not return 'sky' in forward pass. Skipping batch.")
#                             it += 1
#                             continue
#
#                         loss_sky = compute_loss_sky(res, gt)  # user-defined loss
#                         # guard against NaN and inf
#                         loss_sky = torch.nan_to_num(loss_sky, nan=0.0, posinf=1e6, neginf=-1e6)
#                         loss_sky = loss_sky * LAMBDA_CONF
#
#                     scaler.scale(loss_sky).backward()
#                     # unscale grads for clipping
#                     scaler.unscale_(opt_sky)
#                     torch.nn.utils.clip_grad_norm_(sky_params, GRAD_CLIP_NORM)
#                     scaler.step(opt_sky)
#                     scaler.update()
#
#                     scheduler_sky.step()
#
#                     it += 1
#                     step += 1
#                     if (it % 50 == 0 or it == 1) and local_rank == 0:
#                         print(
#                             f"[sky stage] Epoch {epoch + 1} Iter {it}/{ITERS_PER_EPOCH} loss_sky={loss_sky.item():.6f}")
#
#
#
#
#             if local_rank == 0:
#                 os.makedirs(args.save_dir, exist_ok=True)
#
#                 ckpt_path = os.path.join(args.save_dir, "pi3_full_model.safetensors")
#
#                 # ✅ 保存为 safetensors 格式
#                 state_dict = model.module.state_dict()
#                 save_file(state_dict, ckpt_path)
#
#                 print(f"[Rank 0] ✅ Saved full model (safetensors): {ckpt_path}")
#
#     dist.destroy_process_group()
#
#
# # ---------- 启动 / CLI ----------
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gpus", type=int, default=None,
#                         help="Number of GPUs to use on this machine. Default: auto-detect")
#     parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset")
#     parser.add_argument("--save-dir", type=str, required=True, help="Where to save checkpoints")
#     parser.add_argument("--interval", type=int, default=1, help="Dataset interval option")
#     parser.add_argument("--amp", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use AMP (True/False)")
#     parser.add_argument("--num-workers", type=int, default=4, help="num_workers for DataLoader")
#     parser.add_argument("--save-interval", type=int, default=1, help="how many epochs between saves")
#     parser.add_argument("--pretrained-vggt", type=str, default=None, help="path to pretrained vgtt checkpoint")
#     parser.add_argument("--freeze-encoder", action='store_true', help="freeze encoder at start")
#     parser.add_argument("--sky-epochs", type=int, default=20, help="epochs for confidence head training")
#     parser.add_argument("--train-sky", type=lambda x: (str(x).lower() == 'true'), default=True, help="train sky head only (True/False)")
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#     # decide world size
#     if args.gpus is None:
#         world_size = torch.cuda.device_count()
#         if world_size == 0:
#             raise RuntimeError("No GPUs found. Please set --gpus or run on a machine with GPUs.")
#     else:
#         world_size = args.gpus
#         if torch.cuda.device_count() < world_size:
#             print(
#                 f"Warning: torch.cuda.device_count()={torch.cuda.device_count()} < requested --gpus {world_size}. Using available device count instead.")
#             world_size = torch.cuda.device_count()
#
#     if world_size <= 0:
#         raise RuntimeError("No GPUs available.")
#
#     # set env for init_method='env://'
#     os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
#     os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
#
#     print(f"Spawning {world_size} processes for distributed training (single-node).")
#     mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
#
#
# if __name__ == "__main__":
#     main()

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
from loss import *
from dataset.sky import SkyDataset
from safetensors.torch import save_file, load_file
from torchmetrics import JaccardIndex  # ✅ IoU计算
import wandb  # ✅ 新增
import time
# ---------- 超参 ----------
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


try:
    from pi3.models.pi3 import Pi3
except Exception as e:
    print("Warning: failed to import Pi3. Make sure your PYTHONPATH includes the project. Error:", e)
    Pi3 = None


# ---------- 辅助函数 ----------
def sample_random_resolution(total_px_min: int, total_px_max: int,
                             ar_min: float, ar_max: float) -> Tuple[int, int]:
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


def prepare_model_for_training(model: torch.nn.Module, pretrained_vggt_checkpoint: str = None,
                               freeze_encoder: bool = True):
    if pretrained_vggt_checkpoint and os.path.exists(pretrained_vggt_checkpoint):
        if pretrained_vggt_checkpoint.endswith(".safetensors"):
            print(f"Loading safetensors checkpoint from {pretrained_vggt_checkpoint}")
            ckpt = load_file(pretrained_vggt_checkpoint)
        else:
            print(f"Loading torch checkpoint from {pretrained_vggt_checkpoint}")
            ckpt = torch.load(pretrained_vggt_checkpoint, map_location='cpu', weights_only=True)

        try:
            model_state = model.state_dict()
            load_state = {k: v for k, v in ckpt.items() if k in model_state and model_state[k].shape == v.shape}
            model_state.update(load_state)
            model.load_state_dict(model_state)
            print(f"✅ Loaded {len(load_state)}/{len(model_state)} VGGT weights into model.")
        except Exception as e:
            print("⚠️ Warning: failed to load VGGT checkpoint automatically:", e)

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


# ---------- validate() ----------
@torch.no_grad()
def validate(model, dataloader, device, world_size):
    model.eval()
    metric_iou = JaccardIndex(task="binary").to(device)
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        imgs = batch['image'].to(device)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        imgs = imgs.unsqueeze(1)
        with torch.cuda.amp.autocast():
            res = model(imgs, train_sky=True)
            if 'sky' not in res:
                continue
            loss_sky = compute_loss_sky(res, batch)
            loss_sky = torch.nan_to_num(loss_sky, nan=0.0, posinf=1e6, neginf=-1e6)
            total_loss += loss_sky.item()
            count += 1
            pred = res['sky']
            pred_bin = (pred > 0.5).float()
            gt_bin = (batch['depth'] < 0.01).float().unsqueeze(4)
            metric_iou.update(pred_bin, gt_bin)

    avg_loss = total_loss / max(count, 1)
    iou = metric_iou.compute().item()

    tensor = torch.tensor([avg_loss, iou], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.tolist()


# ---------- train_worker ----------
def train_worker(local_rank, world_size, args):
    epoch_best = 0
    best_iou = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)

    if Pi3 is None:
        raise RuntimeError("Pi3 model not imported.")
    model = Pi3().to(device)

    encoder_params, rest_params, conf_params = prepare_model_for_training(
        model, pretrained_vggt_checkpoint=getattr(args, 'pretrained_vggt', None), freeze_encoder=args.freeze_encoder
    )
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    opt_params = []
    rest_trainable = [p for p in rest_params if p.requires_grad]
    if len(rest_trainable) > 0:
        opt_params.append({'params': rest_trainable, 'lr': REST_LR})
    optimizer = torch.optim.AdamW(opt_params, lr=REST_LR, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ✅ 初始化 wandb (仅主进程)
    if local_rank == 0:
        wandb.init(
            project="pi3-sky-training",
            name=f"pi3_sky_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
            mode="online"  # 或 "offline"
        )

    def make_dataloader_for_resolution(h, w, batch_size, split='train'):
        dataset = SkyDataset(args.data_path, split=split)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=(split == 'train'))
        dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=(split == 'train'))
        return dl, sampler

    # === Sky head 训练 ===
    for n, p in model.module.named_parameters():
        p.requires_grad = ('sky' in n.lower() or 'conf' in n.lower())

    opt_sky = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=REST_LR, weight_decay=1e-2)
    scheduler_sky = OneCycleLR(opt_sky, max_lr=REST_LR,
                               total_steps=args.sky_epochs * ITERS_PER_EPOCH,
                               pct_start=0.1, anneal_strategy='cos')

    h, w = sample_random_resolution(STAGE2_PIX_MIN, STAGE2_PIX_MAX, STAGE2_AR_MIN, STAGE2_AR_MAX)
    batch_size =choose_batch_size_for_resolution(h, w)
    dataloader_sky, sampler_sky = make_dataloader_for_resolution(h, w, batch_size, split='train')

    step = 0
    for epoch in range(args.sky_epochs):
        sampler_sky.set_epoch(epoch)
        model.train()
        it = 0
        data_iter = iter(dataloader_sky)

        while it < ITERS_PER_EPOCH:
            try:
                gt = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader_sky)
                gt = next(data_iter)

            imgs = gt['image'].to(device)
            for k, v in gt.items():
                if torch.is_tensor(v):
                    gt[k] = v.to(device)

            opt_sky.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                imgs = imgs.unsqueeze(1)
                res = model(imgs, train_sky=True)
                if 'sky' not in res:
                    it += 1
                    continue
                loss_sky = compute_loss_sky(res, gt)
                loss_sky = torch.nan_to_num(loss_sky, nan=0.0, posinf=1e6, neginf=-1e6)
                loss_sky = loss_sky * LAMBDA_CONF

            scaler.scale(loss_sky).backward()
            scaler.unscale_(opt_sky)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt_sky)
            scaler.update()
            scheduler_sky.step()

            if local_rank == 0 and (it % 50 == 0 or it == 1):
                print(f"[sky stage] Epoch {epoch + 1} Iter {it}/{ITERS_PER_EPOCH} loss_sky={loss_sky.item():.6f}")
                wandb.log({"train/loss_sky": loss_sky.item(), "step": step})

            it += 1
            step += 1

        # === 每轮结束验证 ===
        val_h, val_w = STAGE1_RES
        val_bs = choose_batch_size_for_resolution(val_h, val_w)
        val_loader, _ = make_dataloader_for_resolution(val_h, val_w, val_bs, split='val')
        val_loss, val_iou = validate(model, val_loader, device, world_size)

        if local_rank == 0:
            print(f"[Epoch {epoch + 1}] ✅ Validation done: loss={val_loss:.6f}, IoU={val_iou:.4f}")
            wandb.log({"val/loss": val_loss, "val/iou": val_iou, "epoch": epoch + 1})

            os.makedirs(args.save_dir, exist_ok=True)
            if val_iou > best_iou:
                best_model = model
                best_iou = val_iou
            ckpt_path = os.path.join(args.save_dir, f"pi3_sky_best_1.safetensors")
            save_file(best_model.module.state_dict(), ckpt_path)
            # wandb.save(ckpt_path)
            print(f"[Rank 0] 💾 Saved model to {ckpt_path}")

    if local_rank == 0:
        wandb.finish()
    dist.destroy_process_group()
# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--amp", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--pretrained-vggt", type=str, default=None)
    parser.add_argument("--freeze-encoder", action='store_true')
    parser.add_argument("--sky-epochs", type=int, default=20)
    parser.add_argument("--train-sky", type=lambda x: (str(x).lower() == 'true'), default=True)
    return parser.parse_args()


def main():
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = "0b99aa6ad89849640ad285a61d5cbad3b868a43e"

    wandb.login()
    args = parse_args()
    if args.gpus is None:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No GPUs found.")
    else:
        world_size = args.gpus
        if torch.cuda.device_count() < world_size:
            print(f"Warning: only {torch.cuda.device_count()} GPUs available.")
            world_size = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    print(f"Spawning {world_size} processes for distributed training.")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

