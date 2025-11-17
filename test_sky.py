#!/usr/bin/env python3
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import numpy as np
from safetensors.torch import load_file  # ✅ safetensors 导入
from dataset.sky import SkyDataset
from pi3.models.pi3 import Pi3

# ==============================
# 可视化函数
# ==============================
def visualize_results(img, pred_mask, gt_mask, save_path):
    """保存输入图、预测结果、GT到一张图中"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pred_mask = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    gt_mask = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    concat = np.concatenate([
        img,
        np.stack([pred_mask]*3, axis=-1),
        np.stack([gt_mask]*3, axis=-1)
    ], axis=1)
    Image.fromarray(concat).save(save_path)

# ==============================
# 测试入口
# ==============================
@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='/home/liuwei/mnt/instant_vggt_dataset/mask_train', help='验证集路径')
    parser.add_argument('--ckpt', type=str, required=True, help='模型权重路径 (.safetensors)')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=496)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==============================
    # 加载数据集
    # ==============================
    dataset = SkyDataset(
        root_dir=args.val_dir,
        split='val',
    )
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # ==============================
    # 加载模型
    # ==============================
    model = Pi3().to(device)
    print(f"🔹 Loading safetensors checkpoint: {args.ckpt}")
    ckpt = load_file(args.ckpt, device=device)  # ✅ 使用 safetensors 加载
    # safetensors 文件一般直接存储 state_dict，无需再 ckpt['model']
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # ==============================
    # 随机选10张
    # ==============================
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), args.num_samples)
    print(f"Visualizing {args.num_samples} samples ...")

    for i, idx in enumerate(tqdm(indices)):
        ret = dataset[idx]
        img = ret['image']
        gt_mask = ret['depth']
        img = img.unsqueeze(0).to(device)
        img = img.unsqueeze(1).to(device)
        gt_mask = gt_mask.to(device)

        # 前向推理
        pred = model(img, train_sky=True)
        if isinstance(pred, dict) and 'sky' in pred:
            pred_mask = pred['sky']
        else:
            pred_mask = pred
        pred_mask = (pred_mask > 0.5).float()

        save_path = os.path.join(args.save_dir, f'sample_{i:02d}.png')
        visualize_results(img[0][0], pred_mask[0], gt_mask, save_path)

    print(f"✅ Results saved to {args.save_dir}")

if __name__ == '__main__':
    main()
