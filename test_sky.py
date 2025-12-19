# #!/usr/bin/env python3
# import os
# import random
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# from safetensors.torch import load_file  # ✅ safetensors 导入
# from dataset.sky import SkyDataset
# from pi3.models.pi3 import Pi3
#
# # ==============================
# # 可视化函数
# # ==============================
# def visualize_results(img, pred_mask, gt_mask, save_path):
#     """保存输入图、预测结果、GT到一张图中"""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     img = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#     pred_mask = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
#     gt_mask = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
#
#     concat = np.concatenate([
#         img,
#         np.stack([pred_mask]*3, axis=-1),
#         np.stack([gt_mask]*3, axis=-1)
#     ], axis=1)
#     Image.fromarray(concat).save(save_path)
#
# # ==============================
# # 测试入口
# # ==============================
# @torch.no_grad()
# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--val_dir', type=str, default='/home/liuwei/mnt/instant_vggt_dataset/mask_train', help='验证集路径')
#     parser.add_argument('--ckpt', type=str, required=True, help='模型权重路径 (.safetensors)')
#     parser.add_argument('--save_dir', type=str, default='results')
#     parser.add_argument('--num_samples', type=int, default=10)
#     parser.add_argument('--img_height', type=int, default=256)
#     parser.add_argument('--img_width', type=int, default=496)
#     args = parser.parse_args()
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # ==============================
#     # 加载数据集
#     # ==============================
#     dataset = SkyDataset(
#         root_dir=args.val_dir,
#         split='val',
#     )
#     val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
#
#     # ==============================
#     # 加载模型
#     # ==============================
#     model = Pi3().to(device)
#     print(f"🔹 Loading safetensors checkpoint: {args.ckpt}")
#     ckpt = load_file(args.ckpt, device=device)  # ✅ 使用 safetensors 加载
#     # safetensors 文件一般直接存储 state_dict，无需再 ckpt['model']
#     if "model" in ckpt:
#         ckpt = ckpt["model"]
#     model.load_state_dict(ckpt, strict=False)
#     model.eval()
#
#     # ==============================
#     # 随机选10张
#     # ==============================
#     total_samples = len(dataset)
#     indices = random.sample(range(total_samples), args.num_samples)
#     print(f"Visualizing {args.num_samples} samples ...")
#
#     for i, idx in enumerate(tqdm(indices)):
#         ret = dataset[idx]
#         img = ret['image']
#         gt_mask = ret['depth']
#         img = img.unsqueeze(0).to(device)
#         img = img.unsqueeze(1).to(device)
#         gt_mask = gt_mask.to(device)
#
#         # 前向推理
#         pred = model(img, train_sky=True)
#         if isinstance(pred, dict) and 'sky' in pred:
#             pred_mask = pred['sky']
#         else:
#             pred_mask = pred
#         pred_mask = (pred_mask > 0.5).float()
#
#         save_path = os.path.join(args.save_dir, f'sample_{i:02d}.png')
#         visualize_results(img[0][0], pred_mask[0], gt_mask, save_path)
#
#     print(f"✅ Results saved to {args.save_dir}")
#
# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
import os
import glob
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from tqdm import tqdm
from pi3.models.pi3 import Pi3

# ==============================
# 可视化（使用原图尺寸）
# ==============================
def visualize_results(orig_img, pred_mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    orig = np.array(orig_img)  # H,W,3 uint8
    pred_mask = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    concat = np.concatenate([
        orig,
        np.stack([pred_mask]*3, axis=-1),
    ], axis=1)

    Image.fromarray(concat).save(save_path)

# ==============================
# 图片读取工具
# ==============================
def load_image(path, img_h, img_w):
    """返回 transform 后的张量 + 未缩放原图 (PIL)"""
    orig_img = Image.open(path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()
    ])
    return tf(orig_img), orig_img   # 返回两份

# ==============================
# 主函数
# ==============================
@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--img_height', type=int, default=392)
    parser.add_argument('--img_width', type=int, default=392)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_list = sorted(glob.glob(os.path.join(args.img_dir, "*.*")))
    print(f"🔹 Found {len(img_list)} images in {args.img_dir}")

    # ==============================
    # 加载模型
    # ==============================
    model = Pi3().to(device)

    print(f"🔹 Loading checkpoint: {args.ckpt}")
    ckpt = load_file(args.ckpt, device=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    # ==============================
    # 推理
    # ==============================
    for path in tqdm(img_list):
        # 加载图片（缩放版 + 原图）
        img_tensor, orig_img = load_image(path, args.img_height, args.img_width)
        H, W = orig_img.size[1], orig_img.size[0]  # PIL: (W, H)

        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # 模型输出
        pred = model({"img": img_tensor}, train_sky=True)

        pred = pred["sky"] if isinstance(pred, dict) and "sky" in pred else pred
        pred = (pred > 0.5).float()

        # ==========================
        # 🔥 关键修改：resize mask 回原图大小
        # ==========================
        pred = pred[0][0]
        if pred.dim() == 4:
            pred_4d = pred
        elif pred.dim() == 3:
            if pred.shape[0] in [1, 3]:  # (C,H,W)
                pred_4d = pred.unsqueeze(0)
            else:  # (H,W,C)
                pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
        elif pred.dim() == 2:  # (H,W)
            pred_4d = pred.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected pred dim: {pred.shape}")

        # 获取原图尺寸
        H, W = orig_img.size[1], orig_img.size[0]

        # 重新插值到原图大小
        pred_resized = torch.nn.functional.interpolate(
            pred_4d, size=(H, W), mode="nearest"
        )

        # 保存 mask（单独）
        mask_save_path = os.path.join(
            args.save_dir, f"{os.path.basename(path).split('.')[0]}.png"
        )
        Image.fromarray(
            (pred_resized.squeeze().cpu().numpy() * 255).astype(np.uint8)
        ).save(mask_save_path)

        # 保存可视化图
        viz_save_path = os.path.join(
            args.save_dir, f"{os.path.basename(path).split('.')[0]}_viz.png"
        )
        visualize_results(orig_img, pred_resized[0], viz_save_path)

    print(f"✅ Done! Results saved to: {args.save_dir}")

if __name__ == '__main__':
    main()

