import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SkyDataset(Dataset):
    """
    用于天空识别（淹没检测）的数据集类。
    数据结构:
        root/
            train/
                imgs/
                gts/
            val/
                imgs/
                gts/
    """

    def __init__(self, root_dir, split='train', img_size=(512, 512)):
        """
        Args:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'val'
            img_size (tuple): 统一的图像尺寸 (H, W)
        """
        super().__init__()
        assert split in ['train', 'val'], "split 必须是 'train' 或 'val'"

        self.img_dir = os.path.join(root_dir, split, "imgs")
        self.gt_dir = os.path.join(root_dir, split, "gts")
        self.img_size = img_size
        self.split = split

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        self.gt_files = sorted([
            f for f in os.listdir(self.gt_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        assert len(self.img_files) == len(self.gt_files), \
            f"图片数量与标签数量不匹配: {len(self.img_files)} vs {len(self.gt_files)}"

        # 图像转换
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 掩膜转换（使用最近邻插值，防止混合灰度）
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 读取图像与标签
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        # 应用transform（统一尺寸）
        img = self.img_transform(img)
        mask = self.mask_transform(gt)

        # 二值化（天空=0，淹没=1）
        mask = (mask > 0.01).float()

        return dict(image=img, depth=mask)
