from torch.utils.data import ConcatDataset, DataLoader
from lightning import LightningDataModule
from training.dataset.wild_generic_dataset import WildGenericDataset
from training.dataset.transforms import DynamicTransform
import math

class MultiDatasetModule(LightningDataModule):

    def __init__(self, roots, split="train",
                 batch_size=4, num_views=3,
                 num_workers=8,
                 min_res=256, max_res=512,
                 curriculum_steps=20000):

        super().__init__()
        self.roots = roots
        self.split = split
        self.batch_size = batch_size
        self.num_views = num_views
        self.num_workers = num_workers

        # 动态增强
        self.transform = DynamicTransform(
            base_res=min_res,
            max_res=max_res,
            enable_crop=True
        )

        self.min_res = min_res
        self.max_res = max_res
        self.curriculum_steps = curriculum_steps

        self.current_train_step = 0

    # Lightning 每个 batch 前都会调用这个 hook
    def on_train_batch_start(self, batch, batch_idx):
        """动态调整分辨率 (curriculum)"""
        progress = min(1.0, self.current_train_step / self.curriculum_steps)
        res = int(self.min_res + (self.max_res - self.min_res) * progress)

        self.transform.set_resolution(res)
        self.current_train_step += 1

    def setup(self, stage=None):
        datasets = []
        for r in self.roots:
            ds = WildGenericDataset(
                root=r,
                split=self.split,
                num_views=self.num_views,
                transform=self.transform    # 加入 transform
            )
            datasets.append(ds)

        self.dataset = ConcatDataset(datasets)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )