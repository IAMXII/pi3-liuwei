import os
from torch.utils.data import ConcatDataset, DataLoader
from lightning import LightningDataModule
from training.data.wild_generic_dataset import WildGenericDataset


class MultiDatasetModule(LightningDataModule):

    def __init__(self, roots, split="train",
                 batch_size=4, num_views=3,
                 num_workers=8):

        super().__init__()
        self.roots = roots
        self.split = split
        self.batch_size = batch_size
        self.num_views = num_views
        self.num_workers = num_workers


    def setup(self, stage=None):
        datasets = []
        for r in self.roots:
            ds = WildGenericDataset(
                root=r,
                split=self.split,
                num_views=self.num_views,
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