import os
import os.path as osp
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
from training.utils.image import imread_cv2


class WildGenericDataset(Dataset):
    """
    通用版：支持任意 processed_xxx 数据集，只要目录结构一致即可
    """

    def __init__(self, root, split="train", num_views=3, max_interval=4):
        self.root = root
        self.split = split
        self.num_views = num_views
        self.max_interval = max_interval

        # load split json
        split_file = osp.join(root, f"selected_seqs_{split}.json")
        with open(split_file, "r") as f:
            self.seqs = json.load(f)

        self.seq_frames = []
        self.start_ids = []

        for seq in self.seqs:
            seq_dir = osp.join(root, seq)
            rgb_dir = osp.join(seq_dir, "rgb")

            frames = sorted([f.replace(".jpg","")
                    for f in os.listdir(rgb_dir) if f.endswith(".jpg")])

            if len(frames) < self.num_views:
                continue

            full_ids = [osp.join(seq, fid) for fid in frames]
            self.seq_frames.append(full_ids)

            # 可作为 start-index 的帧
            for sid in range(len(full_ids) - self.num_views + 1):
                self.start_ids.append((len(self.seq_frames)-1, sid))


    def __len__(self):
        return len(self.start_ids)


    def _load_frame(self, frame_id):
        # frame_id:  "category/seq_0001/00010"
        seq_dir, frame = osp.split(frame_id)
        p = osp.join(self.root, seq_dir)

        rgb = imread_cv2(osp.join(p, "rgb", frame + ".jpg"))
        depth = cv2.imread(osp.join(p, "depth", frame + ".png"), -1)
        mask = cv2.imread(osp.join(p, "masks", frame + ".png"), -1)

        meta = np.load(osp.join(p, "metadata", frame + ".npz"))
        intr = meta["intrinsics"].astype(np.float32)
        pose = meta["pose"].astype(np.float32)

        return rgb, depth, mask, intr, pose


    def __getitem__(self, idx):
        seq_id, sid = self.start_ids[idx]
        frames = self.seq_frames[seq_id]

        # sample multiple views
        selected = frames[sid : sid + self.num_views]

        views = []
        for fid in selected:
            rgb, depth, mask, intr, pose = self._load_frame(fid)
            views.append(dict(
                img=rgb,
                depthmap=depth,
                mask=mask,
                camera_intrs=intr,
                camera_poses=pose,
                label=fid,
                dataset=os.path.basename(self.root),
                is_metric=True,
                nvs_sample=True
            ))

        return views