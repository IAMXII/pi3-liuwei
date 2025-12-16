import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque
from training.data.train import HyperSim_Multi
from training.utils.image import imread_cv2


class HyperSim_Eval(HyperSim_Multi):
    def __init__(self, *args, split='test', **kwargs):
        """
        HyperSim Evaluation class.
        Inherits from HyperSim_Multi to reuse caching and loading logic,
        but overrides sampling for deterministic evaluation.
        """
        super().__init__(*args, split=split, **kwargs)
        self.split = split
        self.is_metric = True
        self.video = True

    def __len__(self):
        # Override: Evaluate each sequence exactly once (remove the * 10 multiplier from train)
        return len(self.start_img_ids)

    def _fetch_views(self, idx, resolution, rng, num_views, *args, **kwargs):
        """
        Fetch a deterministic sequence of views for evaluation.
        """
        # 1. Determine the sequence
        # Note: In Eval, idx maps 1-to-1 with the sequence start_ids, no (idx // 10)
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        # 2. Deterministic Sampling
        # Instead of extract_view_sequence (which handles random shuffling/intervals),
        # we simply take the sequential frames starting from the sequence head.
        # Ensure we don't go out of bounds.
        total_frames = len(all_image_ids)
        if total_frames >= num_views:
            # Take the first 'num_views' frames
            image_idxs = all_image_ids[:num_views]
        else:
            # Fallback for short sequences (though HyperSim is usually long enough)
            # Cycle or take available
            image_idxs = all_image_ids
            print(
                f"[HyperSim Eval] Warning: Scene {scene_id} has fewer frames ({total_frames}) than requested ({num_views}).")

        # 3. Load Views (Logic copied and adapted from HyperSim_Multi)
        views = []
        # We set ordered_video=True because we are picking sequential frames
        ordered_video = True

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            # Load Data
            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, depth_path)).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            cam_file = np.load(osp.join(scene_dir, cam_path))
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            # Apply Crop and Resize
            rgb_image, depthmap, intrinsics = self._apply_crop_and_resize(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_poses=camera_pose.astype(np.float32),
                    camera_intrs=intrinsics.astype(np.float32),
                    dataset="hypersim",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=True,  # Explicitly True for Eval
                    is_video=True,  # Explicitly True for Eval
                    nvs_sample=False,  # Not NVS sampling during reconstruction eval
                    scale_norm=True,
                    cam_align=True
                )
            )

        assert len(views) > 0
        return views