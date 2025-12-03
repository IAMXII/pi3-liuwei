import cv2
import numpy as np
import random

class DynamicTransform:
    """
    动态训练增强：分辨率、随机 crop、随机缩放
    """
    def __init__(self, base_res=256, max_res=512, enable_crop=True):
        self.base_res = base_res
        self.max_res = max_res
        self.enable_crop = enable_crop

    def set_resolution(self, res):
        """DataModule 会动态调用它改变分辨率"""
        self.base_res = res

    def __call__(self, rgb, depth, mask):
        h, w = rgb.shape[:2]

        # 1) 随机 scale
        scale = random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale), int(w * scale)

        rgb = cv2.resize(rgb, (new_w, new_h))
        depth = cv2.resize(depth, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h))

        # 2) 随机 crop (保证裁剪能满足 base_res)
        if self.enable_crop:
            if new_h > self.base_res and new_w > self.base_res:
                sy = random.randint(0, new_h - self.base_res)
                sx = random.randint(0, new_w - self.base_res)

                rgb = rgb[sy:sy+self.base_res, sx:sx+self.base_res]
                depth = depth[sy:sy+self.base_res, sx:sx+self.base_res]
                mask = mask[sy:sy+self.base_res, sx:sx+self.base_res]
        else:
            # 3) 否则直接 resize 到 base_res
            rgb = cv2.resize(rgb, (self.base_res, self.base_res))
            depth = cv2.resize(depth, (self.base_res, self.base_res))
            mask = cv2.resize(mask, (self.base_res, self.base_res))

        return rgb, depth, mask