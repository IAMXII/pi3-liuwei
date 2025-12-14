import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


class CameraHead(nn.Module):
    """
    Camera Head predicting 6D Rotation and standard 3D Translation (T).

    Changes:
    1. Rotation: Replaced 9D + SVD with 6D continuous rotation representation.
    2. Translation: Reverted to predicting standard 3D translation (T) without
       explicit scale separation (as requested).
    """

    def __init__(self, dim=512):
        super().__init__()
        output_dim = dim
        # 使用深拷贝确保 ResConvBlock 实例独立
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim))
                                       for _ in range(2)])

        # Global Average Pooling to reduce feature dimension
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Shared MLPs
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

        # --- V3 Prediction Heads (6D Rot + Standard T) ---
        # 1. 6D Rotation (r1, r2 vectors)
        self.fc_rot_6d = nn.Linear(output_dim, 6)
        # 2. Standard 3D Translation (T)
        self.fc_t = nn.Linear(output_dim, 3)

    # =========================================================================
    # --- Utility Functions for 6D Rotation ---
    # =========================================================================
    @staticmethod
    def _6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D continuous rotation representation to 3x3 rotation matrix.
        (Based on Zhou et al., 2018: On the Continuity of Rotation Representations)
        Input: [B, 6] tensor.
        Output: [B, 3, 3] tensor.
        """
        a1, a2 = d6[..., :3], d6[..., 3:]

        # Orthogonalize the first vector
        b1 = F.normalize(a1, p=2, dim=-1)

        # Get the second vector orthogonal to the first
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, p=2, dim=-1)

        # Get the third vector (cross product)
        b3 = torch.cross(b1, b2, dim=-1)

        # Stack them to form the rotation matrix [B, 3, 3]
        return torch.stack((b1, b2, b3), dim=-1)

    # =========================================================================
    # --- Pose Construction ---
    # =========================================================================
    def convert_pose_to_4x4(self, B, out_r_mat, out_t, device):
        """
        Assembles the 4x4 homogenous pose matrix.
        Args:
            out_r_mat: [B, 3, 3] Rotation matrix.
            out_t: [B, 3] Translation vector (full T with scale).
        """
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r_mat  # [B, 3, 3]
        pose[:, :3, 3] = out_t  # [B, 3]
        pose[:, 3, 3] = 1.
        return pose

    # =========================================================================
    # --- Forward Pass ---
    # =========================================================================
    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        # 1. Feature Processing (ResConv Blocks)
        for i in range(2):
            feat = self.res_conv[i](feat)

        # 2. Global Pooling (Reshape [B, hw, c] -> [B, c, h, w] -> [B, c])
        # Reshape to NCHW format: [BN, hw, c] -> [BN, c, hw] -> [BN, c, H, W]
        feat_pooled = feat.permute(0, 2, 1).reshape(BN, c, patch_h, patch_w).contiguous()
        feat_pooled = self.avgpool(feat_pooled)
        feat_pooled = feat_pooled.view(feat_pooled.size(0), -1)  # [B, D]

        # 3. Shared MLPs
        feat_head = self.more_mlps(feat_pooled)  # [B, D_]

        # 使用 autocast 保持 float 精度进行预测 (与原始代码保持一致)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # --- V3 Predictions ---
            # R: 6D Rotation -> 3x3 Matrix
            out_r_6d = self.fc_rot_6d(feat_head.float())  # [B, 6]
            out_r_mat = self._6d_to_matrix(out_r_6d)  # [B, 3, 3]

            # T: Standard 3D Translation
            out_t = self.fc_t(feat_head.float())  # [B, 3]

            # Assemble the final pose (R, T)
            pose = self.convert_pose_to_4x4(BN, out_r_mat, out_t, feat.device)

        # Return the 4x4 pose matrix containing the predicted Rotation and full Translation (T).
        return pose