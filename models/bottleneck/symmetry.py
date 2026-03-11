import torch
import torch.nn as nn
from .base import BaseBottleneck

class SymmetryAwareBottleneck(BaseBottleneck):
    """
    Bottleneck that explicitly models brain hemisphere symmetry
    NO Transformer - pure CNN with symmetry constraints
    """
    def __init__(self, in_channels: int = 1024, num_heads: int = 8):
        super().__init__(in_channels)

        # Dual-branch processing
        self.left_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

        self.right_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

        # Cross-hemisphere attention (simplified)
        self.cross_attention = nn.Conv3d(in_channels * 2, in_channels, 1)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]

        B, C, T, H, W = x.shape
        half_w = W // 2

        # Split into hemispheres
        left = x[..., :half_w]
        right = x[..., half_w:]

        # Flip right hemisphere to match left orientation
        right_flipped = torch.flip(right, dims=[-1])

        # Extract features
        feat_left = self.left_branch(left)
        feat_right = self.right_branch(right_flipped)

        # Cross-hemisphere attention
        concat_feats = torch.cat([feat_left, feat_right], dim=1)
        attention_weights = torch.sigmoid(self.cross_attention(concat_feats))

        # Apply attention and combine
        att_left = feat_left * attention_weights
        att_right = feat_right * attention_weights

        # Flip right back
        att_right_original = torch.flip(att_right, dims=[-1])

        # Reconstruct full brain
        reconstructed = torch.cat([att_left, att_right_original], dim=-1)

        # Global fusion
        out = self.fusion(torch.cat([x, reconstructed], dim=1))

        return out.squeeze(2) if out.shape[2] == 1 else out
