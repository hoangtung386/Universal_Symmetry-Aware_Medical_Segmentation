import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class kMaXBlock(nn.Module):
    """
    k-Means Mask Transformer Block (kMaXU)
    Performs clustering between pixel features and cluster centers
    """
    def __init__(self, in_channels: int, num_heads: int = 8, embed_dim: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Projection for keys/values
        self.proj_kv = nn.Conv2d(in_channels, embed_dim * 2, 1)

        # Projection for queries (cluster centers)
        self.proj_q = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Cluster update gate
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, H, W] - Pixel features
        centers: [B, N, C] - Cluster centers (queries)
        """
        B, C, H, W = x.shape
        N = centers.shape[1]

        # Extract Keys and Values from pixels
        kv = self.proj_kv(x).flatten(2).transpose(1, 2)  # [B, H*W, 2D]
        k, v = kv.chunk(2, dim=-1)  # [B, H*W, D] each

        # Reshape for multi-head attention
        head_dim = self.embed_dim // self.num_heads
        k = k.view(B, H*W, self.num_heads, head_dim).transpose(1, 2)  # [B, heads, H*W, head_dim]
        v = v.view(B, H*W, self.num_heads, head_dim).transpose(1, 2)  # [B, heads, H*W, head_dim]

        # Project Queries from centers
        q = self.proj_q(centers)  # [B, N, D]
        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)    # [B, heads, N, head_dim]

        # Cluster Assignment (Affinity Matrix) -> k-means step
        # Distance calculation (L2 norm) instead of dot product for clustering
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k)

        # kMaX key step: Argmax assignment instead of softmax
        # Softmax with high temperature approximates hard assignment
        scale = head_dim ** -0.5
        attn = (attn * scale).softmax(dim=-1)  # Soft assignment [B, heads, N, H*W]

        # Hard assignment for masking
        mask = torch.argmax(attn.mean(dim=1), dim=1)  # [B, H*W] -> each pixel belongs to 1 center

        # Update Cluster Centers -> k-means centroid update
        # Weighted sum of values based on assignment
        centers_updated = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        centers_updated = centers_updated.transpose(1, 2).reshape(B, N, self.embed_dim)

        # Residual update and FFN
        centers = centers + self.gate * centers_updated
        centers = self.norm(centers)
        centers = centers + self.ffn(centers)

        return centers, mask.view(B, H, W)

class HVTDecoder(nn.Module):
    """
    Hybrid Vision Transformer (HVT) Decoder
    Combines hierarchical kMaX clustering with standard CNN decoder
    """
    def __init__(self, num_classes: int = 3, in_channels: List[int] = [1024, 512, 256, 128, 64]):
        super().__init__()

        self.num_classes = num_classes
        embed_dim = 256
        num_clusters = 64  # Total number of clusters

        # Initial cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(1, num_clusters, embed_dim))

        # kMaX Blocks for multi-scale clustering
        self.kmax_blocks = nn.ModuleList([
            kMaXBlock(in_channels[0], embed_dim=embed_dim),
            kMaXBlock(in_channels[1], embed_dim=embed_dim),
            kMaXBlock(in_channels[2], embed_dim=embed_dim),
            kMaXBlock(in_channels[3], embed_dim=embed_dim)
        ])

        # Standard Decoder blocks for feature upsampling
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(in_channels[0], in_channels[1]),  # 1024 -> 512
            DecoderBlock(in_channels[1], in_channels[2]),  # 512 -> 256
            DecoderBlock(in_channels[2], in_channels[3]),  # 256 -> 128
            DecoderBlock(in_channels[3], in_channels[4])   # 128 -> 64
        ])

        # Classification heads
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels[1], num_classes, 1),
            nn.Conv2d(in_channels[2], num_classes, 1),
            nn.Conv2d(in_channels[3], num_classes, 1),
            nn.Conv2d(in_channels[4], num_classes, 1)
        ])

        self.final = nn.Conv2d(in_channels[4], num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        features: [enc1, enc2, enc3, enc4, bottleneck]
                  [64,   128,  256,  512,  1024]
        Returns:
            List of predictions at multiple scales
        """
        x = features[-1]  # Bottleneck features
        skips = features[:-1][::-1]  # Reverse order: [enc4, enc3, enc2, enc1]

        B = x.shape[0]
        centers = self.cluster_centers.expand(B, -1, -1)

        predictions = []

        # Hierarchical decoding
        for i, (kmax, dec, skip, head) in enumerate(zip(
            self.kmax_blocks, self.dec_blocks, skips, self.heads
        )):
            # 1. Update cluster centers and get cluster masks at current scale
            centers, mask = kmax(x, centers)

            # 2. Upsample features and fuse with skip connection
            x = dec(x, skip)

            # 3. Generate predictions at this scale
            pred = head(x)
            predictions.append(pred)

        # Final refinement
        final_pred = self.final(x)
        predictions.append(final_pred)

        # Return all predictions (multiscale supervision)
        # Reverse to get [Final(largest), Scale4, Scale3, Scale2, Scale1(smallest)]
        return predictions[::-1]

class DecoderBlock(nn.Module):
    """Standard 2D decoder block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
