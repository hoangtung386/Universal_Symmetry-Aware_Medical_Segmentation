import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Internal components
from models.components import EncoderBlock3D, AlignmentNetwork
from models.bottleneck import get_bottleneck
from models.decoder.hvt import HVTDecoder

# SOTA Components
try:
    from models.decoder.kan import KANDecoderHead
    SOTA_AVAILABLE = True
except ImportError:
    print("Warning: SOTA components not found in models/decoder/kan")
    SOTA_AVAILABLE = False


# ============================================================================
# SYMFORMER - Complete Architecture
# ============================================================================

class SymFormer(nn.Module):
    """
    Complete SymFormer Architecture
    
    Pipeline:
    1. Alignment Network (unchanged)
    2. 3D CNN Encoder
    3. Symmetry-Aware Bottleneck (NO Transformer!)
    4. HVT Decoder with k-Means clustering
    5. Multi-scale fusion
    """
    def __init__(self, in_channels=1, num_classes=2, T=1, input_size=(512, 512),
                 bottleneck_type='mamba', use_kan=True, use_conditioning=False, **kwargs):
        super().__init__()
        
        self.T = T
        self.use_conditioning = use_conditioning
        
        # Encoder
        self.alignment_net = AlignmentNetwork(input_size)
        
        self.enc1 = EncoderBlock3D(1, 64)
        self.enc2 = EncoderBlock3D(64, 128)
        self.enc3 = EncoderBlock3D(128, 256)
        self.enc4 = EncoderBlock3D(256, 512)
        
        # Symmetry-Aware Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)
        )

        # Instantiate Bottleneck dynamically
        self.bottleneck = get_bottleneck(bottleneck_type, in_channels=1024, **kwargs)
            
        # Clinical Conditioning is now handled via the ConditionedSymFormer wrapper,
        # so we don't instantiate it here.
        
        # HVT Decoder with k-Means
        self.decoder = HVTDecoder(num_classes=num_classes)
        
        if use_kan and SOTA_AVAILABLE:
             # Replace standard heads with KAN heads
             self.decoder.heads = nn.ModuleList([
                KANDecoderHead(512, num_classes, use_rational=True),
                KANDecoderHead(256, num_classes, use_rational=True),
                KANDecoderHead(128, num_classes, use_rational=True),
                KANDecoderHead(64, num_classes, use_rational=True),
            ])
             self.decoder.final = KANDecoderHead(64, num_classes, use_rational=True)
        
    def forward(self, x):
        """
        x: [B, C*T, H, W] - Usually C=1, T=1 -> [B, 1, H, W]
        For BraTS/CPAISD with T=3 (adjacent slices), x is [B, 3, H, W]
        """
        # 1. Alignment to Canonical Pose
        aligned_x, center_params = self.alignment_net(x)

        # Reshape for 3D Encoder: [B, C*T, H, W] -> [B, 1, T, H, W]
        B, CT, H, W = aligned_x.shape
        C = 1
        T_val = CT // C

        if CT > 1 and T_val > 1:
            x_3d = aligned_x.view(B, C, T_val, H, W)
        else:
            x_3d = aligned_x.unsqueeze(2)

        # 2. Hybrid Encoder (Local Features)
        # EncoderBlock3D returns (skip_features, pooled_features)
        s1, enc1 = self.enc1(x_3d)
        s2, enc2 = self.enc2(enc1)
        s3, enc3 = self.enc3(enc2)
        s4, enc4 = self.enc4(enc3)

        # 3. Symmetry-Aware Bottleneck
        bottleneck = self.bottleneck_conv(enc4)
        bottleneck = self.bottleneck(bottleneck)

        # Project 3D features back to 2D for decoder
        def to_2d(t):
            if t.dim() == 4:
                return t
            if t.shape[2] > 1:
                return t[:, :, t.shape[2] // 2, :, :]
            return t.squeeze(2)

        s1_2d = to_2d(s1)
        s2_2d = to_2d(s2)
        s3_2d = to_2d(s3)
        s4_2d = to_2d(s4)
        bneck_2d = to_2d(bottleneck)

        features = [s1_2d, s2_2d, s3_2d, s4_2d, bneck_2d]

        # 4. HVT Decoder
        predictions = self.decoder(features)

        final_pred = predictions[0]

        # 5. Inverse Transformation (eval only)
        if not self.training:
            final_pred, _ = self.alignment_net.inverse_transform(final_pred, center_params)

        return {
            'pred': final_pred,
            'align_params': center_params,
            'multiscale_preds': predictions[1:],
        }
