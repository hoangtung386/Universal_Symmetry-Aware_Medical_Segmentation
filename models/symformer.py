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
    from models.layers.kan import KANDecoderHead
    SOTA_AVAILABLE = True
except ImportError:
    print("Warning: SOTA components not found in models/layers/")
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
        if x.dim() == 4:
             pass # Standard input
        
        # 1. Alignment to Canonical Pose
        # Output: aligned_x [B, 1, H, W], center_params (dict)
        aligned_x, center_params = self.alignment_net(x)
        
        # Reshape for 3D Encoder: [B, C*T, H, W] -> [B, 1, T, H, W]
        # Our encoder is 3D but we often use T=1 for 2D processing
        B, CT, H, W = aligned_x.shape
        C = 1 # Assume single channel processing per modality/slice sequence
        T_val = CT // C
        
        if CT > 1 and T_val > 1:
            # We have adjacent slices
            x_3d = aligned_x.view(B, C, T_val, H, W)
        else:
            # Add dummy T dimension for 3D convolutions
            x_3d = aligned_x.unsqueeze(2)  # [B, CT, 1, H, W]
        
        # 2. Hybrid Encoder (Local Features)
        enc1 = self.enc1(x_3d)  # [B, 64, T, H/2, W/2]
        enc2 = self.enc2(enc1)  # [B, 128, T, H/4, W/4]
        enc3 = self.enc3(enc2)  # [B, 256, T, H/8, W/8]
        enc4 = self.enc4(enc3)  # [B, 512, T, H/16, W/16]
        
        # 3. Symmetry-Aware Bottleneck (Global Context + Symmetry)
        bottleneck = self.bottleneck_conv(enc4) # [B, 1024, T, H/16, W/16]
        bottleneck = self.bottleneck(bottleneck) # [B, 1024, H/16, W/16] (T reduced to 1)
        
        # Project 3D encoder features back to 2D for decoder
        # Take center slice if T > 1, or just squeeze if T=1
        if enc1.shape[2] > 1:
            center_t = enc1.shape[2] // 2
            enc1_2d = enc1[:, :, center_t, :, :]
            enc2_2d = enc2[:, :, center_t, :, :]
            enc3_2d = enc3[:, :, center_t, :, :]
            enc4_2d = enc4[:, :, center_t, :, :]
        else:
            enc1_2d = enc1.squeeze(2)
            enc2_2d = enc2.squeeze(2)
            enc3_2d = enc3.squeeze(2)
            enc4_2d = enc4.squeeze(2)

        # Reverse list for HVT decoder [bottleneck, enc4, enc3, enc2, enc1]
        features = [enc1_2d, enc2_2d, enc3_2d, enc4_2d, bottleneck]

        # 4. HVT Decoder (Multi-scale Clustering & Decoding)
        predictions = self.decoder(features)

        # predictions = [Final(largest), Scale4, Scale3, Scale2, Scale1(smallest)]
        final_pred = predictions[0]

        # 5. Inverse Transformation (Critical for testing, but NOT used during training)
        # Note: During training, we ALIGN the mask to the prediction instead of
        # inverse-transforming the prediction to the mask. This avoids interpolation errors.
        if not self.training:
            # Only inverse transform during evaluation
            final_pred, _ = self.alignment_net.inverse_transform(final_pred, center_params)
            # You might want to inverse transform multiscale preds too if needed

        return {
            'pred': final_pred,
            'align_params': center_params,
            'multiscale_preds': predictions[1:] # Return for deep supervision
        }
