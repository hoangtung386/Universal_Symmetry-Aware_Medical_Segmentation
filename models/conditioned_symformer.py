import torch
import torch.nn as nn
from models.symformer import SymFormer
from models.layers.conditioning import ClinicalConditionEncoder, ConditionalCrossAttention

class ConditionedSymFormer(nn.Module):
    """
    Wrapper around SymFormer that handles Clinical Metadata Conditioning
    Allows easy toggle between conditioned and standard training
    """
    def __init__(self, in_channels=1, num_classes=3, T=1, embed_dim=256, bottleneck_dim=1024, **kwargs):
        super().__init__()

        # 1. Base Model
        self.symformer = SymFormer(
            in_channels=in_channels,
            num_classes=num_classes,
            T=T,
            use_conditioning=True, # Force conditioning on
            **kwargs
        )

        # 2. Conditioning Components (moved to wrapper)
        self.condition_encoder = ClinicalConditionEncoder(embed_dim=embed_dim)
        self.bottleneck_condition = ConditionalCrossAttention(bottleneck_dim, embed_dim)

        # Disable internal conditioning in SymFormer to prevent double-application
        self.symformer.use_conditioning = False

    def forward(self, x, metadata_dict=None):
        """
        Args:
            x: Input images [B, C*T, H, W]
            metadata_dict: Dictionary containing clinical data
                           Must have: 'nihss', 'age', 'sex', 'time', 'dsa'
        """
        # 1. Alignment
        aligned_x, center_params = self.symformer.alignment_net(x)

        # 2. Encoder
        enc1 = self.symformer.enc1(aligned_x)
        enc2 = self.symformer.enc2(enc1)
        enc3 = self.symformer.enc3(enc2)
        enc4 = self.symformer.enc4(enc3)

        # 3. Bottleneck
        bottleneck = self.symformer.bottleneck_conv(enc4)
        bottleneck = self.symformer.bottleneck(bottleneck)

        # ==========================================================
        # 4. CLINICAL CONDITIONING INJECTION
        # ==========================================================
        if metadata_dict is not None:
            # Encode metadata
            cond_embedding = self.condition_encoder(metadata_dict)

            # Inject into bottleneck features
            bottleneck = self.bottleneck_condition(bottleneck, cond_embedding)
        # ==========================================================

        # 5. Decoder
        # Reverse list for HVT decoder [bottleneck, enc4, enc3, enc2, enc1]
        features = [enc1, enc2, enc3, enc4, bottleneck]

        # Get multiscale predictions
        predictions = self.symformer.decoder(features)

        # First prediction is full resolution
        final_pred = predictions[0]

        # 6. Inverse Alignment
        final_pred, _ = self.symformer.alignment_net.inverse_transform(final_pred, center_params)

        return {
            'pred': final_pred,
            'align_params': center_params,
            'multiscale_preds': predictions[1:] # Return for deep supervision
        }
