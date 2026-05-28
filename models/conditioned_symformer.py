import torch
import torch.nn as nn
from models.symformer import SymFormer
from models.layers.conditioning import ClinicalConditionEncoder, ConditionalCrossAttention


class ConditionedSymFormer(nn.Module):
    """
    Wrapper around SymFormer that handles Clinical Metadata Conditioning.
    """
    def __init__(self, in_channels=1, num_classes=3, T=1, embed_dim=256, bottleneck_dim=1024, **kwargs):
        super().__init__()

        self.T = T

        self.symformer = SymFormer(
            in_channels=in_channels,
            num_classes=num_classes,
            T=T,
            use_conditioning=True,
            **kwargs
        )

        self.condition_encoder = ClinicalConditionEncoder(embed_dim=embed_dim)
        self.bottleneck_condition = ConditionalCrossAttention(bottleneck_dim, embed_dim)

        self.symformer.use_conditioning = False

    def forward(self, x, metadata_dict=None):
        # 1. Alignment
        aligned_x, center_params = self.symformer.alignment_net(x)

        # Reshape for 3D Encoder (same as SymFormer.forward)
        B, CT, H, W = aligned_x.shape
        C = 1
        T_val = CT // C

        if CT > 1 and T_val > 1:
            x_3d = aligned_x.view(B, C, T_val, H, W)
        else:
            x_3d = aligned_x.unsqueeze(2)

        # 2. Encoder
        s1, enc1 = self.symformer.enc1(x_3d)
        s2, enc2 = self.symformer.enc2(enc1)
        s3, enc3 = self.symformer.enc3(enc2)
        s4, enc4 = self.symformer.enc4(enc3)

        # 3. Bottleneck
        bottleneck = self.symformer.bottleneck_conv(enc4)
        bottleneck = self.symformer.bottleneck(bottleneck)
        if bottleneck.dim() == 5 and bottleneck.shape[2] == 1:
            bottleneck = bottleneck.squeeze(2)

        # 4. CLINICAL CONDITIONING INJECTION
        if metadata_dict is not None:
            cond_embedding = self.condition_encoder(metadata_dict)
            bottleneck = self.bottleneck_condition(bottleneck, cond_embedding)

        # 5. Decoder — project 3D features to 2D
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
        predictions = self.symformer.decoder(features)

        final_pred = predictions[0]

        # 6. Inverse Alignment
        final_pred, _ = self.symformer.alignment_net.inverse_transform(final_pred, center_params)

        return {
            'pred': final_pred,
            'align_params': center_params,
            'multiscale_preds': predictions[1:],
        }
