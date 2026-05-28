import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.decoder.hvt import DecoderBlock, kMaXBlock


class EfficientKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order)
        )
        self.register_buffer(
            'grid',
            torch.linspace(-1, 1, grid_size + 1).expand(in_features, -1)
        )
        self.spline_order = spline_order
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(self.spline_weight, gain=0.1)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid.unsqueeze(0)
        bases = torch.zeros(
            x.size(0), self.in_features, self.grid_size + self.spline_order,
            device=x.device
        )
        for i in range(self.grid_size):
            mask = (x >= grid[:, :, i:i+1]) & (x < grid[:, :, i+1:i+2])
            bases[:, :, i] = mask.float().squeeze(-1)
        return bases

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        bases = self.b_splines(x)
        spline_output = torch.einsum(
            'oig,big->bo',
            self.spline_weight,
            bases
        )
        return base_output + spline_output


class RationalKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.P_coef = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1) * 0.1
        )
        self.Q_coef = nn.Parameter(
            torch.randn(out_features, in_features, degree) * 0.1
        )
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.base_bias)
        with torch.no_grad():
            self.P_coef[:, :, 0] = 0.0
            self.Q_coef[:, :, 0] = 1.0

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        x_powers = torch.stack([x ** i for i in range(self.degree + 1)], dim=-1)
        P = torch.einsum('oip,bip->bio', self.P_coef, x_powers)
        Q = 1.0 + torch.einsum('oip,bip->bio', self.Q_coef, x_powers[:, :, 1:])
        rational = P / (Q + 1e-6)
        rational_output = rational.sum(dim=1)

        base_output = F.linear(x, self.base_weight, self.base_bias)
        output = base_output + rational_output
        output = output.view(*original_shape[:-1], self.out_features)

        return output


class KANDecoderHead(nn.Module):
    def __init__(self, in_channels, num_classes, use_rational=True, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(in_channels // 2, num_classes * 4)

        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)

        KANLayer = RationalKANLayer if use_rational else EfficientKANLayer
        self.kan1 = KANLayer(in_channels, hidden_dim)
        self.kan2 = KANLayer(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.norm(self.spatial_conv(x))
        feat = feat.permute(0, 2, 3, 1)
        feat = self.kan1(feat)
        feat = self.dropout(feat)
        output = self.kan2(feat)
        output = output.permute(0, 3, 1, 2)
        return output


class KANHVTDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 256, 512, 1024],
                 num_classes=2, num_heads=8, use_kan_heads=True):
        super().__init__()

        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.kmax_blocks = nn.ModuleList([
            kMaXBlock(512, num_classes, num_heads),
            kMaXBlock(256, num_classes, num_heads),
            kMaXBlock(128, num_classes, num_heads),
            kMaXBlock(64, num_classes, num_heads),
        ])

        if use_kan_heads:
            self.heads = nn.ModuleList([
                KANDecoderHead(512, num_classes, use_rational=True),
                KANDecoderHead(256, num_classes, use_rational=True),
                KANDecoderHead(128, num_classes, use_rational=True),
                KANDecoderHead(64, num_classes, use_rational=True),
            ])
            self.final = KANDecoderHead(64, num_classes, use_rational=True)
        else:
            self.heads = nn.ModuleList([
                nn.Conv2d(num_classes, num_classes, 1) for _ in range(4)
            ])
            self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, bottleneck, skip_connections):
        s1, s2, s3, s4 = skip_connections

        def get_mid_slice(x):
            return x[:, :, x.size(2)//2, :, :]

        x = get_mid_slice(bottleneck)
        cluster_outputs = []

        x = self.dec4(x, get_mid_slice(s4))
        c4, _ = self.kmax_blocks[0](x)
        cluster_outputs.append(self.heads[0](c4))

        x = self.dec3(x, get_mid_slice(s3))
        c3, _ = self.kmax_blocks[1](x)
        cluster_outputs.append(self.heads[1](c3))

        x = self.dec2(x, get_mid_slice(s2))
        c2, _ = self.kmax_blocks[2](x)
        cluster_outputs.append(self.heads[2](c2))

        x = self.dec1(x, get_mid_slice(s1))
        c1, _ = self.kmax_blocks[3](x)
        cluster_outputs.append(self.heads[3](c1))

        final_output = self.final(x)
        return final_output, cluster_outputs
