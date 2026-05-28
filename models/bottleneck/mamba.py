import torch
import torch.nn as nn
from .base import BaseBottleneck

class MambaBottleneckWrapper(BaseBottleneck):
    """
    Wrapper around the MambaBottleneck to ensure it conforms to the
    BaseBottleneck interface.
    """
    def __init__(self, in_channels: int = 1024, mamba_depth: int = 4):
        super().__init__(in_channels)

        try:
            from models.layers.mamba import MambaBottleneck
            self.mamba = MambaBottleneck(channels=in_channels, depth=mamba_depth)
        except ImportError:
            print("Warning: Mamba layer not found. Falling back to SymmetryAwareBottleneck.")
            from .symmetry import SymmetryAwareBottleneck
            self.mamba = SymmetryAwareBottleneck(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.mamba(x)
        if isinstance(result, tuple):
            return result[0]
        return result
