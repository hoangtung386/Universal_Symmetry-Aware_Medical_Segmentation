from .base import BaseBottleneck
from .symmetry import SymmetryAwareBottleneck
from .mamba import MambaBottleneckWrapper

def get_bottleneck(bottleneck_type: str, in_channels: int, **kwargs) -> BaseBottleneck:
    if bottleneck_type.lower() == 'symmetry':
        return SymmetryAwareBottleneck(in_channels=in_channels, **kwargs)
    elif bottleneck_type.lower() == 'mamba':
        return MambaBottleneckWrapper(in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown bottleneck type: {bottleneck_type}")

__all__ = ['BaseBottleneck', 'SymmetryAwareBottleneck', 'MambaBottleneckWrapper', 'get_bottleneck']
