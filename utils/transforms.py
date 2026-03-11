import torch
import torch.nn as nn

def apply_aligned_mask(
    mask: torch.Tensor,
    center_params: dict,
    align_net: nn.Module,
    mode: str = 'nearest'
) -> torch.Tensor:
    """
    Applies spatial transformation to ground truth masks to match
    the aligned input images during training/evaluation.

    Args:
        mask (torch.Tensor): Original ground truth mask, shape (B, 1, H, W)
        center_params (dict): Dictionary from alignment network (theta, translation, etc.)
        align_net (nn.Module): The alignment network module
        mode (str): Interpolation mode ('nearest' for masks to preserve discrete labels)

    Returns:
        torch.Tensor: Aligned mask, same shape as input
    """
    # Cast to float for grid_sample
    mask_float = mask.float()

    # Ensure correct shape (B, C, H, W)
    if mask_float.dim() == 3:
        mask_float = mask_float.unsqueeze(1)

    # Apply transformation via align_net
    aligned_masks, _ = align_net.apply_transform(mask_float, center_params, mode=mode)

    # Cast back to original dtype (usually long for masks)
    return aligned_masks.to(mask.dtype)
