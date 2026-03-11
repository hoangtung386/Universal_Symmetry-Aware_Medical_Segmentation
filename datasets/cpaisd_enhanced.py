import os
import numpy as np
import torch
from pathlib import Path
from .cpaisd import CPAISDDataset

class CPAISDEnhancedDataset(CPAISDDataset):
    """
    Enhanced version of CPAISD Dataset that returns 3-channel input:
    Channel 0: Stroke Window (W:40, L:40) - Optimized for early infarct
    Channel 1: Brain Window (W:80, L:40) - Standard brain tissue
    Channel 2: Subdural Window (W:200, L:40) or Bone Window
    """
    def __init__(self, dataset_root, split='train', T=1, transform=None, config=None):
        # We enforce T=0 and 3 channels in this specific class
        super().__init__(dataset_root, split, T=0, use_hu_window=True, transform=transform, config=config)
        self.config = config
        print("  Multi-channel enhancement: ENABLED (Stroke, Brain, Context)")

    def _apply_window(self, hu_array, center, width):
        """Apply CT windowing and normalize to [0,1]"""
        lower = center - (width / 2)
        upper = center + (width / 2)
        windowed = np.clip(hu_array, lower, upper)
        normalized = (windowed - lower) / (upper - lower)
        return normalized

    def __getitem__(self, idx):
        sample = self.samples[idx]
        slice_dir = sample['slice_path']
        
        # Load Raw DICOM if available, else fallback to NPZ approximation
        # Ideally, we need the raw HU values for proper multi-windowing
        
        raw_dcm_path = slice_dir / "raw.dcm"
        image_npz_path = slice_dir / "image.npz"
        
        # Load mask
        mask_npz = np.load(slice_dir / "mask.npz")
        mask = mask_npz['mask'].astype(np.int64)

        # Determine HU values
        if raw_dcm_path.exists():
            import pydicom
            dcm = pydicom.dcmread(raw_dcm_path)
            pixels = dcm.pixel_array.astype(np.float32)
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            hu = pixels * slope + intercept
        else:
            # Fallback: assume NPZ contains HU, or approximate
            image_npz = np.load(image_npz_path)
            hu = image_npz['image'].astype(np.float32)
            # If already normalized to [0,1], reverse engineering is hard
            if hu.max() <= 1.0 and hu.min() >= 0.0:
                 # This is a hack - if data was saved as [0,1] from Stroke Window (40/40)
                 # We try to recover it, but it's lossy outside the window
                 hu = hu * 40 + 20

        # Create 3 channels
        c1 = self._apply_window(hu, center=40, width=40)  # Stroke
        c2 = self._apply_window(hu, center=40, width=80)  # Brain
        c3 = self._apply_window(hu, center=40, width=200) # Subdural/Context
        
        # Stack channels (C, H, W)
        images = np.stack([c1, c2, c3], axis=0)

        metadata = self._load_metadata(slice_dir)

        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(mask).long()
        
        return images, mask, metadata
