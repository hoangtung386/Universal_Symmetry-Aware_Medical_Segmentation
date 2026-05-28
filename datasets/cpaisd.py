import os
import numpy as np
import torch
from pathlib import Path
from .base import BaseDataset
import pydicom

class CPAISDDataset(BaseDataset):
    """
    Dataset loader cho CPAISD (Dataset APIS)
    """
    def __init__(self, dataset_root, split='train', T=1, 
                 use_hu_window=True, transform=None, config=None):
        super().__init__(dataset_root, split, T, transform)
        self.config = config
        self.use_hu_window = use_hu_window
        
        # Brain window parameters
        if config and hasattr(config, 'WINDOW_CENTER'):
            self.window_center = config.WINDOW_CENTER
            self.window_width = config.WINDOW_WIDTH
        else:
            self.window_center = 40
            self.window_width = 80
        
        # Filtering parameters
        self.skip_empty_slices = getattr(config, 'SKIP_EMPTY_SLICES', False) if config else False
        self.negative_sample_ratio = getattr(config, 'NEGATIVE_SAMPLE_RATIO', 0.2) if config else 0.2
        
        # Build sample index
        self.samples, self.filter_stats = self._build_index()

        print(f"\n{split.upper()} Dataset (CPAISD):")
        print(f"  Root: {self.dataset_root}")
        print(f"  Final dataset size: {len(self.samples)}")
    
    def _build_index(self):
        import random
        random.seed(42)

        samples = []
        filter_stats = {'total': 0, 'empty': 0, 'non_empty': 0, 'dropped_empty': 0}

        split_path = Path(self.dataset_root) / self.split
        if not split_path.exists():
            return [], filter_stats

        for study_dir in sorted(split_path.iterdir()):
            if not study_dir.is_dir():
                continue

            slice_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
            num_slices = len(slice_dirs)

            for idx, slice_dir in enumerate(slice_dirs):
                if not (slice_dir / "image.npz").exists() or not (slice_dir / "mask.npz").exists():
                    continue

                filter_stats['total'] += 1

                try:
                    mask_data = np.load(slice_dir / "mask.npz")
                    mask_key = list(mask_data.keys())[0]
                    mask = mask_data[mask_key]
                except Exception:
                    continue

                has_core = np.any(mask == 1)
                has_penumbra = np.any(mask == 2)
                is_bg_only = not has_core and not has_penumbra

                sample = {
                    'study': study_dir.name,
                    'slice_idx': idx,
                    'slice_path': slice_dir,
                    'all_slices': slice_dirs,
                    'num_slices': num_slices,
                }

                if is_bg_only:
                    filter_stats['empty'] += 1
                    if random.random() < self.negative_sample_ratio:
                        samples.append(sample)
                    else:
                        filter_stats['dropped_empty'] += 1
                else:
                    filter_stats['non_empty'] += 1
                    samples.append(sample)
                    if has_core:
                        for _ in range(3):
                            samples.append(sample)

        random.shuffle(samples)
        print(f"  Core-oversampled dataset: {len(samples)} samples "
              f"(non-empty: {filter_stats['non_empty']}, "
              f"bg kept: {filter_stats['total'] - filter_stats['dropped_empty'] - filter_stats['non_empty']})")
        return samples, filter_stats

    def _load_slice(self, slice_path, use_raw_dicom=False):
        if use_raw_dicom and (slice_path / "raw.dcm").exists():
            dcm = pydicom.dcmread(slice_path / "raw.dcm")
            pixels = dcm.pixel_array.astype(np.float32)
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            hu = pixels * slope + intercept
            
            if self.use_hu_window:
                lower = self.window_center - (self.window_width / 2)
                upper = self.window_center + (self.window_width / 2)
                hu = np.clip(hu, lower, upper)
                hu = (hu - lower) / (upper - lower)
            else:
                hu = (hu + 1024) / (3072 + 1024)
                hu = np.clip(hu, 0, 1)
            image = hu
        else:
            image_npz = np.load(slice_path / "image.npz")
            image = image_npz['image'].astype(np.float32)
            
            if image.max() > 1.0 or image.min() < 0.0:
                if self.use_hu_window:
                    lower = self.window_center - (self.window_width / 2)
                    upper = self.window_center + (self.window_width / 2)
                    image = np.clip(image, lower, upper)
                    image = (image - lower) / (upper - lower)
                else:
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        mask_npz = np.load(slice_path / "mask.npz")
        mask = mask_npz['mask'].astype(np.int64)
        
        return image, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center_idx = sample['slice_idx']
        all_slices = sample['all_slices']
        num_slices = sample['num_slices']
        
        images = []
        for offset in range(-self.T, self.T + 1):
            slice_idx = center_idx + offset
            slice_idx = max(0, min(num_slices - 1, slice_idx))
            image, _ = self._load_slice(all_slices[slice_idx])
            images.append(image)
        
        center_slice_path = all_slices[center_idx]
        _, mask = self._load_slice(center_slice_path)
        metadata = self._load_metadata(center_slice_path)
        
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(mask).long()
        
        return images, mask, metadata
