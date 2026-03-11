import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import os
import glob
from .base import BaseDataset

class BraTSDataset(BaseDataset):
    def __init__(self, dataset_root, split='train', T=1, transform=None,
                 modality='t2f', label_mode='native', use_cache=True, config=None, val_ratio=0.2):
        super().__init__(dataset_root, split, T, transform)
        
        self.dataset_root = Path(dataset_root)
        self.modality = modality
        self.config = config
        self.label_mode = label_mode
        self.use_cache = use_cache
        
        if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData').exists():
            self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
        else:
            self.data_dir = self.dataset_root
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"BraTS TrainingData directory not found: {self.data_dir}")
        
        self.volume_cache = {} if use_cache else None
        self.max_cache_size = 3
        
        self.skip_empty_slices = getattr(config, 'SKIP_EMPTY_SLICES', False) if config else False
        self.negative_sample_ratio = getattr(config, 'NEGATIVE_SAMPLE_RATIO', 0.2) if config else 0.2
        
        self._split_train_val(val_ratio)
        self.samples, self.filter_stats = self._build_index()
        self.norm_stats = self._load_norm_stats()
        
        print(f"\n{split.upper()} Dataset (BraTS - Improved):")
        print(f"  Root: {self.data_dir}")
        print(f"  Modality: {self.modality}")
        print(f"  Label mode: {self.label_mode}")
        if self.skip_empty_slices:
            print(f"  Empty slice filtering: ENABLED")
            print(f"  Final dataset size: {len(self.samples)}")
        else:
            print(f"  Total slices: {len(self.samples)}")

    def _split_train_val(self, val_ratio, seed=42):
        all_studies = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        np.random.seed(seed)
        np.random.shuffle(all_studies)

        num_val = int(len(all_studies) * val_ratio)

        if self.split == 'train':
            self.study_ids = all_studies[num_val:]
        else:
            self.study_ids = all_studies[:num_val]

    def _load_norm_stats(self):
        if self.config and hasattr(self.config, 'GLOBAL_STATS') and self.config.GLOBAL_STATS:
            stats = self.config.GLOBAL_STATS.get(self.modality)
            if stats:
                return stats
        return None

    def _build_index(self):
        import random
        random.seed(42)
        samples = []
        filter_stats = {'total': 0, 'empty': 0, 'non_empty': 0, 'dropped_empty': 0}
        
        for study_id in self.study_ids:
            study_path = self.data_dir / study_id
            mask_path = study_path / f"{study_id}-seg.nii.gz"
            if not mask_path.exists(): continue
            
            mask_data = nib.load(str(mask_path)).get_fdata()
            num_slices = mask_data.shape[2]
            
            for z in range(num_slices):
                filter_stats['total'] += 1
                slice_mask = mask_data[:, :, z]
                
                if self.skip_empty_slices:
                    is_empty = (slice_mask.sum() == 0)
                    if is_empty:
                        filter_stats['empty'] += 1
                        if random.random() > self.negative_sample_ratio:
                            filter_stats['dropped_empty'] += 1
                            continue
                    else:
                        filter_stats['non_empty'] += 1

                samples.append({
                    'study_id': study_id,
                    'z': z,
                    'num_slices': num_slices
                })
        return samples, filter_stats

    def _get_volume(self, study_id):
        if self.use_cache and study_id in self.volume_cache:
            return self.volume_cache[study_id]
            
        study_path = self.data_dir / study_id
        img_path = study_path / f"{study_id}-{self.modality}.nii.gz"
        mask_path = study_path / f"{study_id}-seg.nii.gz"
        
        img_data = nib.load(str(img_path)).get_fdata().astype(np.float32)
        mask_data = nib.load(str(mask_path)).get_fdata().astype(np.int64)
        
        if self.norm_stats and getattr(self.config, 'NORMALIZATION_MODE', '') == 'global':
            g_mean = self.norm_stats['mean']
            g_std = self.norm_stats['std']
            img_data = (img_data - g_mean) / g_std
        else:
            brain_mask = img_data > 0
            if brain_mask.any():
                mean = img_data[brain_mask].mean()
                std = img_data[brain_mask].std()
                img_data[brain_mask] = (img_data[brain_mask] - mean) / (std + 1e-8)
                
        clip_range = getattr(self.config, 'CLIP_RANGE', [-3.0, 3.0]) if self.config else [-3.0, 3.0]
        img_data = np.clip(img_data, clip_range[0], clip_range[1])
        
        target_range = getattr(self.config, 'TARGET_RANGE', [0.0, 1.0]) if self.config else [0.0, 1.0]
        img_data = (img_data - clip_range[0]) / (clip_range[1] - clip_range[0])
        img_data = img_data * (target_range[1] - target_range[0]) + target_range[0]
        
        if self.label_mode == 'wt':
            mask_data = (mask_data > 0).astype(np.int64)
        elif self.label_mode == 'tc':
            mask_data = np.isin(mask_data, [1, 3]).astype(np.int64)
        elif self.label_mode == 'et':
            mask_data = (mask_data == 3).astype(np.int64)
            
        if self.use_cache:
            if len(self.volume_cache) >= self.max_cache_size:
                self.volume_cache.pop(next(iter(self.volume_cache)))
            self.volume_cache[study_id] = (img_data, mask_data)

        return img_data, mask_data

    def __getitem__(self, idx):
        sample = self.samples[idx]
        study_id = sample['study_id']
        z = sample['z']
        num_slices = sample['num_slices']
        
        img_vol, mask_vol = self._get_volume(study_id)
        
        images = []
        for offset in range(-self.T, self.T + 1):
            curr_z = max(0, min(num_slices - 1, z + offset))
            slice_img = img_vol[:, :, curr_z]
            images.append(slice_img)

        mask = mask_vol[:, :, z]
        
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(mask).long()
        
        # Rot90 to match expected orientation
        images = torch.rot90(images, k=1, dims=[-2, -1])
        mask = torch.rot90(mask, k=1, dims=[-2, -1])
        
        # Resize to expected 240x240
        if images.shape[-2:] != (240, 240):
             import torch.nn.functional as F
             images = F.interpolate(images.unsqueeze(0), size=(240, 240), mode='bilinear', align_corners=False).squeeze(0)
             mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(240, 240), mode='nearest').squeeze(0).squeeze(0).long()

        metadata = {'study_id': study_id, 'z': z}

        return images, mask, metadata
