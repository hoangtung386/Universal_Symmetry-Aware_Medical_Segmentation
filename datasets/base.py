import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class BaseDataset(Dataset):
    """
    Base generic dataset for derived classes.
    Implements common logic for loading slices and metadata.
    """
    def __init__(self, dataset_root, split='train', T=1, transform=None):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.T = T
        self.transform = transform
        
    def __len__(self):
        return len(self.samples) if hasattr(self, 'samples') else 0

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _parse_time(self, time_str):
        if isinstance(time_str, (int, float)):
            return float(time_str)
        try:
            if '-' in str(time_str):
                parts = str(time_str).split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(time_str)
        except:
            return 0.0

    def _load_metadata(self, slice_path):
        """Load clinical metadata for the slice's study (common for CPAISD datasets)"""
        study_path = slice_path.parent
        meta_path = study_path / "metadata.json"

        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except json.JSONDecodeError:
                pass

        # Parse into standard format
        parsed_meta = {
            'nihss': self._safe_float(meta.get('nihss', 0)),
            'age': self._safe_float(meta.get('age', 60)),
            'sex': 0 if meta.get('sex', 'M') == 'M' else 1,
            'time': self._parse_time(meta.get('time', '0')),
            'dsa': 1 if meta.get('dsa', False) else 0
        }
        return parsed_meta

    def _build_index(self):
        """
        Base method for building index.
        Child classes should override this or call it via super() and extend.
        """
        raise NotImplementedError("Child classes must implement _build_index")

    def _load_slice(self, slice_path, **kwargs):
        """
        Base method for loading a slice.
        Child classes must implement this.
        """
        raise NotImplementedError("Child classes must implement _load_slice")
        
    def __getitem__(self, idx):
        """
        Base getitem. Uses samples built in _build_index.
        Child classes can override or use this common implementation
        if their samples structure matches (center_idx, all_slices, num_slices).
        """
        if not hasattr(self, 'samples'):
            raise NotImplementedError("Dataset must have 'samples' attribute initialized")

        sample = self.samples[idx]

        # Check if sample has the required keys for this generic implementation
        if 'slice_idx' in sample and 'all_slices' in sample and 'num_slices' in sample:
            center_idx = sample['slice_idx']
            all_slices = sample['all_slices']
            num_slices = sample['num_slices']

            images = []
            for offset in range(-self.T, self.T + 1):
                slice_idx = center_idx + offset
                # Clamp to valid range
                slice_idx = max(0, min(num_slices - 1, slice_idx))

                image, _ = self._load_slice(all_slices[slice_idx])
                images.append(image)

            center_slice_path = all_slices[center_idx]
            _, mask = self._load_slice(center_slice_path)

            # Use metadata if available/applicable (often needed for conditioning)
            metadata = self._load_metadata(center_slice_path)

            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()
            mask = torch.from_numpy(mask).long()

            if self.transform:
                # Apply transforms here if implemented
                pass

            return images, mask, metadata
        else:
             raise NotImplementedError("Child classes must implement __getitem__ if sample structure differs")
