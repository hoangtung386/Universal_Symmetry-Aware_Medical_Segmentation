from .base import BaseDataset

class RSNADataset(BaseDataset):
    """Placeholder for RSNA Dataset"""
    def __init__(self, dataset_root, split='train', T=1, transform=None, config=None):
        super().__init__(dataset_root, split, T, transform)
        self.config = config

    def _build_index(self):
        return [], {}

    def _load_slice(self, slice_path, **kwargs):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
