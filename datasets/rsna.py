"""
RSNA Dataset — Not yet implemented.
This dataset requires DICOM-to-NPZ preprocessing similar to CPAISD.
"""
from .base import BaseDataset


class RSNADataset(BaseDataset):
    """
    RSNA Abdominal CT Dataset (Stub).
    Implement _build_index, _load_slice, and __getitem__ when data is available.
    """
    def __init__(self, dataset_root, split='train', T=1, transform=None, config=None):
        super().__init__(dataset_root, split, T, transform)
        raise NotImplementedError(
            "RSNADataset is not yet implemented. "
            "See CPAISDDataset for a reference implementation."
        )

    def __getitem__(self, idx):
        raise NotImplementedError("RSNADataset is not yet implemented.")
