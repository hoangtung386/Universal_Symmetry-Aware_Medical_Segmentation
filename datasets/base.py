from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    """
    Base generic dataset for derived classes
    """
    def __init__(self, dataset_root, split='train', T=1, transform=None):
        self.dataset_root = dataset_root
        self.split = split
        self.T = T
        self.transform = transform
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError
