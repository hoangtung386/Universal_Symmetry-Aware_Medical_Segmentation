from .cpaisd import CPAISDDataset
from .cpaisd_enhanced import CPAISDEnhancedDataset
from .brats import BraTSDataset
from .rsna import RSNADataset

def get_dataset_class(dataset_name):
    """Factory function to get the correct dataset class"""
    datasets = {
        'cpaisd': CPAISDDataset,
        'cpaisd_enhanced': CPAISDEnhancedDataset,
        'brats': BraTSDataset,
        'rsna': RSNADataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
        
    return datasets[dataset_name]
