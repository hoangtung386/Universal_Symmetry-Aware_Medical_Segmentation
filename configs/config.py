import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple

@dataclass
class TrainingConfig:
    # Basic settings
    SEED: int
    
    # Data paths
    DATA_PATHS: Dict[str, str]
    BASE_PATH: str
    IMAGE_DIR: str
    MASK_DIR: str
    OUTPUT_DIR: str
    CHECKPOINT_DIR: str
    DATASET_NAME: str = ""
    
    # Data split
    TRAIN_SPLIT: float = 0.8
    
    # Model parameters
    NUM_CHANNELS: int = 1
    NUM_CLASSES: int = 3
    INIT_FEATURES: int = 32
    IMAGE_SIZE: Tuple[int, int] = (512, 512)
    MULTICHANNEL: bool = False
    
    # HU Windowing
    USE_HU_WINDOW: bool = True
    WINDOW_CENTER: int = 40
    WINDOW_WIDTH: int = 40
    
    # Dataset filtering
    SKIP_EMPTY_SLICES: bool = True
    NEGATIVE_SAMPLE_RATIO: float = 0.05
    
    # Loss Weights
    FP_PENALTY_WEIGHT: float = 0.3
    
    # Training
    BATCH_SIZE: int = 20
    NUM_EPOCHS: int = 150
    LEARNING_RATE: float = 1e-4
    
    # DataLoader parameters
    NUM_WORKERS: int = 8
    CACHE_RATE: float = 0.0
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    
    # Model architecture
    T: int = 1
    NUM_PARTITIONS_H: int = 4
    NUM_PARTITIONS_W: int = 4
    GLOBAL_IMPACT: float = 0.3
    LOCAL_IMPACT: float = 0.7
    
    # Transformer Parameters
    TRANSFORMER_NUM_HEADS: int = 4
    TRANSFORMER_NUM_LAYERS: int = 2
    TRANSFORMER_EMBED_DIM: int = 1024
    
    # Normalization
    MEAN: Optional[List[float]] = None
    STD: Optional[List[float]] = None
    
    WEIGHT_DECAY: float = 1e-4
    
    # Training stability
    GRAD_CLIP_NORM: float = 1.0
    USE_AMP: bool = False
    DEBUG_MODE: bool = False
    DETECT_ANOMALY: bool = False
    
    # SOTA Components
    USE_MAMBA: bool = True
    USE_KAN: bool = True
    USE_CONDITIONING: bool = True
    
    # Component-specific settings
    MAMBA_DEPTH: int = 4
    KAN_DEGREE: int = 3
    
    # SymFormer specific
    KMAX_NUM_HEADS: int = 8
    KMAX_NUM_LAYERS: int = 2
    SYMMETRY_WEIGHT: float = 0.05
    CLUSTER_WEIGHT: float = 0.1
    
    # Loss weights
    DICE_WEIGHT: float = 0.7
    CE_WEIGHT: float = 0.3
    FOCAL_WEIGHT: float = 1.0
    ALIGNMENT_WEIGHT: float = 0.05
    PERCEPTUAL_WEIGHT: float = 0.1
    
    # W&B settings
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "OmniSym-dataset-"
    WANDB_ENTITY: Optional[str] = None
    WANDB_MODE: str = "online"
    
    # Scheduler parameters
    SCHEDULER_T0: int = 10
    SCHEDULER_T_MULT: int = 2
    SCHEDULER_ETA_MIN: float = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 30

    # BraTS Specific
    NORMALIZATION_MODE: Optional[str] = None
    GLOBAL_STATS: Optional[Dict[str, Dict[str, float]]] = None
    CLIP_RANGE: Optional[List[float]] = None
    TARGET_RANGE: Optional[List[float]] = None
    CUSTOM_CLASS_WEIGHTS: Optional[List[float]] = None

    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        print(f"Directories created: {self.OUTPUT_DIR}, {self.CHECKPOINT_DIR}")

    def to_dict(self):
        return asdict(self)
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(f"Dataset Name:      {self.DATASET_NAME}")
        print(f"Batch Size:        {self.BATCH_SIZE}")
        print(f"Learning Rate:     {self.LEARNING_RATE}")
        print(f"Epochs:            {self.NUM_EPOCHS}")
        print(f"Image Size:        {self.IMAGE_SIZE}")
        print(f"Gradient Clip:     {self.GRAD_CLIP_NORM}")
        print(f"Alignment Weight:  {self.ALIGNMENT_WEIGHT}")
        print(f"Use AMP:           {self.USE_AMP}")
        print(f"Debug Mode:        {self.DEBUG_MODE}")
        print(f"SOTA Components:   Mamba={self.USE_MAMBA}, KAN={self.USE_KAN}, Cond={self.USE_CONDITIONING}")
        print("="*60 + "\n")


def load_config(dataset_name: str) -> TrainingConfig:
    """Load and merge base configuration with dataset-specific overrides"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load base config
    base_path = os.path.join(config_dir, "base.yaml")
    with open(base_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Load dataset-specific config
    dataset_path = os.path.join(config_dir, "datasets", f"{dataset_name}.yaml")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            dataset_config = yaml.safe_load(f)
            if dataset_config:
                config_dict.update(dataset_config)
    else:
        print(f"Warning: Dataset config {dataset_path} not found. Using base config.")

    # Convert lists to tuples where necessary (like IMAGE_SIZE)
    if 'IMAGE_SIZE' in config_dict and isinstance(config_dict['IMAGE_SIZE'], list):
        config_dict['IMAGE_SIZE'] = tuple(config_dict['IMAGE_SIZE'])

    return TrainingConfig(**config_dict)

