# Brain Stroke Segmentation with SymFormer

A state-of-the-art framework for brain stroke lesion segmentation from CT scans, featuring **SymFormer** (Symmetry-Aware Hybrid Transformer).

## 🌟 Key Features

### Model Architecture: SymFormer
- **Symmetry-Aware Bottleneck**: Specifically designed to catch asymmetrical stroke features by comparing left and right brain hemispheres.
- **Mamba-2 Integration**: Utilizes the Mamba-2 State Space Model for efficient long-range dependency modeling.
- **KAN Decoder**: Implements Kolmogorov-Arnold Networks (KAN) in the decoder for improved boundary precision.
- **Clinical Conditioning**: Incorporates patient metadata (NIHSS, Age, Time from onset) to guide segmentation.

### Project Structure
```
.
├── configs/            # Configuration files (hyperparameters, paths)
├── dataset_APIS/       # Dataset directory (CPAISD)
├── models/             # Neural Network architectures
│   ├── layers/         # SOTA blocks (Mamba, KAN, Attention)
│   ├── symformer.py    # Main SymFormer model
│   └── components.py   # Encoder/Decoder blocks
├── scripts/            # Utility scripts (check_npz.py, explore_dataset.py)
├── utils/              # Helper functions (metrics, logging)
├── train.py            # Main training script
└── evaluate.py         # Evaluation script
```

## 💾 Dataset
The project uses the **CPAISD** dataset. 
- **Download Link**: [Zenodo Record 10892316](https://zenodo.org/records/10892316)
- **Format**: Pre-processed `.npz` files (image/mask) or raw DICOM.

## 🚀 Usage

### 1. Installation

#### Option 1: Auto Setup (Recommended)
This script sets up a virtual environment, installs PyTorch (CUDA-optimized), and dependencies.

```powershell
# Clone repository
git clone https://github.com/hoangtung386/brain-stroke-segmentation.git
cd brain-stroke-segmentation

# Run setup script (if available) or install manually
pip install -r requirements.txt
```

#### Option 2: Manual Installation
```powershell
pip install -r requirements.txt
# Update WandB (Required for v1 keys)
pip install --upgrade wandb
```

### 2. Data Preparation
Ensure the dataset is placed in `dataset_APIS/dataset`. The structure should look like:
```
dataset_APIS/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

### 3. Training

#### Configure Weights & Biases (Optional)
If you want to use Weights & Biases for tracking:
```powershell
$env:WANDB_API_KEY = "your_wandb_api_key_here"
```

#### Adjust Configuration
Edit `configs/config.py` to adjust hyperparameters if needed.

#### Start Training
Run the training script with specific GPU selection:

```powershell
# Run on GPU 0 (Default)
python train.py --devices 0

# Run on multiple GPUs
python train.py --devices 0,1

# Resume from checkpoint
# (Modify train.py to load checkpoint if implemented, or ensure checkpoints/ directory exists)
```

### 4. Monitoring
- **Console**: Training progress bar, loss metrics, and validation scores are printed to the console.
- **WandB**: If enabled, view detailed charts and visualizations on your Weights & Biases dashboard.

### 5. Evaluation
To evaluate the model on the test set:
```powershell
python evaluate.py --checkpoint checkpoints/symformer_best.pth
```

## ❓ Troubleshooting

### Common Issues
- **CUDA not found**: Ensure you have installed the correct PyTorch version for your CUDA driver. Check with `python -c "import torch; print(torch.cuda.is_available())"`.
- **SOTA components warning**: If `Mamba` or `KAN` layers are missing, the model gracefully falls back to standard implementations. Ensure you have installed `causal_conv1d` and `mamba_ssm` if you want to use Mamba.

## 📚 Citation
If you use this code in your research, please cite:
```bibtex
@article{symformer2026,
  title={SymFormer: Symmetry-Aware Hybrid Transformer for Stroke Segmentation},
  author={Hoang Tung et al.},
  journal={arXiv preprint},
  year={2026}
}
```

## 📧 Contact
For questions or support, please contact:
- **Email**: hoangtung386@gmail.com
- **GitHub**: [hoangtung386](https://github.com/hoangtung386)
