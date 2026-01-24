# OmniSym: Universal Symmetry-Aware Medical Segmentation

![Model Architecture](Architectural_model.png)

**OmniSym** is a generalized deep learning framework built upon the **SymFormer** architecture, designed to solve the problem of anomaly segmentation in *any* medical imaging modality that exhibits biological symmetry (Axial Symmetry).

## 🚀 The Universal Advantage

Unlike traditional segmentation models trained for specific organs or pathologies, **OmniSym** leverages a fundamental biological constant: **Symmetry**.

- **One Architecture, Any Dataset**: Whether it's **Brain Strokes (CT)**, **Brain Tumors (MRI)**, or **Abdominal Trauma (CT)**, the core architecture remains identical.
- **Symmetry as a Feature**: The model explicitly compares left and right hemispheres (or body sides) effectively "subtracting" the healthy side from the pathological side to highlight anomalies.
- **Modality Agnostic**: Works seamlessly with CT Hounsfield Units, MRI T1/T2/FLAIR sequences, or any other tomographic slice data.

## 🧠 Core Architecture: SymFormer

The heart of OmniSym is **SymFormer**, a state-of-the-art hybrid transformer featuring:

1.  **Symmetry-Aware Bottleneck**: A dedicated mechanism that computes the difference between symmetric regions to isolate lesions.
2.  **Mamba-2 Backbones**: Utilizes Linear State Space Models for efficient, global context modeling without the quadratic cost of Vision Transformers.
3.  **KAN Decoder**: Kolmogorov-Arnold Networks provide superior boundary precision for irregular lesion shapes.

## 📂 Multi-Dataset Support

This project has been restructured to support a plug-and-play dataset system. It currently supports placeholders and implementations for:

*   **Brain Stroke** (APIS/CPAISD) - *Implemented*
*   **Brain Tumor** (BraTS) - *Ready for Integration*
*   **Abdominal Trauma** (RSNA) - *Ready for Integration*

### Directory Structure
```
.
├── configs/            # Global Configuration
├── datasets/           # 🔌 Universal Dataset Loaders
│   ├── base.py         # The Universal Interface
│   ├── factory.py      # Loader Generator
│   ├── cpaisd.py       # Example: Stroke CT
│   └── brats.py        # Example: Tumor MRI
├── models/             # The OmniSym/SymFormer Engine
└── train.py            # Universal Training Script
```

## 🛠️ Usage

### Installation
```powershell
pip install -r requirements.txt
```

### Universal Training
Switching targets is as simple as changing a flag. The model adapts automatically.

```powershell
# Train for Brain Stroke (CT)
python train.py --dataset cpaisd --devices 0

# Train for Brain Tumor (MRI)
python train.py --dataset brats --devices 0

# Train for Abdominal Trauma (CT)
python train.py --dataset rsna --devices 0
```

## 📚 Citation
If you use **OmniSym** or the **SymFormer** architecture in your research, please cite:

```bibtex
@article{omnisym2026,
  title={OmniSym: A Universal Symmetry-Aware Framework for Medical Image Segmentation},
  author={Hoang Tung et al.},
  journal={arXiv preprint},
  year={2026}
}
```

## 📧 Contact
- **Author**: Hoang Tung
- **Project**: OmniSym (formerly Brain-Stroke-Segmentation)
- **GitHub**: [hoangtung386](https://github.com/hoangtung386)
