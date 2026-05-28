# Universal Symmetry-Aware Medical Segmentation (SymFormer)

Symmetry-aware transformer for medical image segmentation (CT/MRI). Leverages brain
hemisphere symmetry to detect stroke lesions and brain tumors.

## Key Features

- **Symmetry-Aware Bottleneck** — models left-right hemisphere symmetry with Mamba-2 or custom SSM
- **Alignment Network** — automatically aligns inputs to canonical symmetric pose
- **KAN Decoder Heads** — Kolmogorov–Arnold Network layers for sharper boundaries
- **Clinical Conditioning** — injects NIHSS, age, sex, time metadata via cross-attention
- **StrokeLoss** — Tversky(0.7/0.3) + contrastive(Core↔Penumbra) + boundary refinement
- **Multi-Dataset** — CPAISD (Stroke CT), BraTS (Brain Tumor MRI), RSNA (Abdominal CT)
- **3 Bottleneck variants** — Symmetry, Mamba-2, SimplifiedSSM fallback

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n symformer python=3.10
conda activate symformer
pip install -e .
# If you want Mamba-2 support:
# pip install mamba-ssm causal-conv1d
```

### 2. API Keys (optional)

Copy `.env.example` to `.env` and fill in your keys for W&B tracking:

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```

### 3. Dataset Paths

Ensure dataset paths in `configs/base.yaml`:

```yaml
DATA_PATHS:
  cpaisd: "dataset_APIS/dataset"
  cpaisd_enhanced: "dataset_APIS/dataset"
  brats: "Dataset_BraTs"
  rsna: "datasets/RSNA"
```

---

## Training

```bash
python scripts/train.py --devices "0" --dataset cpaisd
python scripts/train.py --devices "0" --dataset cpaisd_enhanced
python scripts/train.py --devices "0,1" --dataset brats
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/symformer_best_cpaisd.pth --dataset cpaisd
python scripts/evaluate.py --checkpoint checkpoints/symformer_best_brats.pth --dataset brats
```

---

## Testing

```bash
pytest tests/ -v
```

---

---

## Project Structure

```
├── models/
│   ├── symformer.py              # Core SymFormer architecture
│   ├── conditioned_symformer.py  # Wrapper with clinical metadata conditioning
│   ├── components.py             # AlignmentNetwork, EncoderBlock3D, Attention
│   ├── losses.py                 # StrokeLoss (Tversky + Contrastive + Boundary)
│   ├── bottleneck/               # 3 variants: Symmetry, Mamba-2, SimplifiedSSM
│   ├── decoder/
│   │   ├── hvt.py                # HVTDecoder, DecoderBlock, kMaXBlock
│   │   └── kan.py                # KANDecoderHead, RationalKANLayer
│   └── layers/                   # Mamba-2, ClinicalConditionEncoder
├── datasets/                     # CPAISD, BraTS, RSNA loaders + factory
├── training/
│   └── trainer.py                # SymFormerTrainer (AMP, W&B, checkpoint)
├── evaluation/                   # Evaluator, metrics, complexity, visualization
├── configs/                      # base.yaml + dataset-specific overrides
├── scripts/
│   ├── train.py                  # Entry point for training
│   └── evaluate.py               # Entry point for evaluation
├── preprocessing/                # Enhancement pipeline, BraTS NIfTI converter
├── tests/                        # Smoke tests (pytest)
├── utils/                        # compute_class_weights, transforms
├── pyproject.toml                # Installable package
└── .env.example                  # API key template
```

## Loss System (StrokeLoss)

| Component | Weight | Purpose |
|-----------|--------|---------|
| Tversky (α=0.7, β=0.3) | 0.5 | Focus on recall for Core class |
| Cross-Entropy | 0.3 | Multi-class discrimination |
| Focal (γ=2.0) | 0.1 | Hard example mining |
| Core↔Penumbra Contrastive | 0.3 | Separate Core from Penumbra |
| Boundary | 0.05 | Edge refinement |
| Multi-scale deep supervision | 0.05 | All decoder stages |

## Config Overrides (CPAISD)

```yaml
FP_PENALTY_WEIGHT: 0.0          # Disabled — was suppressing Core recall
CUSTOM_CLASS_WEIGHTS: [0.05, 200.0, 8.0]  # Aggressive Core weight
NEGATIVE_SAMPLE_RATIO: 0.02     # Only 2% background slices kept
```

## Tests

```bash
pytest tests/ -v    # 12 smoke tests (loss, model, alignment, config)
```

## Checkpoints

Saved to `checkpoints/`:
- `symformer_best_{dataset}.pth` — best validation Dice
- `symformer_{dataset}.pth` — last epoch
