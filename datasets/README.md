# Datasets Module

This module handles data loading for different datasets (CPAISD/APIS, BraTS, RSNA).

## Structure
- `base.py`: Defines `BaseDataset` interface.
- `cpaisd.py`: Implementation for the CPAISD (APIS) dataset.
- `brats.py`: Placeholder for BraTS dataset.
- `rsna.py`: Placeholder for RSNA dataset.
- `factory.py`: Handles dataset selection and loader creation.

## Usage
In `train.py`, use the `--dataset` argument to switch datasets:
```bash
python train.py --dataset cpaisd  # Default
python train.py --dataset brats
python train.py --dataset rsna
```

## Adding a New Dataset
1. Create a new file `datasets/your_dataset.py`.
2. Implement a class inheriting from `BaseDataset`.
3. Update `datasets/factory.py` to include your new dataset class in `get_dataset_class` and mapping.
4. Add the dataset path to `configs/config.py` in `DATA_PATHS`.
