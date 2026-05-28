import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

def early_device_setup():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--devices', type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        print(f"[Early Setup] Setting CUDA_VISIBLE_DEVICES={args.devices}")

early_device_setup()

from datasets.factory import get_dataset_class
from torch.utils.data import DataLoader
from models.conditioned_symformer import ConditionedSymFormer
from training.trainer import SymFormerTrainer
from configs.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train SymFormer")
    parser.add_argument('--devices', type=str, default=None, help="CUDA devices (e.g., '0,1')")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset to train on: cpaisd, cpaisd_enhanced, brats, rsna")
    return parser.parse_args()

def create_dataloaders(config):
    dataset_class = get_dataset_class(config.DATASET_NAME)
    root_path = config.DATA_PATHS.get(config.DATASET_NAME, config.BASE_PATH)

    train_dataset = dataset_class(
        dataset_root=root_path,
        split='train',
        T=config.T,
        config=config,
    )

    val_dataset = dataset_class(
        dataset_root=root_path,
        split='val',
        T=config.T,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader

def main():
    args = parse_args()

    if args.dataset is None:
         print("Please specify a dataset using --dataset")
         return

    config = load_config(args.dataset)
    config.create_directories()
    config.print_config()

    if config.USE_WANDB:
        wandb_project = f"{config.WANDB_PROJECT}{config.DATASET_NAME}"
        wandb.init(
            project=wandb_project,
            config=config.to_dict(),
            name=f"SymFormer_{config.DATASET_NAME}_{'cpu' if not args.devices else 'gpu'+args.devices.replace(',','_')}",
        )

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device = torch.device('cpu')
            device_ids = []
            multi_gpu = False
        else:
            device_ids = list(range(device_count))
            device = torch.device('cuda:0')
            multi_gpu = device_count > 1
    else:
        device = torch.device('cpu')
        device_ids = []
        multi_gpu = False

    train_loader, val_loader = create_dataloaders(config)

    model = ConditionedSymFormer(
        in_channels=config.NUM_CHANNELS,
        num_classes=config.NUM_CLASSES,
        T=config.T,
        input_size=config.IMAGE_SIZE,
        bottleneck_type='mamba' if config.USE_MAMBA else 'symmetry',
        use_kan=config.USE_KAN,
    )

    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)

    trainer = SymFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        multi_gpu=multi_gpu
    )

    trainer.train(num_epochs=config.NUM_EPOCHS)

if __name__ == "__main__":
    main()
