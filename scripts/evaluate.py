import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.factory import get_dataset_class
from torch.utils.data import DataLoader
from models.conditioned_symformer import ConditionedSymFormer
from configs.config import load_config
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SymFormer")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--split', type=str, default='val', help="Dataset split to evaluate on")
    parser.add_argument('--devices', type=str, default="0", help="CUDA devices")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    config = load_config(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_class = get_dataset_class(config.DATASET_NAME)
    root_path = config.DATA_PATHS.get(config.DATASET_NAME, config.BASE_PATH)

    eval_dataset = dataset_class(
        dataset_root=root_path,
        split=args.split,
        T=config.T,
        config=config,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    model = ConditionedSymFormer(
        in_channels=config.NUM_CHANNELS,
        num_classes=config.NUM_CLASSES,
        T=config.T,
        input_size=config.IMAGE_SIZE,
        bottleneck_type='mamba' if config.USE_MAMBA else 'symmetry',
        use_kan=config.USE_KAN,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction='mean')

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {args.split}"):
            if len(batch) == 3:
                images, masks, metadata = batch
            else:
                images, masks = batch
                metadata = None

            images = images.to(device)
            masks = masks.to(device)
            if metadata:
                metadata = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in metadata.items()}

            outputs = model(images, metadata_dict=metadata)
            output = outputs['pred']

            # Since output is INVERSE transformed during eval, it's in the original space
            # So we compare directly with the original mask
            if masks.ndim == 3:
                 masks_metric = masks.unsqueeze(1)
            else:
                 masks_metric = masks

            y_pred_idx = torch.argmax(output, dim=1, keepdim=True)
            y_pred_onehot = one_hot(y_pred_idx, num_classes=config.NUM_CLASSES)
            y_target_onehot = one_hot(masks_metric, num_classes=config.NUM_CLASSES)

            dice_metric(y_pred=y_pred_onehot, y=y_target_onehot)

    final_dice = dice_metric.aggregate().item()
    print(f"Final Dice Score on {args.split}: {final_dice:.4f}")

if __name__ == "__main__":
    main()
