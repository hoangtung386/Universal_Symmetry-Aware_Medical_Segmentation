import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric
from models.losses import StrokeLoss


class SymFormerTrainer:
    """
    Core Trainer for SymFormer
    """
    def __init__(self, model, train_loader, val_loader, config, device, multi_gpu=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.multi_gpu = multi_gpu

        # Priority: Custom weights from config > Computed from dataset
        if hasattr(config, 'CUSTOM_CLASS_WEIGHTS') and config.CUSTOM_CLASS_WEIGHTS:
            print(f"[Using CUSTOM class weights from config: {config.CUSTOM_CLASS_WEIGHTS}")
            class_weights = torch.tensor(config.CUSTOM_CLASS_WEIGHTS, dtype=torch.float32).to(device)
        else:
            from utils.data_utils import compute_class_weights
            print("Computing class weights from dataset (this may take a moment)...")
            class_weights = compute_class_weights(
                train_loader.dataset,
                num_classes=config.NUM_CLASSES,
                num_samples=2000
            ).to(device)

        print(f"[Final class weights: {class_weights.cpu().tolist()}")
        print(f"[NUM_CLASSES: {config.NUM_CLASSES}")

        self.criterion = StrokeLoss(
            num_classes=config.NUM_CLASSES,
            class_weights=class_weights,
            tversky_alpha=0.3, tversky_beta=0.7,
            tversky_weight=0.5, ce_weight=0.3, focal_weight=0.1,
            contrastive_weight=0.3, boundary_weight=0.05,
            multiscale_weight=0.05,
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.dice_metric = DiceMetric(
            include_background=False,
            reduction='mean'
        )

        self.best_dice = 0.0
        self.history = []
        self.loss_accumulators = {}

    def _prepare_batch(self, batch):
        if len(batch) == 3:
            images, masks, metadata = batch
        else:
            images, masks = batch
            metadata = None

        images = images.to(self.device)
        masks = masks.to(self.device)

        if metadata:
            metadata = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in metadata.items()
            }
        return images, masks, metadata

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.loss_accumulators = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            images, masks, metadata = self._prepare_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(images, metadata_dict=metadata)
            output = outputs['pred']
            multiscale = outputs.get('multiscale_preds', None)

            loss, loss_dict = self.criterion(output, masks, multiscale, None)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM
            )
            self.optimizer.step()

            # Early warning
            with torch.no_grad():
                pred_classes = torch.argmax(output, dim=1)
                unique_preds = torch.unique(pred_classes)
                if len(unique_preds) < self.config.NUM_CLASSES:
                    if pbar.n % 50 == 0:
                        print(f"\nWARNING: Model only predicting {len(unique_preds)}/{self.config.NUM_CLASSES} classes: {unique_preds.tolist()}")

            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in self.loss_accumulators:
                    self.loss_accumulators[k] = 0.0
                self.loss_accumulators[k] += v if isinstance(v, (int, float)) else v.item()

            postfix = {'loss': f'{loss.item():.4f}'}
            if 'main' in loss_dict:
                postfix['main'] = f"{loss_dict['main']:.4f}"
            pbar.set_postfix(postfix)

            if self.config.USE_WANDB:
                log_dict = {'batch/train_loss': loss.item(), 'batch/grad_norm': grad_norm, 'epoch': epoch}
                for k, v in loss_dict.items():
                    log_dict[f'batch/{k}'] = v if isinstance(v, (int, float)) else v.item()
                wandb.log(log_dict)

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in self.loss_accumulators.items()}
        return avg_loss, avg_metrics

    def validate(self, epoch):
        self.model.eval()
        self.dice_metric.reset()
        total_val_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                images, masks, metadata = self._prepare_batch(batch)

                outputs = self.model(images, metadata_dict=metadata)
                output = outputs['pred']

                # ConditionedSymFormer always inverse-transforms to original space
                # so we evaluate against original masks directly.
                loss, _ = self.criterion(output, masks, None, None)
                total_val_loss += loss.item()

                # Dice metric
                if masks.ndim == 3:
                     masks_metric = masks.unsqueeze(1)
                else:
                     masks_metric = masks

                y_pred_idx = torch.argmax(output, dim=1, keepdim=True)
                y_pred_onehot = one_hot(y_pred_idx, num_classes=self.config.NUM_CLASSES)
                y_target_onehot = one_hot(masks_metric, num_classes=self.config.NUM_CLASSES)

                self.dice_metric(y_pred=y_pred_onehot, y=y_target_onehot)

                if pbar.n == 0:
                    print(f"\n[DEBUG] Unique predictions: {torch.unique(y_pred_idx).cpu().tolist()}")
                    print(f"[DEBUG] Unique targets: {torch.unique(masks).cpu().tolist()}")

                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        val_dice = self.dice_metric.aggregate().item()
        val_loss = total_val_loss / len(self.val_loader)

        print(f"\nValidation: Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        return val_dice, val_loss

    def train(self, num_epochs):
        print(f"\nStarting SymFormer Training\nEpochs: {num_epochs}\nDevice: {self.device}\n{'='*60}")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_dice, val_loss = self.validate(epoch)
            self.scheduler.step()

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self._save_checkpoint(epoch, val_dice, is_best=True)

            self._save_checkpoint(epoch, val_dice, is_best=False)
            self._log_history(epoch, train_loss, val_loss, val_dice, train_metrics)

        print(f"\nTraining Complete! Best Dice: {self.best_dice:.4f}")

    def _save_checkpoint(self, epoch, val_dice, is_best=False):
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        dataset_suffix = self.config.DATASET_NAME if self.config.DATASET_NAME else "unknown"

        prefix = 'symformer_best_' if is_best else 'symformer_'
        path = os.path.join(self.config.CHECKPOINT_DIR, f'{prefix}{dataset_suffix}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'val_dice': val_dice,
            'config': self.config.to_dict()
        }, path)

        if is_best:
            print(f"[Best model saved to {path}! Dice: {val_dice:.4f}")

    def _log_history(self, epoch, train_loss, val_loss, val_dice, train_metrics):
        history_dict = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_dice': val_dice}
        history_dict.update(train_metrics)
        self.history.append(history_dict)

        log_file = os.path.join(self.config.OUTPUT_DIR, f"training_{self.config.DATASET_NAME}_log.csv")
        mode = 'w' if epoch == 1 else 'a'

        with open(log_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history_dict.keys())
            if epoch == 1:
                writer.writeheader()
            writer.writerow(history_dict)

        print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Best: {self.best_dice:.4f}")

        if self.config.USE_WANDB:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/dice': val_dice,
                'val/best_dice': self.best_dice,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            }
            for k, v in train_metrics.items():
                log_dict[f'train/{k}'] = v
            wandb.log(log_dict)
