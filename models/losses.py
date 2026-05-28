import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import FocalLoss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, include_background=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        start_cls = 0 if self.include_background else 1
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        pred_soft = F.softmax(pred, dim=1)

        axes = tuple(range(2, pred.ndim))
        tp = (pred_soft * target_one_hot).sum(dim=axes)
        fp = (pred_soft * (1 - target_one_hot)).sum(dim=axes)
        fn = ((1 - pred_soft) * target_one_hot).sum(dim=axes)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1 - tversky[:, start_cls:].mean()
        return loss


class CorePenumbraContrastiveLoss(nn.Module):
    def __init__(self, core_idx=1, penumbra_idx=2, margin=0.5):
        super().__init__()
        self.core_idx = core_idx
        self.penumbra_idx = penumbra_idx
        self.margin = margin

    def forward(self, pred, target):
        if pred.shape[1] <= max(self.core_idx, self.penumbra_idx):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)
        prob = F.softmax(pred, dim=1)
        core_prob = prob[:, self.core_idx]
        penu_prob = prob[:, self.penumbra_idx]
        overlap = core_prob * penu_prob
        stroke_mask = (target == self.core_idx) | (target == self.penumbra_idx)
        loss = (overlap * stroke_mask.float()).mean()
        return loss


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        def morphological_gradient(x):
            dilated = F.max_pool2d(x, self.kernel_size, stride=1, padding=self.kernel_size // 2)
            eroded = -F.max_pool2d(-x, self.kernel_size, stride=1, padding=self.kernel_size // 2)
            return (dilated - eroded) > 0.1

        pred_label = torch.argmax(pred, dim=1)
        pred_one_hot = F.one_hot(pred_label, num_classes).permute(0, 3, 1, 2).float()

        pred_boundary = morphological_gradient(pred_one_hot)
        target_boundary = morphological_gradient(target_one_hot)

        intersection = (pred_boundary & target_boundary).float().sum(dim=(2, 3))
        union = pred_boundary.float().sum(dim=(2, 3)) + target_boundary.float().sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return (1 - dice[:, 1:].mean())


class StrokeLoss(nn.Module):
    def __init__(self, num_classes=3, class_weights=None,
                 tversky_alpha=0.3, tversky_beta=0.7,
                 tversky_weight=0.5, ce_weight=0.3, focal_weight=0.1,
                 contrastive_weight=0.3, boundary_weight=0.05,
                 multiscale_weight=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.focal = FocalLoss(include_background=False, to_onehot_y=True, gamma=2.0, reduction='mean')
        self.contrastive = CorePenumbraContrastiveLoss()
        self.boundary = BoundaryLoss()

        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        self.boundary_weight = boundary_weight
        self.multiscale_weight = multiscale_weight

        self.scale_weights = [0.5, 0.3, 0.15, 0.05]

    def forward(self, output, target, cluster_outputs=None, asymmetry_map=None):
        if target.ndim == 4:
            target = target.squeeze(1)

        tv = self.tversky(output, target)
        ce = self.ce(output, target.long())
        foc = self.focal(output, target.unsqueeze(1))
        cont = self.contrastive(output, target)
        bnd = self.boundary(output, target)

        main_loss = (
            self.tversky_weight * tv +
            self.ce_weight * ce +
            self.focal_weight * foc +
            self.contrastive_weight * cont +
            self.boundary_weight * bnd
        )

        loss_dict = {
            'tversky': tv.item(),
            'ce': ce.item(),
            'focal': foc.item(),
            'contrastive': (self.contrastive_weight * cont).item(),
            'boundary': (self.boundary_weight * bnd).item(),
        }

        total_loss = main_loss

        if cluster_outputs:
            ms = self._compute_multiscale_loss(cluster_outputs, target)
            total_loss = total_loss + self.multiscale_weight * ms
            loss_dict['multiscale'] = ms.item()

        loss_dict['total'] = total_loss.item()
        loss_dict['main'] = main_loss.item()
        return total_loss, loss_dict

    def _compute_multiscale_loss(self, cluster_outputs, target):
        total = 0.0
        num_stages = len(cluster_outputs)
        weights = self.scale_weights[:num_stages]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]

        for cluster_out, scale_weight in zip(cluster_outputs, weights):
            H_out, W_out = cluster_out.shape[-2:]
            target_scaled = F.interpolate(
                target.unsqueeze(1).float(), size=(H_out, W_out), mode='nearest'
            ).squeeze(1).long()
            loss = self.tversky(cluster_out, target_scaled)
            total = total + scale_weight * loss
        return total


class SymFormerLoss(StrokeLoss):
    pass


class ImprovedSymFormerLoss(StrokeLoss):
    pass


def create_ablation_losses(num_classes=2, class_weights=None):
    from copy import deepcopy
    base = {
        'num_classes': num_classes, 'class_weights': class_weights,
        'tversky_alpha': 0.3, 'tversky_beta': 0.7,
    }
    configs = {
        'baseline': {**base, 'tversky_weight': 1.0, 'ce_weight': 0.0, 'focal_weight': 0.0,
                     'contrastive_weight': 0.0, 'boundary_weight': 0.0, 'multiscale_weight': 0.0},
        'with_focal': {**base, 'tversky_weight': 0.5, 'ce_weight': 0.0, 'focal_weight': 0.5,
                       'contrastive_weight': 0.0, 'boundary_weight': 0.0, 'multiscale_weight': 0.0},
        'with_contrastive': {**base, 'tversky_weight': 0.5, 'ce_weight': 0.3, 'focal_weight': 0.2,
                             'contrastive_weight': 0.3, 'boundary_weight': 0.0, 'multiscale_weight': 0.0},
        'with_boundary': {**base, 'tversky_weight': 0.5, 'ce_weight': 0.3, 'focal_weight': 0.2,
                          'contrastive_weight': 0.0, 'boundary_weight': 0.1, 'multiscale_weight': 0.0},
        'full': {**base, 'tversky_weight': 0.5, 'ce_weight': 0.3, 'focal_weight': 0.1,
                 'contrastive_weight': 0.3, 'boundary_weight': 0.05, 'multiscale_weight': 0.05},
    }
    return {name: StrokeLoss(**cfg) for name, cfg in configs.items()}
