import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from monai.losses import DiceLoss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            loss_weight=1.0,
            reduction="mean",
            pos_weight=None,
            ignore_index=None,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: (B, C, D, H, W)
        target: (B, C, D, H, W)
        """

        target = target.float()

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            target = torch.clamp(target, 0, 1)
        else:
            valid_mask = None

        if not isinstance(self.pos_weight, torch.Tensor):
            self.pos_weight = torch.tensor([self.pos_weight], device=pred.device, dtype=pred.dtype)
        loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction="none",
            pos_weight=self.pos_weight
        )

        if valid_mask is not None:
            loss = loss * valid_mask
            denom = valid_mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
            loss = loss.sum(dim=(2, 3, 4)) / denom  # [B, 1]
        else:
            loss = loss.mean(dim=(2, 3, 4))  # [B, 1]

        if self.reduction == "mean":
            return self.loss_weight * loss.mean()
        elif self.reduction == "sum":
            return self.loss_weight * loss.sum()
        else:
            return self.loss_weight * loss


def foreground_dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0,
):
    """
    pred: (B, C, D, H, W)
    target: (B, C, D, H, W)
    """
    pred = torch.sigmoid(pred)

    dims = (2, 3, 4)

    intersection = (pred * target).sum(dims)

    dice = (2 * intersection + smooth) / (
            pred.sum(dims) + target.sum(dims) + smooth
    )

    return 1.0 - dice.mean()


def compute_boundary(mask):
    """
    mask: [B, 1, D, H, W]
    """
    max_pool = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = max_pool - min_pool
    return boundary


def boundary_dice_loss(pred, target, smooth=1e-6):
    """
    pred: (B, C, D, H, W)
    target: (B, C, D, H, W)
    """
    pred = torch.sigmoid(pred)
    target = target.float()

    pred_b = compute_boundary(pred)
    target_b = compute_boundary(target)

    dims = (2, 3, 4)

    intersection = (pred_b * target_b).sum(dims)
    pred_sum = pred_b.sum(dims)
    target_sum = target_b.sum(dims)

    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return 1 - dice.mean()


class MultiSegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, boundary_weight=0, pos_weight=None, n_class=1,
                 include_background_dice=True, include_background_ce=False):
        super().__init__()
        self.dice_w = dice_weight
        self.ce_w = ce_weight
        self.boundary_w = boundary_weight

        self.n_class = n_class

        if n_class == 1:
            self.ce_loss = BinaryCrossEntropyLoss(
                pos_weight=pos_weight,
            ) if ce_weight > 0 else None

            self.dice_loss = foreground_dice_loss if dice_weight > 0 else None

        else:
            self.ce_loss = VolumeCrossEntropyLoss(
                pos_weight=pos_weight,
                include_background=include_background_ce,
            ) if ce_weight > 0 else None
            self.dice_loss = VolumeDiceLoss(include_background=include_background_dice) if dice_weight > 0 else None

    def forward(self, pred, gt):
        loss = 0.0
        if self.ce_w > 0:
            loss = loss + self.ce_w * self.ce_loss(pred, gt)

        if self.dice_w > 0:
            loss = loss + self.dice_w * self.dice_loss(pred, gt)
            
        if self.boundary_w > 0 and self.n_class == 1:
            loss = loss + self.boundary_w * boundary_dice_loss(pred, gt)
        return loss


class VolumeDiceLoss(nn.Module):
    def __init__(self, include_background=False):
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=True,
            reduction="mean"
        )

    def forward(self, pred, gt):

        return self.dice(pred, gt)


class VolumeCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=None, include_background=False, ignore_index=255):
        super().__init__()
        self.class_weight = pos_weight
        self.include_background = include_background
        self.ignore_index = ignore_index if not self.include_background else -100

    def forward(self, pred, gt):
        """
        pred: (B, C, D, H, W)
        target: (B, C, D, H, W)
        """
        C = pred.shape[1]
        H, W = pred.shape[-2:]

        # ======== Ignore Background ========
        if not self.include_background:
            gt_sum = gt.sum(dim=1)  # [B, D, H, W]
            is_background = (gt_sum == 0)  # Background voxel

            target = gt.argmax(dim=1)  # [B, D, H, W] 0~4
            target = target.long()

            target[is_background] = self.ignore_index
        else:
            target = gt.argmax(dim=1).long()

        loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )

        return loss
