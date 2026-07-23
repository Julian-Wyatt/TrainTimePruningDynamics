import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationCELoss(nn.Module):
    """Weighted pixel-wise cross-entropy loss for semantic segmentation."""

    def __init__(self, ignore_index: int = 255, weight: torch.Tensor = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits.
            target: (B, H, W) class indices.
        """
        return F.cross_entropy(
            pred, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )


class DiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-class segmentation.

    For binary tasks (num_classes=2) the loss is computed over the foreground
    channel only (channel index 1).  For multi-class tasks it is averaged over
    all foreground channels.

    Args:
        ignore_index: Pixels with this label are excluded from the Dice
            numerator and denominator.
        smooth: Laplace smoothing term to avoid division by zero.
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W) logits.
            target: (B, H, W) integer class indices.
        Returns:
            Scalar Dice loss (1 - mean Dice coefficient).
        """
        num_classes = pred.shape[1]
        prob = pred.softmax(dim=1)  # (B, C, H, W)

        # Build valid pixel mask — ignore_index pixels contribute 0
        valid = (target != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)

        # One-hot encode target; clamp ignored pixels to 0
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        one_hot = F.one_hot(target_clamped, num_classes=num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Only average over foreground channels (skip background at index 0)
        dice_scores = []
        for c in range(1, num_classes):
            p = prob[:, c] * valid[:, 0]   # (B, H, W)
            g = one_hot[:, c] * valid[:, 0]
            intersection = (p * g).sum(dim=(1, 2))
            cardinality = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice.mean())

        if not dice_scores:
            return pred.sum() * 0.0  # degenerate single-class case

        return 1.0 - torch.stack(dice_scores).mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy + Dice loss, weighted by ``dice_weight``.

    Loss = CE + dice_weight * Dice
    """

    def __init__(
        self,
        ignore_index: int = 255,
        weight: torch.Tensor = None,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce = SegmentationCELoss(ignore_index=ignore_index, weight=weight)
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(pred, target) + self.dice_weight * self.dice(pred, target)
