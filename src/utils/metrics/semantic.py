"""Semantic-segmentation metrics: mIoU / Dice / precision / recall, boundary IoU
and centerline Dice.

These are vectorised confusion-matrix computations with no heavy optional
dependencies — safe to import anywhere. The soft-skeleton and clDice follow Shit
et al., "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure
Segmentation" (CVPR 2021); boundary IoU follows Cheng et al. (CVPR 2021).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def binary_pred_from_logits(
    logits: torch.Tensor,
    fg_channel: int = 1,
    threshold: float | None = None,
) -> torch.Tensor:
    """Hard binary prediction from (B, 2, H, W) logits.

    Args:
        logits: (B, 2, H, W) pixel logits.
        fg_channel: index of the foreground channel (default 1).
        threshold: if provided, threshold fg_channel against this value.
            Use threshold=0.5 for Mask2Former binary output where the background
            channel is never explicitly trained and thus cannot be compared against.
            If None, falls back to argmax (fg > bg), suitable for CE-trained models.
    """
    if threshold is not None:
        return (logits[:, fg_channel] > threshold).long()
    bg_channel = 1 - fg_channel
    return (logits[:, fg_channel] > logits[:, bg_channel]).long()


def compute_seg_metrics(
    pred: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    include_background: bool = True,
) -> dict[str, float]:
    """Compute mIoU, foreground Dice, mean Dice, precision, and recall.

    Uses a vectorized confusion matrix approach to correctly handle ignore_index
    without artificial inflation of class 0 results.

    Args:
        pred: (B, H, W) integer prediction tensor.
        masks: (B, H, W) integer ground-truth tensor.
        num_classes: total number of classes (including background at index 0).
        ignore_index: label value to exclude from all metrics.
        include_background: if False, class 0 is excluded from mIoU and
            precision/recall. Dice is always foreground-only.

    Returns:
        Dict with keys: mIoU, dice_fg, mean_dice, precision, recall.
    """
    valid = masks != ignore_index
    pred_valid = pred[valid]
    masks_valid = masks[valid]

    # Handle case where all pixels are ignored
    if masks_valid.numel() == 0:
        return {"mIoU": 0.0, "dice_fg": 0.0, "mean_dice": 0.0, "precision": 0.0, "recall": 0.0}

    # Vectorized confusion matrix computation
    combined = masks_valid * num_classes + pred_valid
    conf = torch.bincount(combined, minlength=num_classes**2).reshape(num_classes, num_classes)
    conf = conf.float()

    tp = conf.diag()                              # (C,) match
    fp = conf.sum(dim=0) - tp                      # predicted c, actually something else
    fn = conf.sum(dim=1) - tp                      # actually c, predicted something else

    # ── mIoU ─────────────────────────────────────────────────────────────
    union = tp + fp + fn
    iou_per_class = tp / union.clamp(min=1)
    # Mask out classes that were not present in GT or Pred (union == 0)
    # iou_per_class[union == 0] will be 0.0, so use nanmean to ignore them
    iou_per_class = iou_per_class.masked_fill(union == 0, float("nan"))

    start_class = 0 if include_background else 1
    miou = torch.nanmean(iou_per_class[start_class:]).item()

    # ── Dice ─────────────────────────────────────────────────────────────
    # Dice = 2TP / (2TP + FP + FN)
    dice_per_class = (2 * tp) / (2 * tp + fp + fn).clamp(min=1)
    dice_per_class = dice_per_class.masked_fill(union == 0, float("nan"))

    mean_dice = torch.nanmean(dice_per_class[start_class:]).item()

    if num_classes > 1:
        dice_fg_raw = torch.nanmean(dice_per_class[1:])
        dice_fg = 0.0 if torch.isnan(dice_fg_raw) else dice_fg_raw.item()
    else:
        dice_fg = 0.0

    # ── Precision & Recall ───────────────────────────────────────────────
    prec_per_class = tp / (tp + fp).clamp(min=1)
    rec_per_class  = tp / (tp + fn).clamp(min=1)

    # Mask absent classes (union == 0) to match nanmean behaviour used for mIoU
    prec_per_class = prec_per_class.masked_fill(union == 0, float("nan"))
    rec_per_class  = rec_per_class.masked_fill(union == 0, float("nan"))
    precision = torch.nanmean(prec_per_class[start_class:]).item()
    recall    = torch.nanmean(rec_per_class[start_class:]).item()

    return {
        "mIoU": miou if not math.isnan(miou) else 0.0,
        "dice_fg": dice_fg,
        "mean_dice": mean_dice if not math.isnan(mean_dice) else 0.0,
        "precision": precision,
        "recall": recall,
    }


class PerClassIoUAccumulator:
    """Accumulate TP/FP/FN per class across batches to compute per-class IoU at epoch end.

    More accurate than averaging per-batch IoU values since class frequencies
    vary across batches.

    Usage::
        acc = PerClassIoUAccumulator(num_classes, ignore_index)
        # each val step:
        acc.accumulate_step(pred, masks)
        # end of epoch:
        iou_dict = acc.compute()   # {"iou_class_0": ..., "iou_class_1": ...}
        acc.reset()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def accumulate_step(self, pred: torch.Tensor, masks: torch.Tensor) -> None:
        """pred, masks: (B, H, W) or (H, W) integer tensors."""
        valid = masks != self.ignore_index
        if not valid.any():
            return

        pred_v = pred[valid]
        mask_v = masks[valid]

        # Vectorized confusion matrix via bincount
        combined = mask_v * self.num_classes + pred_v
        conf = torch.bincount(combined, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        conf = conf.float()

        tp = conf.diag()
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp

        for c in range(self.num_classes):
            self._tp[c] += float(tp[c])
            self._fp[c] += float(fp[c])
            self._fn[c] += float(fn[c])

    def sync(self) -> None:
        """Gather TP/FP/FN from all DDP ranks via all_reduce (sum)."""
        import torch.distributed as dist
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return
        for attr in ("_tp", "_fp", "_fn"):
            t = torch.tensor(getattr(self, attr), device=f"cuda:{dist.get_rank()}")
            dist.all_reduce(t)
            setattr(self, attr, t.cpu().tolist())

    def compute(self) -> Dict[str, float]:
        result = {}
        for c in range(self.num_classes):
            denom = self._tp[c] + self._fp[c] + self._fn[c]
            result[f"iou_class_{c}"] = self._tp[c] / (denom + 1e-6) if denom > 0 else float("nan")
        return result

    def compute_dataset_metrics(self, include_background: bool = True) -> Dict[str, float]:
        """Compute dataset-level mIoU, Dice, etc. from accumulated confusion matrix."""
        tp = torch.tensor(self._tp)
        fp = torch.tensor(self._fp)
        fn = torch.tensor(self._fn)

        union = tp + fp + fn
        iou_per_class = tp / union.clamp(min=1)
        iou_per_class = iou_per_class.masked_fill(union == 0, float("nan"))

        start_class = 0 if include_background else 1
        miou = torch.nanmean(iou_per_class[start_class:]).item()

        dice_per_class = (2 * tp) / (2 * tp + fp + fn).clamp(min=1)
        dice_per_class = dice_per_class.masked_fill(union == 0, float("nan"))
        mean_dice = torch.nanmean(dice_per_class[start_class:]).item()

        if self.num_classes > 1:
            dice_fg_raw = torch.nanmean(dice_per_class[1:])
            dice_fg = 0.0 if torch.isnan(dice_fg_raw) else dice_fg_raw.item()
        else:
            dice_fg = 0.0

        prec_per_class = tp / (tp + fp).clamp(min=1)
        rec_per_class = tp / (tp + fn).clamp(min=1)
        prec_per_class = prec_per_class.masked_fill(union == 0, float("nan"))
        rec_per_class = rec_per_class.masked_fill(union == 0, float("nan"))
        precision = torch.nanmean(prec_per_class[start_class:]).item()
        recall = torch.nanmean(rec_per_class[start_class:]).item()

        return {
            "mIoU":       float(miou) if not math.isnan(miou) else 0.0,
            "dice_fg":    float(dice_fg),
            "mean_dice":  float(mean_dice) if not math.isnan(mean_dice) else 0.0,
            "precision":  float(precision) if not math.isnan(precision) else 0.0,
            "recall":     float(recall) if not math.isnan(recall) else 0.0,
        }

    def reset(self) -> None:
        self._tp = [0.0] * self.num_classes
        self._fp = [0.0] * self.num_classes
        self._fn = [0.0] * self.num_classes


def compute_boundary_iou(
    pred: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    boundary_width: int = 5,
    chunk_size: int = 32,
) -> float:
    """Boundary IoU: evaluates segmentation quality only within boundary regions.

    Optimized via multi-channel max_pool2d and chunking for high performance.

    Args:
        pred: (B, H, W) integer prediction tensor.
        masks: (B, H, W) integer ground-truth tensor.
        num_classes: total number of classes.
        ignore_index: label to exclude.
        boundary_width: dilation width to extract boundary region.
        chunk_size: number of classes to process in parallel (prevents OOM).

    Returns:
        Mean boundary IoU over classes present in GT.
    """
    valid = (masks != ignore_index).unsqueeze(1)    # (B, 1, H, W)
    iou_per_class = []

    ks = 2 * boundary_width + 1
    pad = boundary_width

    for start in range(0, num_classes, chunk_size):
        end = min(start + chunk_size, num_classes)
        curr_classes = torch.arange(start, end, device=pred.device)

        # One-hot expansion: (B, H, W) -> (B, num_curr, H, W)
        # Using [None, :, None, None] broadcasts curr_classes over pred
        pred_c = (pred.unsqueeze(1) == curr_classes[None, :, None, None]).float()
        mask_c = (masks.unsqueeze(1) == curr_classes[None, :, None, None]).float()

        # Check which classes are present in GT in this chunk
        present_in_gt = mask_c.sum(dim=(0, 2, 3)) > 0
        if not present_in_gt.any():
            continue

        # Compare boundary bands rather than full class masks — intersecting full
        # masks makes the zone ineffective, since every class pixel is inside its dilation.
        pred_dilated = F.max_pool2d(pred_c, ks, stride=1, padding=pad)
        mask_dilated = F.max_pool2d(mask_c, ks, stride=1, padding=pad)
        pred_eroded = -F.max_pool2d(-pred_c, ks, stride=1, padding=pad)
        mask_eroded = -F.max_pool2d(-mask_c, ks, stride=1, padding=pad)
        pred_boundary = (pred_dilated - pred_eroded) > 0
        mask_boundary = (mask_dilated - mask_eroded) > 0

        inter = pred_boundary & mask_boundary & valid
        union = (pred_boundary | mask_boundary) & valid

        inter_sum = inter.float().sum(dim=(0, 2, 3))
        union_sum = union.float().sum(dim=(0, 2, 3))

        chunk_ious = inter_sum / union_sum.clamp(min=1e-6)

        # Append only for classes present in GT
        iou_per_class.extend(chunk_ious[present_in_gt].tolist())

    return sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0


def _binary_foreground(pred_or_mask: torch.Tensor, ignore_index: int | None = None) -> torch.Tensor:
    fg = pred_or_mask > 0
    if ignore_index is not None:
        fg = fg & (pred_or_mask != ignore_index)
    return fg.float().unsqueeze(1)


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def _soft_skeleton(x: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    skel = F.relu(x - _soft_open(x))
    for _ in range(iterations):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(max=1.0)


def compute_cldice(
    pred: torch.Tensor,
    masks: torch.Tensor,
    ignore_index: int,
    iterations: int = 20,
) -> float:
    """Centerline Dice for binary/foreground vessel segmentation."""
    valid = masks != ignore_index
    if not valid.any():
        return 0.0

    pred_fg = _binary_foreground(pred) * valid.unsqueeze(1).float()
    mask_fg = _binary_foreground(masks, ignore_index=ignore_index)
    if mask_fg.sum() == 0:
        return 0.0

    pred_skel = _soft_skeleton(pred_fg, iterations=iterations)
    mask_skel = _soft_skeleton(mask_fg, iterations=iterations)

    tprec = (pred_skel * mask_fg).sum() / pred_skel.sum().clamp(min=1e-6)
    tsens = (mask_skel * pred_fg).sum() / mask_skel.sum().clamp(min=1e-6)
    cldice = (2.0 * tprec * tsens) / (tprec + tsens).clamp(min=1e-6)
    return float(cldice.item())


def compute_thin_vessel_metrics(
    pred: torch.Tensor,
    masks: torch.Tensor,
    ignore_index: int,
    skeleton_iterations: int = 20,
) -> dict[str, float]:
    """Dice/recall on the GT vessel skeleton as a thin-vessel proxy."""
    valid = masks != ignore_index
    pred_fg = _binary_foreground(pred) * valid.unsqueeze(1).float()
    mask_fg = _binary_foreground(masks, ignore_index=ignore_index)
    skeleton = (_soft_skeleton(mask_fg, iterations=skeleton_iterations) > 0).float()
    if skeleton.sum() == 0:
        return {"thin_vessel_dice": 0.0, "thin_vessel_recall": 0.0}

    inter = (pred_fg * skeleton).sum()
    recall = inter / skeleton.sum().clamp(min=1e-6)
    dice = (2.0 * inter) / (pred_fg.sum() + skeleton.sum()).clamp(min=1e-6)
    return {
        "thin_vessel_dice": float(dice.item()),
        "thin_vessel_recall": float(recall.item()),
    }
