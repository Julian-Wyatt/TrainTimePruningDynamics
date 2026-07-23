"""Shared FIVES segmentation evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from trainer.base_trainer import BaseTrainer
from trainer.engine import evaluate
from utils.distributed import is_main_process
from utils.metrics import (
    binary_pred_from_logits,
    compute_boundary_iou,
    compute_cldice,
    compute_seg_metrics,
    compute_thin_vessel_metrics,
)


class BaseSegmentationTrainer(BaseTrainer):
    """Base class for the fixed-resolution FIVES segmentation pipeline."""

    def __init__(self, cfg: DictConfig, data_module):
        super().__init__(cfg, data_module)
        self.img_size = tuple(cfg.DATASET.IMG_SIZE)

    def get_pixel_logits(self, model: nn.Module, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def _binary_pred(self, logits: torch.Tensor) -> torch.Tensor:
        threshold = 0.5 if self.cfg.TRAIN.LOSS_TYPE == "mask2former" else None
        return binary_pred_from_logits(
            logits,
            fg_channel=self.cfg.DATASET.BINARY_FG_CHANNEL,
            threshold=threshold,
        )

    def _run_evaluation(self, prefix: str, loader) -> dict:
        self._per_class_iou_acc.reset()
        metrics = evaluate(self, loader, prefix=prefix)
        self._per_class_iou_acc.sync()
        include_background = not bool(self.cfg.DATASET.EXCLUDE_BACKGROUND_CLASS)
        dataset_metrics = self._per_class_iou_acc.compute_dataset_metrics(
            include_background=include_background
        )
        metrics.update(dataset_metrics)
        if is_main_process():
            self.logger.log(
                {f"{prefix}_epoch/{key}": value for key, value in dataset_metrics.items()},
                self.global_step,
            )
        return metrics

    def validate(self) -> dict:
        return self._run_evaluation("val", self.data_module.val_loader())

    def test(self) -> dict:
        loader = self.data_module.test_loader()
        return self._run_evaluation("test", loader) if loader is not None else {}

    def validation_step(self, model: nn.Module, batch: dict) -> dict:
        return self._process_segmentation_outputs(
            self.get_pixel_logits(model, batch), batch
        )

    def _process_segmentation_outputs(
        self, pixel_logits: torch.Tensor, batch: dict
    ) -> dict:
        masks = batch["mask"]
        pixel_logits = self._resize_pixel_logits_to_masks(pixel_logits, masks)
        prediction = self._prediction_from_pixel_logits(pixel_logits)
        include_background = not bool(self.cfg.DATASET.EXCLUDE_BACKGROUND_CLASS)
        self._per_class_iou_acc.accumulate_step(prediction, masks)
        metrics = compute_seg_metrics(
            pred=prediction,
            masks=masks,
            num_classes=self.cfg.DATASET.NUM_CLASSES,
            ignore_index=self.cfg.DATASET.IGNORE_INDEX,
            include_background=include_background,
        )
        if self.cfg.DATASET.NUM_CLASSES != 2:
            return metrics

        if self.cfg.TRAIN.LOG_BOUNDARY_IOU:
            metrics["boundary_iou"] = compute_boundary_iou(
                prediction,
                masks,
                num_classes=2,
                ignore_index=self.cfg.DATASET.IGNORE_INDEX,
                boundary_width=self.cfg.TRAIN.BOUNDARY_WIDTH,
            )
        if self.cfg.TRAIN.LOG_CLDICE:
            metrics["cldice"] = compute_cldice(
                prediction, masks, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
        if self.cfg.TRAIN.LOG_THIN_VESSEL_DICE:
            metrics.update(compute_thin_vessel_metrics(
                prediction, masks, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            ))
        return metrics

    @staticmethod
    def _resize_pixel_logits_to_masks(
        pixel_logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        if pixel_logits.shape[-2:] == masks.shape[-2:]:
            return pixel_logits
        return F.interpolate(
            pixel_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

    def _prediction_from_pixel_logits(self, pixel_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.DATASET.NUM_CLASSES == 2:
            return self._binary_pred(pixel_logits)
        return pixel_logits.argmax(dim=1)
