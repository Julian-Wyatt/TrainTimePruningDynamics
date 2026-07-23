"""EoMT segmentation trainer — EoMT backbone + Mask2Former / CE loss + mIoU metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from data.datamodule import DataModule
from models.backbone import Backbone
from models.EoMT import EoMT
from trainer.base_trainer import register_trainer
from trainer.segmentation_base import BaseSegmentationTrainer


@register_trainer("segmentation")
@register_trainer("eomt")
class EoMTTrainer(BaseSegmentationTrainer):
    def __init__(self, cfg: DictConfig, data_module: DataModule):
        super().__init__(cfg, data_module)

        if cfg.TRAIN.LOSS_TYPE == "cross_entropy":
            from trainer.losses.cross_entropy import SegmentationCELoss
            self.criterion = SegmentationCELoss(ignore_index=cfg.DATASET.IGNORE_INDEX)
        elif cfg.TRAIN.LOSS_TYPE == "ce_dice":
            from trainer.losses.cross_entropy import CEDiceLoss
            dice_weight = cfg.TRAIN.get("DICE_WEIGHT", 1.0)
            self.criterion = CEDiceLoss(
                ignore_index=cfg.DATASET.IGNORE_INDEX,
                dice_weight=float(dice_weight),
            )
        elif cfg.TRAIN.LOSS_TYPE == "mask2former":
            from trainer.losses.mask2former import MaskClassificationLoss
            self.criterion = MaskClassificationLoss(
                num_points=cfg.TRAIN.M2F_NUM_POINTS,
                oversample_ratio=cfg.TRAIN.M2F_OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.TRAIN.M2F_IMPORTANCE_SAMPLE_RATIO,
                mask_coefficient=cfg.TRAIN.M2F_MASK_COEF,
                dice_coefficient=cfg.TRAIN.M2F_DICE_COEF,
                class_coefficient=cfg.TRAIN.M2F_CLASS_COEF,
                num_labels=cfg.DATASET.NUM_CLASSES,
                no_object_coefficient=cfg.TRAIN.M2F_NO_OBJECT_COEF,
            )
        else:
            raise ValueError(f"Unknown loss type: {cfg.TRAIN.LOSS_TYPE}")

        self.criterion = self.criterion.to(self.device)
        self._eomt_pruning_epoch = -1

    def build_model(self, cfg: DictConfig) -> nn.Module:
        encoder = Backbone(
            img_size=tuple(cfg.DATASET.IMG_SIZE),
            patch_size=cfg.MODEL.PATCH_SIZE,
            backbone_name=cfg.MODEL.BACKBONE,
            ckpt_path=cfg.MODEL.CKPT_PATH or None,
            drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
        )
        if cfg.MODEL.get("USE_EOMT_PRUNING", False):
            from models.EoMT.EoMT_pruning import EoMTPruning
            model = EoMTPruning(encoder, cfg)
        else:
            model = EoMT.from_cfg(encoder, cfg)
        self._maybe_freeze_binary_m2f_class_head(model, cfg)
        return model

    @staticmethod
    def _maybe_freeze_binary_m2f_class_head(model: nn.Module, cfg: DictConfig) -> None:
        if (
            int(cfg.DATASET.NUM_CLASSES) == 2
            and str(cfg.TRAIN.LOSS_TYPE).lower() == "mask2former"
            and float(cfg.TRAIN.get("M2F_CLASS_COEF", 1.0)) == 0.0
        ):
            class_head = getattr(model, "class_head", None)
            if class_head is not None:
                # Binary mask-only decoding ignores class logits; with no class
                # loss these params are intentionally unused and trip DDP.
                for param in class_head.parameters():
                    param.requires_grad_(False)

    def shared_step(self, model: nn.Module, batch: dict) -> dict:
        """Forward pass — returns per-layer logits."""
        images = batch["image"]
        mask_logits_per_layer, class_logits_per_layer = model(images)
        return {
            "mask_logits_per_layer": mask_logits_per_layer,
            "class_logits_per_layer": class_logits_per_layer,
        }

    def throughput_step(self, model: nn.Module, batch: dict) -> None:
        """Single-decode forward pass for throughput measurement."""
        model(
            batch["image"],
            throughput_mode=True,
            predict_class=not self._uses_binary_mask_only_decoding(),
        )

    def _compute_seg_loss(self, mask_logits_per_layer, class_logits_per_layer, masks, batch) -> tuple:
        """Compute segmentation loss. Returns (total_loss, loss_dict)."""
        if self.cfg.TRAIN.LOSS_TYPE in ("cross_entropy", "ce_dice"):
            loss = self.criterion(mask_logits_per_layer[-1], masks)
            return loss, {"loss/seg": loss.detach()}

        targets = batch.get("targets") or self._build_targets(masks)
        targets = self._move_targets_to_device(targets, masks.device)
        losses_all = {}
        n = len(mask_logits_per_layer)
        for i, (ml, cl) in enumerate(zip(mask_logits_per_layer, class_logits_per_layer)):
            rel_idx = i - (n - 1)
            for k, v in self.criterion(ml, cl, targets).items():
                losses_all[f"{k}_qblock{rel_idx}"] = v
        return self.criterion.loss_total(losses_all)

    def _maybe_update_pruning_curriculum(self, model: nn.Module) -> None:
        if not self.cfg.MODEL.get("USE_EOMT_PRUNING", False):
            return
        if self.current_epoch == self._eomt_pruning_epoch:
            return
        from utils.device import unwrap_model
        update_fn = getattr(unwrap_model(model), "set_pruning_epoch", None)
        if callable(update_fn):
            update_fn(self.current_epoch)
            self._eomt_pruning_epoch = self.current_epoch

    def training_step(self, model: nn.Module, batch: dict) -> dict:
        self._maybe_update_pruning_curriculum(model)
        masks = batch["mask"]
        out = self.shared_step(model, batch)
        ml_per = out["mask_logits_per_layer"]
        cl_per = out["class_logits_per_layer"]
        total_loss, loss_dict = self._compute_seg_loss(ml_per, cl_per, masks, batch)
        from utils.device import unwrap_model
        pruning_loss_fn = getattr(unwrap_model(model), "get_pruning_loss", None)
        if callable(pruning_loss_fn):
            pruning_loss = pruning_loss_fn()
            if pruning_loss is not None:
                total_loss = total_loss + pruning_loss
                loss_dict["loss/pruning_budget"] = pruning_loss.detach()

        # CROPR: directly supervise the detached cross-attention score at each stage.
        aux_pred_fn = getattr(unwrap_model(model), "get_aux_predictions", None)
        if callable(aux_pred_fn):
            aux_preds = aux_pred_fn()
            if aux_preds:
                aux_spatial_ids_fn = getattr(
                    unwrap_model(model), "get_aux_spatial_ids", None)
                aux_spatial_ids = (
                    aux_spatial_ids_fn() if callable(aux_spatial_ids_fn) else [])
                aux_bce = self._aux_bce_loss(aux_preds, masks, aux_spatial_ids)
                if aux_bce is not None:
                    weight = float(self.cfg.MODEL.EOMT_PRUNE_AUX_LOSS_WEIGHT)
                    loss_dict["loss/aux_bce_raw"] = aux_bce.detach()
                    aux_bce = weight * aux_bce
                    total_loss = total_loss + aux_bce
                    loss_dict["loss/aux_bce"] = aux_bce.detach()

        return {"loss": total_loss, **loss_dict}

    def _aux_bce_loss(
        self,
        aux_preds: list,
        masks: torch.Tensor,
        aux_spatial_ids: list[torch.Tensor],
    ) -> torch.Tensor | None:
        """Mean class-balanced BCE on CROPR's raw scorer logits."""
        if not aux_preds or len(aux_preds) != len(aux_spatial_ids):
            return None

        grid_spatial = aux_spatial_ids[0].shape[1]
        grid_size = int(round(grid_spatial**0.5))
        if grid_size * grid_size != grid_spatial:
            return None
        ignore_index = self.cfg.DATASET.IGNORE_INDEX
        foreground = ((masks > 0) & (masks != ignore_index)).float().unsqueeze(1)
        target = F.adaptive_max_pool2d(foreground, (grid_size, grid_size)).squeeze(1)
        valid = F.adaptive_max_pool2d(
            (masks != ignore_index).float().unsqueeze(1), (grid_size, grid_size)
        ).squeeze(1).bool()
        target[~valid] = ignore_index
        target = target.reshape(target.shape[0], -1)

        terms = []
        for pred, spatial_ids in zip(aux_preds, aux_spatial_ids):
            valid_tokens = target.gather(1, spatial_ids.to(target.device)) != ignore_index
            if not bool(valid_tokens.any()):
                continue
            logits = pred.squeeze(-1)[valid_tokens]
            labels = target.gather(1, spatial_ids.to(target.device))[valid_tokens]
            pos = labels.sum()
            pos_weight = ((labels.numel() - pos) / pos.clamp_min(1.0)).clamp(1.0, 50.0)
            terms.append(
                F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )
            )
        return torch.stack(terms).mean() if terms else None

    @staticmethod
    def _move_targets_to_device(targets: list[dict], device) -> list[dict]:
        return [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

    # ── BaseSegmentationTrainer Implementation ───────────────────────────────

    def get_pixel_logits(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """EoMT-specific: derive pixel logits from last layer query outputs."""
        out = self.shared_step(model, batch)
        return self._derive_pixel_logits(out["mask_logits_per_layer"][-1], out["class_logits_per_layer"][-1])

    def _derive_pixel_logits(self, mask_logits, class_logits) -> torch.Tensor:
        """Derive pixel-level semantic logits from query-based mask and class logits."""
        mask_logits_up = F.interpolate(
            mask_logits, list(self.img_size), mode="bilinear", align_corners=False
        )
        if self._uses_binary_mask_only_decoding():
            fg_probs = mask_logits_up.sigmoid().amax(dim=1)
            fg_channel = int(self.cfg.DATASET.get("BINARY_FG_CHANNEL", 1))
            if fg_channel not in (0, 1):
                raise ValueError(f"BINARY_FG_CHANNEL must be 0 or 1, got {fg_channel}")
            pixel_probs = torch.empty(
                mask_logits_up.shape[0],
                2,
                *mask_logits_up.shape[-2:],
                dtype=fg_probs.dtype,
                device=fg_probs.device,
            )
            pixel_probs[:, fg_channel] = fg_probs
            pixel_probs[:, 1 - fg_channel] = 1.0 - fg_probs
            return pixel_probs

        pixel_logits = torch.einsum(
            "bqhw,bqc->bchw",
            mask_logits_up.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )
        return pixel_logits

    def _uses_binary_mask_only_decoding(self) -> bool:
        loss_type = str(self.cfg.TRAIN.get("LOSS_TYPE", "")).lower()
        return self.cfg.DATASET.NUM_CLASSES == 2 and loss_type == "mask2former"

    # ── M2F Specific logic ───────────────────────────────────────────────────

    def _build_targets(self, masks: torch.Tensor) -> list[dict]:
        """Convert a dense segmentation mask tensor to Mask2Former target format."""
        targets = []
        ignore = self.cfg.DATASET.IGNORE_INDEX
        for mask in masks:
            classes_all = mask.unique()
            classes = classes_all[(classes_all != ignore) & (classes_all > 0)]
            if len(classes) == 0:
                H, W = mask.shape[-2:]
                binary_masks = torch.zeros((0, H, W), dtype=torch.float, device=mask.device)
            else:
                binary_masks = (mask.unsqueeze(0) == classes.view(-1, 1, 1)).float()
            targets.append({
                "labels": classes.long() if len(classes) > 0 else torch.zeros(0, dtype=torch.long, device=mask.device),
                "masks": binary_masks
            })
        return targets

    # ── Annealing ──────────────────────────────────────────────────────────

    def update_attn_mask_annealing(self, step: int) -> None:
        """Delegate attention mask annealing to the EoMT model."""
        from utils.device import unwrap_model
        from utils.distributed import is_main_process

        model = unwrap_model(self.model)
        if not hasattr(model, "update_attn_mask_probs"):
            return

        spe = max(self.state.steps_per_epoch, 1)
        cfg_copy = self.cfg.copy()

        # Scale epoch-based configs to step-based if needed
        for key in ["ATTN_MASK_ANNEALING_START_STEPS", "ATTN_MASK_ANNEALING_END_STEPS"]:
            vals = list(cfg_copy.MODEL.get(key, []))
            # Step-based schedules may start at 0, so inspect the full range.
            if vals and max(vals) <= int(cfg_copy.TRAIN.EPOCHS):
                cfg_copy.MODEL[key] = [v * spe for v in vals]

        log_probs = model.update_attn_mask_probs(step, cfg_copy)
        if log_probs and is_main_process():
            self.log_metrics(log_probs, step)
