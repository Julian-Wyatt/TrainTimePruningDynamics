"""Training and evaluation engine functions.

These are stateless functions that operate on a trainer object, keeping the
core loop logic separate from trainer state management.
"""

from __future__ import annotations

import contextlib
import time

import torch
import torch.nn as nn

from utils.device import precision_dtype
from utils.distributed import is_main_process, reduce_dict
from utils.metrics import RunningStats


def _optimizer_step(trainer) -> None:
    """Clip gradients, then take one optimizer step."""
    cfg = trainer.cfg

    if cfg.TRAIN.CLIP_GRAD_VAL > 0:
        trainer.scaler.unscale_(trainer.optimizer)
        nn.utils.clip_grad_norm_(
            trainer.model.parameters(), cfg.TRAIN.CLIP_GRAD_VAL)

    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad()


def train_one_epoch(trainer, train_loader, epoch: int, total_steps: int, start_step: int = 0):
    """Run one training epoch with gradient accumulation and mixed precision."""
    cfg = trainer.cfg
    use_amp = trainer.precision in ("bf16", "fp16")
    dtype = precision_dtype(trainer.precision)

    trainer.optimizer.zero_grad()
    accum_loss = 0.0
    accum_loss_tensor: torch.Tensor | None = None
    epoch_loss_sum = 0.0
    epoch_steps = 0
    _has_attn_anneal = hasattr(trainer, "update_attn_mask_annealing")
    _has_no_sync = hasattr(trainer.model, "no_sync")
    epoch_start_time = time.time()

    # A multiple of grad_accum_steps, so resume lands on a clean accumulation
    # boundary and every rank skips identically (no DDP allreduce on skipped batches).
    start_batch_idx = start_step * trainer.grad_accum_steps
    effective_batches = (
        min(cfg.DATASET.OVERFIT_BATCHES, len(train_loader))
        if cfg.DATASET.OVERFIT_BATCHES > 0
        else len(train_loader)
    )
    last_batch_idx = effective_batches - 1

    for batch_idx, batch in enumerate(train_loader):
        # OVERFIT_BATCHES: stop after N batches for quick debugging
        if cfg.DATASET.OVERFIT_BATCHES > 0 and batch_idx >= cfg.DATASET.OVERFIT_BATCHES:
            break

        if batch_idx < start_batch_idx:
            continue

        batch = trainer._to_device(batch)

        accum_offset = (batch_idx - start_batch_idx) % trainer.grad_accum_steps
        accum_window_start = batch_idx - accum_offset
        accum_denom = min(
            trainer.grad_accum_steps,
            last_batch_idx - accum_window_start + 1,
        )
        is_last_accum = (
            accum_offset + 1 == accum_denom
            or batch_idx == last_batch_idx
        )
        sync_ctx = (
            contextlib.nullcontext()
            if is_last_accum or not _has_no_sync
            else trainer.model.no_sync()
        )
        with sync_ctx, torch.autocast(
            device_type=trainer.device.type,
            dtype=dtype,
            enabled=use_amp,
        ):
            loss_dict = trainer.training_step(trainer.model, batch)
            loss = loss_dict["loss"] / accum_denom
        trainer.scaler.scale(loss).backward()
        # Accumulate on-GPU; .item() is deferred to after the optimizer step
        # so the per-microbatch CUDA→CPU sync doesn't serialize DDP ranks.
        loss_d = loss.detach()
        accum_loss_tensor = (
            loss_d if accum_loss_tensor is None else accum_loss_tensor + loss_d
        )

        if not is_last_accum:
            continue

        _optimizer_step(trainer)
        trainer.lr_controller.apply(trainer.optimizer, trainer.global_step, total_steps)

        if trainer.ema is not None:
            trainer.ema.update(trainer.model)

        if _has_attn_anneal:
            trainer.update_attn_mask_annealing(trainer.global_step)

        trainer.global_step += 1
        accum_loss = (
            float(accum_loss_tensor) if accum_loss_tensor is not None else 0.0
        )
        accum_loss_tensor = None
        epoch_loss_sum += accum_loss
        epoch_steps += 1

        # Core per-step logging — rank 0 only
        if is_main_process():
            trainer.step_logger.log_train_step(trainer, accum_loss, loss_dict, epoch)

        accum_loss = 0.0

        if trainer.resubmit.coordinate(trainer.device):
            if is_main_process():
                print(f"[Trainer] requeue requested at epoch={epoch} step={trainer.global_step}.")
            break

    # Epoch-level logging — sync across ranks then log on rank 0
    if epoch_steps > 0:
        epoch_avg = {"loss": epoch_loss_sum / epoch_steps}
        epoch_avg = reduce_dict(epoch_avg)
        epoch_time = time.time() - epoch_start_time
        if is_main_process():
            trainer.step_logger.log_train_epoch(trainer, epoch, epoch_avg["loss"], epoch_time)
        return epoch_avg["loss"], epoch_steps
    return None, 0


def aggregate_epoch_metrics(
    running: RunningStats,
    reduce_fn,
    prefix: str,
) -> tuple[dict, dict, dict]:
    """Reduce running stats across DDP ranks and build a log payload.

    Args:
        running:   RunningStats accumulated over all validation batches.
        reduce_fn: reduce_dict — averages tensors/scalars across DDP ranks.
        prefix:    logging prefix, e.g. "val" or "test".

    Returns:
        avg_metrics  — DDP-reduced mean per key.
        std_metrics  — DDP-reduced population std per key (only keys with > 1 batch).
        log_payload  — ready-to-log dict with keys ``{prefix}_epoch/{k}`` (means)
                       and ``{prefix}_epoch/{k}_std`` (stds).
    """
    avg_metrics = reduce_fn(running.means())
    raw_stds = running.stds()
    std_metrics = reduce_fn(raw_stds) if raw_stds else {}
    log_payload = {f"{prefix}_epoch/{k}": v for k, v in avg_metrics.items()}
    log_payload.update(
        {f"{prefix}_epoch/{k}_std": v for k, v in std_metrics.items()})
    return avg_metrics, std_metrics, log_payload


@torch.inference_mode()
def evaluate(trainer, val_loader, prefix: str = "val") -> dict:
    """Run evaluation over a dataloader and return averaged metrics."""
    cfg = trainer.cfg
    use_amp = trainer.precision in ("bf16", "fp16")
    dtype = precision_dtype(trainer.precision)
    trainer.model.eval()
    running = RunningStats()
    epoch_start_time = time.time()

    # Use EMA for validation only if enabled in config (avoid validation collapse from degenerate EMA)
    use_ema = trainer.ema and cfg.MODEL.get("EMA_VALIDATE", True)
    ctx = trainer.ema.ema_scope(
        trainer.model) if use_ema else contextlib.nullcontext()
    with ctx:
        for batch_idx, batch in enumerate(val_loader):
            if cfg.DATASET.OVERFIT_BATCHES > 0 and batch_idx >= cfg.DATASET.OVERFIT_BATCHES:
                break
            batch = trainer._to_device(batch)
            with torch.autocast(device_type=trainer.device.type, dtype=dtype, enabled=use_amp):
                metrics = trainer.validation_step(trainer.model, batch)
            running.update(metrics)

    # Epoch-level val logging — sync across ranks then log on rank 0
    avg_metrics, std_metrics, log_payload = aggregate_epoch_metrics(
        running, reduce_dict, prefix)
    epoch_time = time.time() - epoch_start_time

    if prefix == "val":
        trainer.state.last_val_epoch_time = epoch_time

    trainer.step_logger.log_eval_epoch(trainer, prefix, log_payload, avg_metrics, std_metrics, epoch_time)

    trainer.model.train()
    return {**avg_metrics, **{f"{k}_std": v for k, v in std_metrics.items()}}
