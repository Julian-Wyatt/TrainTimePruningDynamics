"""Base trainer with manual DDP, gradient accumulation, mixed precision, and EMA."""

from __future__ import annotations

import gc
import json
import math
import os
import time
import traceback
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from data.datamodule import DataModule
from optim.factory import build_optimizer
from optim.schedules import build_schedule
from trainer.components import (
    CheckpointManager,
    LRScheduleController,
    StepLogger,
    TrainerState,
)
from trainer.engine import evaluate, train_one_epoch
from utils.device import get_device
from utils.distributed import is_main_process
from utils.ema import ModelEMA
from utils.logger import ExperimentLogger
from utils.resubmit import ResubmitManager


def _set_backbone_stem_frozen(model: nn.Module, cfg: DictConfig, freeze: bool) -> None:
    """Freeze or unfreeze the first N backbone blocks and embedding params.

    Controlled by MODEL.FREEZE_BACKBONE_N_BLOCKS (default 0 = no-op).
    """
    n = int(cfg.MODEL.get("FREEZE_BACKBONE_N_BLOCKS", 0))
    if n <= 0:
        return

    backbone = getattr(getattr(model, "encoder", None), "backbone", None)
    if backbone is None:
        return

    blocks = getattr(backbone, "blocks", [])
    for block in list(blocks)[:n]:
        for p in block.parameters():
            p.requires_grad = not freeze

    _embed_attrs = ("patch_embed", "cls_token", "reg_token", "pos_embed", "dist_token")
    for attr in _embed_attrs:
        m = getattr(backbone, attr, None)
        if m is None:
            continue
        params = [m] if isinstance(m, nn.Parameter) else m.parameters()
        for p in params:
            p.requires_grad = not freeze

    if freeze:
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        print(
            f"[freeze] Froze first {n} backbone blocks + embeddings ({n_frozen} tensors frozen)"
        )
    else:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(
            f"[unfreeze] Unfroze first {n} backbone blocks + embeddings ({n_trainable} tensors trainable)"
        )


TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {}


def register_trainer(name: str):
    """Decorator to register a Trainer class in the global registry."""

    def decorator(cls):
        TRAINER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class BaseTrainer(ABC):
    """Manual DDP trainer — replaces PyTorch Lightning.

    Subclasses must implement:
        - ``build_model(cfg) -> nn.Module``
        - ``compute_loss(model, batch) -> dict`` (must include ``"loss"`` key)
        - ``compute_metrics(model, batch) -> dict``
    """

    def __init__(self, cfg: DictConfig, data_module: DataModule):
        self.cfg = cfg
        self.data_module = data_module
        self.device = torch.device(get_device())

        # Build model
        self.model = self.build_model(cfg).to(self.device)
        _set_backbone_stem_frozen(self.model, cfg, freeze=True)
        if is_main_process():
            self._print_param_summary(self.model)
        self._deferred_compile_pending = False
        self._deferred_compile_dynamic = None
        if cfg.TRAIN.COMPILE:
            self._compile_model()

        self.optimizer = self._build_optimizer()

        # Record initial_lr on each param group for LR scheduling
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        # Schedule (progress-based: 0→1)
        self.schedule_fn = self._build_schedule()
        self.lr_controller = LRScheduleController(cfg, self.schedule_fn)

        # Mixed precision — auto-detect: bf16 on Ampere+, fp16 on older CUDA, fp32 on MPS/CPU
        if get_device() == "cuda":
            self.precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            self.precision = "fp32"
        print(f"[Trainer] precision={self.precision}")
        scaler_device = "cuda" if get_device() == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            scaler_device, enabled=(self.precision == "fp16")
        )

        # EMA — enabled whenever EMA_DECAY != 1.0
        self.ema = None
        if 0 < cfg.MODEL.EMA_DECAY < 1.0:
            self.ema = ModelEMA(self.model, decay=cfg.MODEL.EMA_DECAY)

        # Gradient accumulation
        effective_bs = cfg.TRAIN.EFFECTIVE_BATCH_SIZE or cfg.TRAIN.BATCH_SIZE
        if (
            cfg.TRAIN.EFFECTIVE_BATCH_SIZE
            and cfg.TRAIN.EFFECTIVE_BATCH_SIZE < cfg.TRAIN.BATCH_SIZE
        ):
            import warnings

            warnings.warn(
                f"EFFECTIVE_BATCH_SIZE ({cfg.TRAIN.EFFECTIVE_BATCH_SIZE}) < BATCH_SIZE "
                f"({cfg.TRAIN.BATCH_SIZE}); grad_accum_steps clamped to 1.",
                stacklevel=2,
            )
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        self.grad_accum_steps = max(
            1, effective_bs // (cfg.TRAIN.BATCH_SIZE * num_gpus)
        )

        self.resubmit = ResubmitManager()
        self._load_requeue_state()

        # Logging
        self.logger = ExperimentLogger(cfg)
        self.state = TrainerState()
        self.step_logger = StepLogger(cfg)

        from utils.metrics import PerClassIoUAccumulator

        self._per_class_iou_acc = PerClassIoUAccumulator(
            num_classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_INDEX,
        )

        # Checkpoint manager (depends on optimizer/scaler/ema)
        self.checkpoint_manager = CheckpointManager(
            cfg,
            self.model,
            self.optimizer,
            self.ema,
            self.scaler,
            self.state,
            logger=self.logger,
        )

    def _prepare_compile(self) -> bool:
        if get_device() != "cuda":
            print("[Trainer] torch.compile: disabled on MPS/CPU")
            return False

        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            print(f"[Trainer] torch.compile: skipped (GPU cap {cap[0]}.{cap[1]} < 8.0)")
            return False

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.cache_size_limit = 64
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = max(
                64,
                int(getattr(torch._dynamo.config, "accumulated_cache_size_limit", 0)),
            )
        return True

    def _compile_model(self) -> None:
        try:
            compile_stable = getattr(self.model, "compile_stable_submodules", None)
            strategy, dynamic = self._compile_policy(callable(compile_stable))
            if strategy == "deferred":
                self._deferred_compile_pending = True
                self._deferred_compile_dynamic = dynamic
                ramp_epochs = int(self.cfg.MODEL.get("EOMT_PRUNE_RAMP_EPOCHS", 0))
                print(
                    "[Trainer] torch.compile: deferred routed-pruning compile "
                    f"until epoch >= {ramp_epochs} (dynamic={dynamic})"
                )
                return

            if not self._prepare_compile():
                return

            if strategy == "stable":
                n_compiled = compile_stable(dynamic=dynamic, fullgraph=False)
                print(
                    f"[Trainer] torch.compile: {n_compiled} stable submodule(s) "
                    f"(dynamic={dynamic})"
                )
                return
            self.model = torch.compile(self.model, fullgraph=False, dynamic=dynamic)
            print(f"[Trainer] torch.compile: full model (dynamic={dynamic})")
        except Exception:
            print("[Trainer] torch.compile: FAILED")
            print(traceback.format_exc())

    def _compile_policy(self, has_stable_compile: bool) -> tuple[str, bool | None]:
        """Choose compile scope and shape policy for train/val throughput."""
        model_cfg = self.cfg.MODEL
        train_mode = str(model_cfg.get("EOMT_PRUNE_TRAIN_MODE", "dense")).lower()

        if model_cfg.get("USE_EOMT_PRUNING", False):
            # These keep the full token set, so shapes are static from epoch 0.
            static_shape_modes = {"dense", "soft_mask"}
            if train_mode in static_shape_modes:
                return "full", None
            # Gathering modes ramp keep rates over EOMT_PRUNE_RAMP_EPOCHS, so run
            # eager through the anneal and compile once shapes freeze.
            return "deferred", False

        # Dense EoMT has stable train shapes, so validation can share the compiled
        # wrapper; this also covers future models exposing the same hook.
        if hasattr(self.model, "_num_backbone_blocks"):
            return "full", None

        if has_stable_compile:
            return "stable", None

        return "full", True

    def _maybe_compile_deferred_model(self, epoch: int) -> None:
        if not getattr(self, "_deferred_compile_pending", False):
            return

        ramp_epochs = int(self.cfg.MODEL.get("EOMT_PRUNE_RAMP_EPOCHS", 0))
        if epoch < ramp_epochs:
            return

        self._deferred_compile_pending = False
        dynamic = getattr(self, "_deferred_compile_dynamic", False)
        try:
            if not self._prepare_compile():
                return

            if hasattr(self.model, "module"):
                self.model.module = torch.compile(
                    self.model.module, dynamic=dynamic, fullgraph=False
                )
                print(
                    "[Trainer] torch.compile: DDP inner module "
                    f"(dynamic={dynamic})"
                )
                return

            self.model = torch.compile(self.model, dynamic=dynamic, fullgraph=False)
            self.checkpoint_manager.model = self.model
            print(f"[Trainer] torch.compile: full model (dynamic={dynamic})")
        except Exception:
            print("[Trainer] torch.compile: FAILED")
            print(traceback.format_exc())

    @property
    def global_step(self) -> int:
        return self.state.global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        self.state.global_step = value

    @property
    def current_epoch(self) -> int:
        return self.state.current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        self.state.current_epoch = value

    @abstractmethod
    def build_model(self, cfg: DictConfig) -> nn.Module: ...

    @abstractmethod
    def shared_step(self, model: nn.Module, batch: dict) -> dict:
        """Forward pass shared by train and val. Returns a dict with at minimum
        ``"logits"`` and ``"loss"`` keys (loss may be None during val)."""
        ...

    @abstractmethod
    def training_step(self, model: nn.Module, batch: dict) -> dict:
        """Apply train-only augmentation then call shared_step.
        Must return a dict with a ``"loss"`` key."""
        ...

    @abstractmethod
    def validation_step(self, model: nn.Module, batch: dict) -> dict:
        """Call shared_step and compute eval metrics. Returns a metrics dict."""
        ...

    def throughput_step(self, model: nn.Module, batch: dict) -> None:
        """Minimal forward pass for throughput measurement — no loss, no metrics.

        Override in subclasses to use model-specific optimisations (e.g. EoMT
        single-decode mode).  The default calls the model with the image tensor.
        """
        model(batch["image"])

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = OmegaConf.create(
            {
                "optimizer": {
                    "name": self.cfg.TRAIN.OPTIMIZER,
                    "embedding_lr": self.cfg.TRAIN.LR,
                    "scalar_lr": self.cfg.TRAIN.LR,
                    "weight_decay": self.cfg.TRAIN.WEIGHT_DECAY,
                    "adam_betas": list(self.cfg.TRAIN.ADAM_BETAS),
                    "momentum": self.cfg.TRAIN.MOMENTUM,
                    "llrd": self.cfg.TRAIN.LLRD,
                    "lr_mult": float(self.cfg.TRAIN.get("LR_MULT", 1.0)),
                    "llrd_full_lr": bool(
                        self.cfg.TRAIN.get("LLRD_FULL_LR_LAST_N_BLOCKS", True)
                    ),
                    "original_eomt_lr_mult_compat": bool(
                        self.cfg.TRAIN.get("ORIGINAL_EOMT_LR_MULT_COMPAT", False)
                    ),
                },
            }
        )
        return build_optimizer(self.model, opt_cfg)

    def _build_schedule(self):
        min_lr_frac = float(self.cfg.TRAIN.get("MIN_LR_FRAC", 0.0))
        schedule_name = self.cfg.TRAIN.SCHEDULE_NAME
        if schedule_name == "cosine":
            schedule_name = "polynomial"
        # When SPLIT_WARMUP is active, LRScheduleController handles all warmup phases; the schedule
        # function is only applied to the post-warmup tail and must not add its own warmup.
        warmup_ratio = (
            0.0
            if self.cfg.TRAIN.get("SPLIT_WARMUP", False)
            else self.cfg.TRAIN.WARMUP_RATIO
        )
        schedule_cfg = OmegaConf.create(
            {
                "schedule": {
                    "name": schedule_name,
                    "warmup_ratio": warmup_ratio,
                    "warmdown_ratio": self.cfg.TRAIN.WARMDOWN_RATIO,
                    "final_lr_frac": min_lr_frac,
                    "eta_min": min_lr_frac,
                    "power": 0.9,
                },
            }
        )
        return build_schedule(schedule_cfg)

    def _load_requeue_state(self) -> None:
        """Restore a checkpoint and run identity from a prior Slurm requeue."""
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return
        sidecar = os.path.join(self.cfg.TRAIN.SAVING_ROOT_DIR, f"requeue_{job_id}.json")

        if dist.is_available() and dist.is_initialized():
            payload = [None]
            if is_main_process() and os.path.exists(sidecar):
                with open(sidecar) as file:
                    payload[0] = json.load(file)
            dist.broadcast_object_list(payload, src=0)
            state = payload[0]
        elif os.path.exists(sidecar):
            with open(sidecar) as file:
                state = json.load(file)
        else:
            state = None
        if state is None:
            return

        self.cfg.TRAIN.CHECKPOINT_FILE = state["ckpt_path"]
        self.cfg.TRAIN.RUN_ID = state["run_id"]
        self.cfg.TRAIN.FRESH_RUN = False
        self.cfg.TRAIN.MAX_RESUBMIT = state["resubmits_left"]
        if is_main_process():
            os.remove(sidecar)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def log_metrics(self, metrics: dict, step: int):
        self.logger.log(metrics, step)

    def setup_ddp(self, rank: int, world_size: int):
        self.model = DDP(self.model, device_ids=[rank], gradient_as_bucket_view=True)
        self.checkpoint_manager.model = self.model

    def train(self):
        cfg = self.cfg

        self._job_start_time = time.time()
        self.resubmit.register()
        self.checkpoint_manager.resume_if_available()
        save_dir = self.checkpoint_manager.build_save_dir()

        train_loader = self.data_module.train_loader()
        effective_batches = (
            min(cfg.DATASET.OVERFIT_BATCHES, len(train_loader))
            if cfg.DATASET.OVERFIT_BATCHES > 0
            else len(train_loader)
        )
        self.state.steps_per_epoch = math.ceil(
            effective_batches / self.grad_accum_steps
        )
        total_steps = cfg.TRAIN.EPOCHS * self.state.steps_per_epoch

        self.lr_controller.apply(self.optimizer, self.global_step, total_steps)

        for epoch in range(self.state.start_epoch, cfg.TRAIN.EPOCHS):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            self.current_epoch = (
                epoch  # Always up-to-date for subclass use during training
            )

            # Check if we should unfreeze backbone at this epoch
            unfreeze_epoch = int(cfg.MODEL.get("UNFREEZE_BACKBONE_EPOCH", -1))
            if unfreeze_epoch >= 0 and epoch == unfreeze_epoch:
                _set_backbone_stem_frozen(self.model, cfg, freeze=False)

            self._maybe_compile_deferred_model(epoch)

            self.model.train()
            start_step = (
                self.state.start_step_in_epoch if epoch == self.state.start_epoch else 0
            )
            epoch_loss, epoch_steps_done = train_one_epoch(
                self, train_loader, epoch, total_steps, start_step=start_step
            )
            if epoch_loss is not None:
                self.state.last_epoch_loss = epoch_loss
                if epoch_loss < self.state.best_train_loss:
                    self.state.best_train_loss = epoch_loss

            requeue_requested = self.resubmit.coordinate(self.device)
            if requeue_requested:
                if is_main_process():
                    print("[Trainer] requeue requested; saving current checkpoint.", flush=True)
                del train_loader
                gc.collect()

            val_metrics = {}
            if not requeue_requested and (epoch + 1) % cfg.TRAIN.VAL_EVERY_N_EPOCHS == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self.state.last_val_metrics = val_metrics

            if is_main_process() and val_metrics:
                metric_key = cfg.TRAIN.BEST_METRIC
                metric_val = val_metrics.get(metric_key, None)

                self.step_logger.log_val_summary(
                    self, val_metrics, self.state.last_val_epoch_time
                )
                if metric_val is not None:
                    self.checkpoint_manager.maybe_save_best(
                        save_dir, epoch + 1, metric_key, metric_val
                    )

            # Record either a normal epoch boundary or a requeue-resume point.
            self.state.cumulative_train_seconds += time.time() - self._job_start_time
            self._job_start_time = time.time()
            if requeue_requested:
                self.checkpoint_manager.save_last_complete(
                    save_dir, epoch, step_in_epoch=epoch_steps_done
                )
            elif (epoch + 1) % cfg.TRAIN.CKPT_EVERY_N_EPOCHS == 0:
                self.checkpoint_manager.save_last_complete(
                    save_dir, epoch + 1, step_in_epoch=0
                )

            if requeue_requested:
                if is_main_process():
                    self.resubmit.requeue(
                        self.state.last_complete_ckpt,
                        cfg.TRAIN.get("RUN_ID", ""),
                        cfg.TRAIN.SAVING_ROOT_DIR,
                        int(cfg.TRAIN.get("MAX_RESUBMIT", 0)),
                        epochs_done=epoch,
                        total_epochs=cfg.TRAIN.EPOCHS,
                    )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                break

            gc.collect()
        else:
            self.checkpoint_manager.save_train_end(save_dir, cfg.TRAIN.EPOCHS)

    def validate(self) -> dict:
        return evaluate(self, self.data_module.val_loader(), prefix="val")

    def test(self) -> dict:
        loader = self.data_module.test_loader()
        if loader is not None:
            return evaluate(self, loader, prefix="test")
        return {}

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                out[k] = [t.to(self.device, non_blocking=True) for t in v]
            else:
                out[k] = v
        return out

    @staticmethod
    def _print_param_summary(model: nn.Module) -> None:
        """Print a compact parameter summary: one line per top-level named child module.

        ModuleLists report per-block and total counts. Frozen parameters are annotated.
        """
        lines = []
        for name, module in model.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if isinstance(module, nn.ModuleList):
                n = len(module)
                per_block = total // n if n > 0 else 0
                lines.append(
                    f"  {n}x {name}: {per_block / 1e6:.2f}M each  "
                    f"({total / 1e6:.2f}M total)"
                )
            else:
                frozen = (
                    f"  [{(total - trainable) / 1e6:.2f}M frozen]"
                    if trainable < total
                    else ""
                )
                lines.append(f"  {name}: {total / 1e6:.2f}M params{frozen}")
        grand_total = sum(p.numel() for p in model.parameters())
        trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[Model] Parameter summary:")
        for line in lines:
            print(line)
        print(
            f"  {'─' * 40}\n"
            f"  TOTAL  {grand_total / 1e6:.2f}M "
            f"({trainable_total / 1e6:.2f}M trainable, "
            f"{(grand_total - trainable_total) / 1e6:.2f}M frozen)"
        )
