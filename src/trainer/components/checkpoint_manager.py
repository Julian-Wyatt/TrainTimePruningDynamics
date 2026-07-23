"""Checkpoint handling for trainers."""

from __future__ import annotations

import os

from trainer.components.state import TrainerState
from utils.checkpoint import resume_from_checkpoint, save_checkpoint
from utils.distributed import is_main_process


class CheckpointManager:
    """Encapsulate checkpoint save/resume and best-metric tracking."""

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        ema,
        scaler,
        state: TrainerState,
        logger=None,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self.scaler = scaler
        self.state = state
        self.logger = logger

    def resume_if_available(self) -> None:
        if self.cfg.TRAIN.CHECKPOINT_FILE:
            epoch, step, best_metric_val, cumulative_train_seconds, step_in_epoch = resume_from_checkpoint(
                self.cfg.TRAIN.CHECKPOINT_FILE,
                self.model,
                self.optimizer,
                self.ema,
                self.scaler,
            )
            self.state.start_epoch = epoch
            self.state.global_step = step
            self.state.best_metric_val = best_metric_val
            self.state.cumulative_train_seconds = cumulative_train_seconds
            self.state.start_step_in_epoch = step_in_epoch

    def build_save_dir(self) -> str:
        run_id = self.cfg.TRAIN.get("RUN_ID") or "run"
        save_dir = os.path.join(self.cfg.TRAIN.SAVING_ROOT_DIR, "checkpoints", run_id)
        if is_main_process() and self.cfg.TRAIN.SAVE_CHECKPOINTS == "true":
            os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def maybe_save_best(self, save_dir: str, epoch: int, metric_key: str, metric_val: float) -> None:
        if metric_val <= self.state.best_metric_val:
            return
        self.state.best_metric_val = metric_val
        if self.cfg.TRAIN.SAVE_CHECKPOINTS != "true":
            print(f"  → New best {metric_key}={metric_val:.4f} (checkpointing disabled)")
            return

        ckpt_path = os.path.join(save_dir, f"best_epoch{epoch}.pt")
        save_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            epoch,
            self.state.global_step,
            self.ema,
            self.scaler,
            self.cfg,
            best_metric_val=self.state.best_metric_val,
            cumulative_train_seconds=self.state.cumulative_train_seconds,
            step_in_epoch=0,
        )
        self.state.best_ckpt_path = ckpt_path

        # Symlink best.pt → latest best checkpoint for easy reference
        symlink = os.path.join(save_dir, "best.pt")
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(os.path.basename(ckpt_path), symlink)
        print(f"  → New best {metric_key}={metric_val:.4f}, saved to {ckpt_path}")

        # Top-K checkpoint pruning
        max_k = int(self.cfg.TRAIN.get("MAX_SAVED_CHECKPOINTS", 1))
        self.state.saved_ckpt_paths.append(ckpt_path)
        if max_k > 0 and len(self.state.saved_ckpt_paths) > max_k:
            oldest = self.state.saved_ckpt_paths.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

    def save_last_complete(self, save_dir: str, epoch: int, step_in_epoch: int = 0) -> None:
        if not is_main_process():
            return
        requeue_dir = os.path.join(self.cfg.TRAIN.SAVING_ROOT_DIR, "requeue")
        os.makedirs(requeue_dir, exist_ok=True)
        job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
        self.state.last_complete_ckpt = os.path.join(requeue_dir, f"last_complete_{job_id}.pt")
        save_checkpoint(
            self.state.last_complete_ckpt,
            self.model,
            self.optimizer,
            epoch,
            self.state.global_step,
            self.ema,
            self.scaler,
            self.cfg,
            best_metric_val=self.state.best_metric_val,
            cumulative_train_seconds=self.state.cumulative_train_seconds,
            step_in_epoch=step_in_epoch,
        )

    def save_train_end(self, save_dir: str, epoch: int) -> None:
        if not is_main_process() or self.cfg.TRAIN.SAVE_CHECKPOINTS != "true":
            return

        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "train_end.pt")
        save_checkpoint(
            ckpt_path,
            self.model,
            self.optimizer,
            epoch,
            self.state.global_step,
            self.ema,
            self.scaler,
            self.cfg,
            best_metric_val=self.state.best_metric_val,
            cumulative_train_seconds=self.state.cumulative_train_seconds,
            step_in_epoch=0,
        )
        self.state.train_end_ckpt_path = ckpt_path
        print(f"  → Train-end checkpoint saved to {ckpt_path}")
