"""Experiment logger supporting W&B or TensorBoard."""

from __future__ import annotations

import os
import pathlib
import traceback

from omegaconf import DictConfig, OmegaConf

from .device import get_device, gpu_info
from .distributed import is_main_process


class ExperimentLogger:
    """Thin wrapper around W&B or TensorBoard.

    Only logs on the main process. Safe to call on all ranks.

    Args:
        cfg: Full training config (``cfg.TRAIN.LOG_TYPE`` controls backend).
            Valid values: ``"wandb"``, ``"tensorboard"`` / ``"tb"``, ``"auto"``.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._wandb = None
        self._tb = None

        if not is_main_process():
            return

        log_type = cfg.TRAIN.LOG_TYPE
        if log_type == "auto":
            log_type = "wandb" if get_device() == "cuda" else "tensorboard"
        saving_dir = cfg.TRAIN.SAVING_ROOT_DIR

        if log_type == "wandb":
            self._init_wandb(cfg, saving_dir)
        elif log_type in ("tensorboard", "tb"):
            self._init_tensorboard(cfg, saving_dir)
        elif log_type in ("none", "null"):
            pass  # no-op logger for timing / smoke tests
        else:
            raise ValueError(
                f"Unknown LOG_TYPE {cfg.TRAIN.LOG_TYPE!r}. "
                "Expected 'wandb', 'tensorboard' / 'tb', 'auto', or 'none'."
            )

    def _log_code_to_wandb(self):
        """Log all Python source files in src/ directory to W&B."""
        if self._wandb is None:
            return
        try:
            src_dir = pathlib.Path(__file__).parent.parent
            if not src_dir.exists():
                print(f"Warning: src/ directory not found at {src_dir}")
                return
            print(f"Logging code from {src_dir} to W&B...")
            self._wandb.run.log_code(root=str(src_dir))
            print("Code logging complete.")
        except Exception as e:
            print(f"Code logging failed: {e}")

    def _init_wandb(self, cfg: DictConfig, saving_dir: str):
        try:
            import wandb
            os.makedirs(os.path.join(saving_dir, "wandb"), exist_ok=True)
            run_id = cfg.TRAIN.get("RUN_ID") or None
            config_name = cfg.get("config_name") or ""
            # Build wandb run name: [desc /] config_name / date-suffix
            parts = []
            if cfg.TRAIN.DESCRIPTION:
                parts.append(cfg.TRAIN.DESCRIPTION)
            if config_name:
                # Use only the leaf filename, not the full path
                parts.append(os.path.basename(config_name))
            if run_id:
                parts.append(run_id)
            name = " | ".join(parts) if parts else None
            wandb_id = run_id.replace(
                "/", "-").replace(":", "-") if run_id else None
            # Fresh runs use "allow" so a prior crash with the same run ID can
            # reconnect cleanly; explicit resubmits still require the existing run.
            is_fresh = cfg.TRAIN.get("FRESH_RUN", True)
            wandb_resume = "allow" if is_fresh else "must"
            project = cfg.TRAIN.get("PROJECT", "") or None
            if run_id and project:
                print(
                    f"[W&B] init requested | run_id={run_id} | resume={wandb_resume} | "
                    f"project={project}",
                    flush=True,
                )
            # Group seeds under EXPERIMENT_ID so the W&B group matches the
            # results-table experiment_id column.
            wandb_group = cfg.TRAIN.get("EXPERIMENT_ID", "") or None
            results_group = cfg.TRAIN.get("RESULTS_GROUP", "") or None
            wandb_kwargs = dict(
                id=wandb_id,
                name=name,
                group=wandb_group,
                tags=[results_group] if results_group else None,
                notes=cfg.TRAIN.DESCRIPTION or None,
                dir=os.path.join(saving_dir, "wandb"),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume=wandb_resume,
                settings=wandb.Settings(console="auto"),
            )
            if project:
                wandb_kwargs["project"] = project
            wandb.init(**wandb_kwargs)
            wandb.config.update(
                {f"system/{key}": value for key, value in gpu_info().items()},
                allow_val_change=True,
            )
            self._wandb = wandb
            # Log source code from src/ directory
            self._log_code_to_wandb()
            print("W&B initialized.")
        except Exception as e:
            print(f"W&B init failed: {e}")
            print(traceback.format_exc())

    def _init_tensorboard(self, cfg: DictConfig, saving_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_id = cfg.TRAIN.get("RUN_ID") or "run"
            log_dir = os.path.join(saving_dir, "tb_logs", run_id)
            self._tb = SummaryWriter(log_dir=log_dir, flush_secs=30)
            print(f"TensorBoard writing to {log_dir}")
        except Exception as e:
            print(f"TensorBoard init failed: {e}")

    def log(self, metrics: dict, step: int):
        """Log a dict of scalar metrics at the given global step."""
        if not is_main_process():
            return
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)

    def finish(self):
        """Close all logging backends."""
        if not is_main_process():
            return
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()
