"""Trainer runtime state container."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainerState:
    """Mutable trainer runtime state.

    Centralizes mutable state to reduce BaseTrainer surface area.
    """
    global_step: int = 0
    start_epoch: int = 0
    start_step_in_epoch: int = 0
    current_epoch: int = 0
    steps_per_epoch: int = 1

    best_metric_val: float = -float("inf")
    best_ckpt_path: str | None = None
    saved_ckpt_paths: list[str] = field(default_factory=list)

    best_train_loss: float = float("inf")
    last_epoch_loss: float | None = None
    last_val_metrics: dict = field(default_factory=dict)

    last_complete_ckpt: str = ""
    train_end_ckpt_path: str | None = None
    train_vis_epoch: int = -1
    last_val_epoch_time: float = 0.0
    cumulative_train_seconds: float = 0.0
