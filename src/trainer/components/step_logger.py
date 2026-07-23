"""Logging helpers for training and evaluation loops."""

from __future__ import annotations

import time

from utils.distributed import is_main_process
from utils.metrics import fmt_metrics


def _fmt_time(epoch_time: float) -> str:
    return f" | epoch time: {epoch_time:.2f}s" if epoch_time > 0 else ""


def _fmt_lr(lr: float) -> str:
    if lr == 0:
        return "0"
    mantissa, exponent = f"{lr:.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


class StepLogger:
    """Build and emit log payloads without duplicating formatting logic."""

    PRINT_EVERY_N_STEPS = 200

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _optimizer_lr_payload(trainer) -> dict:
        """Build per-parameter-name LR payload for epoch-level logging.

        Builds a param-id → lr map from optimizer groups, then walks the model's
        named parameters to emit one entry per unique LR (named after the first
        matching parameter). Also logs head/backbone summary min/max scalars.
        """
        from utils.device import unwrap_model

        # Map param id → (lr, group_kind)
        id_to_lr: dict[int, float] = {}
        id_to_kind: dict[int, str] = {}
        head_lrs: list[float] = []
        backbone_lrs: list[float] = []
        for pg in trainer.optimizer.param_groups:
            lr = float(pg["lr"])
            kind = pg.get("group_kind", "ungrouped")
            for p in pg["params"]:
                id_to_lr[id(p)] = lr
                id_to_kind[id(p)] = kind

        # Walk named params; emit one entry per distinct LR (first param name wins)
        seen_lrs: dict[float, str] = {}
        model = unwrap_model(trainer.model)
        for name, param in model.named_parameters():
            pid = id(param)
            if pid not in id_to_lr:
                continue
            lr = id_to_lr[pid]
            if lr not in seen_lrs:
                seen_lrs[lr] = name
            kind = id_to_kind[pid]
            if kind == "head":
                head_lrs.append(lr)
            elif kind == "backbone":
                backbone_lrs.append(lr)

        payload = {f"LRs/{n}": lr for lr, n in seen_lrs.items()}
        if head_lrs:
            payload["LRs/head_max"] = max(head_lrs)
            payload["LRs/head_min"] = min(head_lrs)
        if backbone_lrs:
            payload["LRs/backbone_max"] = max(backbone_lrs)
            payload["LRs/backbone_min"] = min(backbone_lrs)
        return payload

    def log_train_step(self, trainer, accum_loss: float, loss_dict: dict, epoch: int) -> None:
        if not is_main_process():
            return
        lr = trainer.optimizer.param_groups[-1]["lr"]
        job_elapsed = time.time() - getattr(trainer, "_job_start_time", time.time())
        train_hours = (trainer.state.cumulative_train_seconds + job_elapsed) / 3600.0
        log_payload = {
            "train_losses/step_loss": accum_loss,
            "trainer/lr": lr,
            "trainer/step": trainer.state.global_step,
            "trainer/train_hours": train_hours,
        }
        # Log per-layer M2F loss components, stripping "losses/" prefix if present
        for k, v in loss_dict.items():
            if k != "loss":
                clean_k = k.replace("losses/", "")
                log_payload[f"train_losses/{clean_k}"] = v.item(
                ) if hasattr(v, "item") else float(v)
        trainer.log_metrics(log_payload, trainer.state.global_step)
        if trainer.state.global_step % self.PRINT_EVERY_N_STEPS == 0:
            print(
                f"[Epoch {epoch}] step={trainer.state.global_step} loss={accum_loss:.4f} lr={_fmt_lr(lr)}"
            )

    def log_train_epoch(
        self, trainer, epoch: int, epoch_loss: float, epoch_time: float = 0.0
    ) -> None:
        if not is_main_process():
            return
        lr = trainer.optimizer.param_groups[-1]["lr"]
        trainer.log_metrics(
            {
                "train_losses/epoch_loss": epoch_loss,
                "trainer/lr": lr,
                "trainer/epoch": epoch,
                "trainer/step": trainer.state.global_step,
                **self._optimizer_lr_payload(trainer),
            },
            trainer.state.global_step,
        )
        print(
            f"[Train] epoch={epoch} loss={epoch_loss:.4f}{_fmt_time(epoch_time)}")

    def log_eval_epoch(
        self,
        trainer,
        prefix: str,
        log_payload: dict,
        avg_metrics: dict,
        std_metrics: dict,
        epoch_time: float = 0.0,
    ) -> None:
        if not is_main_process():
            return
        trainer.log_metrics(log_payload, trainer.state.global_step)
        parts = []
        for k, v in avg_metrics.items():
            if k in std_metrics:
                parts.append(f"{k}={v:.4f}±{std_metrics[k]:.4f}")
            else:
                parts.append(f"{k}={v:.4f}")
        print(
            f"[{prefix.capitalize()}] step={trainer.state.global_step} "
            + " ".join(parts) + _fmt_time(epoch_time)
        )

    def log_val_summary(self, trainer, val_metrics: dict, val_time: float = 0.0) -> None:
        if not is_main_process():
            return
        val_summary = fmt_metrics(val_metrics)
        best = (
            f" | best_train_loss={trainer.state.best_train_loss:.4f}"
            if trainer.state.best_train_loss < float("inf")
            else ""
        )
        print(
            f"[Epoch {trainer.state.current_epoch + 1}/{trainer.cfg.TRAIN.EPOCHS}] Val: {val_summary}"
            + best + _fmt_time(val_time)
        )
        trainer.logger.log(
            {"train_losses/best_train_loss": trainer.state.best_train_loss},
            trainer.state.global_step,
        )
