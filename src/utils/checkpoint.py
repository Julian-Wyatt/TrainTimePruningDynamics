import contextlib
from typing import Optional

import torch
import torch.nn as nn

from .distributed import is_main_process


def _strip_compile_prefix_from_key(key: str) -> str:
    wrapper_components = {"_orig_mod", "module"}
    return ".".join(part for part in key.split(".") if part not in wrapper_components)


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove wrapper path components added by torch.compile/DDP."""
    return {
        _strip_compile_prefix_from_key(k): v
        for k, v in state_dict.items()
    }


def _adapt_state_dict_to_model(state_dict: dict, model: nn.Module) -> dict:
    """Map canonical checkpoint keys onto the current model wrapper layout."""
    stripped_state = _strip_compile_prefix(state_dict)
    target_keys = {
        _strip_compile_prefix_from_key(k): k
        for k in model.state_dict().keys()
    }
    return {
        target_keys.get(k, k): v
        for k, v in stripped_state.items()
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    ema=None,
    scaler=None,
    cfg=None,
    best_metric_val: Optional[float] = None,
    cumulative_train_seconds: float = 0.0,
    step_in_epoch: int = 0,
):
    if not is_main_process():
        return
    # Swap to EMA weights if available, then restore live weights after extracting state_dict
    ctx = ema.ema_scope(model) if ema is not None else contextlib.nullcontext()
    with ctx:
        raw_sd = _strip_compile_prefix(model.state_dict())

    ckpt = {
        "model": raw_sd,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "step_in_epoch": step_in_epoch,
    }
    if ema is not None:
        ckpt["ema"] = _strip_compile_prefix(ema.state_dict())
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if cfg is not None:
        from omegaconf import OmegaConf
        ckpt["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    if best_metric_val is not None:
        ckpt["best_metric_val"] = best_metric_val
    ckpt["cumulative_train_seconds"] = cumulative_train_seconds
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def resume_from_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema=None,
    scaler=None,
) -> tuple[int, int, float, float, int]:
    ckpt = load_checkpoint(path)
    model.load_state_dict(_adapt_state_dict_to_model(ckpt["model"], model))
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    best_metric_val = ckpt.get("best_metric_val", 0.0)
    cumulative_train_seconds = ckpt.get("cumulative_train_seconds", 0.0)
    step_in_epoch = ckpt.get("step_in_epoch", 0)
    if "step_in_epoch" not in ckpt:
        print(f"Warning: checkpoint {path} has no step_in_epoch; defaulting to 0")
    print(
        f"Resumed from {path} "
        f"(epoch={epoch}, step={step}, step_in_epoch={step_in_epoch}, best_metric_val={best_metric_val})"
    )
    return epoch, step, best_metric_val, cumulative_train_seconds, step_in_epoch
