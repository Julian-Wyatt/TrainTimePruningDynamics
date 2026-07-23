# ---------------------------------------------------------------
# Schedule factory — warmup_warmdown | cosine | polynomial
# ---------------------------------------------------------------

from __future__ import annotations

import math
from typing import Callable

from omegaconf import DictConfig

# ── Schedule strategies ────────────────────────────────────────

def _warmup_warmdown(cfg: DictConfig) -> Callable[[float], float]:
    """Original piecewise-linear schedule from autoresearch."""
    warmup = cfg.schedule.warmup_ratio
    warmdown = cfg.schedule.warmdown_ratio
    final_frac = cfg.schedule.final_lr_frac

    def lr_mult(progress: float) -> float:
        if progress < warmup:
            return progress / warmup if warmup > 0 else 1.0
        elif progress < 1.0 - warmdown:
            return 1.0
        else:
            cooldown = (1.0 - progress) / warmdown
            return cooldown * 1.0 + (1 - cooldown) * final_frac

    return lr_mult


def _cosine_annealing(cfg: DictConfig) -> Callable[[float], float]:
    """Cosine annealing with optional linear warmup.

    ``progress`` is expected in [0, 1].
    """
    warmup = cfg.schedule.warmup_ratio
    eta_min_frac = cfg.schedule.get("eta_min", 0.01)

    def lr_mult(progress: float) -> float:
        if progress < warmup:
            return progress / warmup if warmup > 0 else 1.0
        # Map remaining progress to [0, 1]
        t = (progress - warmup) / max(1.0 - warmup, 1e-8)
        return eta_min_frac + 0.5 * (1.0 - eta_min_frac) * (
            1 + math.cos(math.pi * t)
        )

    return lr_mult


def _polynomial_decay(cfg: DictConfig) -> Callable[[float], float]:
    """Polynomial decay with optional linear warmup.

    ``progress`` is expected in [0, 1].
    """
    warmup = cfg.schedule.warmup_ratio
    power = cfg.schedule.get("power", 1.0)
    final_frac = cfg.schedule.get("final_lr_frac", 0.0)

    def lr_mult(progress: float) -> float:
        if progress < warmup:
            return progress / warmup if warmup > 0 else 1.0
        t = (progress - warmup) / max(1.0 - warmup, 1e-8)
        return (1.0 - final_frac) * (1.0 - t) ** power + final_frac

    return lr_mult


# ── Factory ────────────────────────────────────────────────────

_STRATEGIES = {
    "warmup_warmdown": _warmup_warmdown,
    "cosine": _cosine_annealing,
    "polynomial": _polynomial_decay,
    "constant": lambda cfg: (lambda progress: 1.0),
}


def build_schedule(cfg: DictConfig) -> Callable[[float], float]:
    """Return a ``progress -> lr_multiplier`` callable.

    ``progress`` is ``training_time / time_budget`` in [0, 1].
    """
    name = cfg.schedule.name.lower()
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown schedule: '{name}'. Choose from: "
            f"{', '.join(_STRATEGIES)}"
        )
    return _STRATEGIES[name](cfg)
