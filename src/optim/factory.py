# ---------------------------------------------------------------
# Optimizer factory — adamw | adam | sgd
# Model-specific parameter grouping lives on the model.
# ---------------------------------------------------------------

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig


def _dedupe_param_groups(groups: list[dict]) -> list[dict]:
    """Remove repeated Parameter objects across optimizer groups.

    Shared/tied parameters are legal in modules, but torch optimizers require a
    parameter to appear in exactly one group. Keep the first occurrence, which
    preserves the most specific routing established by the model.
    """
    seen: set[int] = set()
    deduped: list[dict] = []
    n_removed = 0
    for group in groups:
        params = []
        for p in group.get("params", []):
            pid = id(p)
            if pid in seen:
                n_removed += 1
                continue
            seen.add(pid)
            params.append(p)
        if params:
            group = dict(group)
            group["params"] = params
            deduped.append(group)
    if n_removed:
        print(f"[optimizer] removed {n_removed} duplicate tied parameter reference(s)")
    return deduped


# ── Factory ────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module, cfg: DictConfig
) -> torch.optim.Optimizer:
    """Build an optimizer from the Hydra config.

    Supported ``cfg.optimizer.name`` values: ``"adamw"``, ``"adam"``, ``"sgd"``.
    Models exposing ``get_param_groups`` control their own LLRD grouping.
    """
    opt_cfg = cfg.optimizer
    name = opt_cfg.name.lower()
    llrd = opt_cfg.get("llrd", 1.0)
    lr_mult = opt_cfg.get("lr_mult", 1.0)

    if llrd != 1.0 and lr_mult != 1.0:
        warnings.warn("Both LLRD and LR_MULT are set; LLRD takes precedence and LR_MULT is ignored.")
        lr_mult = 1.0

    # ── Build param groups ────────────────────────────────────
    if hasattr(model, "get_param_groups"):
        groups = model.get_param_groups(
            embedding_lr=opt_cfg.embedding_lr,
            scalar_lr=opt_cfg.scalar_lr,
            weight_decay=opt_cfg.weight_decay,
            adam_betas=tuple(opt_cfg.adam_betas),
            llrd=llrd,
            llrd_full_lr=opt_cfg.get("llrd_full_lr", False),
            lr_mult=lr_mult,
            original_eomt_lr_mult_compat=opt_cfg.get("original_eomt_lr_mult_compat", False),
        )
    else:
        if llrd != 1.0 or lr_mult != 1.0:
            warnings.warn(
                "LLRD/LR_MULT requested, but this model does not expose "
                "get_param_groups(); using one flat optimizer group."
            )
        groups = [dict(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=opt_cfg.embedding_lr,
            weight_decay=opt_cfg.weight_decay,
        )]

    groups = _dedupe_param_groups(groups)

    # ── Construct optimizer ───────────────────────────────────
    betas = tuple(opt_cfg.adam_betas)

    if name == "adamw":
        return torch.optim.AdamW(groups, betas=betas, eps=1e-8)

    if name == "adam":
        return torch.optim.Adam(groups, betas=betas, eps=1e-8)

    if name == "sgd":
        return torch.optim.SGD(groups, momentum=opt_cfg.get("momentum", 0.9))

    raise ValueError(
        f"Unknown optimizer: '{name}'. Choose from: adamw, adam, sgd"
    )
