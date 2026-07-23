"""Shared parameter-group builders for models that support LLRD.

This keeps the optimizer factory generic while letting model-specific code
define only which tensors belong to the backbone and which head params should
be excluded from backbone routing.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def build_backbone_head_param_groups(
    model: nn.Module,
    backbone: nn.Module,
    *,
    embedding_lr: float,
    scalar_lr: float,
    weight_decay: float,
    adam_betas: tuple,
    llrd: float = 1.0,
    llrd_full_lr: bool = False,
    lr_mult: float = 1.0,
    original_eomt_lr_mult_compat: bool = False,
    head_param_ids: Iterable[int] | None = None,
    extra_backbone_named_params: Iterable[tuple[str, nn.Parameter]] = (),
    layer_id_fn=None,
    head_group_kind: str = "head",
) -> list[dict]:
    """Build AdamW param groups for a backbone + head style model."""
    head_param_ids = set(head_param_ids or [])
    backbone_param_ids = {id(p) for p in backbone.parameters()}
    extra_backbone_named_params = list(extra_backbone_named_params)
    extra_backbone_param_ids = {id(p) for _, p in extra_backbone_named_params}

    blocks = list(getattr(backbone, "blocks", []))
    n_layers = len(blocks)

    def _default_layer_id(name: str) -> int:
        parts = name.split(".")
        for pattern in ("h", "blocks", "layer", "layers"):
            if pattern in parts:
                idx = parts.index(pattern)
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    return min(int(parts[idx + 1]), n_layers - 1)
        if any(kw in name for kw in ("patch_embed", "cls_token", "pos_embed", "reg_token")):
            return 0
        return n_layers - 1

    _layer_id = layer_id_fn or _default_layer_id

    adamw_backbone = [
        (name, param)
        for name, param in backbone.named_parameters()
        if param.requires_grad
    ]
    for name, param in extra_backbone_named_params:
        if param.requires_grad and id(param) not in backbone_param_ids:
            adamw_backbone.append((name, param))

    param_groups = []

    for name, param in adamw_backbone:
        lid = _layer_id(name)
        if llrd_full_lr and lid >= n_layers - getattr(model, "num_blocks", 0):
            lr = scalar_lr
        elif llrd != 1.0:
            lr = embedding_lr * (llrd ** (n_layers - 1 - lid))
        elif original_eomt_lr_mult_compat:
            lr = embedding_lr
        else:
            lr = embedding_lr * lr_mult
        param_groups.append({
            "kind": "adamw",
            "group_kind": "backbone",
            "params": [param],
            "lr": lr,
            "betas": adam_betas,
            "eps": 1e-8,
            "weight_decay": 0.0 if param.ndim < 2 else weight_decay,
        })

    head_params = [
        p for p in model.parameters()
        if p.requires_grad
        and id(p) not in backbone_param_ids
        and id(p) not in extra_backbone_param_ids
        and id(p) not in head_param_ids
    ]
    if head_params:
        param_groups.append({
            "kind": "adamw",
            "group_kind": head_group_kind,
            "params": head_params,
            "lr": scalar_lr,
            "betas": adam_betas,
            "eps": 1e-8,
            "weight_decay": weight_decay,
        })

    return param_groups
