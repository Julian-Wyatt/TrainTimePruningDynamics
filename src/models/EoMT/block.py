# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted, with modifications, from the EoMT reference implementation:
# https://github.com/tue-mps/eomt
# ---------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn

from .attn import run_attn


def run_block(
    blk: nn.Module,
    x: torch.Tensor,
    attn_mask,
    rope: Optional[torch.Tensor],
    attn_attr: str,
    training: bool,
    policy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one pre-norm ViT block with masked attention and residual connections.

    Handles both RoPE (timm or DINOv3) and no-RoPE backbones by forwarding
    ``rope`` to :func:`run_attn`. Pass ``rope=None`` for the no-RoPE path.

    Assumes the timm/EVA block layout:
    ``norm1 → attention → ls1 → drop_path1`` and
    ``norm2 → mlp → ls2 → drop_path2``.

    Args:
        blk: ViT block module (timm or EVA style).
        x: Input tensor [B, N, C].
        attn_mask: Attention mask forwarded to :func:`run_attn` — bool tensor,
            dense additive mask, or None.
        rope: RoPE embeddings forwarded to :func:`run_attn` (None, timm tensor,
            or DINOv3 (cos, sin) tuple).
        attn_attr: Name of the attention submodule on ``blk`` (``"attn"`` or
            ``"attention"``).
        training: Forwarded to :func:`run_attn` for dropout control.
    Returns:
        Output tensor [B, N, C].
    """
    attn = getattr(blk, attn_attr, None) or getattr(blk, "attn")
    attn_out = run_attn(
        attn,
        blk.norm1(x),
        attn_mask,
        rope,
        training,
        policy,
    )
    if hasattr(blk, "ls1"):
        attn_out = blk.ls1(attn_out)
    elif getattr(blk, "gamma_1", None) is not None:
        attn_out = attn_out * blk.gamma_1
    x = x + blk.drop_path1(attn_out)
    mlp_out = blk.mlp(blk.norm2(x))
    if hasattr(blk, "ls2"):
        mlp_out = blk.ls2(mlp_out)
    elif getattr(blk, "gamma_2", None) is not None:
        mlp_out = mlp_out * blk.gamma_2
    x = x + blk.drop_path2(mlp_out)
    return x
