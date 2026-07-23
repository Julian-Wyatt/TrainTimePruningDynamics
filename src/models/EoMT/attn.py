# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted, with modifications, from the EoMT reference implementation:
# https://github.com/tue-mps/eomt
#
# _softmax_with_policy implements the policy-gated attention softmax introduced
# by DynamicViT: https://github.com/raoyongming/DynamicViT
# ---------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import apply_rot_embed_cat


def _softmax_with_policy(
    attn: torch.Tensor,
    policy: torch.Tensor,
    add_mask: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked softmax that multiplicatively gates keys by a keep ``policy``.

    ``attn`` is the pre-softmax score ``[B, H, N, N]``; ``policy`` is ``[B, N]`` with
    a keep weight per *key* (1 = keep, 0 = drop, straight-through values in between).
    Dropped keys are removed from the normalisation, so kept tokens never attend to
    them, while gradient still flows to the (soft) policy. A diagonal term keeps every
    token attending to itself so a fully-dropped row cannot produce NaNs.
    """
    B, _, N, _ = attn.shape
    if add_mask is not None:
        attn = attn + add_mask
    # All keys kept and no additive mask => the gated softmax is plain softmax.
    if add_mask is None and policy.min() >= 1.0:
        return torch.softmax(attn.float(), dim=-1).type_as(attn)
    # Broadcast per-key keep weights instead of materialising [B, 1, N, N] plus a
    # torch.eye(N); equivalent to ``policy + (1 - policy) * I``, far cheaper at N≈4k.
    keep = policy.reshape(B, 1, 1, N).to(torch.float32)
    max_att = attn.max(dim=-1, keepdim=True).values
    weights = (attn - max_att).to(torch.float32).exp()
    # Keep the diagonal aside so `weights` can be freed before the reduction, halving
    # peak memory. The `*` must stay out-of-place — `weights` is autograd-saved by exp().
    diag_vals = weights.diagonal(dim1=-2, dim2=-1).clone()
    num = weights * keep
    del weights
    # Every token keeps full self-attention, so a fully-dropped row still has a
    # defined softmax. policy_i == 1 on the diagonal, so no gradient reaches the selector.
    num.diagonal(dim1=-2, dim2=-1).copy_(diag_vals)
    del diag_vals
    attn = (num + eps / N) / (num.sum(dim=-1, keepdim=True) + eps)
    return attn.type_as(max_att)


def run_attn(
    module: nn.Module,
    x: torch.Tensor,
    mask,  # bool tensor | dense additive mask | None
    rope: Optional[torch.Tensor],
    training: bool,
    policy: Optional[torch.Tensor] = None,  # [B, N] keep policy | None
) -> torch.Tensor:
    """Run one transformer attention block.

    When ``policy`` is ``None`` the fast fused SDPA path is used.  When a keep
    ``policy`` is supplied (soft-mask pruning) attention is computed manually via
    :func:`_softmax_with_policy` so gradient reaches the token selector.
    """
    B, N, C = x.shape

    # --- QKV ---
    head_dim = getattr(module, "head_dim", C // module.num_heads)
    if module.qkv is not None:
        if getattr(module, "q_bias", None) is None:
            qkv = module.qkv(x)
        else:
            qkv_bias = torch.cat((module.q_bias, module.k_bias, module.v_bias))
            if getattr(module, "qkv_bias_separate", False):
                qkv = module.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=module.qkv.weight, bias=qkv_bias)
        # .contiguous() before unbind collapses three downstream slice-copies into one.
        qkv = qkv.reshape(B, N, 3, module.num_heads,
                          head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
    else:
        q = module.q_proj(x).reshape(
            B, N, module.num_heads, head_dim).transpose(1, 2)
        k = module.k_proj(x).reshape(
            B, N, module.num_heads, head_dim).transpose(1, 2)
        v = module.v_proj(x).reshape(
            B, N, module.num_heads, head_dim).transpose(1, 2)

    q = module.q_norm(q)
    k = module.k_norm(k)

    # --- RoPE ---
    if rope is not None:
        if isinstance(rope, tuple):
            from transformers.models.dinov3_vit.modeling_dinov3_vit import rotate_half
            cos, sin = rope
            if cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            num_rope = sin.shape[-2]
            num_prefix = getattr(module, "num_prefix_tokens", 0)
            q_skip, q_patch = q[:, :, :num_prefix, :], q[:, :, num_prefix:, :]
            k_skip, k_patch = k[:, :, :num_prefix, :], k[:, :, num_prefix:, :]
            n_post = q_patch.shape[-2]
            if n_post > num_rope:
                extra = n_post - num_rope
                q_pref, q_rope = q_patch[:, :, :extra, :], q_patch[:, :, extra:, :]
                k_pref, k_rope = k_patch[:, :, :extra, :], k_patch[:, :, extra:, :]
                q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
                k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)
                q = torch.cat([q_skip, q_pref, q_rope], dim=-2)
                k = torch.cat([k_skip, k_pref, k_rope], dim=-2)
            else:
                cos_s, sin_s = cos[..., :n_post, :], sin[..., :n_post, :]
                q_patch = (q_patch * cos_s) + (rotate_half(q_patch) * sin_s)
                k_patch = (k_patch * cos_s) + (rotate_half(k_patch) * sin_s)
                q = torch.cat([q_skip, q_patch], dim=-2)
                k = torch.cat([k_skip, k_patch], dim=-2)
        else:
            npt = getattr(module, "num_prefix_tokens", 0)
            half = getattr(module, "rotate_half", False)
            rope_apply = rope.unsqueeze(1) if rope.dim() == 3 else rope
            num_rope = rope_apply.shape[-2]
            q_skip, q_patch = q[:, :, :npt, :], q[:, :, npt:, :]
            k_skip, k_patch = k[:, :, :npt, :], k[:, :, npt:, :]
            n_post = q_patch.shape[-2]
            if n_post > num_rope:
                extra = n_post - num_rope
                q_pref, q_rope = q_patch[:, :, :extra, :], q_patch[:, :, extra:, :]
                k_pref, k_rope = k_patch[:, :, :extra, :], k_patch[:, :, extra:, :]
                q_rope = apply_rot_embed_cat(q_rope, rope_apply, half=half)
                k_rope = apply_rot_embed_cat(k_rope, rope_apply, half=half)
                q = torch.cat([q_skip, q_pref, q_rope], dim=2).type_as(v)
                k = torch.cat([k_skip, k_pref, k_rope], dim=2).type_as(v)
            else:
                rope_s = rope_apply[..., :n_post, :]
                q_patch = apply_rot_embed_cat(q_patch, rope_s, half=half)
                k_patch = apply_rot_embed_cat(k_patch, rope_s, half=half)
                q = torch.cat([q_skip, q_patch], dim=2).type_as(v)
                k = torch.cat([k_skip, k_patch], dim=2).type_as(v)

    # --- Attention ---
    attn_mask = None
    if mask is not None:
        if mask.dtype == torch.bool:
            if policy is None:
                # SDPA broadcasts a bool mask over heads and converts it to the same
                # additive -inf/0 internally, so skip building the [B, H, N, N] float mask.
                attn_mask = mask.unsqueeze(1)
            else:
                # Manual softmax-with-policy needs an additive float mask.
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
                attn_mask = mask.to(q.dtype).masked_fill(
                    ~mask, float("-inf")).masked_fill(mask, 0.0)
        else:
            attn_mask = mask
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
    dropout_p = module.attn_drop.p if training else 0.0
    if policy is None:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    else:
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = _softmax_with_policy(attn, policy, attn_mask)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=True)
        out = attn @ v

    x = out.transpose(1, 2).reshape(B, N, C)
    x = module.norm(x)
    x = module.proj(x)
    x = module.proj_drop(x)

    return x


@torch.compiler.disable
def disable_attn_mask(
    attn_mask: torch.Tensor,
    prob: float | torch.Tensor,
    num_q: int,
    sp_start: int,
) -> torch.Tensor:
    """Randomly release masked query rows to open attention (annealing dropout).

    When prob < 1, each query independently has a (1 - prob) chance of having its
    spatial mask cleared, allowing it to attend everywhere regardless of mask_logits.
    Decorated with @torch.compiler.disable to prevent compilation of the random sampling.

    Args:
        attn_mask: Bool mask [B, N, N] to mutate in-place.
        prob: Current mask hardness in [0, 1]. 1 = fully hard, 0 = fully open.
        num_q: Number of query tokens.
        sp_start: Index of first spatial (non-query, non-prefix) token.

    Returns:
        The mutated attn_mask (same tensor, mutated in-place).
    """
    if prob < 1:
        random_queries = (
            torch.rand(attn_mask.shape[0], num_q,
                       device=attn_mask.device) > prob
        )
        attn_mask[:, :num_q, sp_start:][random_queries] = True
    return attn_mask


def build_attn_mask(
    x: torch.Tensor,
    mask_logits: torch.Tensor,
    num_q: int,
    sp_start: int,
    grid_size: tuple[int, int],
    block_idx: int,
    attn_mask_probs: torch.Tensor,
) -> torch.Tensor:
    """Build query→spatial attention mask from mask_logits for one EoMT query block.

    Builds a fresh (B, N, N) bool mask each call via concatenation. Only the
    query→spatial block carries the predicted mask; all other entries are True
    (permitting full attention).

    A fresh tensor is constructed every call rather than mutating a reused
    buffer in place: the query blocks in a forward pass each save their mask
    for backward, so a single persistent buffer mutated across blocks trips the
    autograd version counter (fatal under torch.compile / AOTAutograd).

    Args:
        x: Current token sequence [B, N, C] (used for shape and device only).
        mask_logits: Predicted mask logits [B, Q, H, W] from the mask head.
        num_q: Number of query tokens.
        sp_start: Index of first spatial token (num_q + num_prefix_tokens).
        grid_size: (H, W) patch grid to interpolate mask_logits to.
        block_idx: Index into attn_mask_probs for annealing.
        attn_mask_probs: Per-block mask hardness buffer [num_blocks].

    Returns:
        attn_mask — bool [B, N, N].
    """
    B, N = x.shape[:2]
    # The ``> 0`` below already blocks the gradient, so detaching is equivalent and
    # skips retaining the interpolation in the forward graph.
    interpolated = F.interpolate(mask_logits.detach(), grid_size, mode="bilinear")
    interpolated = interpolated.reshape(
        interpolated.size(0), interpolated.size(1), -1)
    # Query rows: [True over query+prefix cols | (mask_logits > 0) over spatial].
    q_spatial = interpolated > 0  # [B, num_q, N - sp_start]
    q_prefix = q_spatial.new_ones(B, num_q, sp_start)
    q_rows = torch.cat((q_prefix, q_spatial), dim=2)  # [B, num_q, N]
    # All non-query rows attend everywhere.
    other_rows = q_spatial.new_ones(B, N - num_q, N)
    mask = torch.cat((q_rows, other_rows), dim=1)  # [B, N, N]

    return disable_attn_mask(
        mask, attn_mask_probs[block_idx], num_q, sp_start)
