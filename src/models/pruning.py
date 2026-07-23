"""Token-selection primitives shared by the pruning model.

``keep_score`` and ``ratio_loss`` implement the Gumbel-Softmax keep decision and
budget loss from DynamicViT (https://github.com/raoyongming/DynamicViT).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def round_kept_tokens_to_total_multiple(
    keep_count: float | int,
    spatial_len: int,
    *,
    total_tokens_offset: int = 0,
    multiple: int = 8,
    min_keep: int = 1,
) -> int:
    """Round kept spatial tokens so ``offset + kept`` hits a token multiple.

    ``total_tokens_offset`` should include non-spatial tokens that stay in the
    active sequence, such as prefix/register tokens and learned query tokens.
    Set ``multiple=1`` to disable alignment while preserving integer rounding.
    """
    spatial_len = max(0, int(spatial_len))
    min_keep = max(0, int(min_keep))
    raw_keep = max(min_keep, min(float(keep_count), float(spatial_len)))
    multiple = max(1, int(multiple))
    if multiple == 1:
        rounded = int(math.floor(raw_keep + 0.5))
    else:
        target_total = raw_keep + int(total_tokens_offset)
        rounded_total = int(math.floor((target_total / multiple) + 0.5) * multiple)
        rounded = rounded_total - int(total_tokens_offset)
    return max(min_keep, min(rounded, spatial_len))


def round_pruned_tokens_to_total_multiple(
    prune_count: float | int,
    spatial_len: int,
    *,
    total_tokens_offset: int = 0,
    multiple: int = 8,
    min_keep: int = 1,
) -> int:
    """Round a prune count by aligning the resulting active total-token count."""
    spatial_len = max(0, int(spatial_len))
    if float(prune_count) <= 0:
        return 0
    kept = round_kept_tokens_to_total_multiple(
        spatial_len - float(prune_count),
        spatial_len,
        total_tokens_offset=total_tokens_offset,
        multiple=multiple,
        min_keep=min_keep,
    )
    return spatial_len - kept


def keep_tokens_from_keep_rate(
    spatial_len: int,
    keep_rate: float,
    *,
    multiple: int = 8,
    min_keep: int = 1,
    total_tokens_offset: int = 0,
) -> int:
    """Convert a keep rate to a kept-token count with optional total alignment."""
    keep_rate = max(0.0, min(float(keep_rate), 1.0))
    return round_kept_tokens_to_total_multiple(
        int(spatial_len) * keep_rate,
        spatial_len,
        multiple=multiple,
        min_keep=min_keep,
        total_tokens_offset=total_tokens_offset,
    )


def keep_score(logits: torch.Tensor, tau: float, training: bool) -> torch.Tensor:
    """Differentiable per-token keep score in [0, 1] from 2-way keep/drop logits.

    During training a Gumbel-softmax sample injects exploration noise; at eval the
    plain softmax keep-probability is used.  Channel 0 is the *keep* class.
    """
    if training and tau > 0:
        return F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)[..., 0]
    return logits.softmax(dim=-1)[..., 0]


def straight_through_gumbel_keep(
    logits: torch.Tensor,
    tau: float,
    training: bool,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token keep/drop via a straight-through (Gumbel-)argmax over 2-way logits.

    Returns ``(policy, score)`` where ``policy`` is ``[B, N]`` carrying the hard
    {0, 1} keep decision in the forward pass and the soft keep-probability gradient
    on the backward pass, and ``score`` is ``[B, N]`` the soft keep-probability
    (channel 0) for diagnostics / capture.

    Unlike :func:`straight_through_topk` there is **no fixed budget**: each token
    independently picks keep vs drop (argmax of the two classes, with Gumbel noise
    during training), so the kept count is *emergent*.  The budget must therefore
    be enforced by a ratio loss applied to ``policy`` — this is the DynamicViT
    train-time mechanism, kept distinct from the eval-only top-k readout.
    ``valid_mask`` (``[B, N]``) nests the decision inside a previously-kept subset
    (strict hierarchy); invalid tokens are forced to drop.  Channel 0 is the
    *keep* class, matching :func:`keep_score`.
    """
    if training and tau > 0:
        soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
    else:
        soft = logits.softmax(dim=-1)
    score = soft[..., 0]
    hard = (soft[..., 0] >= soft[..., 1]).to(score.dtype)
    policy = hard - score.detach() + score
    if valid_mask is not None:
        policy = policy * valid_mask.to(policy.dtype)
    return policy, score


def randomize_ids(
    ids_keep: torch.Tensor,
    spatial_len: int,
    random_ratio: float,
) -> torch.Tensor:
    """Randomly perturb a set of kept spatial ids while preserving their count.

    ``random_ratio`` interpolates between the learned selection and pure noise:
    ``0`` returns ``ids_keep`` unchanged, ``1`` returns a fully random ``k``-subset,
    and ``0 < r < 1`` swaps ``floor(k * r)`` kept tokens for random dropped ones.
    All operations stay on-device (no host syncs).
    """
    B, k = ids_keep.shape
    device = ids_keep.device
    if random_ratio <= 0.0 or k == 0:
        return ids_keep
    if random_ratio >= 1.0:
        noise = torch.rand(B, spatial_len, device=device)
        return noise.argsort(dim=1)[:, :k]

    n_swap = int(k * random_ratio)
    if n_swap <= 0:
        return ids_keep

    keep_mask = torch.zeros(B, spatial_len, dtype=torch.bool, device=device)
    keep_mask.scatter_(1, ids_keep, True)
    # Drop n_swap random kept tokens.
    drop = torch.rand(B, spatial_len, device=device).masked_fill(~keep_mask, -1.0)
    keep_mask.scatter_(1, drop.topk(n_swap, dim=1).indices, False)
    # Reactivate n_swap random previously-dropped tokens.
    add = torch.rand(B, spatial_len, device=device).masked_fill(keep_mask, -1.0)
    keep_mask.scatter_(1, add.topk(n_swap, dim=1).indices, True)
    # Recover exactly k ids from the (k-True) mask.
    return keep_mask.float().topk(k, dim=1).indices


def ratio_loss(
    keep: torch.Tensor,
    target: float,
    *,
    per_sample: bool = True,
) -> torch.Tensor:
    """MSE between the realised keep ratio and the target keep rate.

    ``keep`` is ``[B, N]`` (hard or soft keep decisions over spatial tokens).
    With ``per_sample`` the ratio is averaged per sample before the MSE so the
    budget is enforced image-by-image rather than only in expectation.
    """
    if keep.numel() == 0:
        return keep.new_zeros(())
    ratio = keep.mean(dim=1) if per_sample else keep.mean()
    return F.mse_loss(ratio, torch.full_like(ratio, float(target)))
