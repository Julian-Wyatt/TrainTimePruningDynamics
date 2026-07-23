"""Paper Figure 1 rendering from EoMT pruning masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from .helpers import _denorm, _np_to_tensor

# Paper Figure 1: pruning at blocks 3, 6, and 9 is black, cyan, and orange.
# Extra colours keep the helper usable for a different number of prune stages.
PRUNE_STAGE_COLOURS = np.array(
    [(0, 0, 0), (0, 210, 210), (255, 160, 0), (215, 70, 180), (225, 70, 70)],
    dtype=np.uint8,
)


def _first_pruned_stage(keep_maps: torch.Tensor) -> torch.Tensor:
    """Return the first stage that drops every token; ``S`` means retained."""
    if keep_maps.ndim != 3:
        raise ValueError(f"Expected [stages, gh, gw] keep maps, got {tuple(keep_maps.shape)}.")
    stage_count = keep_maps.shape[0]
    policy = torch.full_like(keep_maps[0], stage_count, dtype=torch.long)
    still_kept = torch.ones_like(keep_maps[0], dtype=torch.bool)
    for stage, keep in enumerate(keep_maps.bool()):
        dropped_here = still_kept & ~keep
        policy[dropped_here] = stage
        still_kept &= keep
    return policy


def prune_policy_overlay(img_np: np.ndarray, keep_maps: torch.Tensor) -> np.ndarray:
    """Mark each token by the stage that first pruned it, preserving retained pixels."""
    if keep_maps.shape[0] > len(PRUNE_STAGE_COLOURS):
        raise ValueError("No display colour is defined for every pruning stage.")
    policy = _first_pruned_stage(keep_maps)
    if policy.shape != img_np.shape[:2]:
        policy = F.interpolate(
            policy.float()[None, None], size=img_np.shape[:2], mode="nearest"
        )[0, 0].long()
    policy_np = policy.cpu().numpy()
    out = img_np.copy()
    for stage, colour in enumerate(PRUNE_STAGE_COLOURS[: keep_maps.shape[0]]):
        out[policy_np == stage] = colour
    return out


def visualise_prune_policy(
    images: torch.Tensor,
    selection_maps: Sequence[dict[str, torch.Tensor]],
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    max_samples: int = 4,
) -> torch.Tensor | None:
    """Render input/policy pairs for a small Figure 1-style diagnostic grid."""
    if not selection_maps:
        return None

    keeps = torch.stack([stage["keep"] for stage in selection_maps], dim=0)
    batch_size = min(images.shape[0], max_samples)
    panels: list[torch.Tensor] = []
    for image_idx in range(batch_size):
        image = _denorm(images[image_idx], mean, std)
        panels.extend(
            [
                _np_to_tensor(image),
                _np_to_tensor(prune_policy_overlay(image, keeps[:, image_idx])),
            ]
        )
    return make_grid(panels, nrow=2, padding=2)
