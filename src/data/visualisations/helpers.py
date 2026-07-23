"""Small conversion helpers for the paper pruning-policy panel."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _denorm(
    img: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a normalised image tensor to an RGB uint8 array."""
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    image = (img.float().cpu() * std_t + mean_t).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _np_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert an RGB uint8 array to a float image tensor."""
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
