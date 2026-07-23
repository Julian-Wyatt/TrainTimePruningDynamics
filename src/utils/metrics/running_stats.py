"""Welford's online mean and population standard deviation.

O(1) memory — suitable for large datasets where storing all batch values
would be prohibitive.
"""

from __future__ import annotations

import math


class RunningStats:
    """Accumulate per-key mean and population std using Welford's algorithm.

    Usage::
        stats = RunningStats()
        for batch_metrics in val_batches:
            stats.update(batch_metrics)
        means = stats.means()
        stds  = stats.stds()   # empty if fewer than 2 updates per key
    """

    def __init__(self) -> None:
        # key → (count, mean, M2)  where M2 = sum of squared deviations from mean
        self._stats: dict[str, tuple[int, float, float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        """Add one observation for each key in *metrics*. Skips NaN/Inf values."""
        for k, x in metrics.items():
            if not math.isfinite(x):
                continue
            count, mean, M2 = self._stats.get(k, (0, 0.0, 0.0))
            count += 1
            delta = x - mean
            mean += delta / count
            M2 += delta * (x - mean)  # uses updated mean (Welford's formula)
            self._stats[k] = (count, mean, M2)

    def means(self) -> dict[str, float]:
        """Return the running mean for each key seen so far."""
        return {k: mean for k, (count, mean, M2) in self._stats.items()}

    def stds(self) -> dict[str, float]:
        """Return population std for keys that have received more than one update.

        Keys with only a single observation are omitted (std is undefined / zero
        by convention but meaningless as an uncertainty estimate).
        """
        return {
            k: math.sqrt(M2 / count)
            for k, (count, mean, M2) in self._stats.items()
            if count > 1
        }

    def reset(self) -> None:
        self._stats.clear()


def fmt_metrics(metrics: dict[str, float]) -> str:
    """Format a metrics dict as ``'k=mean±std | ...'`` pairs.

    Keys ending in ``_std`` are consumed as the uncertainty for their base key
    and are not printed separately.

    Example::
        >>> fmt_metrics({"mIoU": 0.82, "mIoU_std": 0.01, "loss": 0.5})
        'mIoU=0.8200±0.0100 | loss=0.5000'
    """
    parts = []
    for k, v in metrics.items():
        if k.endswith("_std"):
            continue
        std = metrics.get(f"{k}_std")
        parts.append(f"{k}={v:.4f}±{std:.4f}" if std is not None else f"{k}={v:.4f}")
    return " | ".join(parts)
