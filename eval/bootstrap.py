"""Bootstrap confidence intervals (>= 10,000 resamples by default)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return the (low, high) percentile bootstrap CI for ``statistic``.

    Args:
        data: observed sample (non-empty).
        statistic: maps a resample to a scalar (default: mean).
        n_resamples: number of bootstrap resamples (>= 1).
        ci: central confidence level in (0, 1).
        seed: RNG seed for reproducibility.

    Raises:
        ValueError: if ``data`` is empty.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must be non-empty")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    stats = np.array([statistic(arr[row]) for row in idx])
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(low), float(high)
