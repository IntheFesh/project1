"""Statistics for headline comparisons: paired bootstrap + Holm-Bonferroni.

Headline results (Phase 5/8) use these with >=10k resamples; the CI gate (Phase 6) does
NOT gate on these noisy estimates — it uses a deterministic point check instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


class PairedDiff(BaseModel):
    """Paired-difference summary with a bootstrap CI and two-sided p-value."""

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> PairedDiff:
    """Bootstrap the mean paired difference ``a - b`` (CI + two-sided p-value).

    Args:
        a, b: paired observations of equal, non-zero length.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        raise ValueError("a and b must be non-empty and the same length")
    diff = arr_a - arr_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    p_value = 2.0 * min(float((means >= 0).mean()), float((means <= 0).mean()))
    return PairedDiff(
        mean_diff=float(diff.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p_value=min(1.0, p_value),
    )


def holm_bonferroni(
    pvalues: dict[str, float], alpha: float = 0.05
) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down correction.

    Returns ``{name: (adjusted_p, reject_null)}`` controlling family-wise error at ``alpha``.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        adjusted = max(running, min(1.0, (m - rank) * p))
        running = adjusted
        out[name] = (adjusted, adjusted <= alpha)
    return out
