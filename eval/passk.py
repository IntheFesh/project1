"""pass^k estimator (tau-bench style): probability that ALL k sampled trials pass.

Uses the unbiased combinatorial estimator E[C(c, k) / C(n, k)], where ``n`` is the number
of independent trials and ``c`` the number of successes among them.
"""

from __future__ import annotations

from math import comb


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Return pass^k = C(c, k) / C(n, k) (0.0 when c < k).

    Args:
        n: total independent trials (n >= 1).
        c: number of successful trials (0 <= c <= n).
        k: subset size (1 <= k <= n).

    Raises:
        ValueError: if the inputs violate 1 <= k <= n or 0 <= c <= n.
    """
    if not 1 <= k <= n:
        raise ValueError(f"require 1 <= k <= n, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"require 0 <= c <= n, got n={n}, c={c}")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)
