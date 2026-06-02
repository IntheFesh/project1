"""pass^k combinatorial estimator E[C(c,k)/C(n,k)]."""

import pytest

from eval.passk import pass_hat_k


def test_all_success() -> None:
    assert pass_hat_k(5, 5, 1) == 1.0
    assert pass_hat_k(5, 5, 5) == 1.0


def test_partial() -> None:
    assert pass_hat_k(4, 3, 2) == pytest.approx(0.5)   # C(3,2)/C(4,2) = 3/6
    assert pass_hat_k(5, 0, 1) == 0.0
    assert pass_hat_k(4, 2, 4) == 0.0                  # c < k


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        pass_hat_k(3, 1, 0)   # k < 1
    with pytest.raises(ValueError):
        pass_hat_k(3, 5, 1)   # c > n
