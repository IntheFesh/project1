"""Bootstrap CI sanity + reproducibility."""

from eval.bootstrap import bootstrap_ci


def test_ci_brackets_mean() -> None:
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=2000, seed=0)
    assert low <= 3.0 <= high


def test_reproducible() -> None:
    data = [0.1, 0.2, 0.9, 0.4, 0.7, 0.3]
    a = bootstrap_ci(data, n_resamples=1000, seed=123)
    b = bootstrap_ci(data, n_resamples=1000, seed=123)
    assert a == b
