"""Tests for headline-results aggregation + the benchmark output parsers."""

from __future__ import annotations

import json

from eval.results import aggregate_pass_k, build_report, compare_metric
from eval.run_bfcl import parse_bfcl_summary
from eval.run_tau2 import parse_tau2_trials, pass_k_from_trials


def test_aggregate_pass_k_perfect_and_partial() -> None:
    # All tasks pass every attempt -> pass^k == 1.0 for any k.
    assert aggregate_pass_k([(4, 4), (4, 4)], k=2).mean == 1.0
    # c < k -> 0 for that task; two tasks, one 0 one 1 -> mean 0.5.
    res = aggregate_pass_k([(4, 1), (4, 4)], k=2)
    assert 0.0 <= res.ci_low <= res.mean <= res.ci_high <= 1.0
    assert res.n_tasks == 2


def test_compare_metric_and_holm() -> None:
    base = [0, 0, 1, 0, 1, 0, 0, 1]
    lora = [1, 1, 1, 1, 1, 0, 1, 1]
    comp = compare_metric("tool_accuracy", base=base, lora=lora)
    assert comp.lora_mean > comp.base_mean
    assert comp.mean_diff > 0
    report = build_report("zh-base-vs-lora", [comp])
    assert report.comparisons[0].adjusted_p is not None
    assert "tool_accuracy" in report.to_markdown()


def test_build_report_holm_marks_significance() -> None:
    a = compare_metric("m1", base=[0] * 20, lora=[1] * 20)  # strong effect
    b = compare_metric("m2", base=[0, 1] * 10, lora=[0, 1] * 10)  # no effect
    report = build_report("multi", [a, b])
    by = {c.name: c for c in report.comparisons}
    assert by["m2"].significant is False


def test_parse_bfcl_summary_nested_and_flat() -> None:
    nested = parse_bfcl_summary(json.dumps(
        {"version": "BFCL-V4@abc123", "overall_accuracy": 87.5,
         "categories": {"simple": 92.0, "parallel": 83.0}}))
    assert nested.version == "BFCL-V4@abc123"
    assert abs(nested.overall_ast_accuracy - 0.875) < 1e-9
    assert nested.per_category["simple"] == 0.92
    # flat map, already in [0,1]
    flat = parse_bfcl_summary(json.dumps({"simple": 0.9, "multiple": 0.8}))
    assert abs(flat.overall_ast_accuracy - 0.85) < 1e-9


def test_parse_tau2_trials_variants() -> None:
    rewards = parse_tau2_trials(json.dumps([
        {"id": "t1", "rewards": [1.0, 0.0, 1.0, 1.0]},
        {"id": "t2", "trials": [{"success": True}, {"success": False}]},
        {"id": "t3", "results": [{"reward": 1.0}, {"reward": 1.0}]},
    ]))
    assert rewards == [(4, 3), (2, 1), (2, 2)]
    wrapped = parse_tau2_trials(json.dumps({"results": [{"rewards": [1.0, 1.0]}]}))
    assert wrapped == [(2, 2)]


def test_pass_k_from_trials_skips_k_above_attempts() -> None:
    trials = [(2, 2), (2, 1)]  # min attempts = 2
    pk = pass_k_from_trials(trials, k_values=(1, 2, 4))
    assert set(pk) == {1, 2}  # k=4 skipped (only 2 attempts)
    assert pk[1].mean >= pk[2].mean  # pass^k is non-increasing in k
