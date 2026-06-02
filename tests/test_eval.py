"""Eval harness, statistics, RAG-triad proxies, and table rendering."""

import pytest

from agent.tools.services import ServiceDesk
from eval import run_bfcl, run_tau2
from eval.harness import evaluate_tasks, summarize
from eval.metrics import schema_valid_rate, tool_accuracy
from eval.rag_triad import groundedness
from eval.report import render_table
from eval.stats import holm_bonferroni, paired_bootstrap_diff
from eval.tasks import SMOKE_TASKS
from serving.client import ScriptedLLMClient


def test_smoke_slice_is_all_correct() -> None:
    records = evaluate_tasks(SMOKE_TASKS, ScriptedLLMClient(), ServiceDesk())
    assert tool_accuracy(records) == 1.0
    assert schema_valid_rate(records) == 1.0


def test_summarize_reports_counts() -> None:
    summary = summarize(
        evaluate_tasks(SMOKE_TASKS, ScriptedLLMClient(), ServiceDesk()), "slice"
    )
    assert summary.n == len(SMOKE_TASKS)
    assert summary.label == "slice"


def test_holm_bonferroni_rejects_and_is_monotonic() -> None:
    result = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.2})
    assert result["a"][1] is True
    assert result["c"][1] is False
    assert result["a"][0] <= result["b"][0] <= result["c"][0]


def test_paired_bootstrap_detects_difference() -> None:
    result = paired_bootstrap_diff([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], n_resamples=1000)
    assert result.mean_diff == 1.0
    assert result.p_value < 0.05


def test_groundedness_bounds() -> None:
    assert groundedness("退款 七 天", ["退款 七 天 政策"]) == 1.0
    assert groundedness("无关内容", ["完全不同"]) == 0.0


def test_render_table_fills_tbd() -> None:
    table = render_table([{"metric": "acc", "base": None}], ["metric", "base", "sft"])
    assert "acc" in table
    assert table.count("TBD") == 2


def test_runners_smoke_works_and_real_run_is_guarded() -> None:
    summary = run_tau2.smoke(ScriptedLLMClient())
    assert summary.tool_accuracy == 1.0
    assert "synthetic" in summary.label
    with pytest.raises(NotImplementedError):
        run_tau2.run()
    with pytest.raises(NotImplementedError):
        run_bfcl.run()
