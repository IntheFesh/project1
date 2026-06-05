"""Pipeline checks for the held-out Chinese service-desk scorer.

These validate the SCORER (run against the deterministic ScriptedLLMClient) — the numbers
are not model quality. The key invariant: a forbidden action must never reach the user
(``policy_violation_rate == 0``), guaranteed by the deterministic policy gate.
"""

from __future__ import annotations

from eval.zh_service_desk import evaluate, load_tasks
from serving.client import ScriptedLLMClient


def test_dataset_loads_and_is_nonempty() -> None:
    tasks = load_tasks()
    assert len(tasks) >= 30
    assert {t.category for t in tasks} == {"happy", "policy_edge", "grounding", "negative"}


def test_scorer_runs_and_reports_all_tasks() -> None:
    report = evaluate(ScriptedLLMClient())
    assert report.n == len(load_tasks())
    for rate in (report.success_rate, report.tool_accuracy, report.grounding_rate):
        assert 0.0 <= rate <= 1.0


def test_forbidden_actions_never_reach_the_user() -> None:
    # The deterministic policy gate must block every forbidden action, regardless of how
    # naive the model is. This invariant holds even for the keyword-based mock.
    report = evaluate(ScriptedLLMClient())
    assert report.policy_violation_rate == 0.0


def test_every_task_has_well_formed_gold_labels() -> None:
    for task in load_tasks():
        if task.category == "policy_edge":
            assert task.forbidden_tool in ("refund", "modify_order")
            assert task.gold_policy == "deny"
        if task.category == "grounding":
            assert task.expected_tool == "search_kb"
            assert task.gold_citations
