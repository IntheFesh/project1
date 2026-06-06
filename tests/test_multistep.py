"""Multi-step agent loop (max_steps > 1) — the basis for tau2-bench.

Verifies that the loop (a) terminates when the model answers after observing a tool
result, (b) stops immediately on a policy violation (refuse, don't keep going), and
(c) never exceeds the step budget.
"""

from __future__ import annotations

from typing import Any

from agent.graph import run_turn
from agent.tools.services import ServiceDesk
from rag.pipeline import build_default_kb_search
from serving.client import LLMResponse, LLMToolCall, ScriptedLLMClient


def _svc() -> ServiceDesk:
    return ServiceDesk(kb_search=build_default_kb_search())


def test_loop_terminates_after_observing_a_result() -> None:
    state = run_turn("帮我查一下订单 E9001 的状态", ScriptedLLMClient(), _svc(), max_steps=4)
    assert [t.name for t in state.executed_tools] == ["query_order"]
    assert state.steps == 1
    assert state.final_answer  # answered after observing the tool result


def test_policy_violation_stops_the_loop() -> None:
    # E9003 is 30 days old -> refund is past the window. The loop must stop and refuse,
    # not execute further tools.
    state = run_turn("订单 E9003 我要退款", ScriptedLLMClient(), _svc(), max_steps=4)
    assert [t.name for t in state.executed_tools] == ["refund"]
    assert state.violations and state.violations[0].rule_id == "refund_window"
    assert "无法直接执行" in (state.final_answer or "")


class _AlwaysCallsTool:
    """Stub client that never stops calling a tool — exercises the step cap."""

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> LLMResponse:
        return LLMResponse(tool_calls=[LLMToolCall(name="query_order", arguments={"order_id": "E9001"})])


def test_step_budget_is_respected() -> None:
    state = run_turn("查询订单 E9001", _AlwaysCallsTool(), _svc(), max_steps=3)
    assert state.steps == 3
    assert len(state.executed_tools) == 3  # capped, no runaway loop


def test_single_step_default_is_unchanged() -> None:
    # max_steps defaults to 1: exactly one tool, no loop-back.
    state = run_turn("订单 E9001 我要退款", ScriptedLLMClient(), _svc())
    assert state.steps == 1
    assert [t.name for t in state.executed_tools] == ["refund"]
