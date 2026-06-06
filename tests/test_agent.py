"""End-to-end agent graph scenarios (deterministic mock client)."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph, run_turn
from agent.tools.services import ServiceDesk
from rag.pipeline import build_default_kb_search
from serving.client import ScriptedLLMClient


@pytest.fixture
def client() -> ScriptedLLMClient:
    return ScriptedLLMClient()


@pytest.fixture
def services() -> ServiceDesk:
    return ServiceDesk()


def test_query_order_scenario(client, services) -> None:
    # max_steps>1: the turn ends with a direct answer, so read acted_tool (what ran),
    # not selected_tool (the final, empty proposal).
    state = run_turn("帮我查询订单 A1001 的状态", client, services)
    assert state.acted_tool is not None
    assert state.acted_tool.name == "query_order"
    assert state.policy_ok
    assert state.final_answer


def test_refund_within_window_allowed(client, services) -> None:
    state = run_turn("订单 A1001 我要退款", client, services)
    assert state.acted_tool.name == "refund"
    assert state.policy_ok
    assert "无法" not in (state.final_answer or "")


def test_refund_past_window_refused(client, services) -> None:
    state = run_turn("订单 A1009 我要退款", client, services)
    assert state.selected_tool.name == "refund"
    assert not state.policy_ok
    assert any(v.rule_id == "refund_window" for v in state.violations)
    assert "无法" in (state.final_answer or "")


def test_modify_shipped_order_refused(client, services) -> None:
    state = run_turn("把订单 A1002 的地址改一下", client, services)
    assert state.selected_tool.name == "modify_order"
    assert not state.policy_ok
    assert any(v.rule_id == "modify_after_ship" for v in state.violations)


def test_knowledge_query_is_grounded_and_cited(client) -> None:
    services = ServiceDesk(kb_search=build_default_kb_search())
    state = run_turn("请问运费是怎么计算的？", client, services)
    assert state.acted_tool.name == "search_kb"
    assert state.citations  # grounding: KB docs were retrieved + cited (structured signal)
    assert state.final_answer  # multi-step: the answer text is the model's, citations live in state


def test_same_thread_does_not_leak_state_across_turns(client) -> None:
    services = ServiceDesk(kb_search=build_default_kb_search())
    graph = build_graph(client, services, checkpointer=MemorySaver())
    first = run_turn("订单 A1009 我要退款", client, services, thread_id="t", graph=graph)
    assert not first.policy_ok  # refund past window is refused
    second = run_turn("请问运费是怎么计算的？", client, services, thread_id="t", graph=graph)
    assert second.acted_tool.name == "search_kb"
    assert second.policy_ok  # no stale violation carried over
    assert "无法" not in (second.final_answer or "")  # no stale refusal carried over
    assert second.citations
