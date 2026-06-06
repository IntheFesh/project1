"""LangGraph wiring: planner -> tool_select -> tool_executor -> policy_check -> responder.

The graph uses ``AgentState`` (pydantic) as its schema and an injected LLM client +
ServiceDesk, so the same graph runs against SGLang/vLLM/Ollama or the offline mock.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.planner import planner
from agent.nodes.policy_check import policy_check
from agent.nodes.responder import responder
from agent.nodes.tool_executor import tool_executor
from agent.nodes.tool_select import tool_select
from agent.state import AgentState
from agent.tools.services import ServiceDesk
from serving.client import LLMClient

NODE_ORDER: tuple[str, ...] = (
    "planner",
    "tool_select",
    "tool_executor",
    "policy_check",
    "responder",
)


def _route_after_select(state: AgentState) -> str:
    """Skip execution/policy when the model answered directly (no tool)."""
    return "tool_executor" if state.selected_tool is not None else "responder"


def _route_after_policy(state: AgentState, max_steps: int) -> str:
    """Loop back for another tool while within budget; stop on a violation or the cap.

    With ``max_steps == 1`` this always routes to the responder, preserving the original
    single-tool-per-turn behavior. With ``max_steps > 1`` the agent may call another tool
    after observing the result (multi-step agentic loop), unless policy blocked this step.
    """
    if state.violations:
        return "responder"  # a violation stops the turn -> refuse
    if state.steps >= max_steps:
        return "responder"
    return "tool_select"


def build_graph(
    client: LLMClient,
    services: ServiceDesk,
    checkpointer: Any | None = None,
    max_steps: int = 2,
) -> CompiledStateGraph:
    """Build and compile the agent graph with checkpointing.

    ``max_steps`` caps the number of tool calls per turn (1 = single-step; >1 enables the
    multi-step loop needed for tau2-bench).
    """
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("tool_select", lambda s: tool_select(s, client))
    builder.add_node("tool_executor", lambda s: tool_executor(s, client, services))
    builder.add_node("policy_check", lambda s: policy_check(s, services))
    builder.add_node("responder", responder)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "tool_select")
    builder.add_conditional_edges(
        "tool_select",
        _route_after_select,
        {"tool_executor": "tool_executor", "responder": "responder"},
    )
    builder.add_edge("tool_executor", "policy_check")
    builder.add_conditional_edges(
        "policy_check",
        lambda s: _route_after_policy(s, max_steps),
        {"tool_select": "tool_select", "responder": "responder"},
    )
    builder.add_edge("responder", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def run_turn(
    text: str,
    client: LLMClient,
    services: ServiceDesk,
    thread_id: str = "default",
    graph: CompiledStateGraph | None = None,
    max_steps: int = 2,
    tool_choice_override: str | None = None,
) -> AgentState:
    """Run one user turn through the graph and return the final ``AgentState``."""
    graph = graph or build_graph(client, services, max_steps=max_steps)
    initial: dict[str, Any] = {"messages": [{"role": "user", "content": text}]}
    if tool_choice_override is not None:
        initial["tool_choice_override"] = tool_choice_override
    result = graph.invoke(
        initial,
        config={"configurable": {"thread_id": thread_id}},
    )
    return result if isinstance(result, AgentState) else AgentState.model_validate(result)
