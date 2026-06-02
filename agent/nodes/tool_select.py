"""Tool-select node: choose a tool (or the RAG subgraph) for the current plan."""

from __future__ import annotations

from agent.state import AgentState


def tool_select(state: AgentState) -> AgentState:
    """Select the next tool call via the server-side tool parser (Phase 2)."""
    raise NotImplementedError("tool_select node is implemented in Phase 2.")
