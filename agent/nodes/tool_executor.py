"""Tool-executor node: validate arguments against the JSON schema, run, retry on invalid."""

from __future__ import annotations

from agent.state import AgentState


def tool_executor(state: AgentState) -> AgentState:
    """Execute the selected tool with schema validation + retry (Phase 2)."""
    raise NotImplementedError("tool_executor node is implemented in Phase 2.")
