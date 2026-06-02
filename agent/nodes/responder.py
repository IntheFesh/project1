"""Responder node: compose the final, policy-checked, optionally cited answer."""

from __future__ import annotations

from agent.state import AgentState


def responder(state: AgentState) -> AgentState:
    """Render the final answer from tool results + citations (Phase 2)."""
    raise NotImplementedError("responder node is implemented in Phase 2.")
