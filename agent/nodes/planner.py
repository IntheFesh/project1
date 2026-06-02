"""Planner node: turn the latest user request into a short execution plan."""

from __future__ import annotations

from agent.state import AgentState


def planner(state: AgentState) -> AgentState:
    """Produce a plan for the current turn (implemented in Phase 2)."""
    raise NotImplementedError("planner node is implemented in Phase 2.")
