"""Planner node: turn the latest user request into a short execution plan."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState


def planner(state: AgentState) -> dict[str, Any]:
    """Produce a one-line plan for the current turn."""
    user = state.last_user_text()
    return {"plan": f"理解用户意图并选择合适工具处理：{user[:60]}"}
