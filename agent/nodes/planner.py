"""Planner node: turn the latest user request into a short execution plan."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState


def planner(state: AgentState) -> dict[str, Any]:
    """Produce a one-line plan and reset transient per-turn fields.

    Runs first each turn, so it clears any state left over from a previous turn on the
    same checkpointed thread (otherwise an earlier answer/violation could leak forward).
    """
    user = state.last_user_text()
    return {
        "plan": f"理解用户意图并选择合适工具处理：{user[:60]}",
        "selected_tool": None,
        "tool_result": None,
        "final_answer": None,
        "violations": [],
        "citations": [],
    }
