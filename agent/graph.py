"""LangGraph wiring: planner -> tool_select -> tool_executor -> policy_check -> responder.

The compiled graph (with checkpointing) is built in Phase 2; this module fixes the node
order and exposes a factory so the package imports cleanly during scaffolding.
"""

from __future__ import annotations

from typing import Any

NODE_ORDER: tuple[str, ...] = (
    "planner",
    "tool_select",
    "tool_executor",
    "policy_check",
    "responder",
)


def build_graph() -> Any:
    """Build and compile the LangGraph state machine (implemented in Phase 2)."""
    raise NotImplementedError("Graph construction is implemented in Phase 2.")
