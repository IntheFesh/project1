"""Policy-check node: enforce the service-desk policy docs; mark violations.

A response that violates any policy (e.g. refund past the window) is a FAILURE.
"""

from __future__ import annotations

from agent.state import AgentState


def policy_check(state: AgentState) -> AgentState:
    """Evaluate policy rules and append any violations to the state (Phase 2)."""
    raise NotImplementedError("policy_check node is implemented in Phase 2.")
