"""Policy-check node: enforce the service-desk policy docs; mark violations.

A response that violates any policy (e.g. refund past the window, modifying a shipped
order) is a FAILURE. The responder turns recorded violations into a refusal.
"""

from __future__ import annotations

from typing import Any

from agent.policies.rules import modify_allowed, refund_allowed
from agent.state import AgentState, PolicyViolation
from agent.tools.services import ServiceDesk


def policy_check(state: AgentState, services: ServiceDesk) -> dict[str, Any]:
    """Evaluate policy rules for the selected tool and record any violations."""
    tool = state.selected_tool
    if tool is None:
        return {}

    violations: list[PolicyViolation] = []
    order_id = tool.arguments.get("order_id")
    order = services.get_order(order_id) if isinstance(order_id, str) else None

    if tool.name == "refund" and order is not None:
        decision = refund_allowed(order.days_since_purchase)
        if not decision.allowed:
            violations.append(PolicyViolation(rule_id=decision.rule_id, message=decision.reason))

    if tool.name == "modify_order" and order is not None:
        decision = modify_allowed(order.shipped)
        if not decision.allowed:
            violations.append(PolicyViolation(rule_id=decision.rule_id, message=decision.reason))

    return {"violations": violations}
