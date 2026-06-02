"""Policy rules. A response that violates any rule counts as a FAILURE.

Rules are intentionally simple and fully typed so they unit-test deterministically.
The human-readable Chinese policy documents live under ``policy_docs/``.
"""

from __future__ import annotations

from pydantic import BaseModel

# Refunds are only allowed within this many days of purchase (service-desk policy).
REFUND_WINDOW_DAYS: int = 7


class PolicyDecision(BaseModel):
    """Outcome of a single policy evaluation."""

    rule_id: str
    allowed: bool
    reason: str


def refund_allowed(
    days_since_purchase: int, window_days: int = REFUND_WINDOW_DAYS
) -> PolicyDecision:
    """Decide whether a refund is permitted under the refund-window policy."""
    allowed = 0 <= days_since_purchase <= window_days
    reason = (
        f"within the {window_days}-day refund window"
        if allowed
        else f"refund window ({window_days}d) exceeded: {days_since_purchase}d since purchase"
    )
    return PolicyDecision(rule_id="refund_window", allowed=allowed, reason=reason)
