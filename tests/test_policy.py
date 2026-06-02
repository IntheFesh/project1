"""The refund-window policy is enforced deterministically."""

from agent.policies.rules import REFUND_WINDOW_DAYS, refund_allowed


def test_within_window_allowed() -> None:
    assert refund_allowed(0).allowed
    assert refund_allowed(REFUND_WINDOW_DAYS).allowed


def test_past_window_refused() -> None:
    decision = refund_allowed(REFUND_WINDOW_DAYS + 1)
    assert not decision.allowed
    assert "window" in decision.reason
