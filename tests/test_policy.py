"""The service-desk policy rules are enforced deterministically."""

from agent.policies.rules import REFUND_WINDOW_DAYS, modify_allowed, refund_allowed


def test_refund_within_window_allowed() -> None:
    assert refund_allowed(0).allowed
    assert refund_allowed(REFUND_WINDOW_DAYS).allowed


def test_refund_past_window_refused() -> None:
    decision = refund_allowed(REFUND_WINDOW_DAYS + 1)
    assert not decision.allowed
    assert "window" in decision.reason


def test_modify_blocked_when_shipped() -> None:
    assert modify_allowed(False).allowed
    assert not modify_allowed(True).allowed
