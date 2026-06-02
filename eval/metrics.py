"""Tool-calling evaluation metrics over a list of records."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel


class EvalRecord(BaseModel):
    """One evaluated task: what tool was expected vs predicted, and policy outcome."""

    prompt: str
    expected_tool: str | None
    predicted_tool: str | None
    args_valid: bool
    policy_ok: bool

    @property
    def correct(self) -> bool:
        """True iff the predicted tool matches the expected tool."""
        return self.predicted_tool == self.expected_tool


def _mean(values: Sequence[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def tool_accuracy(records: Sequence[EvalRecord]) -> float:
    """Fraction of records whose predicted tool matches the expected tool."""
    return _mean([r.correct for r in records])


def schema_valid_rate(records: Sequence[EvalRecord]) -> float:
    """Fraction of tool calls whose arguments validate (1.0 if no tool was called)."""
    flags = [r.args_valid for r in records if r.predicted_tool is not None]
    return _mean(flags) if flags else 1.0


def policy_violation_rate(records: Sequence[EvalRecord]) -> float:
    """Fraction of records that ended in a policy violation (lower is better)."""
    return _mean([not r.policy_ok for r in records])
