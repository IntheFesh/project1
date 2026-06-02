"""The CI eval gate passes on the smoke slice and is deterministic."""

from eval.gate import main, run_gate
from serving.client import ScriptedLLMClient


def test_gate_passes_on_smoke_slice() -> None:
    result = run_gate(ScriptedLLMClient())
    assert result.passed
    assert result.tool_accuracy >= result.thresholds["tool_accuracy"]
    assert result.schema_valid_rate >= result.thresholds["schema_valid_rate"]


def test_gate_decision_is_deterministic() -> None:
    first = run_gate(ScriptedLLMClient())
    second = run_gate(ScriptedLLMClient())
    assert first.model_dump() == second.model_dump()


def test_gate_cli_returns_zero_on_pass() -> None:
    assert main() == 0
