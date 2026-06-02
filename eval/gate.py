"""Deterministic CI eval gate (Phase 6).

Fixed task slice, greedy decoding (temperature=0) -> the SAME input yields the SAME
pass/fail decision every run. It checks schema-valid rate and tool accuracy against the
thresholds in ``configs/eval.yaml``. In CI it runs the deterministic ScriptedLLMClient (no
GPU); on the box it would run against SGLang at temperature=0. This gate intentionally does
NOT use noisy multi-seed/bootstrap estimates — those are for headline results (Phase 5/8).
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from agent.tools.services import ServiceDesk
from common.config import CIGateConfig, load_eval_config
from eval.harness import evaluate_tasks
from eval.metrics import schema_valid_rate, tool_accuracy
from eval.tasks import SMOKE_TASKS
from serving.client import LLMClient, ScriptedLLMClient


class GateResult(BaseModel):
    """Outcome of the deterministic gate."""

    passed: bool
    tool_accuracy: float
    schema_valid_rate: float
    thresholds: dict[str, float]
    failures: list[str]


def run_gate(
    client: LLMClient,
    services: ServiceDesk | None = None,
    config: CIGateConfig | None = None,
) -> GateResult:
    """Evaluate the fixed slice and decide pass/fail against the thresholds."""
    config = config or load_eval_config().ci_gate
    records = evaluate_tasks(SMOKE_TASKS, client, services or ServiceDesk())
    ta = tool_accuracy(records)
    sv = schema_valid_rate(records)

    failures: list[str] = []
    if ta < config.thresholds.tool_accuracy:
        failures.append(f"tool_accuracy {ta:.3f} < {config.thresholds.tool_accuracy}")
    if sv < config.thresholds.schema_valid_rate:
        failures.append(f"schema_valid_rate {sv:.3f} < {config.thresholds.schema_valid_rate}")

    return GateResult(
        passed=not failures,
        tool_accuracy=ta,
        schema_valid_rate=sv,
        thresholds={
            "tool_accuracy": config.thresholds.tool_accuracy,
            "schema_valid_rate": config.thresholds.schema_valid_rate,
        },
        failures=failures,
    )


def main() -> int:
    """CLI entry point: run the gate with the deterministic mock; exit non-zero on fail."""
    result = run_gate(ScriptedLLMClient())
    status = "PASS" if result.passed else "FAIL"
    print(
        f"[eval-gate] {status} "
        f"tool_accuracy={result.tool_accuracy:.3f} "
        f"schema_valid_rate={result.schema_valid_rate:.3f}"
    )
    for failure in result.failures:
        print(f"  - {failure}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
