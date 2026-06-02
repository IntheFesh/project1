"""Run task slices through the agent graph and summarize tool-calling metrics."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from agent.graph import run_turn
from agent.tools.registry import is_valid
from agent.tools.services import ServiceDesk
from eval.metrics import (
    EvalRecord,
    policy_violation_rate,
    schema_valid_rate,
    tool_accuracy,
)
from serving.client import LLMClient

Task = dict[str, str]


class EvalSummary(BaseModel):
    """Aggregate metrics for one evaluated task slice."""

    label: str
    n: int
    tool_accuracy: float
    schema_valid_rate: float
    policy_violation_rate: float


def evaluate_tasks(
    tasks: Sequence[Task], client: LLMClient, services: ServiceDesk
) -> list[EvalRecord]:
    """Run each task through the agent and collect per-task eval records."""
    records: list[EvalRecord] = []
    for i, task in enumerate(tasks):
        state = run_turn(task["prompt"], client, services, thread_id=f"eval-{i}")
        tool = state.selected_tool
        predicted = tool.name if tool else None
        args_valid = is_valid(tool.name, tool.arguments) if tool else True
        records.append(
            EvalRecord(
                prompt=task["prompt"],
                expected_tool=task.get("expected_tool"),
                predicted_tool=predicted,
                args_valid=args_valid,
                policy_ok=state.policy_ok,
            )
        )
    return records


def summarize(records: Sequence[EvalRecord], label: str) -> EvalSummary:
    """Aggregate eval records into a labeled summary."""
    return EvalSummary(
        label=label,
        n=len(records),
        tool_accuracy=tool_accuracy(records),
        schema_valid_rate=schema_valid_rate(records),
        policy_violation_rate=policy_violation_rate(records),
    )
