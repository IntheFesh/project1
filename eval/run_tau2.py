"""Run tau2-bench (retail + knowledge). Real runs need the repo + a served model.

``run()`` is the real benchmark and is executed on the GPU box (it drives the served model
through github.com/sierra-research/tau2-bench). ``smoke()`` runs a tiny SYNTHETIC slice to
validate the harness off-GPU — its numbers are NOT tau2-bench results.
"""

from __future__ import annotations

from typing import Any

from agent.tools.services import ServiceDesk
from eval.harness import EvalSummary, evaluate_tasks, summarize
from eval.tasks import SMOKE_TASKS
from serving.client import LLMClient


def run(domain: str = "retail", n_tasks: int | None = None) -> dict[str, Any]:
    """Run the real tau2-bench domain slice (GPU box; Phase 5)."""
    raise NotImplementedError(
        "Real tau2-bench runs via github.com/sierra-research/tau2-bench against a served "
        "model on the GPU box. Use smoke() for off-GPU pipeline validation."
    )


def smoke(client: LLMClient, services: ServiceDesk | None = None) -> EvalSummary:
    """Run the synthetic smoke slice (off-GPU pipeline check, not a benchmark)."""
    records = evaluate_tasks(SMOKE_TASKS, client, services or ServiceDesk())
    return summarize(records, label="tau2-smoke-synthetic")
