"""Run BFCL-V4 and report AST accuracy. Real runs need the BFCL repo + a served model.

``run()`` is the real benchmark (GPU box; record the exact V4 version). ``smoke()`` runs a
tiny SYNTHETIC slice to validate the harness off-GPU — not a BFCL result.
"""

from __future__ import annotations

from typing import Any

from agent.tools.services import ServiceDesk
from eval.harness import EvalSummary, evaluate_tasks, summarize
from eval.tasks import SMOKE_TASKS
from serving.client import LLMClient


def run(n_tasks: int | None = None) -> dict[str, Any]:
    """Run the real BFCL-V4 AST-accuracy slice (GPU box; Phase 5)."""
    raise NotImplementedError(
        "Real BFCL-V4 runs via the gorilla berkeley-function-call-leaderboard against a "
        "served model on the GPU box. Use smoke() for off-GPU pipeline validation."
    )


def smoke(client: LLMClient, services: ServiceDesk | None = None) -> EvalSummary:
    """Run the synthetic smoke slice (off-GPU pipeline check, not a benchmark)."""
    records = evaluate_tasks(SMOKE_TASKS, client, services or ServiceDesk())
    return summarize(records, label="bfcl-smoke-synthetic")
