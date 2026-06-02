"""Run tau2-bench (retail + knowledge domains) and emit per-task results.

Wraps github.com/sierra-research/tau2-bench. Implemented in Phase 5.
"""

from __future__ import annotations

from typing import Any


def run(domain: str = "retail", n_tasks: int | None = None) -> dict[str, Any]:
    """Execute a tau2-bench domain slice and return raw results (Phase 5)."""
    raise NotImplementedError("run_tau2 is implemented in Phase 5.")
