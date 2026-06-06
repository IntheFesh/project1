"""Run tau2-bench (retail + knowledge) against our served model; report pass^k.

``run()`` drives the *sierra-research/tau2-bench* CLI with the **agent** LLM pointed at our
SGLang endpoint (the model under test) and the **user-simulator** LLM pointed at an external
API (cheap, saves the GPU's VRAM/time). It collects per-task pass/fail and aggregates pass^k
with bootstrap CIs (``eval/results.py``). Executed on the GPU box; ``smoke()`` validates the
harness off-GPU. The results parser is pure and unit-tested.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from agent.tools.services import ServiceDesk
from eval.harness import EvalSummary, evaluate_tasks, summarize
from eval.results import PassKResult, aggregate_pass_k
from eval.tasks import SMOKE_TASKS
from serving.client import LLMClient


class Tau2Result(BaseModel):
    """Per-domain tau2-bench outcome: trials + pass^k at several k."""

    domain: str
    n_tasks: int
    pass_hat: dict[int, PassKResult]


def parse_tau2_trials(text: str) -> list[tuple[int, int]]:
    """Parse tau2-bench results JSON into per-task ``(n_attempts, n_successes)``.

    Accepts a list of task entries (or ``{"results": [...]}``); each entry has a list of
    trial rewards/successes under ``trials`` | ``rewards`` | ``results``, where success is a
    reward of 1.0 / a truthy ``success``. This is the input to the pass^k estimator.
    """
    data = json.loads(text)
    entries = data["results"] if isinstance(data, dict) and "results" in data else data
    trials: list[tuple[int, int]] = []
    for entry in entries:
        outcomes = entry.get("trials", entry.get("rewards", entry.get("results", [])))
        flags = [_is_success(o) for o in outcomes]
        if flags:
            trials.append((len(flags), sum(flags)))
    return trials


def _is_success(outcome: Any) -> bool:
    if isinstance(outcome, dict):
        if "success" in outcome:
            return bool(outcome["success"])
        outcome = outcome.get("reward", 0)
    return float(outcome) >= 1.0


def pass_k_from_trials(trials: Sequence[tuple[int, int]], k_values: Sequence[int]) -> dict[int, PassKResult]:
    """Aggregate pass^k over tasks for each k (skipping k > min attempts)."""
    if not trials:
        return {}
    min_n = min(n for n, _ in trials)
    return {k: aggregate_pass_k(trials, k) for k in k_values if k <= min_n}


def run(
    domain: str = "retail",
    *,
    agent_base_url: str = "http://localhost:30000/v1",
    agent_model: str = "Qwen/Qwen3-8B",
    user_model: str = "deepseek-chat",
    user_base_url: str | None = None,
    results_path: str | None = None,
    k_values: Sequence[int] = (1, 2, 4),
    num_trials: int = 4,
) -> Tau2Result:
    """Drive tau2-bench for ``domain`` and return pass^k.

    The agent LLM is the served model under test; the user-simulator runs on an external
    API. Requires the ``tau2`` CLI (``pip install tau2-bench`` or the repo). See
    ``eval/README.md`` for the exact, version-pinned commands. Raises an actionable error
    when a prerequisite is missing.
    """
    if shutil.which("tau2") is None:
        raise RuntimeError(
            "tau2 CLI not found. Install it on the GPU box: `pip install tau2-bench` (or clone "
            "sierra-research/tau2-bench). See eval/README.md for the pinned version + commands. "
            "Use smoke() for off-GPU pipeline validation."
        )
    cmd = ["tau2", "run", "--domain", domain,
           "--agent-llm", agent_model, "--agent-base-url", agent_base_url,
           "--user-llm", user_model, "--num-trials", str(num_trials)]
    if user_base_url:
        cmd += ["--user-base-url", user_base_url]
    subprocess.run(cmd, check=True)  # noqa: S603 - args constructed, not shell
    if results_path is None:
        raise RuntimeError("Pass results_path pointing at tau2's results JSON (see eval/README.md).")
    with open(results_path, encoding="utf-8") as handle:
        trials = parse_tau2_trials(handle.read())
    return Tau2Result(domain=domain, n_tasks=len(trials), pass_hat=pass_k_from_trials(trials, k_values))


def smoke(client: LLMClient, services: ServiceDesk | None = None) -> EvalSummary:
    """Run the synthetic smoke slice (off-GPU pipeline check, not a benchmark)."""
    records = evaluate_tasks(SMOKE_TASKS, client, services or ServiceDesk())
    return summarize(records, label="tau2-smoke-synthetic")
