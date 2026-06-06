"""Run BFCL-V4 against our served model and report AST accuracy.

``run()`` drives the gorilla *berkeley-function-call-leaderboard* CLI against our
OpenAI-compatible vLLM endpoint, then parses the score summary. It is executed on the
GPU box; the exact BFCL-V4 commit is recorded in ``eval/README.md`` and the result.
``smoke()`` runs a tiny synthetic slice to validate the harness off-GPU (NOT a BFCL number).
The score parser is pure and unit-tested.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from pydantic import BaseModel

from agent.tools.services import ServiceDesk
from eval.harness import EvalSummary, evaluate_tasks, summarize
from eval.tasks import SMOKE_TASKS
from serving.client import LLMClient


class BFCLResult(BaseModel):
    """Parsed BFCL-V4 AST-accuracy summary."""

    version: str
    overall_ast_accuracy: float
    per_category: dict[str, float]


def parse_bfcl_summary(text: str) -> BFCLResult:
    """Parse a BFCL score summary (JSON) into AST accuracy + per-category breakdown.

    Accepts either ``{"version", "overall_accuracy"|"ast_accuracy", "categories": {...}}``
    or a flat ``{category: accuracy}`` map. Percentages (>1) are normalized to [0, 1].
    """
    data = json.loads(text)

    def _norm(v: float) -> float:
        return v / 100.0 if v > 1.0 else v

    categories_raw = data.get("categories", data.get("per_category"))
    if isinstance(categories_raw, dict):
        per_cat = {k: _norm(float(v)) for k, v in categories_raw.items()}
        overall = data.get("overall_accuracy", data.get("ast_accuracy"))
        overall = _norm(float(overall)) if overall is not None else (
            sum(per_cat.values()) / len(per_cat) if per_cat else 0.0
        )
        version = str(data.get("version", "BFCL-V4"))
    else:  # flat {category: accuracy} map
        per_cat = {k: _norm(float(v)) for k, v in data.items() if isinstance(v, int | float)}
        overall = sum(per_cat.values()) / len(per_cat) if per_cat else 0.0
        version = "BFCL-V4"
    return BFCLResult(version=version, overall_ast_accuracy=overall, per_category=per_cat)


def run(
    base_url: str = "http://localhost:30000/v1",
    model: str = "Qwen/Qwen3-8B",
    *,
    summary_path: str | None = None,
    test_category: str = "ast",
) -> BFCLResult:
    """Drive the BFCL-V4 CLI against the served model and return parsed AST accuracy.

    Requires the ``bfcl`` CLI (``pip install bfcl-eval`` or the gorilla repo) and a running
    OpenAI-compatible endpoint at ``base_url``. See ``eval/README.md`` for the exact,
    version-pinned commands. Raises an actionable error if a prerequisite is missing.
    """
    if shutil.which("bfcl") is None:
        raise RuntimeError(
            "BFCL CLI not found. Install it on the GPU box: `pip install bfcl-eval` (or clone "
            "gorilla/berkeley-function-call-leaderboard). See eval/README.md for the pinned "
            "version + commands. Use smoke() for off-GPU pipeline validation."
        )
    # Documented two-step BFCL flow (generate then evaluate). The handler points BFCL at our
    # OpenAI-compatible endpoint via env; see eval/README.md.
    env_cmd = ["bfcl", "generate", "--model", model, "--test-category", test_category,
               "--backend", "openai", "--base-url", base_url]
    subprocess.run(env_cmd, check=True)  # noqa: S603 - args are constructed, not shell
    subprocess.run(["bfcl", "evaluate", "--model", model, "--test-category", test_category],
                   check=True)  # noqa: S603
    if summary_path is None:
        raise RuntimeError(
            "Pass summary_path pointing at BFCL's score summary JSON (see eval/README.md)."
        )
    with open(summary_path, encoding="utf-8") as handle:
        return parse_bfcl_summary(handle.read())


def smoke(client: LLMClient, services: ServiceDesk | None = None) -> EvalSummary:
    """Run the synthetic smoke slice (off-GPU pipeline check, not a benchmark)."""
    records = evaluate_tasks(SMOKE_TASKS, client, services or ServiceDesk())
    return summarize(records, label="bfcl-smoke-synthetic")
