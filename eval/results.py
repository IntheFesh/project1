"""Headline results: base vs +LoRA with bootstrap CIs, pass^k, and corrected p-values.

This is the layer that turns raw per-task outcomes (from the zh service-desk scorer,
BFCL, and tau2-bench) into the report tables — every number carries a 95% bootstrap CI,
multi-metric comparisons are Holm-Bonferroni corrected, and pass^k is the unbiased
combinatorial estimator aggregated across tasks. Pure + deterministic, so it is fully
unit-tested off-GPU; on the box it consumes the real run outputs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from eval.bootstrap import bootstrap_ci
from eval.passk import pass_hat_k
from eval.stats import holm_bonferroni, paired_bootstrap_diff


class PassKResult(BaseModel):
    """pass^k aggregated over tasks, with a bootstrap CI over tasks."""

    k: int
    n_tasks: int
    mean: float
    ci_low: float
    ci_high: float


class MetricComparison(BaseModel):
    """One metric compared between base and +LoRA (paired across the same tasks)."""

    name: str
    base_mean: float
    lora_mean: float
    base_ci: tuple[float, float]
    lora_ci: tuple[float, float]
    mean_diff: float
    diff_ci: tuple[float, float]
    p_value: float
    adjusted_p: float | None = None
    significant: bool | None = None
    higher_is_better: bool = True


class ResultsReport(BaseModel):
    """A full base-vs-LoRA comparison across metrics (Holm-Bonferroni corrected)."""

    label: str
    comparisons: list[MetricComparison]

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown table for the report."""
        rows = ["| Metric | Base [95% CI] | +LoRA [95% CI] | Δ [95% CI] | adj. p | sig |",
                "| --- | --- | --- | --- | --- | --- |"]
        for c in self.comparisons:
            sig = "—" if c.significant is None else ("✓" if c.significant else "·")
            adj = "—" if c.adjusted_p is None else f"{c.adjusted_p:.3f}"
            rows.append(
                f"| {c.name} | {c.base_mean:.3f} [{c.base_ci[0]:.3f}, {c.base_ci[1]:.3f}] "
                f"| {c.lora_mean:.3f} [{c.lora_ci[0]:.3f}, {c.lora_ci[1]:.3f}] "
                f"| {c.mean_diff:+.3f} [{c.diff_ci[0]:+.3f}, {c.diff_ci[1]:+.3f}] | {adj} | {sig} |"
            )
        return "\n".join(rows)


def aggregate_pass_k(trials: Sequence[tuple[int, int]], k: int, seed: int = 42) -> PassKResult:
    """Aggregate pass^k over tasks. ``trials`` is per-task ``(n_attempts, n_successes)``."""
    per_task = [pass_hat_k(n, c, k) for n, c in trials]
    low, high = bootstrap_ci(per_task, seed=seed) if per_task else (0.0, 0.0)
    mean = sum(per_task) / len(per_task) if per_task else 0.0
    return PassKResult(k=k, n_tasks=len(per_task), mean=mean, ci_low=low, ci_high=high)


def compare_metric(
    name: str,
    base: Sequence[float],
    lora: Sequence[float],
    higher_is_better: bool = True,
    seed: int = 42,
) -> MetricComparison:
    """Compare one paired metric (same tasks) between base and +LoRA."""
    base_ci = bootstrap_ci(base, seed=seed)
    lora_ci = bootstrap_ci(lora, seed=seed)
    diff = paired_bootstrap_diff(lora, base, seed=seed)
    return MetricComparison(
        name=name,
        base_mean=sum(base) / len(base),
        lora_mean=sum(lora) / len(lora),
        base_ci=base_ci,
        lora_ci=lora_ci,
        mean_diff=diff.mean_diff,
        diff_ci=(diff.ci_low, diff.ci_high),
        p_value=diff.p_value,
        higher_is_better=higher_is_better,
    )


def build_report(label: str, comparisons: list[MetricComparison], alpha: float = 0.05) -> ResultsReport:
    """Apply Holm-Bonferroni across the comparisons' p-values and fill significance."""
    adjusted = holm_bonferroni({c.name: c.p_value for c in comparisons}, alpha=alpha)
    for c in comparisons:
        c.adjusted_p, c.significant = adjusted[c.name]
    return ResultsReport(label=label, comparisons=comparisons)


def write_results(report: ResultsReport, path: str) -> None:
    """Write the report JSON + a Markdown table next to it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(report.to_markdown() + "\n", encoding="utf-8")
