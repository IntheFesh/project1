# What changed (and why) — credibility & results upgrade

This branch turns the original scaffold (excellent engineering, but **every quality metric was
`TBD` and not actually producible**) into a system that can yield **trustworthy, SOTA-class
results** on a rented **RTX 5090 (AutoDL, ≤5 GPU-days)**. Summary of the work on top of the
original `main`.

## Why it was needed (findings from the code review)
1. **Leakage risk.** SFT trained on the *same* order ids / prompts that the (would-be) eval used
   → accuracy could be silently inflated. This is the #1 thing a reviewer/committee checks.
2. **No real benchmark.** `eval/run_tau2.run()` / `eval/run_bfcl.run()` were bare
   `NotImplementedError` — nothing could ever fill the `TBD`s.
3. **No held-out quality set** for the self-built Chinese domain (only a 5-task CI smoke slice).
4. **Single-step agent** couldn't run τ²-bench (which needs multi-step agentic loops).
5. **Too little SFT data** (7 templated samples) to produce a meaningful QLoRA gain.
6. **Hardware mismatch** — repo defaulted to a 96 GB PRO 6000; the real box is a 32 GB 5090.

## What changed, by theme

### Data hygiene (foundation)
- `agent/tools/order_data.py` — **disjoint train (A-series) / eval (E-series) order pools**.
- `tests/test_leakage.py` — **fails CI** if the pools or any SFT↔eval prompts overlap. It
  caught two real overlapping prompts, now reworded.

### Held-out self-built benchmark
- `eval/datasets/zh_service_desk_eval.jsonl` — **32 held-out tasks** (E-pool only) across
  happy / policy-edge / grounding / negative, incl. the 7-day refund boundary.
- `eval/zh_service_desk.py` — scorer for success / tool-accuracy / args-match / grounding /
  negative handling, and the **policy story**: `policy_violation_rate` (reaches user — **0 by
  construction**) vs `unsafe_selection_rate` (did the model *try*? — the signal QLoRA drives → 0).

### Multi-step agent (enables τ²-bench)
- `agent/graph.py` + `state.py` + nodes — a `policy_check → tool_select` **loop guarded by
  `max_steps`** (default 1 preserves all prior behavior; >1 enables multi-step). A violation
  stops the loop and refuses. `configs/server.yaml: agent.max_steps`.

### Real benchmark integration + statistics
- `eval/run_bfcl.py` / `eval/run_tau2.py` — real **subprocess drivers** (BFCL-V4 CLI; τ²-bench
  with the user-simulator on an **external API**, agent on the 5090) + **tested output parsers**;
  missing prereqs now raise **actionable** errors pointing at `eval/README.md`.
- `eval/results.py` — base-vs-+LoRA with **bootstrap CIs**, **paired bootstrap + Holm–Bonferroni**,
  and **pass^k aggregated over tasks**, rendered to a Markdown table. Fully unit-tested.
- `eval/README.md` — reproducible runbook with **pinned versions** and exact commands.

### Training data + hardware
- `finetune/build_sft_data.py` — scaled to **144 samples** (14 train orders × varied phrasings,
  refusals, multi-step, cited KB) + `ingest_external` (license check + dedup) + `train_val_split`.
- `configs/lora.yaml` — default flipped to **5090 / QLoRA 4-bit** (bs=2 × grad-accum 8).
- `scripts/gpu_run.sh` — **one-command AutoDL pipeline** (setup→data→train→serve→eval→results),
  HF mirror, cu128 sm_120 check.

### Docs / framing
- `README.md`, `report/technical_report.md` — **honest SOTA framing** (competitive among open
  ≤10B models + statistically significant base→+QLoRA gain + policy-violation rate → 0), a
  **data-hygiene** subsection, and AutoDL notes.
- `report/improvement_plan.md`, `report/case_study.md` — the GPU runbook and a recruiter/
  committee-facing one-pager.

## Verification
- **127 deterministic tests green** (was ~105), **ruff clean**, **deterministic eval gate PASS**.
- New off-GPU coverage: leakage guards, held-out scorer (incl. the policy-violation = 0
  invariant), multi-step loop, results aggregation + benchmark parsers, SFT scale + license/dedup.
- **No fabricated numbers** — all quality metrics remain `TBD` until the real 5090 run.

## What's left (on the 5090, per `eval/README.md`)
QLoRA train → serve base/+adapter → zh eval + BFCL-V4 + τ²-bench retail → fill the tables with
real numbers + CIs + a failure analysis. ≈10–20 GPU-hours; power off between runs.
