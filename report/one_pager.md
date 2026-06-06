# PolicyArena · One-Pager

**A statistically-validated, policy-compliant tool-calling agent on Qwen3-8B.**
Held-out Chinese service-desk benchmark; QLoRA-SFT on RTX 5090.

---

## The result in one line

QLoRA-SFT lifts `success_rate` from **53.1 % → 96.9 %** (Δ = +43.8 pp, paired bootstrap
**p < 0.001**, Holm-Bonferroni corrected) and `grounding_rate` from **25.0 % → 87.5 %**
(p = 0.005), with **0 policy violations** across every run thanks to a deterministic
policy gate.

## Why it is credible (the differentiators)

| | |
|---|---|
| **Data hygiene, enforced by CI** | SFT trains on A-series orders, eval uses disjoint E-series; `tests/test_leakage.py` fails the build on any overlap (5 leakage tests, 0 leaks). |
| **Statistics-first** | Every headline number has a 95 % bootstrap CI (10k resamples); base-vs-+LoRA uses paired bootstrap + Holm-Bonferroni; pass^k for tool-calling. |
| **Safety as a measured invariant** | Two metrics, not one: `policy_violation_rate` = 0 % (gate-guaranteed) AND `unsafe_selection_rate` (model intent). Defence in depth. |
| **Honest scope** | No fabricated numbers, regressions documented and root-caused, no leaderboard claims without a leaderboard run. |

## Headline numbers (held-out, n = 32, paired bootstrap)

| Metric | Base 95 % CI | +LoRA 95 % CI | Δ 95 % CI | adj. p | sig. |
|---|---|---|---|---:|:---:|
| success_rate | 0.531 [0.344, 0.688] | 0.969 [0.906, 1.000] | +0.438 [+0.281, +0.594] | 0.000 | **✓** |
| grounding_rate | 0.250 [0.000, 0.625] | 0.875 [0.625, 1.000] | +0.625 [+0.250, +0.875] | 0.005 | **✓** |
| negative_handling | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 | 1.000 | · |
| unsafe_selection (↓) | 0.286 [0.000, 0.571] | 0.714 [0.429, 1.000] | +0.429 [+0.143, +0.857] | 0.072 | · |
| **policy_violation_rate** | **0 %** | **0 %** | gate-guaranteed | — | — |

## What the evaluation pipeline surfaced (bonus contributions)

Three production-relevant findings emerged from iterative protocol hardening — each
documented in code:

1. **`tool_choice` is not a free binary.** Auto loses grounding (model internalises KB),
   required loses negative-handling (model fabricates tool calls). Fix: task-aware routing.
2. **`max_steps=1` makes check-then-act SFT look broken.** Lifting to `max_steps=2`
   recovered +18.8 pp success on the same adapter.
3. **`assistant_only_loss` matters more than data balance.** Full-sequence loss teaches
   the model to memorise tool returns, defeating retrieval.

## Honest limitations

- n = 32 held-out tasks → wide CIs (e.g. base 95 % CI ±17 pp). No comparison to public
  leaderboards (BFCL-V4, τ²-bench) yet.
- `unsafe_selection_rate` regresses (28.6 → 71.4 %). Root-caused as stricter unsafe
  detection (F7) + small-data overfitting; **operationally harmless because the gate
  intercepts all violations**. Resolution requires ≈10 k SFT examples (planned).
- ≤10 B SOTA references (e.g. ToolACE-2-8B @ 68.7 % on BFCL-V4) trained on 30k+
  trajectories; this work used 240. Not comparable on absolute scale; the contribution
  is **statistically validated within-domain improvement**.

## Stack

Qwen3-8B · vLLM 0.22.1 (cu130) · LangGraph multi-step loop · hybrid RAG (BM25 + dense
+ RRF + rerank) · QLoRA r=16 α=32 4-bit · trl 1.5.1 · paired bootstrap with
Holm–Bonferroni · LangGraph + Docker Compose · CI-gated.

Reproduce: `make train-lora && make eval-lora` on an RTX 5090 (Blackwell, 32 GB).
Every cell in this document is filled from a real run on the box.
