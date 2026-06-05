# PolicyArena — Case Study

*A policy-compliant, tool-calling + RAG agent for an enterprise service desk — built to be
trusted, not just to score. (Qwen3-8B · SGLang · LangGraph · QLoRA · statistics-first eval.)*

## The problem
Customer-service agents don't just need to call the right tool — they must **obey written
policy**. Refunding past the 7-day window or editing a shipped order isn't a style slip; it's a
**failure** that costs money and trust. Most tool-calling demos optimize accuracy and ignore
this. PolicyArena treats **any policy violation as a hard failure** and asks: can a small,
cheap, open model be made both *capable* and *safe*?

## The approach
- **Agent** — a LangGraph state machine (`planner → tool_select → tool_executor → policy_check
  → responder`) with a **multi-step loop**: the model calls a tool, observes the result, and
  decides again, up to a step budget. A **deterministic policy gate** sits in the loop — a
  forbidden action is blocked and refused, so it **never reaches the user**.
- **RAG** — hybrid retrieval (dense + BM25 + reciprocal-rank fusion → rerank) returns **cited**
  knowledge answers; an answer without grounding is not trusted.
- **Training** — **QLoRA-SFT** (4-bit, fits a 32 GB RTX 5090) on Chinese service-desk
  trajectories that teach the correct calls *and* **policy-compliant refusals** (check, then
  refuse — never call the forbidden tool).
- **Serving** — one OpenAI-compatible client runs the same graph against SGLang / vLLM / Ollama
  or a deterministic mock, so the whole system is testable without a GPU.

## What makes it credible (the differentiators)
1. **Data hygiene, enforced by CI.** SFT trains on an **A-series** order pool; evaluation uses a
   **disjoint E-series** pool. `tests/test_leakage.py` fails the build if the pools or any
   prompts overlap — it has already caught real leaks. No inflated accuracy.
2. **Statistics-first.** Every headline number carries a **95% bootstrap CI**; base-vs-+LoRA
   uses a **paired bootstrap with Holm–Bonferroni** correction; tool-calling uses the unbiased
   **pass^k** estimator. (`eval/results.py`, all unit-tested.)
3. **Safety as a measured invariant.** The benchmark separates **`policy_violation_rate`**
   (forbidden action reaching the user — **0 by construction**) from **`unsafe_selection_rate`**
   (did the *model* even try the forbidden tool?) — the learning signal QLoRA should drive to
   zero. "Small model + zero policy violations at a fraction of the cost" is the thesis.
4. **Honest scope.** No fabricated numbers — unrun metrics are `TBD`; the report records exact
   BFCL-V4 / τ²-bench commits.

## Results
> Filled from the real GPU run (AutoDL RTX 5090). Target: *competitive among open ≤10B models*,
> with a **statistically significant** base→+QLoRA gain and **policy-violation rate = 0**.

| Metric | Base | +QLoRA | 95% CI / sig | Source |
| --- | --- | --- | --- | --- |
| Service-desk success (held-out) | _TBD_ | _TBD_ | _TBD_ | `eval/zh_service_desk.py` |
| Unsafe-tool-selection rate ↓ | _TBD_ | _TBD_ | _TBD_ | learning signal → 0 |
| Policy-violation rate (reaches user) | **0** | **0** | — | gate-guaranteed |
| BFCL-V4 · AST accuracy | _TBD_ | _TBD_ | _TBD_ | `eval/run_bfcl.py` |
| τ²-bench retail · pass^1 / pass^4 | _TBD_ | _TBD_ | _TBD_ | `eval/run_tau2.py` |

## Failure analysis (template — fill from the run)
Categorize every miss, with examples and a fix:
1. **Wrong tool** (e.g., `modify_order` vs `create_ticket` on a shipped order) — *cause → fix*.
2. **Bad arguments** (schema-invalid / wrong order id) — caught by the executor's retry?
3. **Missed refusal** (model called a forbidden tool; gate caught it) — does QLoRA reduce it?
4. **Weak grounding** (KB answer cited the wrong doc) — retrieval vs generation?
5. **Over-refusal** (refused an allowed action) — precision/recall trade-off.

## Limitations & next steps
- Absolute scores won't beat closed frontier models; the claim is *open-≤10B competitiveness* +
  *zero policy violations*, not "beats GPT".
- The off-GPU retriever is a lexical stand-in; production uses bge-m3 + reranker.
- Next: scale SFT with licensed external corpora, a GRPO run on a larger-memory card, and a
  Qwen3.5 comparison — each only after the CORE numbers are in.

*Repo: agent/ rag/ eval/ finetune/ serving/ · 127 deterministic tests · GPU runbook in
`report/improvement_plan.md` and `eval/README.md`.*
