# PolicyArena — Technical Report

> Skeleton. Sections are filled as phases complete. **No fabricated numbers**: every metric
> here is copied from a real run's output and reported with a 95% bootstrap CI.

## 1. Problem & scope
Policy-compliant tool-calling + RAG agent. An answer that violates a written policy is a FAILURE.
Two domains: τ²-bench (retail/knowledge) + a self-built Chinese enterprise service desk.

## 2. System design
- Serving: SGLang (Qwen3-8B, OpenAI-compatible) — _details Phase 1_.
- Agent: LangGraph (planner → tool_select → tool_executor → policy_check → responder) — _Phase 2_.
- RAG: LlamaIndex + Milvus + bge hybrid + reranker — _Phase 3_.
- Observability: Langfuse — _Phase 4_.

## 3. Model choice — stability vs novelty
Why **Qwen3-8B** is CORE rather than Qwen3.5-9B (parser stability, tooling maturity, fit on a
single 5090). Qwen3.5-9B is a STRETCH comparison, gated behind a smoke test. _To be written._

## 4. Hardware & reproducibility
Blackwell sm_120, CUDA 12.8+; QLoRA (5090) / LoRA-bf16 (PRO 6000). Seeds, configs, commit hashes.

## 5. Evaluation
- τ²-bench (retail/knowledge), pass^k via E[C(c,k)/C(n,k)].
- BFCL-V4 AST accuracy (record exact V4 version).
- RAG triad (TruLens).
- Bootstrap 95% CIs (≥10k resamples); paired comparison + Holm–Bonferroni.

| Metric | Base | + LoRA-SFT | 95% CI | Notes |
| --- | --- | --- | --- | --- |
| (filled from real runs) | TBD | TBD | TBD | |

## 6. Fine-tuning
QLoRA-SFT data mix + licenses; before/after schema-valid rate and tool accuracy. _Phase 7._

## 7. Failures & limitations
Honest list of what didn't work. _To be written._

## 8. Appendix
Configs, exact commands, dataset subset sizes, run logs.
