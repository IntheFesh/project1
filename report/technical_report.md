# PolicyArena — Technical Report

> **No fabricated numbers.** Every metric is `TBD` until copied from a real run, and headline
> numbers are reported with 95% bootstrap CIs. This report documents the system as built and
> the methodology for the numbers that the GPU box will produce.

## 1. Problem & scope
A policy-compliant tool-calling + RAG agent for an enterprise service desk. An answer that
violates a written policy (e.g. refunding past the 7-day window, modifying a shipped order) is
scored as a **FAILURE**, not a stylistic issue. Two domains: τ²-bench (retail/knowledge) +
BFCL-V4 for standardized eval, and a self-built **Chinese 企业服务台** domain with five tools
(`query_order`, `modify_order`, `refund`, `create_ticket`, `search_kb`), policy docs, and a FAQ KB.

## 2. System architecture
- **Serving** (`serving/`): one OpenAI-compatible client (`OpenAICompatibleClient`) works
  against SGLang (primary), vLLM, or Ollama by `base_url`; `ScriptedLLMClient` is a
  deterministic offline stand-in (not a model) for tests/demos.
- **Agent** (`agent/`): a LangGraph state machine over a pydantic `AgentState` with
  checkpointing — `planner → tool_select → tool_executor → policy_check → responder`, with a
  conditional skip to `responder` when the model answers directly. `tool_executor`
  JSON-schema-validates arguments and retries once on invalid. Transient per-turn fields are
  reset by `planner` so a reused `thread_id` cannot leak state across turns.
- **Tools** (`agent/tools/`): pydantic argument schemas double as the OpenAI `tools=[...]`
  spec; a registry validates + dispatches to an in-memory `ServiceDesk`.
- **Policy** (`agent/policies/`): typed, unit-tested rules (refund window, modify-after-ship)
  plus a Chinese policy doc; any violation marks the turn a FAILURE and the responder refuses.
- **RAG** (`rag/`): chunk → index → hybrid retrieve (dense + Okapi BM25, reciprocal-rank
  fusion) → rerank → cited results, exposed as `search_kb`. Offline path uses a deterministic
  `HashEmbedder` + in-memory index; production uses bge-m3 + bge-reranker + Milvus.
- **API** (`api/`): FastAPI `/agent/query`, SSE `/agent/stream`, `/health`; bearer auth +
  SlowAPI rate limiting; per-turn latency + structured trace.
- **Observability** (`observability/`): Langfuse tracing that degrades to a no-op when
  unconfigured; versioned prompt registry.

## 3. Model & serving choice — stability vs novelty
CORE model is **Qwen/Qwen3-8B** (no `-Instruct` suffix; that ID does not exist). It is chosen
over Qwen3.5-9B for **tooling maturity and parser stability**: SGLang/vLLM expose a tested
`qwen25` tool-call parser and `qwen3` reasoning parser, and `tool_choice:"required"` works via
xgrammar. Qwen3.5-9B is a **STRETCH** comparison, gated behind a smoke test confirming (a)
LoRA-SFT converges and (b) its tool parser emits valid structured calls — only then is it worth
the risk. The tool-selection/policy path defaults to **non-thinking** for clean structured
output; if thinking is enabled, `--reasoning-parser qwen3` separates the trace from the answer.

## 4. Hardware & reproducibility
Target: **RTX 5090 (32 GB)** default — the actual box, rented on **AutoDL** with a ≤5 GPU-day
budget (QLoRA 4-bit SFT, bf16 8B serving ≈16 GB, FP8 throughput ablation; GRPO out of scope at
32 GB); **RTX PRO 6000 (96 GB)** is the larger-memory alternative (bf16 LoRA). Both are
Blackwell **sm_120 → CUDA 12.8+** (cu128 wheels; cu124/cu126 fail at runtime). Serving via
`lmsysorg/sglang:blackwell` with `--attention-backend flashinfer`; on AutoDL set
`HF_ENDPOINT=https://hf-mirror.com` and run the τ²-bench user-simulator on an external API so
the 5090 only serves the model under test. All knobs in `configs/*.yaml`; seeds fixed (42); the
light runtime is locked in `uv.lock`; heavy CUDA stacks pinned in `requirements/*.txt`.

**Data hygiene (no leakage).** SFT draws from a train order pool (A-series); the held-out
benchmark uses a disjoint eval pool (E-series). `tests/test_leakage.py` fails CI if the pools or
any prompts overlap, so accuracy cannot be inflated by training on the evaluation set.

## 5. Data
Chinese service-desk trajectories are generated as **real tool-calling traces** in Qwen3
chat/tool format (`finetune/build_sft_data.py`): query, allowed/denied refund (the denied case
is taught as a policy-compliant refusal), modify allowed/refused, ticket, and a cited knowledge
answer — then rule-filtered (valid tool args + non-empty final answer). On the GPU box this seed
set is scaled with ToolACE (arXiv 2409.00920) and APIGen-MT / xLAM subsets; **each dataset's
license is checked and recorded before use** (some are research-only).

## 6. Training
QLoRA-SFT via PEFT + TRL (`finetune/train_lora.py`): r=16, α=32, all-linear targets, 4-bit NF4
base + bf16 compute (5090 default; bf16 LoRA on a PRO 6000), adapter saved only, effective batch
16 (bs=2 × grad-accum 8). `--dry-run` validates config + data off-GPU. The SFT seed set is
generated over the train pool (`finetune/build_sft_data.py`) and mixed with licensed external
corpora via `ingest_external` (license-checked + deduped against the eval prompts). GRPO
(`finetune/train_grpo.py`) is **out of scope on the 5090** (won't fit 32 GB), but the
**verifiable reward** (policy violation ⇒ 0; else `0.6·tool_correct + 0.4·args_valid`) is
implemented and tested for a future PRO 6000 run.

## 7. Evaluation methodology
- **τ²-bench** (retail/knowledge) and **BFCL-V4** (AST accuracy; exact V4 version recorded).
- **pass^k** via the unbiased combinatorial estimator E[C(c,k)/C(n,k)] (`eval/passk.py`).
- **Uncertainty**: ≥10k-resample bootstrap CIs (`eval/bootstrap.py`); paired bootstrap +
  Holm–Bonferroni for multi-metric comparisons (`eval/stats.py`).
- **RAG triad** via TruLens (groundedness / context- / answer-relevance); off-GPU uses
  deterministic lexical proxies for pipeline checks only.
- **CI gate** (`eval/gate.py`): deterministic (fixed slice, greedy, temperature=0) so the same
  input yields the same pass/fail decision; it does **not** gate on noisy point estimates.
- **Latency**: p50/p95 measured only on an **exclusive (non-time-sliced) GPU**.

## 8. Results
SOTA target (honest): *competitive among open ≤10B models*, with **+QLoRA beating base by a
statistically significant margin** (paired bootstrap + Holm–Bonferroni, `eval/results.py`), and
**policy-violation rate (reaching the user) = 0** by construction. Numbers are `TBD` until the
real GPU run; record the BFCL-V4 and τ²-bench commit hashes here.

| Metric | Base | + QLoRA-SFT | 95% CI | Notes |
| --- | --- | --- | --- | --- |
| τ²-bench retail · pass^1 | TBD | TBD | TBD | |
| τ²-bench retail · pass^4 | TBD | TBD | TBD | |
| BFCL-V4 · AST accuracy | TBD | TBD | TBD | record V4 commit |
| service-desk · success rate | TBD | TBD | TBD | held-out E-pool |
| service-desk · unsafe-selection rate | TBD | TBD | TBD | learning signal → 0 |
| service-desk · policy-violation rate (reaches user) | 0 | 0 | — | gate-guaranteed |
| RAG · groundedness | TBD | TBD | TBD | TruLens |
| serving · p50/p95 latency | TBD | TBD | TBD | exclusive GPU |

## 9. What is verified vs pending
- **Verified off-GPU** (127 tests, deterministic): agent scenarios incl. policy refusals and
  the **multi-step loop** (termination, violation-stops-loop, step cap), hybrid RAG grounding +
  citations, API auth/SSE, the **held-out zh service-desk scorer** (incl. the policy-violation =
  0 invariant), **leakage guards** (disjoint pools + prompts), eval statistics + **headline
  aggregation** (CIs, paired tests, pass^k) + **benchmark output parsers**, the deterministic
  gate, SFT data building + external-ingest license/dedup contract, QLoRA dry-run, and a
  containerized app-only deploy.
- **Pending on the GPU box**: live SGLang serving + tool-call check, real τ²-bench/BFCL numbers
  + CIs, the actual QLoRA training run, and live Langfuse traces.

## 10. Failures & limitations
The off-GPU embedder/reranker are lexical stand-ins (no semantics) — real retrieval quality
needs bge on the GPU box. Multi-step tool loops are single-tool-per-turn in the demo. All
quality numbers are pending real runs and are intentionally `TBD`.

## 11. Roadmap / STRETCH
LiteLLM gateway, GRPO training run, Next.js frontend, Prometheus+Grafana, K8s/Helm, and the
Qwen3.5-9B comparison — each started only after CORE works and with the owner's go-ahead.
