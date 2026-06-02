# PolicyArena — One-Pager

**What:** a policy-compliant tool-calling + RAG agent. Policy violations count as FAILURES.

**Stack:** Qwen3-8B · SGLang · LangGraph · LlamaIndex+Milvus · FastAPI · Langfuse · QLoRA · a
statistics-first eval harness (τ²-bench, BFCL-V4, RAG-triad, bootstrap CIs, pass^k).

**Why it's different:** rigorous, reproducible evaluation (95% bootstrap CIs, combinatorial
pass^k, Holm–Bonferroni) + a real Chinese service-desk domain with native tool-calling
trajectories + a deterministic CI eval gate.

**Headline results:** _TBD — filled only from real runs._

**Hardware:** single RTX 5090 (QLoRA / FP8) or RTX PRO 6000 (bf16 / GRPO). Blackwell, CUDA 12.8+.
