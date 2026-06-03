# PolicyArena — One-Pager

**What.** A policy-compliant tool-calling + RAG agent for an enterprise service desk. Policy
violations (e.g. refunding past the window) count as **FAILURES**, not style nits.

**Stack.** Qwen3-8B · SGLang (OpenAI-compatible, Blackwell image) · LangGraph (stateful +
checkpointing) · LlamaIndex/Milvus + bge hybrid retrieval · FastAPI + SSE · Langfuse · PEFT
LoRA/QLoRA · a statistics-first eval harness (τ²-bench, BFCL-V4, RAG-triad, bootstrap CIs, pass^k).

**Why it's different.**
- Rigorous, reproducible evaluation: 95% bootstrap CIs, combinatorial pass^k, paired
  bootstrap + Holm–Bonferroni, and a **deterministic CI eval gate**.
- A real **Chinese service-desk** domain with native tool-calling trajectories + policy compliance.
- Runs end-to-end **off-GPU** (deterministic mock backend) for instant demo/CI, and on a single
  **Blackwell** card (RTX PRO 6000 default / RTX 5090 fallback) for real serving + training.

**Run it.**
- Off-GPU: `./scripts/quickstart.sh` (sync → test → gate → API on :8000), or `make up` (Docker).
- GPU box: `./scripts/deploy.sh full` (SGLang + Milvus + Langfuse + API + UI).

**Status.** System implemented + verified off-GPU (100 tests). On-GPU runs produce the real
metrics — **headline results: TBD until then** (never fabricated).

**Hardware.** Single RTX PRO 6000 (96 GB, bf16 + GRPO) or RTX 5090 (32 GB, QLoRA + FP8).
Blackwell sm_120, CUDA 12.8+.
