# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-06

First public, results-bearing release: a policy-compliant tool-calling + RAG agent with
real, reproducible evaluation numbers.

### Added
- **Final QLoRA-SFT run + results** on an AutoDL RTX 5090 (Blackwell sm_120, 32 GB),
  PyTorch 2.11+cu130 · vLLM 0.22.1 · trl 1.5.1. Held-out benchmark (n=32): success_rate
  **53.1% → 96.9%**, grounding_rate **25.0% → 87.5%** (paired bootstrap, Holm–Bonferroni),
  **0%** policy-violation rate across every run. Raw artifacts in `results/`.
- **Evaluation-protocol findings (F1–F7):** multi-step loop (`max_steps=2`), task-aware
  `tool_choice` routing, an executed-tools unsafe detector, and `assistant_only_loss=True`.
- **Held-out Chinese service-desk benchmark + scorer** (`eval/zh_service_desk.py`) and a
  statistics-first headline layer (`eval/results.py`: paired bootstrap + Holm + pass^k).
- **CI-enforced data-hygiene** (`tests/test_leakage.py`): disjoint A-/E-series order pools.
- Reports: `report/case_study.md`, `one_pager.md`, `technical_report.md`,
  `resume_bullets.md`; `results/README.md`; `docs/AUDIT.md`; `docs/REPRODUCIBILITY.md`;
  `CITATION.cff`; `CONTRIBUTING.md`; `BLACKWELL_NOTES.md`.

### Changed
- **Documentation now matches reality:** serving is **vLLM** (SGLang abandoned on Blackwell —
  no sm120 kernel), the CUDA stack is **cu130** (not cu128), and RAG is an in-memory hybrid
  index (Milvus/bge = production swap). The README leads with the real results and the honestly
  documented `unsafe_selection_rate` regression. τ²-bench / BFCL-V4 / TruLens are marked as
  future work, never as run.
- `AgentState.acted_tool` makes the eval harness, API trace, and scenario tests multi-step
  aware (the loop ends with a direct answer, so `selected_tool` is `None`).

### Fixed
- Green test suite (128 tests) and deterministic eval gate under the new `max_steps=2`
  default; the API now reports the executed tool instead of `null` on successful actions.

### Known issues
- `unsafe_selection_rate` regresses 28.6% → 71.4% (not statistically significant) under the
  240-sample SFT regime; operationally harmless (the gate keeps policy violations at 0).
  Resolution requires ≈10k SFT examples (planned). See `report/case_study.md`.
