# PolicyArena — Claim Audit (Phase A, read-only)

**Audited state:** `claude/compassionate-franklin-Masnj`, fast-forwarded to `origin/main`
@ `f551655` (the branch was 5 commits behind; the GPU run lives on `main`). Method: read the
four `report/*` reports, `BLACKWELL_NOTES.md`, every `results/*.json`, `README.md`, configs,
requirements, and the F1–F7 code; cross-check each public claim against the actual artifacts.
Only this file was written in Phase A.

## 0. Reconciliation — resolved; red line holds

The "finished GPU run" artifacts the brief treats as ground truth **do exist** — they were on
`main`, not on the stale feature branch. After syncing, the headline numbers **match the saved
JSON exactly** (verified, not transcribed):

| Metric (n=32) | `results/zh_base_final_detailed.json` | `results/zh_lora_final_detailed.json` | report/case_study.md |
| --- | --- | --- | --- |
| success_rate | 0.531 | 0.969 | 53.1% → 96.9% ✓ |
| grounding_rate | 0.250 | 0.875 | 25.0% → 87.5% ✓ |
| unsafe_selection_rate | 0.286 | 0.714 | 28.6% → 71.4% ✓ |
| policy_violation_rate | 0.0 | 0.0 | 0% → 0% ✓ |
| tool_accuracy | 0.286 | 0.952 | (consistent) |
| args_match_rate | 0.333 | 0.889 | (consistent) |

`results/zh_paired_final.md` reproduces the paired-bootstrap CIs + Holm-adjusted p-values
(success p≈0.000 ✓, grounding p=0.005 ✓). **Every reported number is backed by a saved JSON.**

---

## 1. 🔴 MOST URGENT — the test suite is RED on `main`

A reviewer who clones and runs `make check` currently gets **failures**, and the deterministic
**eval gate FAILS** (`[eval-gate] FAIL tool_accuracy=0.000 < 0.7`). This makes the README's
"**127 tests green**" claim false *today*, and is a credibility landmine for a portfolio repo.

**9 failing tests** (all on the deterministic mock path; the real vLLM results are unaffected):
```
test_agent.py::test_query_order_scenario, ::test_refund_within_window_allowed,
  ::test_knowledge_query_is_grounded_and_cited, ::test_same_thread_does_not_leak_state_across_turns
test_api.py::test_query_refund_allowed
test_eval.py::test_smoke_slice_is_all_correct, ::test_runners_smoke_works_and_real_run_is_guarded
test_gate.py::test_gate_passes_on_smoke_slice, ::test_gate_cli_returns_zero_on_pass
```
**Root cause (single).** F1 changed the default to `max_steps=2` and F2/F4 added task-aware
`tool_choice`; the **fixed `eval/zh_service_desk.py` reads `executed_tools`**, but the *older*
`eval/harness.py` (used by the gate + smoke + these tests) still reads `state.selected_tool`,
which is now `None` after the loop's final "answer" step → predicted tool `None` → 0 accuracy.
**Fix (Phase D):** align `eval/harness.py` (and the mock-path tests/gate) with the new
multi-step `executed_tools` semantics. This touches **only the offline mock/test path** — it does
**not** change the reported vLLM numbers (those come from the fixed scorer + saved JSON).

> **Sequencing recommendation:** do this fix *before* Phase B, so the rewritten README can state a
> true "N tests green". Writing "tests green" over a red suite would itself violate the red line.

---

## 2. Claim inventory — current `main`

Legend: **FALSE** (contradicts the real run) · **ASPIRATIONAL** (described as done, never run) ·
**STALE-TBD** (placeholder superseded by a real number) · **TRUE**.

| # | Claim · location | Status | Correct statement |
| --- | --- | --- | --- |
| 1 | "Qwen3 · **SGLang** · LlamaIndex/**Milvus**" · README:4; `pyproject.toml`:4,9; one_pager:6 | **FALSE** | Served on **vLLM 0.22.1** (SGLang attempted, abandoned — `sgl-kernel` 0.3.21 has no sm120 build); RAG = **in-memory hashed-emb + BM25 + RRF + rerank** (Milvus/bge = production swap). |
| 2 | "τ²-bench, BFCL-V4, TruLens RAG-triad" headline · README:5,43,265–270 | **ASPIRATIONAL** | **Not run.** Mark as future work; present no number. |
| 3 | "live **SGLang** serving … `lmsysorg/sglang:blackwell` … flashinfer" · README:13,58,64,97,179,188,212–213 | **FALSE** | `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3` (per `BLACKWELL_NOTES.md`). |
| 4 | "**cu128** / CUDA 12.8+" · README:96,122,282; requirements/train.txt:3–7; rag.txt | **FALSE** | **cu130 / CUDA 13.0**: torch 2.11+cu130, vLLM 0.22.1, flashinfer 0.6.11.post2+cu130, bitsandbytes 0.49.2. |
| 5 | "**127 tests green** / 100+ tests" · README:8,84,139 | **FALSE → fixed** | Was red (9 failing, see §1); after the harness fix the suite is **green at 128 tests** — the count the README now states. |
| 6 | Results rows = `TBD` · README:264–270 | **STALE-TBD** | Fill **service-desk** rows from `results/*.json` (§0). Keep τ²/BFCL/TruLens/latency rows explicitly "not yet run". |
| 7 | "competitive among open ≤10B" as current · README:247 | **ASPIRATIONAL** | Reframe as future target (no leaderboard run). Current claim: *significant within-domain gain + zero policy violations*. |
| 8 | mermaid `SGLang / vLLM`, `Milvus / in-memory` · README:56–58 | **misleading** | Redraw: **vLLM** engine, **in-memory hybrid RAG**, loop `max_steps=2`. |
| 9 | unsafe_selection regression unmentioned in README | **OMISSION** | Carry over case_study's honest account (28.6→71.4%, p=0.072 n.s.; root cause overfit + stricter detector; gate keeps violations at 0). |
| 10 | one_pager.md / technical_report.md / resume_bullets.md | **TRUE (post-sync)** | Already updated with the real numbers + vLLM/cu130 (this row was flagged from the pre-sync stale commit; corrected after fast-forwarding to `main`). |

---

## 3. Already honest — do NOT regress these (the GPU work)

- `report/case_study.md` — real numbers, CIs, the U-shape ablation, the failure analysis, the
  three evaluation-protocol findings. **Excellent; the README should mirror it.**
- `report/resume_bullets.md`, `report/technical_report.md` — carry the real numbers.
- `BLACKWELL_NOTES.md` — the true stack + launch command + the SGLang-sm120 rationale.
- `results/*.json` + `zh_paired_final.md` — the backing evidence.
- `configs/lora.yaml` — **epochs=4, lr 2e-4, dropout 0.05, qlora, bs2×ga8** — matches the final run ✓.
- F1–F7 code present: `agent/graph.py` (`max_steps=2`), `agent/state.py`/`tool_select.py`
  (`tool_choice_override`), `eval/zh_service_desk.py` (`_tool_choice_for`, executed-tools unsafe
  detector), `finetune/train_lora.py` (`assistant_only_loss=True`, `max_length`).
- `.gitignore` already excludes `outputs/`, `*.log`, `*.bak`; no backups committed ✓.

---

## 4. Per-file action map (Phases B–E)

- **Phase D-pre (recommended first):** fix `eval/harness.py` + the 9 mock/gate tests → green suite.
- **Phase B (README):** rewrite per claims #1–#10; results-first top fold from §0; vLLM/cu130
  reproduce section from `BLACKWELL_NOTES.md`; honest limitations incl. the regression.
- **Phase C:** add `results/README.md`; make `one_pager.md` consistent; reconcile
  `report/CHANGES.md` + `improvement_plan.md` (the "scale to ~1k / run τ²/BFCL" items are
  partially done at 240 samples / still future).
- **Phase D:** purge cu128/sglang/milvus from `requirements/*.txt`, `pyproject.toml`, and stale
  docstrings (`train_lora.py` line 5 still says cu128; `serving/sglang_server.sh` → label "not
  used on Blackwell"); confirm true test count.
- **Phase E:** `CITATION.cff`, `CONTRIBUTING.md`, root `CHANGELOG.md`, badges,
  `docs/REPRODUCIBILITY.md`, PR/issue templates.

## 5. Open question for the owner

The 9 test failures are a real regression from the F1–F7 commit. **Recommend I fix the suite
green first (offline mock path only, numbers untouched), then rewrite the README.** Confirm, and
I'll proceed Phase by Phase.
