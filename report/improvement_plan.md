# PolicyArena — Plan to Reach Credible, SOTA-Class Results

> Goal: take the project from "complete scaffold, all metrics `TBD`" to **trustworthy,
> publishable results** on (1) standardized benchmarks and (2) a self-built Chinese
> service-desk domain. Author has a Blackwell GPU, so the on-GPU steps are runnable.
> This plan is reviewed and approved **before** any system code is changed.

## 0. What "SOTA" means here (honest framing)

A Qwen3-8B + LoRA system will **not** beat GPT-5 / Claude on absolute benchmark scores.
The defensible, still-impressive claims we will target:

- **Standardized track (comparable):** competitive **among open models ≤10B** on
  **BFCL-V4** (AST accuracy) and **τ²-bench** (retail/knowledge, pass^k), reported with
  **95% bootstrap CIs** and a **base vs +LoRA** delta that is **statistically significant**
  (paired bootstrap + Holm–Bonferroni). Position against the public-leaderboard tier of
  similar-size open models — never against closed frontier models.
- **Self-built track (research depth):** **policy-violation rate → 0** (a hard, verifiable
  constraint), tool accuracy + args-match with CIs, grounded/cited RAG answers, and a real
  **failure analysis**. This "small model + zero policy violations at a fraction of the cost"
  story is the differentiator for PhD/job applications.

A reviewer's #1 trust test is **data hygiene**. So the plan front-loads leakage prevention.

---

## Phase A — Leakage guard + train/eval data split  (off-GPU, FOUNDATION)

**Problem found in review:** `finetune/build_sft_data.py` trains on the *same* order ids
(`A1001`, `A1002`, `A1009`) and near-identical prompts (`"订单 A1001 我要退款"`,
`"请问运费是怎么计算的？"`) that also appear in `eval/tasks.py`. If the self-built domain
is ever scored on those prompts, results are inflated by leakage — the exact thing that
destroys credibility.

**Changes**
- Introduce **disjoint id pools** in `agent/tools/services.py`:
  - train pool: `A1xxx` (e.g. `A1001–A1099`)
  - eval pool: `E9xxx` (e.g. `E9001–E9099`) — never seen in training.
- `build_sft_data.py` draws **only** from the train pool.
- New held-out eval set draws **only** from the eval pool (Phase B).
- New `tests/test_leakage.py`:
  - `normalize(SFT prompts) ∩ normalize(eval prompts) == ∅`
  - train id pool ∩ eval id pool == ∅
  - (normalize = strip whitespace/punctuation, lowercase)

**Validation:** new test fails on today's data, passes after the split. `make check` green.

---

## Phase B — Real held-out Chinese service-desk benchmark + scorer  (off-GPU harness)

**Problem found in review:** there is no held-out zh benchmark — only 5 `SMOKE_TASKS`
(reused by the CI gate) and 7 SFT seed trajectories. Nothing to actually score quality on.

**Changes**
- New `eval/datasets/zh_service_desk_eval.jsonl` — hand-authored, **eval-pool ids only**,
  ~40–80 tasks covering:
  - happy path per tool (query / modify-unshipped / refund-in-window / ticket / KB)
  - **policy edges**: refund past 7-day window (must refuse), modify shipped order (must
    refuse), refund already-refunded, partial refund
  - **grounding**: KB questions with a gold citation doc id
  - **negatives**: out-of-scope request (no tool / clarify), ambiguous order id
- Task schema (one JSON per line):
  ```json
  {"id":"...", "prompt":"...", "expected_tool":"refund|...|null",
   "arg_constraints":{"order_id":"E9001"}, "gold_policy":"allow|deny",
   "gold_citations":["refund_faq"], "category":"happy|policy_edge|grounding|negative"}
  ```
- New `eval/zh_service_desk.py`: loader + scorer producing per-task records and aggregate
  **tool accuracy, args-match rate, policy-violation rate, citation grounding** (reuse
  `eval/metrics.py`, extend with args-match + grounding).
- `tests/test_zh_eval.py`: scorer runs on the deterministic `ScriptedLLMClient` (pipeline
  check only — not a quality number).

**Validation:** scorer runs end-to-end off-GPU on the mock; emits a populated `EvalSummary`.

---

## Phase C — Multi-step agent loop  (off-GPU)

**Problem found in review:** the graph is single-tool, single-turn
(`tool_select → tool_executor → policy_check → responder → END`). τ²-bench needs
multi-step agentic loops (call a tool, observe, decide again). Current architecture can't
run it properly.

**Changes**
- `agent/state.py`: add `steps: int`, `tool_history: list[...]`, `max_steps` handling.
- `agent/graph.py`: add a loop edge `policy_check → tool_select` that continues while the
  model proposes another tool and `steps < max_steps` and policy is OK; otherwise route to
  `responder`. Keep per-step policy enforcement (a violation stops the loop → refuse).
- `agent/nodes/tool_select.py`: feed accumulated tool results back into `messages` so the
  model conditions on observations.
- `configs/server.yaml` (or `model.yaml`): `agent.max_steps: 6`.
- Tests: a 2-step trajectory (query_order → then refund/refuse) terminates correctly; the
  reused-`thread_id` state-reset guarantee still holds; existing single-step tests pass.

**Validation:** new multi-step test green; full suite still green.

---

## Phase D — Real benchmark integration  (code off-GPU, run on GPU)

**Problem found in review:** `eval/run_tau2.run()` and `eval/run_bfcl.run()` are
`raise NotImplementedError`. No code actually drives the benchmarks. This is the #1 reason
`TBD` can never be filled.

**Changes**
- `eval/run_bfcl.py::run()`: implement a BFCL model handler that calls our SGLang
  OpenAI-compatible endpoint with our tool schemas, runs the gorilla
  berkeley-function-call-leaderboard, parses **AST accuracy**, and **records the exact
  BFCL-V4 commit/version**. Document exact `pip`/clone/run commands in `eval/README.md`.
- `eval/run_tau2.py::run()`: adapter exposing our multi-step agent (Phase C) as a τ²-bench
  solver against the served model; run retail + knowledge; collect per-task pass/fail for
  **pass^k** (`eval/passk.py`) and **bootstrap CIs** (`eval/bootstrap.py`).
- `eval/results.py` (new): writes a results JSON (`base` vs `+LoRA`) with CIs + paired
  tests, ready to paste into the report tables.
- Pin versions in `requirements/eval.txt` and record commit hashes in the report.

**Validation (off-GPU):** import + dry-run wiring covered by tests with a stub client;
`run()` raises a clear "needs served model + repo" error only when the deps are absent
(no more bare `NotImplementedError`). Real numbers produced on the GPU box.

---

## Phase E — Scale & diversify SFT data  (off-GPU)

**Problem found in review:** 7 templated trajectories won't move LoRA. Scaling code
("ToolACE/APIGen") is referenced but absent.

**Changes**
- Parametrize `build_sft_data.py` over the **train id pool**, multiple products,
  addresses, reasons, and **paraphrased** user phrasings → a few hundred–~1k samples.
- Add refusal/negative and **multi-step** trajectories (matching Phase C).
- Add an ingestion hook for **ToolACE / APIGen-MT / xLAM** subsets with a license-check
  stub and a **dedup-against-eval** filter (reuses the Phase A normalizer).
- Add an internal **train/val split** (val never used for the held-out benchmark).
- Keep the existing rule-filter (valid args + non-empty final answer).

**Validation:** builder emits N≫7 samples, all rule-valid, **zero overlap** with the eval
set (asserted by the Phase A test extended to the scaled output).

---

## Phase F — Rewrite results framing & docs  (off-GPU)

**Changes**
- `README.md` + `report/technical_report.md`: replace the implicit "100%/beat everything"
  expectation with the **§0 honest SOTA framing**; add a **comparison-tier** column
  (open ≤10B), a **data-hygiene / no-leakage** subsection, and make
  **policy-violation rate → 0** the headline self-built result.
- Results tables keep `TBD` until the GPU run; add CI + paired-test + pass^k columns.

---

## Phase G — Job/PhD case study  (off-GPU)

**Changes**
- `report/case_study.md` (1–2 pages): problem → approach → architecture diagram →
  results (wired to real numbers) → **failure analysis** (error taxonomy + examples +
  fixes) → limitations → what I'd do next. This is the artifact a recruiter/committee
  reads in 5 minutes.

---

## On-GPU run checklist (author runs on the Blackwell box)

1. `uv pip install -r requirements/{train,rag,eval}.txt` (cu128 torch first).
2. `python -m finetune.build_sft_data` (scaled set).
3. `python -m finetune.train_lora` → adapter in `outputs/adapters/...`.
4. Serve **base** and **+adapter** via `serving/sglang_server.sh`.
5. `python -m eval.zh_service_desk` (base vs +LoRA) → zh metrics.
6. `python -c "from eval.run_bfcl import run; run()"` → BFCL-V4 AST accuracy.
7. `python -c "from eval.run_tau2 import run; run('retail'); run('knowledge')"` → pass^k.
8. `python -m eval.results` → bootstrap CIs + paired bootstrap + Holm–Bonferroni.
9. Latency p50/p95 on the **exclusive** GPU.
10. Failure analysis → fill `report/*` tables (real numbers only).

---

## Suggested execution order & sequencing

Off-GPU (I implement now, in this order): **A → B → C → E → D → F → G**.
(A is the foundation; B/C unblock real scoring; E feeds training; D wires the external
benchmarks; F/G are the write-up.) Each phase: code + tests + `make check` green +
commit/push to `claude/compassionate-franklin-Masnj`. Then the GPU checklist produces the
real numbers, and we fill the tables together.

**Non-goals / risks:** no fabricated numbers; absolute scores won't beat closed models;
τ²-bench/BFCL versions pinned and recorded; all external datasets license-checked.
