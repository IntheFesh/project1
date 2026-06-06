# Evaluation runbook

Reproducible, statistics-first evaluation. **No fabricated numbers** — every headline
figure comes from a real run and carries a 95% bootstrap CI; multi-metric comparisons are
Holm–Bonferroni corrected (`eval/results.py`). Off-GPU `smoke()` paths validate the
*pipeline*, not quality.

## What runs where

| Track | Module | Off-GPU | On GPU (AutoDL RTX 5090) |
| --- | --- | --- | --- |
| Self-built zh service-desk | `eval/zh_service_desk.py` | scorer pipeline check (mock) | base vs +LoRA, real model |
| BFCL-V4 (AST accuracy) | `eval/run_bfcl.py` | `smoke()` + parser tests | `run()` against served model |
| τ²-bench (pass^k) | `eval/run_tau2.py` | `smoke()` + parser tests | `run()`; user-sim on external API |
| Headline aggregation | `eval/results.py` | fully unit-tested | consumes real run outputs |

## 0. Serve the model (5090)

```bash
export HF_ENDPOINT=https://hf-mirror.com          # AutoDL: HF mirror
bash serving/sglang_server.sh                      # base Qwen3-8B on :30000
# +LoRA: serve with the adapter (see serving notes), or merge then serve.
```

## 1. Self-built Chinese service-desk (P0)

```bash
SERVING_BACKEND=sglang OPENAI_BASE_URL=http://localhost:30000/v1 \
  python -m eval.zh_service_desk            # prints success/tool/grounding + policy metrics
```
Run once against the **base** model and once against **+LoRA**; feed the per-task
`success` vectors into `eval.results.compare_metric` for the CI + paired test. Headline:
`policy_violation_rate == 0` (gate-guaranteed) and `unsafe_selection_rate` dropping
base → +LoRA (the model learned the policy).

## 2. BFCL-V4 (P0)

Pin the version and **record the commit** in the report.

```bash
pip install bfcl-eval                       # or: git clone gorilla; pin the BFCL-V4 commit
bfcl generate --model Qwen/Qwen3-8B --test-category ast \
  --backend openai --base-url http://localhost:30000/v1
bfcl evaluate --model Qwen/Qwen3-8B --test-category ast
python -c "from eval.run_bfcl import parse_bfcl_summary; \
  import pathlib,sys; print(parse_bfcl_summary(pathlib.Path(sys.argv[1]).read_text()))" \
  <path-to-score-summary.json>
```
Record: BFCL commit hash, `--test-category`, overall + per-category AST accuracy.

## 3. τ²-bench retail (P1 headline) / knowledge (P2)

Agent = served 5090 model (under test); user-simulator = **external API** (cheap, frees
VRAM). Export your key first (e.g. `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`).

```bash
pip install tau2-bench                       # or clone sierra-research/tau2-bench; pin commit
python - <<'PY'
from eval.run_tau2 import run
res = run(domain="retail",
          agent_base_url="http://localhost:30000/v1", agent_model="Qwen/Qwen3-8B",
          user_model="deepseek-chat", user_base_url="https://api.deepseek.com/v1",
          results_path="results/tau2_retail.json", k_values=(1,2,4), num_trials=4)
print({k: v.model_dump() for k, v in res.pass_hat.items()})
PY
```
Record: τ²-bench commit, domains, `num_trials`, pass^1/2/4 with CIs.

## 4. Headline tables (base vs +LoRA)

```bash
python - <<'PY'
from eval.results import compare_metric, build_report, write_results
# success vectors aligned per task from step 1 (and BFCL/tau2 per-task pass/fail):
cmps = [compare_metric("zh_success", base=BASE, lora=LORA)]   # add tool_accuracy, etc.
report = build_report("policyarena base vs +LoRA", cmps)
write_results(report, "results/headline.json")               # + headline.md table
print(report.to_markdown())
PY
```

## 5. Latency

`p50/p95/p99` on an **exclusive** (non-time-sliced) GPU only; record alongside the
serving config (bf16 vs fp8). Power the AutoDL instance **off** between runs to save budget.

## Version pinning

Record exact commits/versions in `report/technical_report.md` §8:
BFCL-V4 (`bfcl-eval==<ver>` or commit), τ²-bench commit, Qwen3-8B revision, torch cu128.
