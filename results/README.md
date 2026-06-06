# Evaluation results

Raw, per-task evaluation artifacts. **Every number in `README.md`, `report/case_study.md`,
`report/one_pager.md`, and `report/technical_report.md` is reproduced from these files — no table
cell is hand-edited.** All from the final run on an AutoDL RTX 5090 (Blackwell sm_120, 32 GB),
PyTorch 2.11+cu130 · vLLM 0.22.1 · trl 1.5.1.

## Headline artifacts (the reported numbers)

| File | What it is |
| --- | --- |
| `zh_base_final_detailed.json` | Base Qwen3-8B on the 32-task held-out set, production protocol (`max_steps=2`, task-aware `tool_choice`), per-task records + aggregate rates. |
| `zh_lora_final_detailed.json` | The final QLoRA-SFT adapter (240 samples × 4 epochs) on the same set, same protocol. |
| `zh_paired_final.json` / `.md` | Base-vs-+LoRA paired bootstrap (10 000 resamples) + Holm–Bonferroni: Δ, 95 % CIs, adjusted p-values. The `.md` is the table pasted into the reports. |

Aggregate values in the headline files (verifiable with `jq . <file>`):

| | base | +LoRA |
| --- | ---: | ---: |
| success_rate | 0.531 | 0.969 |
| grounding_rate | 0.250 | 0.875 |
| tool_accuracy | 0.286 | 0.952 |
| args_match_rate | 0.333 | 0.889 |
| negative_handling_rate | 1.000 | 1.000 |
| unsafe_selection_rate | 0.286 | 0.714 |
| policy_violation_rate | 0.000 | 0.000 |

## Ablation artifacts (the U-shape / protocol findings)

| File | Configuration |
| --- | --- |
| `zh_base.json`, `zh_base_detailed.json` | Base under earlier (less-strict) protocol. |
| `zh_lora_detailed.json` | Early LoRA run. |
| `zh_lora_72ep_detailed.json` | 72-step over-trained collapse (U-shape bottom). |
| `zh_lora_4ep_balanced_detailed.json`, `zh_lora_4ep_stage2_detailed.json` | Intermediate 4-epoch runs. |
| `zh_lora_regularized_detailed.json`, `zh_paired_regularized.{json,md}` | Regularisation experiment (epochs↓, lr↓, dropout↑) — degraded both success and safety. |
| `zh_lora_stage2_toolrequired_detailed.json` | `tool_choice="required"` ablation (grounding recovers, negative-handling collapses). |

> Earlier-protocol rows are **not** comparable to the headline `*_final_*` files; they used
> `tool_choice="auto"`, `max_steps=1`, and the narrower unsafe detector. See
> `report/case_study.md` § "Evaluation-protocol findings".

## Regenerate

On the served model (see the README "Reproduce" section):

```bash
SERVING_BACKEND=vllm OPENAI_BASE_URL=http://localhost:30000/v1 \
  uv run python -m eval.zh_service_desk      # -> zh_<label>_detailed.json
uv run python -m eval.results                # paired bootstrap + Holm -> zh_paired_final.{json,md}
```

Scorer: `eval/zh_service_desk.py`; statistics: `eval/results.py`, `eval/stats.py`,
`eval/bootstrap.py` (all unit-tested off-GPU).
