# Reproducibility checklist

What makes the PolicyArena numbers reproducible and trustworthy. Every item is enforced in
code or pinned in config.

## Determinism
- [x] **Greedy decoding** — `temperature = 0` (`configs/server.yaml: sampling`).
- [x] **`--generation-config vllm`** at serve time, so Qwen3's packaged config does *not*
      override to `temperature = 0.6`.
- [x] **Fixed seeds** — `seed: 42` for training (`configs/lora.yaml`) and for every bootstrap
      (`eval/bootstrap.py`, `eval/results.py`, `eval/stats.py`).
- [x] **Deterministic offline path** — `ScriptedLLMClient` (rule-based, not a model) drives
      the CI eval gate to a reproducible pass/fail without a GPU.

## Data hygiene (no leakage)
- [x] **Disjoint pools** — SFT uses A-series order ids, the held-out benchmark uses E-series
      (`agent/tools/order_data.py`).
- [x] **CI guard** — `tests/test_leakage.py` fails the build if the pools or any prompts
      overlap.

## Evaluation protocol (must match the saved runs)
- [x] `max_steps = 2` and **task-aware `tool_choice`** (required for happy/grounding, auto for
      policy_edge/negative) — see `eval/zh_service_desk.py`.
- [x] Held-out set: `eval/datasets/zh_service_desk_eval.jsonl` (n = 32, four categories).
- [x] Comparison: **paired bootstrap (10 000 resamples) + Holm–Bonferroni** (`eval/results.py`).
- [x] **Saved artifacts** — per-task records + aggregates in `results/*.json`; no table cell
      is hand-edited (see `results/README.md`).

## Exact stack (the run that produced the numbers)
- Hardware: AutoDL **RTX 5090** (Blackwell **sm_120**, 32 GB).
- **PyTorch 2.11+cu130 · vLLM 0.22.1 · trl 1.5.1 · bitsandbytes 0.49.2 · transformers 5.10 ·
  flashinfer 0.6.11.post2+cu130**; uv-managed Python. See `BLACKWELL_NOTES.md`.
- Training: QLoRA r=16, α=32, dropout=0.05, all-linear, 4-bit NF4, bf16 compute,
  `assistant_only_loss=True`, 240 trajectories × 4 epochs (`configs/lora.yaml`).
- Serve env: `VLLM_USE_FLASHINFER_SAMPLER=0` (flashinfer mis-detects sm120).

## Regenerate
```bash
make train-lora                                   # build SFT data + QLoRA-SFT (GPU box)
SERVING_BACKEND=vllm OPENAI_BASE_URL=http://localhost:30000/v1 make eval-lora
```
