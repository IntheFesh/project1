#!/usr/bin/env bash
# PRIMARY serving path on Blackwell (sm120): vLLM (OpenAI-compatible). Requires the cu130
# stack (torch 2.11+cu130, vLLM 0.22.1); see BLACKWELL_NOTES.md. SGLang is NOT used on
# Blackwell (no sm120 kernel in sgl-kernel).
set -euo pipefail

MODEL_PATH="${MODEL_ID:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-30000}"

# VLLM_USE_FLASHINFER_SAMPLER=0 bypasses flashinfer's sm120 arch mis-detection;
# --generation-config vllm stops Qwen3's packaged config forcing temperature=0.6.
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 --port "${PORT}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --generation-config vllm
