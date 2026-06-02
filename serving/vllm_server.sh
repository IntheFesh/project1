#!/usr/bin/env bash
# Alternative serving path: vLLM (OpenAI-compatible). Keep CUDA 12.8+ on Blackwell.
# SGLang is the PRIMARY path (see sglang_server.sh).
set -euo pipefail

MODEL_PATH="${MODEL_ID:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 --port "${PORT}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
