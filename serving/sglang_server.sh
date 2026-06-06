#!/usr/bin/env bash
# REFERENCE ONLY — NOT used on the RTX 5090 (Blackwell sm_120). SGLang's sgl-kernel
# (0.3.21) ships no sm120 build, so this path was abandoned; the project serves with
# vLLM instead (serving/vllm_server.sh, BLACKWELL_NOTES.md). Kept for non-Blackwell use.
#
# Launch SGLang (OpenAI-compatible) for Qwen3-8B. Prefer the prebuilt image:
#   lmsysorg/sglang:blackwell
set -euo pipefail

MODEL_PATH="${MODEL_ID:-Qwen/Qwen3-8B}"
PORT="${SGLANG_PORT:-30000}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

docker run --rm --gpus all \
  --shm-size 16g --ipc host \
  -p "${PORT}:30000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  lmsysorg/sglang:blackwell \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port 30000 \
    --tool-call-parser qwen25 \
    --reasoning-parser qwen3 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.85
