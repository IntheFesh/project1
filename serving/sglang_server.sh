#!/usr/bin/env bash
# Launch SGLang (OpenAI-compatible) for Qwen3-8B on a Blackwell GPU (sm_120).
# PRIMARY serving path. Prefer the prebuilt image to avoid JIT/PTX build breakage:
#   lmsysorg/sglang:blackwell  (AOT-compiled sm_120 kernels)
#
# Requires CUDA 12.8+. Do NOT use cu124/cu126 wheels on Blackwell — they only
# compile up to sm_90 and fail at runtime with:
#   "no kernel image is available for execution on the device".
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
