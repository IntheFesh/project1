# Blackwell (RTX 5090 / sm120) Notes

## Working stack
- torch 2.11.0+cu130
- vllm 0.22.1
- flashinfer 0.6.11.post2 + flashinfer-jit-cache 0.6.11.post2+cu130

## Required env var
VLLM_USE_FLASHINFER_SAMPLER=0 — bypasses flashinfer's broken arch detection on sm120.

## Why not SGLang
sgl-kernel 0.3.21 ships only sm90/sm100 .so; no sm120 variant as of 2026/06.
vLLM 0.22.1 has mature sm120 support.

## Launch command
VLLM_USE_FLASHINFER_SAMPLER=0 uv run vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 30000 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.85 --max-model-len 8192
