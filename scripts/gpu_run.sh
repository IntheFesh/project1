#!/usr/bin/env bash
# One-command GPU pipeline for the AutoDL RTX 5090 (Blackwell sm_120, ≤5 GPU-day budget).
# Orchestrates the steps documented in eval/README.md and report/improvement_plan.md so you
# don't waste paid GPU minutes wiring the environment by hand.
#
# Usage:
#   scripts/gpu_run.sh setup        # cu130 torch + train/rag/eval requirements
#   scripts/gpu_run.sh data         # build the SFT dataset (CPU; can run before power-on)
#   scripts/gpu_run.sh train        # QLoRA-SFT -> adapter
#   scripts/gpu_run.sh serve base   # launch SGLang for the base model (foreground)
#   scripts/gpu_run.sh serve lora   # launch SGLang for base + the trained adapter
#   scripts/gpu_run.sh eval-zh base # held-out zh service-desk scorer -> results/zh_<label>.json
#   scripts/gpu_run.sh eval-bfcl base
#   scripts/gpu_run.sh eval-tau2 base
#   scripts/gpu_run.sh results      # base-vs-LoRA CIs + paired tests -> results/headline.{json,md}
#
# Recommended order (power OFF the instance between long idle gaps to save budget):
#   setup -> data -> train -> (serve base &) -> eval-zh base -> eval-bfcl base -> eval-tau2 base
#         -> (serve lora &) -> eval-zh lora -> eval-bfcl lora -> eval-tau2 lora -> results
set -euo pipefail
cd "$(dirname "$0")/.."

# --- config (override via env) ----------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # AutoDL intl. network mirror
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
BASE_URL="${BASE_URL:-http://localhost:${SGLANG_PORT}/v1}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/adapters/qwen3-8b-policyarena}"
SFT_DATA="${SFT_DATA:-outputs/sft/zh_service_desk.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-results}"
# tau2-bench user-simulator on a cheap EXTERNAL API (keeps the 5090 for the model under test):
USER_MODEL="${USER_MODEL:-deepseek-chat}"
USER_BASE_URL="${USER_BASE_URL:-https://api.deepseek.com/v1}"
mkdir -p "${RESULTS_DIR}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }

wait_for_server() {
  log "waiting for SGLang at ${BASE_URL} ..."
  for _ in $(seq 1 120); do
    if curl -sf "${BASE_URL%/v1}/health" >/dev/null 2>&1 || curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "server is up."; return 0
    fi
    sleep 5
  done
  echo "server did not come up in time"; return 1
}

step_setup() {
  log "installing cu130 torch + heavy CUDA stacks (Blackwell)"
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  python -c "import torch; al=torch.cuda.get_arch_list(); print('arch_list:', al); assert any('sm_120' in a for a in al), 'need sm_120 (cu130) wheels'"
  uv pip install -r requirements/train.txt
  uv pip install -r requirements/rag.txt
  uv pip install -r requirements/eval.txt
}

step_data() {
  log "building SFT dataset -> ${SFT_DATA}"
  uv run python -m finetune.build_sft_data --out "${SFT_DATA}"
  uv run python -m finetune.train_lora --dry-run --data "${SFT_DATA}"
}

step_train() {
  log "QLoRA-SFT (configs/lora.yaml) -> ${ADAPTER_DIR}"
  uv run python -m finetune.train_lora --data "${SFT_DATA}"
}

serve() {
  local which="${1:-base}"
  need docker
  if [ "${which}" = "lora" ]; then
    log "serving base + adapter (${ADAPTER_DIR}) on :${SGLANG_PORT}"
    docker run --rm --gpus all --shm-size 16g --ipc host \
      -p "${SGLANG_PORT}:30000" \
      -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
      -v "$(pwd)/${ADAPTER_DIR}:/adapter" \
      -e HF_TOKEN="${HF_TOKEN:-}" lmsysorg/sglang:blackwell \
      python3 -m sglang.launch_server --model-path "${MODEL_ID}" \
        --lora-paths policyarena=/adapter \
        --host 0.0.0.0 --port 30000 --tool-call-parser qwen25 \
        --reasoning-parser qwen3 --attention-backend flashinfer --mem-fraction-static 0.85
  else
    log "serving base model on :${SGLANG_PORT}"
    MODEL_ID="${MODEL_ID}" SGLANG_PORT="${SGLANG_PORT}" bash serving/sglang_server.sh
  fi
}

eval_zh() {
  local label="${1:-base}"
  log "held-out zh service-desk eval (${label})"
  SERVING_BACKEND=sglang OPENAI_BASE_URL="${BASE_URL}" \
    uv run python -m eval.zh_service_desk | tee "${RESULTS_DIR}/zh_${label}.json"
  echo "saved ${RESULTS_DIR}/zh_${label}.json"
}

eval_bfcl() {
  local label="${1:-base}"
  log "BFCL-V4 (${label}) — see eval/README.md for the pinned commands"
  need bfcl
  bfcl generate --model "${MODEL_ID}" --test-category ast --backend openai --base-url "${BASE_URL}"
  bfcl evaluate --model "${MODEL_ID}" --test-category ast
  echo "parse the score summary with eval.run_bfcl.parse_bfcl_summary -> ${RESULTS_DIR}/bfcl_${label}.json"
}

eval_tau2() {
  local label="${1:-base}"
  log "tau2-bench retail (${label}); user-simulator on ${USER_MODEL}"
  need tau2
  uv run python - "$label" <<PY
import sys, json
from eval.run_tau2 import run
label = sys.argv[1]
res = run(domain="retail", agent_base_url="${BASE_URL}", agent_model="${MODEL_ID}",
          user_model="${USER_MODEL}", user_base_url="${USER_BASE_URL}",
          results_path="${RESULTS_DIR}/tau2_retail_raw.json", k_values=(1,2,4), num_trials=4)
open("${RESULTS_DIR}/tau2_%s.json" % label, "w").write(
    json.dumps({k: v.model_dump() for k, v in res.pass_hat.items()}, ensure_ascii=False, indent=2))
print("saved ${RESULTS_DIR}/tau2_%s.json" % label)
PY
}

step_results() {
  log "headline base-vs-LoRA aggregation (edit the success vectors per eval/README.md)"
  echo "Use eval.results.compare_metric on the per-task vectors in ${RESULTS_DIR}/ then"
  echo "eval.results.build_report + write_results -> ${RESULTS_DIR}/headline.{json,md}"
}

cmd="${1:-help}"; shift || true
case "${cmd}" in
  setup)      step_setup ;;
  data)       step_data ;;
  train)      step_train ;;
  serve)      serve "${1:-base}" ;;
  eval-zh)    eval_zh "${1:-base}" ;;
  eval-bfcl)  eval_bfcl "${1:-base}" ;;
  eval-tau2)  eval_tau2 "${1:-base}" ;;
  results)    step_results ;;
  *) sed -n '2,22p' "$0" ;;   # print the usage header
esac
