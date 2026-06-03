#!/usr/bin/env bash
# One-command Docker deploy.
#   ./scripts/deploy.sh            # app-only, off-GPU (API :8000 + UI :7860, mock backend)
#   ./scripts/deploy.sh full       # FULL stack on a Blackwell GPU box (sglang + milvus + langfuse)
#   ./scripts/deploy.sh down       # stop everything
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-app}"
case "$MODE" in
  full)
    [ -f .env ] || cp .env.example .env
    echo "==> FULL stack (needs Blackwell GPU + NVIDIA Container Toolkit)"
    exec docker compose up --build
    ;;
  app|"")
    echo "==> app-only stack (off-GPU, SERVING_BACKEND=mock)"
    exec docker compose -f docker-compose.app.yml up --build
    ;;
  down)
    docker compose -f docker-compose.app.yml down --remove-orphans || true
    docker compose down --remove-orphans || true
    ;;
  *)
    echo "usage: $0 [app|full|down]" >&2
    exit 2
    ;;
esac
