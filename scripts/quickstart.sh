#!/usr/bin/env bash
# One-command OFF-GPU quickstart (no Docker): install, verify, then launch the API.
# Uses the deterministic mock backend so it runs anywhere.
#
#   ./scripts/quickstart.sh            # API on :8000 (mock)
#   PORT=9000 ./scripts/quickstart.sh  # custom port
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> uv sync (light runtime + dev tools)"
uv sync

echo "==> ruff"
uv run ruff check .

echo "==> pytest"
uv run pytest -q

echo "==> deterministic eval gate"
uv run python -m eval.gate

export SERVING_BACKEND="${SERVING_BACKEND:-mock}"
export API_AUTH_TOKEN="${API_AUTH_TOKEN:-dev-token}"
PORT="${PORT:-8000}"
echo "==> launching API on :${PORT} (SERVING_BACKEND=${SERVING_BACKEND}, token=${API_AUTH_TOKEN})"
echo "    try: curl -s -X POST localhost:${PORT}/agent/query \\"
echo "           -H 'Authorization: Bearer ${API_AUTH_TOKEN}' -H 'Content-Type: application/json' \\"
echo "           -d '{\"message\":\"订单 A1009 我要退款\"}'"
exec uv run uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
