# PolicyArena task shortcuts. `make help` lists targets.
# Uses .RECIPEPREFIX = > so recipes don't depend on literal tabs.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help
.PHONY: help setup lint test gate check demo ui sft dry-run up up-full down clean

help:  ## list targets
>@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create .venv and install deps
>uv sync

lint:  ## ruff
>uv run ruff check .

test:  ## pytest
>uv run pytest -q

gate:  ## deterministic eval gate
>uv run python -m eval.gate

check: lint test gate  ## lint + test + gate

demo:  ## run the API off-GPU (mock backend) on :8000
>SERVING_BACKEND=mock API_AUTH_TOKEN=dev-token uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:  ## run the Gradio UI off-GPU (mock backend) on :7860
>uv sync --extra ui && SERVING_BACKEND=mock uv run python frontend/app.py

sft:  ## build the Chinese SFT dataset
>uv run python -m finetune.build_sft_data

dry-run:  ## validate LoRA config + data without a GPU
>uv run python -m finetune.train_lora --dry-run

up:  ## docker: app-only stack (off-GPU)
>docker compose -f docker-compose.app.yml up --build

up-full:  ## docker: full stack (Blackwell GPU box)
>docker compose up --build

down:  ## docker: stop everything
>docker compose -f docker-compose.app.yml down --remove-orphans || true
>docker compose down --remove-orphans || true

clean:  ## remove venv + caches + generated outputs
>rm -rf .venv .pytest_cache .ruff_cache outputs
