# PolicyArena Gradio UI image. Build context is the repo root:
#   docker build -t policyarena:ui -f frontend/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SERVING_BACKEND=mock \
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ui

COPY . .

EXPOSE 7860
CMD [".venv/bin/python", "frontend/app.py"]
