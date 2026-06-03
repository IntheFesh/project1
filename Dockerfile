# PolicyArena agent API image (CPU; light runtime). The model is served separately by
# SGLang (Blackwell GPU). Build: docker build -t policyarena:api .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SERVING_BACKEND=mock

WORKDIR /app

# uv binary (pinned) for reproducible installs from the committed lockfile.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

# Dependency layer (cached unless pyproject/uv.lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code.
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD [".venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
