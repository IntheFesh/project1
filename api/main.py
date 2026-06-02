"""FastAPI app exposing the PolicyArena agent (/agent/query, /health) with SSE.

The /agent/query endpoint is wired to the LangGraph agent in Phase 2; here we expose a
health check so the service is importable and runnable during scaffolding.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(title="PolicyArena API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
