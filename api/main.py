"""FastAPI app exposing the PolicyArena agent (/agent/query, /agent/stream, /health).

Auth is a bearer token (API_AUTH_TOKEN); rate limiting is per-client via SlowAPI. The LLM
backend is resolved from settings (SERVING_BACKEND); set it to ``mock`` to run off-GPU.
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse

from agent.graph import build_graph
from agent.state import AgentState
from agent.tools.services import ServiceDesk
from api.auth import verify_static_token
from api.ratelimit import build_limiter
from common.config import get_settings
from rag.pipeline import build_default_kb_search
from serving.client import LLMClient, get_client


class QueryRequest(BaseModel):
    """Request body for an agent turn."""

    message: str
    thread_id: str = "default"


class QueryResponse(BaseModel):
    """Structured agent result (final answer + trace fields)."""

    final_answer: str | None
    plan: str | None
    tool: str | None
    tool_result: dict[str, Any] | None
    violations: list[dict[str, str]]
    citations: list[dict[str, Any]]


def require_auth(authorization: str = Header(default="")) -> None:
    """Reject the request unless a valid bearer token is supplied."""
    token = authorization.removeprefix("Bearer ").strip()
    if not verify_static_token(token):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def get_llm_client() -> LLMClient:
    """Dependency: resolve the LLM client from settings (overridable in tests)."""
    return get_client()


def _to_response(state: AgentState) -> QueryResponse:
    return QueryResponse(
        final_answer=state.final_answer,
        plan=state.plan,
        tool=state.selected_tool.name if state.selected_tool else None,
        tool_result=state.tool_result,
        violations=[v.model_dump() for v in state.violations],
        citations=[c.model_dump() for c in state.citations],
    )


def create_app() -> FastAPI:
    """Application factory."""
    load_dotenv()  # populate os.environ from .env (no-op if absent)
    settings = get_settings()

    app = FastAPI(title="PolicyArena API", version="0.1.0")
    limiter = build_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.state.services = ServiceDesk(kb_search=build_default_kb_search())
    app.state.checkpointer = MemorySaver()

    def _run(message: str, thread_id: str, client: LLMClient) -> AgentState:
        graph = build_graph(client, app.state.services, checkpointer=app.state.checkpointer)
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result if isinstance(result, AgentState) else AgentState.model_validate(result)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/agent/query",
        response_model=QueryResponse,
        dependencies=[Depends(require_auth)],
    )
    @limiter.limit(f"{settings.rate_limit_per_minute}/minute")
    def agent_query(
        request: Request,
        body: QueryRequest,
        client: LLMClient = Depends(get_llm_client),
    ) -> QueryResponse:
        return _to_response(_run(body.message, body.thread_id, client))

    @app.post("/agent/stream", dependencies=[Depends(require_auth)])
    async def agent_stream(
        request: Request,
        body: QueryRequest,
        client: LLMClient = Depends(get_llm_client),
    ) -> EventSourceResponse:
        state = _run(body.message, body.thread_id, client)

        async def events() -> Any:
            yield {"event": "plan", "data": state.plan or ""}
            if state.selected_tool is not None:
                yield {
                    "event": "tool",
                    "data": json.dumps(
                        {"name": state.selected_tool.name, "arguments": state.selected_tool.arguments},
                        ensure_ascii=False,
                    ),
                }
            if state.tool_result is not None:
                yield {"event": "tool_result", "data": json.dumps(state.tool_result, ensure_ascii=False)}
            if state.violations:
                yield {
                    "event": "policy",
                    "data": json.dumps([v.model_dump() for v in state.violations], ensure_ascii=False),
                }
            yield {"event": "final", "data": state.final_answer or ""}

        return EventSourceResponse(events())

    return app


app = create_app()
