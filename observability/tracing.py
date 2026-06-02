"""Optional Langfuse tracing. Degrades to a no-op when Langfuse is absent/unconfigured.

On the deployed box, set LANGFUSE_* in ``.env`` and ``uv sync --extra obs``; each agent
turn is then sent to Langfuse as a structured trace (input/output/metadata, latency). The
real OpenAI/Langfuse integration also captures token/cost automatically on the GPU box.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from common.config import Settings, get_settings


@runtime_checkable
class Tracer(Protocol):
    """Records one agent turn as a structured trace."""

    def trace_turn(self, **payload: Any) -> None:
        ...


class NullTracer:
    """Default tracer when Langfuse is unconfigured — records nothing."""

    def trace_turn(self, **payload: Any) -> None:
        return None


class InMemoryTracer:
    """Keeps turns in memory (for tests and local inspection)."""

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def trace_turn(self, **payload: Any) -> None:
        self.turns.append(payload)


class LangfuseTracer:
    """Langfuse-backed tracer (constructed only when keys + the SDK are present)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def trace_turn(self, **payload: Any) -> None:
        self._client.trace(
            name="agent_turn",
            input=payload.get("user"),
            output=payload.get("final_answer"),
            metadata=payload,
        )


def get_tracer(settings: Settings | None = None) -> Tracer:
    """Return a LangfuseTracer when configured, else a NullTracer."""
    s = settings or get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return NullTracer()
    try:
        from langfuse import Langfuse
    except ImportError:
        return NullTracer()
    client = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )
    return LangfuseTracer(client)
