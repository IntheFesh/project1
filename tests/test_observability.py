"""Tracing degrades gracefully; the prompt registry is versioned."""

from common.config import Settings
from observability.prompts import get_prompt
from observability.tracing import InMemoryTracer, NullTracer, get_tracer


def test_null_tracer_is_noop() -> None:
    NullTracer().trace_turn(user="x", tool="refund")  # must not raise


def test_in_memory_tracer_records_turns() -> None:
    tracer = InMemoryTracer()
    tracer.trace_turn(user="hi", tool="refund", policy_ok=False)
    assert tracer.turns[0]["tool"] == "refund"
    assert tracer.turns[0]["policy_ok"] is False


def test_get_tracer_defaults_to_null_when_unconfigured() -> None:
    settings = Settings(langfuse_public_key="", langfuse_secret_key="")
    assert isinstance(get_tracer(settings), NullTracer)


def test_prompt_registry_returns_latest() -> None:
    prompt = get_prompt("system")
    assert prompt.version >= 1
    assert prompt.template
