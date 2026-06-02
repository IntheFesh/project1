"""TruLens RAG triad: groundedness, context relevance, answer relevance. Phase 5."""

from __future__ import annotations

from typing import Any


def evaluate(records: list[dict[str, Any]]) -> dict[str, float]:
    """Score retrieval + generation records on the RAG triad (Phase 5)."""
    raise NotImplementedError("rag_triad is implemented in Phase 5.")
