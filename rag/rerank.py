"""Cross-encoder reranking with bge-reranker (Phase 3)."""

from __future__ import annotations

from rag.retrieve import RetrievedChunk


def rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int = 5
) -> list[RetrievedChunk]:
    """Rerank retrieved candidates and keep the top-n (implemented in Phase 3)."""
    raise NotImplementedError("rerank is implemented in Phase 3.")
