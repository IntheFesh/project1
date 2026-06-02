"""Reranking. A lexical-overlap reranker for offline dev + the bge-reranker for prod."""

from __future__ import annotations

from rag.retrieve import RetrievedChunk
from rag.text import tokenize


def _overlap(query_tokens: set[str], text: str) -> float:
    doc = set(tokenize(text))
    if not query_tokens or not doc:
        return 0.0
    return len(query_tokens & doc) / len(query_tokens)


def rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int = 5
) -> list[RetrievedChunk]:
    """Rerank candidates by query/chunk token overlap and keep the top-n.

    Offline dev stand-in for a cross-encoder. Production uses bge-reranker-v2
    (``requirements/rag.txt``) — swap this function's body to call it on the GPU box.
    """
    query_tokens = set(tokenize(query))
    rescored = [c.model_copy(update={"score": _overlap(query_tokens, c.text)}) for c in candidates]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_n]
