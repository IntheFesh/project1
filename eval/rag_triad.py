"""RAG triad: groundedness, context relevance, answer relevance.

The production metric uses TruLens with an LLM judge (``requirements/eval.txt``, GPU/API).
The functions here are deterministic LEXICAL PROXIES for off-GPU pipeline checks — they
approximate the triad via token overlap and are not the TruLens scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.text import tokenize


def _coverage(target: str, sources: Sequence[str]) -> float:
    target_tokens = set(tokenize(target))
    if not target_tokens:
        return 1.0
    source_tokens: set[str] = set()
    for text in sources:
        source_tokens |= set(tokenize(text))
    return len(target_tokens & source_tokens) / len(target_tokens)


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens supported by the retrieved contexts."""
    return _coverage(answer, contexts)


def context_relevance(query: str, contexts: Sequence[str]) -> float:
    """Fraction of query tokens present in the retrieved contexts."""
    return _coverage(query, contexts)


def answer_relevance(query: str, answer: str) -> float:
    """Fraction of query tokens addressed by the answer."""
    return _coverage(query, [answer])


def evaluate(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Average the triad over records with keys ``query``, ``answer``, ``contexts``."""
    if not records:
        return {"groundedness": 0.0, "context_relevance": 0.0, "answer_relevance": 0.0}
    n = len(records)
    return {
        "groundedness": sum(groundedness(r["answer"], r["contexts"]) for r in records) / n,
        "context_relevance": sum(context_relevance(r["query"], r["contexts"]) for r in records) / n,
        "answer_relevance": sum(answer_relevance(r["query"], r["answer"]) for r in records) / n,
    }
