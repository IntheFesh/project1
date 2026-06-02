"""Hybrid (BM25 + dense) retrieval over the Milvus collection (Phase 3)."""

from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A retrieved chunk paired with its fused relevance score."""

    doc_id: str
    chunk_id: str
    text: str
    score: float


def retrieve(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    """Hybrid-retrieve the top-k chunks for ``query`` (implemented in Phase 3)."""
    raise NotImplementedError("retrieve is implemented in Phase 3.")
