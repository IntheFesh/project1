"""Hybrid (BM25 + dense) retrieval over the in-memory index with rank fusion."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from common.config import RetrievalConfig, load_retrieval_config
from rag.index import InMemoryIndex
from rag.text import tokenize


class RetrievedChunk(BaseModel):
    """A retrieved chunk paired with its (fused) relevance score."""

    doc_id: str
    chunk_id: str
    text: str
    score: float


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion of several ranked index lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def retrieve(
    query: str, index: InMemoryIndex, cfg: RetrievalConfig | None = None
) -> list[RetrievedChunk]:
    """Return the top chunks for ``query`` using dense / BM25 / hybrid (RRF)."""
    cfg = cfg or load_retrieval_config()
    rc = cfg.retrieval
    if len(index) == 0:
        return []

    query_vec = index.embedder.embed([query])[0]
    dense_scores = index.embeddings @ query_vec
    dense_rank = [int(i) for i in np.argsort(-dense_scores)[: rc.dense_top_k]]

    bm25_scores = index.bm25.scores(tokenize(query))
    bm25_rank = [int(i) for i in np.argsort(-bm25_scores)[: rc.bm25_top_k]]

    if rc.mode == "dense":
        fused = {idx: float(dense_scores[idx]) for idx in dense_rank}
    elif rc.mode == "bm25":
        fused = {idx: float(bm25_scores[idx]) for idx in bm25_rank}
    else:
        fused = _rrf([dense_rank, bm25_rank])

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: rc.final_top_k]
    out: list[RetrievedChunk] = []
    for idx, score in ranked:
        chunk = index.chunks[idx]
        out.append(
            RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
            )
        )
    return out
