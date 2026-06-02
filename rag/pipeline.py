"""RAG pipeline: ingest -> index -> hybrid retrieve -> rerank, returning cited results.

Exposed to the agent as the ``search_kb`` backend (``build_default_kb_search``), so a
knowledge query produces a grounded, cited answer and the retrieval hit shows in the trace.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.config import RetrievalConfig, load_retrieval_config
from rag.embeddings import Embedder, HashEmbedder
from rag.index import InMemoryIndex, build_index
from rag.ingest import ingest_docs
from rag.rerank import rerank
from rag.retrieve import retrieve

SAMPLE_KB_DIR = Path(__file__).parent / "sample_kb"


class RagPipeline:
    """Retrieve + rerank wrapper that returns results and citations."""

    def __init__(self, index: InMemoryIndex, cfg: RetrievalConfig) -> None:
        self.index = index
        self.cfg = cfg

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        retrieved = retrieve(query, self.index, self.cfg)
        reranked = rerank(query, retrieved, self.cfg.reranker.top_n)[:top_k]
        citations = [
            {"doc_id": c.doc_id, "score": round(c.score, 4), "snippet": c.text[:80]}
            for c in reranked
        ]
        return {"results": [c.text for c in reranked], "citations": citations}


def load_sample_kb() -> dict[str, str]:
    """Load the bundled Chinese FAQ corpus as ``{doc_id: text}``."""
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(SAMPLE_KB_DIR.glob("*.md"))
    }


def build_pipeline(
    docs: dict[str, str] | None = None, embedder: Embedder | None = None
) -> RagPipeline:
    """Build a pipeline over ``docs`` (defaults to the bundled sample KB)."""
    cfg = load_retrieval_config()
    corpus = docs if docs is not None else load_sample_kb()
    chunks = ingest_docs(corpus, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    index = build_index(chunks, embedder or HashEmbedder())
    return RagPipeline(index, cfg)


@lru_cache
def default_pipeline() -> RagPipeline:
    """Cached pipeline over the bundled sample KB."""
    return build_pipeline()


def build_default_kb_search() -> Callable[[str, int], dict[str, Any]]:
    """Return a ``search_kb`` callable backed by the default pipeline."""
    pipe = default_pipeline()

    def kb_search(query: str, top_k: int = 5) -> dict[str, Any]:
        return pipe.search(query, top_k)

    return kb_search
