"""Vector index. ``InMemoryIndex`` (offline dev/test) + a Milvus path for production.

The in-memory index precomputes dense embeddings and a BM25 model so hybrid retrieval is
cheap. The production path (``MilvusIndex``) uses pymilvus + bge (``requirements/rag.txt``).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from rag.embeddings import Embedder, HashEmbedder
from rag.ingest import Chunk
from rag.text import tokenize


class BM25:
    """Okapi BM25 over a tokenized corpus."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf = [Counter(doc) for doc in corpus_tokens]
        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """Return a BM25 score per document for ``query_tokens``."""
        scores = np.zeros(self.n, dtype=float)
        if self.n == 0:
            return scores
        avgdl = self.avgdl or 1.0
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n):
                freq = self.tf[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class InMemoryIndex:
    """Dense embeddings + BM25 over a chunk list, all held in memory."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [c.text for c in self.chunks]
        self.embeddings = (
            embedder.embed(texts)
            if texts
            else np.zeros((0, embedder.dim), dtype=float)
        )
        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25(self.corpus_tokens)

    def __len__(self) -> int:
        return len(self.chunks)


def build_index(chunks: list[Chunk], embedder: Embedder | None = None) -> InMemoryIndex:
    """Build an in-memory index (defaults to the offline ``HashEmbedder``)."""
    return InMemoryIndex(chunks, embedder or HashEmbedder())


class MilvusIndex:
    """Production Milvus-backed index (lazy pymilvus import; runs on the infra box)."""

    def __init__(self, uri: str, collection: str, embedder: Embedder) -> None:
        self.uri = uri
        self.collection = collection
        self.embedder = embedder

    def build(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into Milvus (Phase 8 / GPU-infra)."""
        raise NotImplementedError(
            "MilvusIndex runs on the infra box with requirements/rag.txt (Phase 8)."
        )
