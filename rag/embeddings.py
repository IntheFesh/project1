"""Embeddings. A deterministic, dependency-free dev embedder + the prod bge interface.

``HashEmbedder`` hashes tokens into a fixed-width L2-normalized vector. It is a
reproducible offline stand-in (no downloads, no GPU) — NOT a semantic model. The
production path uses bge-m3 via ``requirements/rag.txt`` (see ``BGEEmbedder``).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from rag.text import tokenize


@runtime_checkable
class Embedder(Protocol):
    """Maps a list of texts to an ``(n, dim)`` float matrix of unit vectors."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class HashEmbedder:
    """Deterministic hashing bag-of-tokens embedder for offline dev/tests."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                matrix[row, _bucket(token, self.dim)] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


class BGEEmbedder:
    """Production bge embedder (lazy import; install requirements/rag.txt on the GPU box)."""

    def __init__(self, model: str = "BAAI/bge-m3", dim: int = 1024) -> None:
        self.model_name = model
        self.dim = dim
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy / heavy

            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        assert self._model is not None
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))
