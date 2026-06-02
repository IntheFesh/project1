"""Vector indexing into Milvus using bge embeddings (Phase 3)."""

from __future__ import annotations

from rag.ingest import Chunk


def build_index(chunks: list[Chunk]) -> None:
    """Embed and upsert chunks into the Milvus collection (implemented in Phase 3)."""
    raise NotImplementedError("build_index is implemented in Phase 3.")
