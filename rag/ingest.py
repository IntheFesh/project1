"""Document ingestion: chunk source docs and attach provenance metadata (Phase 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable chunk with provenance metadata."""

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


def ingest(path: str) -> list[Chunk]:
    """Load and chunk documents under ``path`` (implemented in Phase 3)."""
    raise NotImplementedError("ingest is implemented in Phase 3.")
