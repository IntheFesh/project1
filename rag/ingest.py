"""Document ingestion: chunk source docs and attach provenance metadata."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable chunk with provenance metadata."""

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split ``text`` into overlapping character windows."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    step = max(1, chunk_size - overlap)
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def ingest_docs(
    docs: dict[str, str], chunk_size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """Chunk a ``{doc_id: text}`` mapping into ``Chunk`` records."""
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        for index, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{index}",
                    text=piece,
                    metadata={"source": doc_id},
                )
            )
    return chunks


def ingest(path: str, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Load and chunk all ``*.md`` / ``*.txt`` files under ``path``."""
    root = Path(path)
    files = sorted([*root.glob("*.md"), *root.glob("*.txt")])
    docs = {f.stem: f.read_text(encoding="utf-8") for f in files}
    return ingest_docs(docs, chunk_size, overlap)
