"""RAG pipeline: chunking, hybrid retrieval, rerank, and citations."""

from agent.tools.services import ServiceDesk
from rag.ingest import chunk_text, ingest_docs
from rag.pipeline import build_default_kb_search, build_pipeline


def test_chunking_respects_size_and_overlap() -> None:
    chunks = chunk_text("a" * 1000, chunk_size=400, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)


def test_ingest_docs_sets_metadata() -> None:
    chunks = ingest_docs({"d1": "你好世界"})
    assert chunks[0].doc_id == "d1"
    assert chunks[0].metadata["source"] == "d1"


def test_retrieves_shipping_doc_for_freight_query() -> None:
    out = build_pipeline().search("运费是怎么计算的", top_k=3)
    assert out["citations"]
    assert any(c["doc_id"] == "shipping_faq" for c in out["citations"])


def test_retrieves_refund_doc_for_refund_query() -> None:
    out = build_pipeline().search("退款时效是多久", top_k=3)
    assert any(c["doc_id"] == "refund_faq" for c in out["citations"])


def test_service_desk_search_kb_returns_citations() -> None:
    services = ServiceDesk(kb_search=build_default_kb_search())
    assert services.search_kb("运费怎么算", top_k=3)["citations"]
