from __future__ import annotations

from typing import Any

import pytest

from tax_talk.core.config import settings
from tax_talk.retrieval.helpers.filters import payload_matches_filters
from tax_talk.retrieval.helpers.tokenization import tokenize
from tax_talk.retrieval.hybrid import HybridRetriever


class DummyStore:
    def __init__(self, payloads: list[dict[str, Any]], dense_hits: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self._dense_hits = dense_hits

    def scroll_payloads(
        self, page_size: int = 512, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not filters:
            return list(self._payloads)
        return [p for p in self._payloads if payload_matches_filters(p, filters)]

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        hits = list(self._dense_hits)
        if filters:
            hits = [h for h in hits if payload_matches_filters(h, filters)]
        return hits[:top_k]


@pytest.fixture(autouse=True)
def _stub_dense_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_dense_search(
        *,
        store: DummyStore,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        _ = query
        return store.search(query_vector=[0.0], top_k=top_k, filters=filters)

    async def _fake_run_dense_search_async(
        *,
        store: DummyStore,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        _ = query
        return store.search(query_vector=[0.0], top_k=top_k, filters=filters)

    monkeypatch.setattr("tax_talk.retrieval.hybrid.run_dense_search", _fake_run_dense_search)
    monkeypatch.setattr("tax_talk.retrieval.hybrid.run_dense_search_async", _fake_run_dense_search_async)


def test_tokenize_keeps_identifiers_and_numbers() -> None:
    tokens = tokenize("Section 54F exemption for FY 2026-27")
    assert "section" in tokens
    assert "54f" in tokens
    assert "2026" in tokens
    assert "27" in tokens


def test_hybrid_retrieval_rrf_fuses_dense_and_bm25() -> None:
    payloads = [
        {
            "chunk_id": "c1",
            "text": "GST applies on free samples under specific circumstances",
            "doc_type": "act",
        },
        {
            "chunk_id": "c2",
            "text": "Section 54F exemption conditions for house property",
            "doc_type": "act",
        },
        {
            "chunk_id": "c3",
            "text": "TDS for technical services paid to non residents",
            "doc_type": "act",
        },
    ]

    dense_hits = [
        {"chunk_id": "c2", "text": payloads[1]["text"], "score": 0.95, "doc_type": "act"},
        {"chunk_id": "c1", "text": payloads[0]["text"], "score": 0.90, "doc_type": "act"},
    ]

    retriever = HybridRetriever(
        store=DummyStore(payloads=payloads, dense_hits=dense_hits), rrf_k=60
    )
    results = retriever.retrieve("section 54F exemption", top_k=2, dense_top_k=2, bm25_top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"
    assert "fused_score" in results[0]


def test_hybrid_retrieval_respects_filters() -> None:
    payloads = [
        {
            "chunk_id": "a1",
            "text": "CGST section about valuation",
            "doc_type": "act",
            "act_status": "current",
        },
        {
            "chunk_id": "a2",
            "text": "Old act legacy section",
            "doc_type": "act",
            "act_status": "legacy",
        },
    ]

    dense_hits = [
        {
            "chunk_id": "a1",
            "text": payloads[0]["text"],
            "score": 0.8,
            "doc_type": "act",
            "act_status": "current",
        },
        {
            "chunk_id": "a2",
            "text": payloads[1]["text"],
            "score": 0.7,
            "doc_type": "act",
            "act_status": "legacy",
        },
    ]

    retriever = HybridRetriever(store=DummyStore(payloads=payloads, dense_hits=dense_hits))
    results = retriever.retrieve("valuation", top_k=5, filters={"act_status": "current"})

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a1"


def test_retrieval_skips_cohere_when_no_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cohere_api_key", "")

    payloads = [
        {"chunk_id": "c1", "text": "section 54f exemption details", "doc_type": "act"},
        {"chunk_id": "c2", "text": "valuation rules under cgst", "doc_type": "rules"},
    ]
    dense_hits = [
        {"chunk_id": "c1", "text": payloads[0]["text"], "score": 0.95, "doc_type": "act"},
        {"chunk_id": "c2", "text": payloads[1]["text"], "score": 0.90, "doc_type": "rules"},
    ]

    retriever = HybridRetriever(
        store=DummyStore(payloads=payloads, dense_hits=dense_hits),
        rerank_enabled=True,
        rerank_top_k=5,
        rerank_top_n=2,
    )
    results = retriever.retrieve("section 54f", top_k=2, dense_top_k=2, bm25_top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert "rerank_score" not in results[0]


def test_retrieval_applies_cohere_rerank(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cohere_api_key", "test-key")

    class DummyResult:
        def __init__(self, index: int, relevance_score: float) -> None:
            self.index = index
            self.relevance_score = relevance_score

    class DummyResponse:
        def __init__(self) -> None:
            self.results = [
                DummyResult(index=1, relevance_score=0.99),
                DummyResult(index=0, relevance_score=0.88),
            ]

    class DummyCohereClient:
        def rerank(self, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["model"] == "rerank-v4.0-pro"
            assert kwargs["top_n"] == 2
            return DummyResponse()

    monkeypatch.setattr("tax_talk.retrieval.hybrid.get_cohere_client", lambda: DummyCohereClient())

    payloads = [
        {"chunk_id": "c1", "text": "valuation rules under cgst", "doc_type": "rules"},
        {"chunk_id": "c2", "text": "section 54f exemption details", "doc_type": "act"},
    ]
    dense_hits = [
        {"chunk_id": "c2", "text": payloads[1]["text"], "score": 0.95, "doc_type": "act"},
        {"chunk_id": "c1", "text": payloads[0]["text"], "score": 0.90, "doc_type": "rules"},
    ]

    retriever = HybridRetriever(
        store=DummyStore(payloads=payloads, dense_hits=dense_hits),
        rerank_enabled=True,
        rerank_top_k=5,
        rerank_top_n=2,
    )
    results = retriever.retrieve("section 54f", top_k=2, dense_top_k=2, bm25_top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_rank"] == 0
    assert "rerank_score" in results[0]


def test_retrieval_fallback_when_cohere_rerank_errors(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cohere_api_key", "test-key")

    class FailingCohereClient:
        def rerank(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("cohere unavailable")

    monkeypatch.setattr(
        "tax_talk.retrieval.hybrid.get_cohere_client", lambda: FailingCohereClient()
    )

    payloads = [
        {"chunk_id": "c1", "text": "section 54f exemption details", "doc_type": "act"},
        {"chunk_id": "c2", "text": "valuation rules under cgst", "doc_type": "rules"},
    ]
    dense_hits = [
        {"chunk_id": "c1", "text": payloads[0]["text"], "score": 0.95, "doc_type": "act"},
        {"chunk_id": "c2", "text": payloads[1]["text"], "score": 0.90, "doc_type": "rules"},
    ]

    retriever = HybridRetriever(
        store=DummyStore(payloads=payloads, dense_hits=dense_hits),
        rerank_enabled=True,
        rerank_top_k=5,
        rerank_top_n=2,
    )
    results = retriever.retrieve("section 54f", top_k=2, dense_top_k=2, bm25_top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert "rerank_score" not in results[0]


@pytest.mark.asyncio
async def test_bm25_search_async_matches_sync_results(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = [
        {"chunk_id": "c1", "text": "section 54f exemption details", "doc_type": "act"},
    ]
    retriever = HybridRetriever(store=DummyStore(payloads=payloads, dense_hits=[]))

    def _fake_bm25_search(
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        _ = (query, top_k, filters)
        return [{"chunk_id": "c1", "bm25_score": 0.5, "bm25_rank": 0}]

    monkeypatch.setattr(retriever, "_bm25_search", _fake_bm25_search)

    result = await retriever._bm25_search_async(
        query="section 54f",
        top_k=1,
        filters=None,
    )

    assert result == [{"chunk_id": "c1", "bm25_score": 0.5, "bm25_rank": 0}]
