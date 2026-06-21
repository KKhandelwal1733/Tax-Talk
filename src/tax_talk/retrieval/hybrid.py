"""Hybrid retrieval: dense + BM25 with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import asyncio
from typing import Any

from langfuse import observe

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_cohere_client, get_logger
from tax_talk.ingestion.qdrant_store import QdrantStore
from tax_talk.retrieval.bm25_index import Bm25Searcher
from tax_talk.retrieval.dense_search import run_dense_search, run_dense_search_async
from tax_talk.retrieval.helpers.fusion import reciprocal_rank_fusion
from tax_talk.retrieval.rerank import cohere_rerank_candidates

log = get_logger(__name__)


class HybridRetriever:
    """Retrieve candidates from dense and BM25 retrievers, then fuse with RRF."""

    def __init__(
        self,
        store: QdrantStore | None = None,
        *,
        collection_name: str | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rerank_enabled: bool | None = None,
        rerank_model: str | None = None,
        rerank_top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if dense_weight <= 0 or bm25_weight <= 0:
            raise ValueError("dense_weight and bm25_weight must be > 0")

        resolved_rerank_top_k = settings.rerank_top_k if rerank_top_k is None else rerank_top_k
        resolved_rerank_top_n = settings.rerank_top_n if rerank_top_n is None else rerank_top_n
        if resolved_rerank_top_k <= 0 or resolved_rerank_top_n <= 0:
            raise ValueError("rerank_top_k and rerank_top_n must be > 0")

        self._store = store or QdrantStore(collection_name=collection_name)
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._rerank_enabled = settings.rerank_enabled if rerank_enabled is None else rerank_enabled
        self._rerank_model = settings.rerank_model if rerank_model is None else rerank_model
        self._rerank_top_k = resolved_rerank_top_k
        self._rerank_top_n = resolved_rerank_top_n

        self._bm25 = Bm25Searcher(store=self._store, logger=log)

    @observe(name="retrieval-hybrid", as_type="retriever", capture_input=True, capture_output=True)
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int = 30,
        bm25_top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return fused retrieval hits sorted by descending RRF score."""
        if not query.strip():
            return []
        if top_k <= 0:
            return []

        fusion_top_k = max(top_k, self._rerank_top_k) if self._rerank_enabled else top_k

        dense_hits = self._dense_search(
            query=query, top_k=max(fusion_top_k, dense_top_k), filters=filters
        )
        bm25_hits = self._bm25_search(
            query=query, top_k=max(fusion_top_k, bm25_top_k), filters=filters
        )

        fused = self._reciprocal_rank_fusion(
            ranked_lists=[dense_hits, bm25_hits],
            weights=[self._dense_weight, self._bm25_weight],
            top_k=fusion_top_k,
        )

        if not self._rerank_enabled:
            return fused[:top_k]

        return self._cohere_rerank(query=query, candidates=fused, top_k=top_k)

    @observe(
        name="retrieval-hybrid-async", as_type="retriever", capture_input=True, capture_output=True
    )
    async def retrieve_async(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int = 30,
        bm25_top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return fused retrieval hits via async dense and BM25 paths."""
        if not query.strip():
            return []
        if top_k <= 0:
            return []

        fusion_top_k = max(top_k, self._rerank_top_k) if self._rerank_enabled else top_k

        dense_task = self._dense_search_async(
            query=query,
            top_k=max(fusion_top_k, dense_top_k),
            filters=filters,
        )
        bm25_task = self._bm25_search_async(
            query=query,
            top_k=max(fusion_top_k, bm25_top_k),
            filters=filters,
        )
        dense_hits, bm25_hits = await asyncio.gather(dense_task, bm25_task)

        fused = self._reciprocal_rank_fusion(
            ranked_lists=[dense_hits, bm25_hits],
            weights=[self._dense_weight, self._bm25_weight],
            top_k=fusion_top_k,
        )

        if not self._rerank_enabled:
            return fused[:top_k]

        return await asyncio.to_thread(
            self._cohere_rerank, query=query, candidates=fused, top_k=top_k
        )

    @observe(
        name="retrieval-cohere-rerank",
        as_type="retriever",
        capture_input=False,
        capture_output=False,
    )
    def _cohere_rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return cohere_rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=top_k,
            rerank_top_n=self._rerank_top_n,
            rerank_model=self._rerank_model,
            rerank_max_tokens_per_doc=settings.rerank_max_tokens_per_doc,
            cohere_api_key=settings.cohere_api_key,
            get_client=get_cohere_client,
            logger=log,
        )

    def refresh_bm25_index(self) -> None:
        """Force reload BM25 corpus from Qdrant payloads."""
        self._bm25.refresh()

    @observe(
        name="retrieval-dense-search", as_type="span", capture_input=False, capture_output=False
    )
    def _dense_search(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return run_dense_search(store=self._store, query=query, top_k=top_k, filters=filters)

    @observe(
        name="retrieval-dense-search-async",
        as_type="span",
        capture_input=False,
        capture_output=False,
    )
    async def _dense_search_async(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return await run_dense_search_async(
            store=self._store,
            query=query,
            top_k=top_k,
            filters=filters,
        )

    @observe(
        name="retrieval-bm25-search", as_type="span", capture_input=False, capture_output=False
    )
    def _bm25_search(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        self._ensure_bm25_index()
        return self._bm25.search(query=query, top_k=top_k, filters=filters)

    @observe(
        name="retrieval-bm25-search-async",
        as_type="span",
        capture_input=False,
        capture_output=False,
    )
    async def _bm25_search_async(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._bm25_search,
            query=query,
            top_k=top_k,
            filters=filters,
        )

    def _ensure_bm25_index(self) -> None:
        self._load_bm25_index_locked(force=False)

    @observe(
        name="retrieval-bm25-load-index", as_type="span", capture_input=False, capture_output=False
    )
    def _load_bm25_index_locked(self, *, force: bool) -> None:
        self._bm25.load(force=force)

    @observe(name="retrieval-rrf-fusion", as_type="span", capture_input=False, capture_output=False)
    def _reciprocal_rank_fusion(
        self,
        *,
        ranked_lists: list[list[dict[str, Any]]],
        weights: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            weights=weights,
            rrf_k=self._rrf_k,
            top_k=top_k,
        )
