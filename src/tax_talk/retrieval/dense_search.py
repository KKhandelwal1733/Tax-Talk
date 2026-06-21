"""Dense search adapter for hybrid retrieval."""

from __future__ import annotations

import asyncio
from typing import Any

from tax_talk.ingestion.embeddings import embed_query_async
from tax_talk.ingestion.qdrant_store import QdrantStore


def run_dense_search(
    *,
    store: QdrantStore,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run query embedding + dense vector search and shape ranked hits.

    Note: Uses async embedding internally via asyncio.run().
    """
    return asyncio.run(
        run_dense_search_async(store=store, query=query, top_k=top_k, filters=filters)
    )


async def run_dense_search_async(
    *,
    store: QdrantStore,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run async dense search with native async embedding."""
    query_vector = await embed_query_async(query)
    hits = await store.search_async(query_vector=query_vector, top_k=top_k, filters=filters)

    ranked: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits):
        chunk_id = str(hit.get("chunk_id", ""))
        if not chunk_id:
            continue
        row = dict(hit)
        row["chunk_id"] = chunk_id
        row["dense_score"] = float(hit.get("score", 0.0))
        row["dense_rank"] = rank
        ranked.append(row)

    return ranked
