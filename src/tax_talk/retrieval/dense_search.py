"""Dense search adapter for hybrid retrieval."""

from __future__ import annotations

from typing import Any

from tax_talk.ingestion.embeddings import embed_query
from tax_talk.ingestion.qdrant_store import QdrantStore


def run_dense_search(
    *,
    store: QdrantStore,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run query embedding + dense vector search and shape ranked hits."""
    query_vector = embed_query(query)
    hits = store.search(query_vector=query_vector, top_k=top_k, filters=filters)

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
