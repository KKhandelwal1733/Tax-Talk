"""Reciprocal rank fusion for hybrid retrieval."""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    *,
    ranked_lists: list[list[dict[str, Any]]],
    weights: list[float],
    rrf_k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge ranked result lists using Reciprocal Rank Fusion (RRF)."""
    by_chunk: dict[str, dict[str, Any]] = {}
    fused_scores: dict[str, float] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx]
        for rank, item in enumerate(ranked):
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id:
                continue

            contribution = weight / float(rrf_k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + contribution

            if chunk_id not in by_chunk:
                by_chunk[chunk_id] = dict(item)
            else:
                by_chunk[chunk_id].update(item)

    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

    result: list[dict[str, Any]] = []
    for fused_rank, (chunk_id, score) in enumerate(ordered[:top_k]):
        payload = dict(by_chunk[chunk_id])
        payload["chunk_id"] = chunk_id
        payload["fused_score"] = score
        payload["fused_rank"] = fused_rank
        result.append(payload)

    return result



