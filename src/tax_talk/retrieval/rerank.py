"""Optional Cohere reranking helpers for retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def cohere_rerank_candidates(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    rerank_top_n: int,
    rerank_model: str,
    rerank_max_tokens_per_doc: int,
    cohere_api_key: str,
    get_client: Callable[[], Any],
    logger: Any,
    relevance_threshold: float = 0.35,  # <-- ADDED: Calibrated gatekeeper threshold
) -> list[dict[str, Any]]:
    """Rerank candidates with Cohere; return clean filtered results with fail-open fallback."""
    if not candidates:
        return []

    # If no API key, do a clean fallback of the top candidates without padding noise
    if not cohere_api_key:
        return candidates[:top_k]

    indexed_docs: list[tuple[int, str]] = []
    for idx, candidate in enumerate(candidates):
        text = candidate.get("text")
        if isinstance(text, str) and text.strip():
            indexed_docs.append((idx, text))

    if not indexed_docs:
        return candidates[:top_k]

    docs = [text for _, text in indexed_docs]
    
    # CRITICAL CHANGE: Widen top_n parameter. 
    # Let Cohere grade your ENTIRE candidate pool (e.g., top 20-30 chunks) 
    # instead of truncating early.
    top_n = min(rerank_top_n, len(docs))
    if top_n <= 0:
        return candidates[:top_k]

    try:
        response = get_client().rerank(
            model=rerank_model,
            query=query,
            documents=docs,
            top_n=top_n,
            max_tokens_per_doc=rerank_max_tokens_per_doc,
        )
        response_results = getattr(response, "results", [])
    except Exception as exc:  # pragma: no cover
        logger.warning("Cohere rerank failed; using fused ranking fallback. %s", exc)
        return candidates[:top_k]

    selected: list[dict[str, Any]] = []
    selected_global_idxs: set[int] = set()
    
    for rerank_rank, item in enumerate(response_results):
        local_idx = getattr(item, "index", None)
        if local_idx is None and isinstance(item, dict):
            local_idx = item.get("index")
        if not isinstance(local_idx, int) or local_idx < 0 or local_idx >= len(indexed_docs):
            continue

        relevance_score = getattr(item, "relevance_score", None)
        if relevance_score is None and isinstance(item, dict):
            relevance_score = item.get("relevance_score")

        if isinstance(relevance_score, (int, float)): # noqa: UP038
            if float(relevance_score) < relevance_threshold:
                continue  

        candidate_idx = indexed_docs[local_idx][0]
        if candidate_idx in selected_global_idxs:
            continue

        row = dict(candidates[candidate_idx])
        if isinstance(relevance_score, (int, float)):# noqa: UP038
            row["rerank_score"] = float(relevance_score)
        row["rerank_rank"] = rerank_rank
        selected.append(row)
        selected_global_idxs.add(candidate_idx)

    final_results = selected[:top_k]
    
    logger.info(
        "Reranker processed %d candidates -> returned %d clean chunks clearing threshold %s",
        len(candidates), len(final_results), relevance_threshold
    )
    return final_results