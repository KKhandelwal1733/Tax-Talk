"""Entrypoint for the tax_talk API."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from langfuse import observe

from tax_talk.core.runtime import get_logger
from tax_talk.models.api import HealthResponse, RetrieveRequest, RetrieveResponse
from tax_talk.retrieval import HybridRetriever

log = get_logger(__name__)


app = FastAPI(title="domain-oracle API", version="0.1.0")


@lru_cache(maxsize=1)
def _get_retriever() -> HybridRetriever:
    """Build and cache the retriever instance used by API requests."""
    return HybridRetriever()


@observe(name="api-health", as_type="span", capture_input=False)
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Return simple API health status."""
    return HealthResponse()


@observe(name="api-retrieve", as_type="span", capture_input=True, capture_output=False)
@app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    """Return hybrid retrieval hits for a query."""
    try:
        hits = _get_retriever().retrieve(
            payload.query,
            top_k=payload.top_k,
            dense_top_k=payload.dense_top_k,
            bm25_top_k=payload.bm25_top_k,
            filters=payload.filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary
        log.exception("Retrieval failed for API request.")
        raise HTTPException(status_code=500, detail="retrieval failed") from exc

    return RetrieveResponse(hits=hits)


def main() -> None:
    """CLI placeholder when module is executed directly."""
    log.info("Run with: uv run uvicorn tax_talk.api.main:app --reload")
