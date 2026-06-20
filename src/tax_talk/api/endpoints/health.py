"""Health endpoints for liveness and readiness checks."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from langfuse import observe
from tax_talk.core.runtime import get_async_qdrant_client
from tax_talk.models.api import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
@observe(name="api-health-live", as_type="span", capture_input=False, capture_output=True)
async def live() -> HealthResponse:
    """Lightweight process liveness probe."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
@observe(name="api-health-ready", as_type="span", capture_input=False, capture_output=True)
async def ready() -> HealthResponse:
    """Dependency readiness probe for Qdrant connectivity."""
    client = get_async_qdrant_client()
    try:
        await client.get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"readiness check failed: {exc}") from exc
    return HealthResponse(status="ok")
