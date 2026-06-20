"""Entrypoint for the tax_talk API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tax_talk.api.endpoints.chat import router as chat_router
from tax_talk.api.endpoints.health import router as health_router
from tax_talk.api.middleware.observability import ObservabilityMiddleware
from tax_talk.core.runtime import (
    close_gemini_client,
    flush_langfuse_client,
    get_logger,
    get_qdrant_client,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm lightweight shared clients at startup."""
    try:
        get_qdrant_client()
    except Exception as exc:  # pragma: no cover
        log.warning("Qdrant warmup skipped: %s", exc)
    yield
    try:
        flush_langfuse_client()
    except Exception as exc:  # pragma: no cover
        log.warning("Langfuse flush skipped: %s", exc)
    try:
        await close_gemini_client()
    except Exception as exc:  # pragma: no cover
        log.warning("Gemini client close skipped: %s", exc)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="tax-talk API", version="0.3.2", lifespan=lifespan)
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()


def main() -> None:
    """CLI placeholder when module is executed directly."""
    log.info("Run with: uv run uvicorn tax_talk.api.main:app --reload")
