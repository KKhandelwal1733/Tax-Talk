from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response model for liveness checks."""

    status: str = "ok"


class RetrieveRequest(BaseModel):
    """Request model for hybrid retrieval."""

    query: str = Field(min_length=1, description="Natural-language query to search for.")
    top_k: int = Field(default=10, ge=1, le=100)
    dense_top_k: int = Field(default=30, ge=1, le=300)
    bm25_top_k: int = Field(default=30, ge=1, le=300)
    filters: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class RetrieveResponse(BaseModel):
    """Response model for hybrid retrieval results."""

    hits: list[dict[str, Any]]
