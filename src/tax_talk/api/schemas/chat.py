"""Chat API request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request body for synchronous and streaming chat endpoints."""

    query: str = Field(min_length=1, description="Natural-language user query.")
    top_k: int = Field(default=8, ge=1, le=50)
    dense_top_k: int = Field(default=24, ge=1, le=200)
    bm25_top_k: int = Field(default=24, ge=1, le=200)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    answer: str
    citations: list[dict[str, Any]]
    faithfulness: dict[str, Any] | None = None


class ChatStreamEvent(BaseModel):
    """Server-sent event payload for streamed chat responses."""

    id: str | None = None
    event: str
    text: str = ""
    citations: list[dict[str, Any]] | None = None
    faithfulness: dict[str, Any] | None = None
