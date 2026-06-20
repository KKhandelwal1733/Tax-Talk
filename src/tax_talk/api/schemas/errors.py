"""Shared API error response schema."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    detail: str
