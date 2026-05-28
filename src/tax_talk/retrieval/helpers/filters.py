"""Payload filter helpers for retrieval results."""

from __future__ import annotations

from typing import Any


def payload_matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return true when payload satisfies all equality filters."""
    for key, expected in filters.items():
        if payload.get(key) != expected:
            return False
    return True



