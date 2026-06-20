"""Service dependency providers for API routes."""

from __future__ import annotations

from functools import lru_cache

from tax_talk.api.services.chat_services import ChatService
from tax_talk.retrieval import HybridRetriever


@lru_cache(maxsize=1)
def _build_chat_service() -> ChatService:
    return ChatService(retriever=HybridRetriever())


def get_chat_service() -> ChatService:
    """Return process-wide chat service singleton for route injection."""
    return _build_chat_service()
