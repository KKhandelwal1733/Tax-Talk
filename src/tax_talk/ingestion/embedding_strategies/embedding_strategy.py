"""Common strategy interface for all embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text. All vectors same length."""
        raise NotImplementedError

    @abstractmethod
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async version: return one vector per text. All vectors same length."""
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Default query embedding behavior for providers without query/document split."""
        return self.embed([query])[0]

    async def embed_query_async(self, query: str) -> list[float]:
        """Async query embedding using default behavior."""
        result = await self.embed_async([query])
        return result[0]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector size. Must match Qdrant collection setting."""
        raise NotImplementedError

    def get_usage_stats(self) -> dict[str, int]:
        """Optional provider usage counters. Empty for providers without tracking."""
        return {}

    def reset_usage_stats(self) -> None:
        """Optional reset hook for provider usage counters."""
        return
