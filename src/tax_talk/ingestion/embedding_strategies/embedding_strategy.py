"""Common strategy interface for all embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text. All vectors same length."""
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Default query embedding behavior for providers without query/document split."""
        return self.embed([query])[0]

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
