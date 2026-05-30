from __future__ import annotations

from abc import ABC, abstractmethod

from tax_talk.ingestion.contextual_summary import ContextualSummaryResult
from tax_talk.ingestion.loader import SourceDocument


class ChunkingStrategy(ABC):
    """Contract for chunking strategies."""

    name: str

    @abstractmethod
    def split_text(self, cleaned_text: str) -> list[tuple[str, int, int]]:
        """Return chunk text windows with source character offsets."""

    def build_summary(self, doc: SourceDocument, cleaned_text: str) -> ContextualSummaryResult:
        """Build optional summary applied to chunks from the same document."""
        return ContextualSummaryResult()

    def render_chunk(self, chunk_text: str, summary: ContextualSummaryResult) -> str:
        """Render final chunk text before embedding/upsert."""
        return chunk_text
