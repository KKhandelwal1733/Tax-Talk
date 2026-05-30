from __future__ import annotations

from tax_talk.core.config import settings
from tax_talk.ingestion.chunking_strategies.base import ChunkingStrategy
from tax_talk.ingestion.chunking_strategies.helpers import split_semantic_text


class SemanticChunkingStrategy(ChunkingStrategy):
    """Heading/section-aware semantic chunking."""

    name = "semantic"

    def split_text(self, cleaned_text: str) -> list[tuple[str, int, int]]:
        return split_semantic_text(
            cleaned_text,
            min_chunk_chars=settings.semantic_chunk_min_chars,
            max_chunk_chars=settings.semantic_chunk_max_chars,
        )
