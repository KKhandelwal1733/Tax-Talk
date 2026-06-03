from __future__ import annotations

from tax_talk.core.config import settings
from tax_talk.ingestion.chunking_strategies.base import ChunkingStrategy
from tax_talk.ingestion.chunking_strategies.helpers import split_fixed_text


class FixedChunkingStrategy(ChunkingStrategy):
    """Fixed-size chunking with overlap."""

    name = "fixed"

    def split_text(self, cleaned_text: str) -> list[tuple[str, int, int]]:
        return split_fixed_text(
            cleaned_text,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
