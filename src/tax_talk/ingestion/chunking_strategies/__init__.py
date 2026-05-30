"""Chunking strategy package for ingestion."""

from __future__ import annotations

from tax_talk.ingestion.chunking_strategies.base import ChunkingStrategy
from tax_talk.ingestion.chunking_strategies.contextual import ContextualChunkingStrategy
from tax_talk.ingestion.chunking_strategies.fixed import FixedChunkingStrategy
from tax_talk.ingestion.chunking_strategies.registry import (
    CHUNKING_STRATEGIES,
    get_chunking_strategy,
    resolve_chunking_strategy,
)
from tax_talk.ingestion.chunking_strategies.semantic import SemanticChunkingStrategy

__all__ = [
    "CHUNKING_STRATEGIES",
    "ChunkingStrategy",
    "ContextualChunkingStrategy",
    "FixedChunkingStrategy",
    "SemanticChunkingStrategy",
    "get_chunking_strategy",
    "resolve_chunking_strategy",
]
