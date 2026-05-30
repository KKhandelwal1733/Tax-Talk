from __future__ import annotations

from tax_talk.core.config import settings
from tax_talk.ingestion.chunking_strategies.base import ChunkingStrategy
from tax_talk.ingestion.chunking_strategies.contextual import ContextualChunkingStrategy
from tax_talk.ingestion.chunking_strategies.fixed import FixedChunkingStrategy
from tax_talk.ingestion.chunking_strategies.semantic import SemanticChunkingStrategy

CHUNKING_STRATEGIES: tuple[str, str, str] = ("fixed", "semantic", "contextual")

_STRATEGY_REGISTRY: dict[str, ChunkingStrategy] = {
    "fixed": FixedChunkingStrategy(),
    "semantic": SemanticChunkingStrategy(),
    "contextual": ContextualChunkingStrategy(),
}


def resolve_chunking_strategy(strategy: str | None = None) -> str:
    """Return a validated chunking strategy key."""
    candidate = (strategy or settings.chunking_strategy).strip().lower()
    if candidate not in CHUNKING_STRATEGIES:
        options = ", ".join(CHUNKING_STRATEGIES)
        raise ValueError(f"Unsupported chunking strategy '{candidate}'. Choose one of: {options}.")
    return candidate


def get_chunking_strategy(strategy: str | None = None) -> ChunkingStrategy:
    """Return a chunking strategy instance from the registry."""
    resolved = resolve_chunking_strategy(strategy)
    return _STRATEGY_REGISTRY[resolved]
