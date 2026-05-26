"""Embedding strategy implementations and selection logic."""

from tax_talk.ingestion.embedding_strategies.embedding_strategy import EmbeddingStrategy
from tax_talk.ingestion.embedding_strategies.factory import get_embedding_strategy

__all__ = ["EmbeddingStrategy", "get_embedding_strategy"]
