from __future__ import annotations

from pathlib import Path

import pytest

from tax_talk.ingestion.chunker import (
    ContextualChunkingStrategy,
    FixedChunkingStrategy,
    SemanticChunkingStrategy,
    chunk_document,
    get_chunking_strategy,
    resolve_chunking_strategy,
)
from tax_talk.ingestion.loader import SourceDocument


def _doc() -> SourceDocument:
    return SourceDocument(
        source_key="test_source",
        file_path=Path("data/raw/test_source/source.txt"),
        text=(
            "CHAPTER I\n\n"
            "Section 1. Preliminary and short title.\n\n"
            "This section describes the short title and commencement.\n\n"
            "CHAPTER II\n\n"
            "Section 2. Definitions and interpretation clauses.\n\n"
            "This section defines key terms used throughout the statute."
        ),
        metadata={"filename": "source.txt"},
    )


def test_resolve_chunking_strategy_accepts_supported_values() -> None:
    assert resolve_chunking_strategy("fixed") == "fixed"
    assert resolve_chunking_strategy("semantic") == "semantic"
    assert resolve_chunking_strategy("contextual") == "contextual"


def test_resolve_chunking_strategy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        resolve_chunking_strategy("random")


def test_get_chunking_strategy_returns_registered_implementation() -> None:
    assert isinstance(get_chunking_strategy("fixed"), FixedChunkingStrategy)
    assert isinstance(get_chunking_strategy("semantic"), SemanticChunkingStrategy)
    assert isinstance(get_chunking_strategy("contextual"), ContextualChunkingStrategy)


def test_chunk_document_sets_strategy_metadata() -> None:
    chunks = chunk_document(_doc(), chunking_strategy="semantic")
    assert chunks
    assert all(c.chunking_strategy == "semantic" for c in chunks)
