"""
src/tax_talk/ingestion/embeddings.py

Backward-compatible facade over the nested embedding strategy package.

Usage:
    from tax_talk.ingestion.embeddings import embed_texts, get_embedder

    # Embed a batch of texts
    vectors = embed_texts(["GST on free samples?", "TDS under Section 194C"])

    # Or get the embedder object for reuse
    embedder = get_embedder()
    vectors = embedder.embed(["some text"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from langfuse import observe

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.ingestion.embedding_strategies import EmbeddingStrategy, get_embedding_strategy
from tax_talk.models.ingestion import EmbeddingManifest

log = get_logger(__name__)


def _is_langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _int_stat(stats: dict[str, int], key: str) -> int:
    value = stats.get(key, 0)
    return value if isinstance(value, int) else 0


def _sanitize_text(text: str) -> str:
    """Replace invalid Unicode surrogates so embedding payloads are JSON-safe."""
    result = []
    for char in text:
        try:
            # Valid UTF-8 encodable characters pass through
            char.encode("utf-8")
            result.append(char)
        except UnicodeEncodeError:
            # Unencod able characters (like surrogates) become spaces
            result.append(" ")
    return "".join(result)


def _normalize_embedding_inputs(texts: list[object]) -> list[str]:
    """Coerce embedding inputs to strings so tokenizer calls are stable."""
    normalized: list[str] = []
    coerced = 0

    for text in texts:
        if isinstance(text, str):
            value = text
        elif isinstance(text, bytes):
            value = text.decode("utf-8", errors="ignore")
            coerced += 1
        elif text is None:
            value = ""
            coerced += 1
        else:
            value = str(text)
            coerced += 1

        value = _sanitize_text(value)
        # Keep cardinality stable for downstream upsert alignment.
        normalized.append(value if value else " ")

    if coerced:
        log.warning("Coerced %d non-string embedding inputs to strings.", coerced)

    return normalized


def _track_embedding_usage(
    *,
    operation: str,
    text_count: int,
    before_stats: dict[str, int],
    after_stats: dict[str, int],
) -> None:
    if not _is_langfuse_enabled():
        return

    # Langfuse embedding tracing removed.
    return


def get_embedder() -> EmbeddingStrategy:
    """
    Return the singleton embedder for the configured provider.
    Call this once at startup; the local model loads only on first call.
    """
    return get_embedding_strategy()


@observe(name="embed-texts", as_type="embedding", capture_input=False, capture_output=False)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convenience wrapper. Embeds texts using the configured provider.

    Args:
        texts: list of strings to embed (pass full chunks, not sentence fragments)

    Returns:
        list of float vectors, one per input text
    """
    if not texts:
        return []
    safe_texts = _normalize_embedding_inputs(list(texts))
    embedder = get_embedder()
    before = embedder.get_usage_stats()
    vectors = embedder.embed(safe_texts)
    after = embedder.get_usage_stats()
    _track_embedding_usage(
        operation="texts",
        text_count=len(safe_texts),
        before_stats=before,
        after_stats=after,
    )
    return vectors


@observe(name="embed-query", as_type="embedding", capture_input=False, capture_output=False)
def embed_query(query: str) -> list[float]:
    """
    Embed a single query string at retrieval time.

    Uses the selected strategy's query behavior.
    """
    strategy = get_embedder()
    before = strategy.get_usage_stats()
    vector = strategy.embed_query(query)
    after = strategy.get_usage_stats()
    _track_embedding_usage(
        operation="query",
        text_count=1,
        before_stats=before,
        after_stats=after,
    )
    return vector


def get_embedding_usage_stats() -> dict[str, int]:
    """Return provider usage counters (includes estimated HF requests for local strategy)."""
    return get_embedder().get_usage_stats()


def reset_embedding_usage_stats() -> None:
    """Reset provider usage counters."""
    get_embedder().reset_usage_stats()


def write_embeddings_npy(vectors: list[list[float]], output_path: Path) -> None:
    """Persist embedding vectors as .npy for processed-stage reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(vectors, dtype=np.float32))


def read_embeddings_npy(input_path: Path) -> list[list[float]]:
    """Load embedding vectors from .npy artifact."""
    array = np.load(input_path)
    if array.ndim != 2:
        raise ValueError(f"Invalid embeddings array shape {array.shape} in {input_path}.")
    return array.tolist()


def write_embedding_manifest(
    manifest_path: Path, payload: EmbeddingManifest | dict[str, Any]
) -> None:
    """Write embedding metadata used for stage validation and resume."""
    manifest = (
        payload
        if isinstance(payload, EmbeddingManifest)
        else EmbeddingManifest.model_validate(payload)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_embedding_manifest(manifest_path: Path) -> EmbeddingManifest:
    """Read embedding metadata written during processed embed stage."""
    return EmbeddingManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
