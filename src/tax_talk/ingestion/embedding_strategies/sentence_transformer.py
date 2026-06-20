"""Sentence-transformer model embedding strategy via HF Inference API."""

from __future__ import annotations

import time
from math import ceil
from threading import Lock, Semaphore
from typing import Any

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.ingestion.embedding_strategies.embedding_strategy import EmbeddingStrategy


def _sanitize_text(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8").replace("\ufffd", " ")


log = get_logger(__name__)


class LocalEmbeddingStrategy(EmbeddingStrategy):
    """Runs sentence-transformer embeddings through HF Inference API."""

    def __init__(self, model_name: str = settings.embedding_model_sentence_transformer) -> None:
        self._usage_lock = Lock()
        self._embed_calls = 0
        self._texts_embedded = 0
        self._estimated_hf_requests = 0
        from huggingface_hub import InferenceClient  # lazy import

        if not settings.hf_token:
            raise ValueError(
                "HF_TOKEN not set in .env - required for sentence_transformer embedding."
            )

        log.info("Using HF Inference API for embedding model: %s", model_name)
        self._client = InferenceClient(token=settings.hf_token)
        self._model_name = model_name
        self._dim = settings.embedding_dimensions
        self._hf_request_limiter = Semaphore(max(1, settings.hf_max_concurrent_requests))
        log.info("HF inference embedder ready. Expected dimensions: %d", self._dim)

    def _normalize_vectors(self, result: Any) -> list[list[float]]:
        if hasattr(result, "tolist"):
            result = result.tolist()

        if not isinstance(result, list) or not result:
            raise ValueError("HF inference returned empty embeddings payload.")

        first = result[0]
        if isinstance(first, (int, float)):  # noqa: UP038
            return [[float(v) for v in result]]

        vectors: list[list[float]] = []
        for row in result:
            if not isinstance(row, list):
                raise ValueError("HF inference returned malformed embedding row.")
            vectors.append([float(v) for v in row])
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        safe_texts: list[str] = []
        coerced = 0
        for text in texts:
            if isinstance(text, str):
                value = _sanitize_text(text)
            elif isinstance(text, bytes):
                value = _sanitize_text(text.decode("utf-8", errors="ignore"))
                coerced += 1
            elif text is None:
                value = ""
                coerced += 1
            else:
                value = _sanitize_text(str(text))
                coerced += 1

            safe_texts.append(value if value else " ")

        if coerced:
            log.warning("Coerced %d non-string local embedding inputs.", coerced)

        batch_size = max(1, settings.embedding_batch_size)
        estimated_requests = ceil(len(safe_texts) / batch_size)

        all_vectors: list[list[float]] = []
        for i in range(0, len(safe_texts), batch_size):
            batch = safe_texts[i : i + batch_size]
            response: Any | None = None
            last_error: Exception | None = None
            max_attempts = max(1, settings.hf_retry_max_attempts)
            for attempt in range(1, max_attempts + 1):
                try:
                    with self._hf_request_limiter:
                        response = self._client.feature_extraction(
                            text=batch,
                            model=self._model_name,
                        )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    retryable_status = {429, 500, 502, 503, 504}
                    should_retry = status_code in retryable_status or status_code is None
                    if not should_retry or attempt >= max_attempts:
                        raise

                    backoff = min(
                        settings.hf_retry_max_delay_seconds,
                        settings.hf_retry_initial_delay_seconds * (2 ** (attempt - 1)),
                    )
                    log.warning(
                        "HF embedding request failed (attempt %d/%d, status=%s). Retrying in %.1fs.",
                        attempt,
                        max_attempts,
                        status_code,
                        backoff,
                    )
                    time.sleep(backoff)

            if response is None:
                raise RuntimeError("HF embedding request returned no response") from last_error
            vectors = self._normalize_vectors(response)
            all_vectors.extend(vectors)

        if all_vectors and len(all_vectors[0]) != self._dim:
            raise ValueError(
                f"HF inference embedding dimensions mismatch: got {len(all_vectors[0])}, expected {self._dim}."
            )

        with self._usage_lock:
            self._embed_calls += 1
            self._texts_embedded += len(safe_texts)
            self._estimated_hf_requests += estimated_requests

        return all_vectors

    @property
    def dimensions(self) -> int:
        return self._dim

    def get_usage_stats(self) -> dict[str, int]:
        with self._usage_lock:
            return {
                "embed_calls": self._embed_calls,
                "texts_embedded": self._texts_embedded,
                "estimated_hf_requests": self._estimated_hf_requests,
                "hf_request_batch_size": max(1, settings.embedding_batch_size),
            }

    def reset_usage_stats(self) -> None:
        with self._usage_lock:
            self._embed_calls = 0
            self._texts_embedded = 0
            self._estimated_hf_requests = 0
