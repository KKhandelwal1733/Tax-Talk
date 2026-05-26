"""Local sentence-transformers embedding strategy."""

from __future__ import annotations

from math import ceil
from threading import Lock
from typing import Any

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.ingestion.embedding_strategies.embedding_strategy import EmbeddingStrategy


def _sanitize_text(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8").replace("\ufffd", " ")

log = get_logger(__name__)


class LocalEmbeddingStrategy(EmbeddingStrategy):
    """Runs sentence-transformers locally or through HF Inference API."""

    def __init__(self, model_name: str = settings.embedding_model_local) -> None:
        self._mode = settings.embedding_local_mode.lower().strip()
        self._usage_lock = Lock()
        self._embed_calls = 0
        self._texts_embedded = 0
        self._estimated_hf_requests = 0

        if self._mode == "hf_inference":
            from huggingface_hub import InferenceClient  # lazy import

            if not settings.hf_token:
                raise ValueError("HF_TOKEN not set in .env - required for EMBEDDING_LOCAL_MODE=hf_inference.")

            log.info("Using HF Inference API for embedding model: %s", model_name)
            self._client = InferenceClient(token=settings.hf_token)
            self._model_name = model_name
            self._dim = settings.embedding_dimensions
            log.info("HF inference embedder ready. Expected dimensions: %d", self._dim)
            return

        if self._mode != "local":
            raise ValueError(
                "Invalid EMBEDDING_LOCAL_MODE: "
                f"'{settings.embedding_local_mode}'. Choose: local | hf_inference"
            )

        from sentence_transformers import SentenceTransformer  # lazy import

        log.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise ValueError(f"Embedding model '{model_name}' did not report output dimensions.")
        self._dim = int(dim)
        log.info("Model loaded. Dimensions: %d", self._dim)

    def _normalize_vectors(self, result: Any) -> list[list[float]]:
        if hasattr(result, "tolist"):
            result = result.tolist()

        if not isinstance(result, list) or not result:
            raise ValueError("HF inference returned empty embeddings payload.")

        first = result[0]
        if isinstance(first, (int, float)):
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

        if self._mode == "hf_inference":
            all_vectors: list[list[float]] = []
            for i in range(0, len(safe_texts), batch_size):
                batch = safe_texts[i : i + batch_size]
                response = self._client.feature_extraction(
                    text=batch,
                    model=self._model_name,
                )
                vectors = self._normalize_vectors(response)
                all_vectors.extend(vectors)

            if all_vectors and len(all_vectors[0]) != self._dim:
                raise ValueError(
                    f"HF inference embedding dimensions mismatch: got {len(all_vectors[0])}, expected {self._dim}."
                )
        else:
            vectors = self._model.encode(
                safe_texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_vectors = vectors.tolist()

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
