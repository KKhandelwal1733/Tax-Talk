"""Gemini text-embedding strategy."""

from __future__ import annotations

import asyncio
import time

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.ingestion.embedding_strategies.embedding_strategy import EmbeddingStrategy

log = get_logger(__name__)


class GeminiEmbeddingStrategy(EmbeddingStrategy):
    """Uses Google Gemini text-embedding API."""

    _MAX_BATCH = 100

    def __init__(self) -> None:
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=settings.gemini_api_key)
        self._genai = genai
        self._model = settings.embedding_model_gemini
        self._dim = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._MAX_BATCH):
            batch = texts[i : i + self._MAX_BATCH]
            for attempt in range(3):
                try:
                    result = self._genai.embed_content(
                        model=self._model,
                        content=batch,
                        task_type="retrieval_document",
                    )
                    embeddings = result["embedding"]
                    if isinstance(embeddings[0], float):
                        embeddings = [embeddings]
                    all_vectors.extend(embeddings)
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        wait = 60 * (attempt + 1)
                        log.warning("Gemini rate limit hit. Sleeping %ds...", wait)
                        time.sleep(wait)
                    else:
                        log.error("Gemini embedding error: %s", e)
                        raise

            time.sleep(0.05)

        return all_vectors

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async embedding using thread pool to avoid blocking."""
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._MAX_BATCH):
            batch = texts[i : i + self._MAX_BATCH]
            for attempt in range(3):
                try:
                    result = await asyncio.to_thread(
                        self._genai.embed_content,
                        model=self._model,
                        content=batch,
                        task_type="retrieval_document",
                    )
                    embeddings = result["embedding"]
                    if isinstance(embeddings[0], float):
                        embeddings = [embeddings]
                    all_vectors.extend(embeddings)
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        wait = 60 * (attempt + 1)
                        log.warning("Gemini rate limit hit. Sleeping %ds...", wait)
                        await asyncio.sleep(wait)
                    else:
                        log.error("Gemini embedding error: %s", e)
                        raise

            await asyncio.sleep(0.05)

        return all_vectors

    @property
    def dimensions(self) -> int:
        return self._dim
