"""
src/tax_talk/ingestion/qdrant_store.py

Creates the Qdrant collection (if needed) and upserts/searches chunks.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from langfuse import observe
from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger, get_qdrant_client
from tax_talk.models.ingestion import Chunk

log = get_logger(__name__)


class QdrantStore:
    """Thin wrapper around QdrantClient for this project."""

    def __init__(self, collection_name: str | None = None) -> None:
        self._client = get_qdrant_client()
        chosen = (collection_name or settings.qdrant_collection).strip()
        if not chosen:
            raise ValueError("Qdrant collection name must not be blank")
        self.collection = chosen

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self.collection)

    def create_collection_if_not_exists(self) -> None:
        """
        Create the collection if missing.
        If it already exists, validate that dimensions and distance still match.
        """
        dims = settings.embedding_dimensions

        if self.collection_exists():
            self._validate_collection_config(expected_dims=dims)
            log.info("Collection '%s' already exists and matches expected config.", self.collection)
            return

        log.info("Creating collection '%s' with %d dimensions...", self.collection, dims)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
        )
        log.info("Collection created.")

    def _validate_collection_config(self, expected_dims: int) -> None:
        info = self._client.get_collection(self.collection)
        vectors_config = info.config.params.vectors

        if isinstance(vectors_config, dict):
            raise ValueError(
                f"Collection '{self.collection}' uses named vectors, but this project expects one unnamed dense vector."
            )

        actual_dims = getattr(vectors_config, "size", None)
        actual_distance = self._normalize_distance(getattr(vectors_config, "distance", None))

        if actual_dims != expected_dims:
            raise ValueError(
                f"Collection '{self.collection}' dimension mismatch: "
                f"got {actual_dims}, expected {expected_dims}."
            )

        if actual_distance != "cosine":
            raise ValueError(
                f"Collection '{self.collection}' distance mismatch: "
                f"got {actual_distance}, expected cosine."
            )

    @staticmethod
    def _normalize_distance(value: Any) -> str:
        if isinstance(value, Distance):
            return value.name.lower()
        return str(value).split(".")[-1].lower()

    @observe(name="qdrant-upsert-chunks", capture_input=False)
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) must be the same length."
            )

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            points = [
                PointStruct(
                    id=_chunk_id_to_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=chunk.to_qdrant_payload(),
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=False)
            ]

            self._client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )

            total += len(points)
            log.info("Upserted batch %d-%d (%d total so far)", i + 1, i + len(batch_chunks), total)

        return total

    def count(self, exact: bool = True) -> int:
        result = self._client.count(collection_name=self.collection, exact=exact)
        return result.count

    @observe(name="qdrant-search", capture_input=False, capture_output=False)
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            )

        response = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        points = getattr(response, "points", [])
        return [{"score": hit.score, **(hit.payload or {})} for hit in points]

    def scroll_payloads(
        self,
        page_size: int = 512,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read all payloads from the collection, optionally applying equality filters."""
        if page_size <= 0:
            raise ValueError("page_size must be > 0")

        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            )

        payloads: list[dict[str, Any]] = []
        offset: Any = None

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
                limit=page_size,
                offset=offset,
            )

            if not points:
                break

            for point in points:
                payload = dict(point.payload or {})
                if "chunk_id" not in payload and getattr(point, "id", None) is not None:
                    payload["chunk_id"] = str(point.id)
                payloads.append(payload)

            if next_offset is None:
                break

            offset = next_offset

        return payloads


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """
    Convert chunk_id into a stable UUID string for Qdrant point IDs.
    This avoids silent collisions from truncating hash strings.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
