from __future__ import annotations

import pytest

from tax_talk.ingestion.qdrant_store import QdrantStore
from tax_talk.models.ingestion import Chunk


class DummyQdrantClient:
    def upsert(self, **kwargs) -> None:
        _ = kwargs


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="chunk text",
        source_key="source-a",
        filename="doc.txt",
    )


def test_upsert_chunks_rejects_batch_size_one_for_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tax_talk.ingestion.qdrant_store.get_qdrant_client", lambda: DummyQdrantClient()
    )
    monkeypatch.setattr("tax_talk.ingestion.qdrant_store.get_async_qdrant_client", lambda: object())

    store = QdrantStore(collection_name="demo")

    with pytest.raises(ValueError, match="batch_size=1"):
        store.upsert_chunks(
            chunks=[_make_chunk("c-1"), _make_chunk("c-2")],
            vectors=[[0.1, 0.2], [0.2, 0.3]],
            batch_size=1,
        )
