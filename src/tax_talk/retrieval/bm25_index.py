"""BM25 index lifecycle and search behavior for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from rank_bm25 import BM25Okapi

from tax_talk.ingestion.qdrant_store import QdrantStore
from tax_talk.retrieval.helpers.filters import payload_matches_filters
from tax_talk.retrieval.helpers.tokenization import tokenize


@dataclass
class Bm25Doc:
    """Internal BM25 document wrapper with payload and tokens."""

    payload: dict[str, Any]
    tokens: list[str]


class Bm25Searcher:
    """Stateful lazy BM25 index built from Qdrant payloads."""

    def __init__(self, store: QdrantStore, logger: Any) -> None:
        self._store = store
        self._log = logger
        self._lock = Lock()
        self._docs: list[Bm25Doc] = []
        self._index: BM25Okapi | None = None

    def refresh(self) -> None:
        """Force reload BM25 corpus from Qdrant payloads."""
        with self._lock:
            self.load(force=True)

    def load(self, *, force: bool) -> None:
        """Load BM25 index from Qdrant payloads when required."""
        if self._index is not None and not force:
            return

        payloads = self._store.scroll_payloads(page_size=512)

        docs: list[Bm25Doc] = []
        for payload in payloads:
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue

            tokens = tokenize(text)
            if not tokens:
                continue

            docs.append(Bm25Doc(payload=dict(payload), tokens=tokens))

        if not docs:
            self._log.warning(
                "BM25 index not built because no valid payloads were found in Qdrant."
            )
            self._docs = []
            self._index = None
            return

        self._docs = docs
        self._index = BM25Okapi([d.tokens for d in docs])
        self._log.info("BM25 index ready with %d chunk(s).", len(docs))

    def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Search BM25 index and return scored, ranked payload rows."""
        with self._lock:
            self.load(force=False)

        if not self._docs or self._index is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        indexed_scores = list(enumerate(scores))

        if filters:
            indexed_scores = [
                (idx, score)
                for idx, score in indexed_scores
                if payload_matches_filters(self._docs[idx].payload, filters)
            ]

        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        ranked: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k]):
            payload = dict(self._docs[idx].payload)
            chunk_id = str(payload.get("chunk_id", ""))
            if not chunk_id:
                continue
            payload["chunk_id"] = chunk_id
            payload["bm25_score"] = float(score)
            payload["bm25_rank"] = rank
            ranked.append(payload)

        return ranked
