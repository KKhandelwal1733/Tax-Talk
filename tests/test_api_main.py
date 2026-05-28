from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tax_talk.api.main import app


class DummyRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        dense_top_k: int,
        bm25_top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "dense_top_k": dense_top_k,
                "bm25_top_k": bm25_top_k,
                "filters": filters,
            }
        )
        return [
            {
                "chunk_id": "c-1",
                "text": "dummy text",
                "fused_score": 0.42,
                "fused_rank": 0,
            }
        ]


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_retrieve_endpoint_uses_retriever_dependency(monkeypatch) -> None:
    dummy = DummyRetriever()
    monkeypatch.setattr("tax_talk.api.main._get_retriever", lambda: dummy)

    response = client.post(
        "/retrieve",
        json={
            "query": "section 54f exemption",
            "top_k": 5,
            "dense_top_k": 7,
            "bm25_top_k": 9,
            "filters": {"act_status": "current"},
        },
    )

    assert response.status_code == 200
    assert response.json()["hits"][0]["chunk_id"] == "c-1"
    assert dummy.calls[0] == {
        "query": "section 54f exemption",
        "top_k": 5,
        "dense_top_k": 7,
        "bm25_top_k": 9,
        "filters": {"act_status": "current"},
    }


def test_retrieve_rejects_blank_query() -> None:
    response = client.post(
        "/retrieve",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 422
