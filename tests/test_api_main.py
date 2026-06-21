"""API endpoint tests with mocked JWT verification."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from tax_talk.api.dependencies.services import get_chat_service
from tax_talk.api.main import app, create_app
from tax_talk.api.schemas.chat import ChatStreamEvent


class DummyChatService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def answer(self, request, *, current_user: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"query": request.query, "user": current_user.get("user_id")})
        return {
            "answer": "dummy answer",
            "citations": [{"chunk_id": "c-1", "text": "dummy text"}],
        }

    async def stream_answer(
        self, request, *, current_user: dict[str, Any]
    ) -> AsyncIterator[ChatStreamEvent]:
        _ = request
        _ = current_user
        yield ChatStreamEvent(event="token", text="dummy ")
        yield ChatStreamEvent(event="token", text="stream")
        yield ChatStreamEvent(event="done", citations=[{"chunk_id": "c-1"}])


class DummyQdrantAsyncClient:
    async def get_collections(self) -> dict[str, Any]:
        return {"collections": []}


client = TestClient(app)


def test_health_live_endpoint_returns_ok() -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_endpoint_returns_ok(monkeypatch) -> None:
    monkeypatch.setattr(
        "tax_talk.api.endpoints.health.get_async_qdrant_client",
        lambda: DummyQdrantAsyncClient(),
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_bearer_token() -> None:
    response = client.post("/chat", json={"query": "section 54f exemption"})
    assert response.status_code == 401


def test_chat_endpoint_uses_chat_service_dependency(monkeypatch) -> None:
    dummy = DummyChatService()
    app.dependency_overrides[get_chat_service] = lambda: dummy

    payload = {
        "sub": "user-1",
        "email": "user@example.com",
        "role": "authenticated",
        "exp": 9999999999,
    }

    def mock_verify_token(token: str) -> dict:
        return payload

    monkeypatch.setattr(
        "tax_talk.api.dependencies.auth.verify_supabase_token",
        mock_verify_token,
    )

    try:
        response = client.post(
            "/chat",
            json={
                "query": "section 54f exemption",
                "top_k": 4,
            },
            headers={"Authorization": "Bearer fake-token"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "dummy answer"
    assert dummy.calls[0]["query"] == "section 54f exemption"
    assert dummy.calls[0]["user"] == "user-1"


def test_chat_stream_returns_event_stream(monkeypatch) -> None:
    dummy = DummyChatService()
    app.dependency_overrides[get_chat_service] = lambda: dummy

    payload = {
        "sub": "user-1",
        "email": "user@example.com",
        "role": "authenticated",
        "exp": 9999999999,
    }

    def mock_verify_token(token: str) -> dict:
        return payload

    monkeypatch.setattr(
        "tax_talk.api.dependencies.auth.verify_supabase_token",
        mock_verify_token,
    )

    try:
        with client.stream(
            "POST",
            "/chat/stream",
            json={"query": "stream this"},
            headers={"Authorization": "Bearer fake-token"},
        ) as response:
            body = "".join(chunk for chunk in response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "event: token" in body
    assert "event: done" in body
    assert "id: 1" in body
    assert "id: 3" in body


def test_chat_rejects_expired_bearer_token(monkeypatch) -> None:
    from fastapi import HTTPException, status

    def mock_verify_expired(token: str) -> dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )

    monkeypatch.setattr(
        "tax_talk.api.dependencies.auth.verify_supabase_token",
        mock_verify_expired,
    )

    response = client.post(
        "/chat",
        json={"query": "section 54f exemption"},
        headers={"Authorization": "Bearer expired-token"},
    )

    assert response.status_code == 401


def test_lifespan_runs_flush_and_close(monkeypatch) -> None:
    calls: list[str] = []

    async def _fake_close_gemini_client() -> None:
        calls.append("close_gemini")

    def _fake_flush_langfuse_client() -> None:
        calls.append("flush_langfuse")

    monkeypatch.setattr("tax_talk.api.main.close_gemini_client", _fake_close_gemini_client)
    monkeypatch.setattr("tax_talk.api.main.flush_langfuse_client", _fake_flush_langfuse_client)
    monkeypatch.setattr("tax_talk.api.main.get_qdrant_client", lambda: object())

    with TestClient(create_app()) as local_client:
        response = local_client.get("/health/live")
        assert response.status_code == 200

    assert "flush_langfuse" in calls
    assert "close_gemini" in calls


def test_chat_rejects_blank_query(monkeypatch) -> None:
    payload = {
        "sub": "user-1",
        "email": "user@example.com",
        "role": "authenticated",
        "exp": 9999999999,
    }

    def mock_verify_token(token: str) -> dict:
        return payload

    monkeypatch.setattr(
        "tax_talk.api.dependencies.auth.verify_supabase_token",
        mock_verify_token,
    )

    response = client.post(
        "/chat",
        json={"query": "   "},
        headers={"Authorization": "Bearer fake-token"},
    )

    assert response.status_code == 422
