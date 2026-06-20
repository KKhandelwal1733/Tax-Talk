"""Chat service orchestration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from tax_talk.api.schemas.chat import ChatRequest
from tax_talk.api.services.chat_services import ChatService
from tax_talk.core.config import settings


class DummyRetriever:
    def __init__(self) -> None:
        self.last_query = ""

    async def retrieve_async(
        self,
        query: str,
        *,
        top_k: int,
        dense_top_k: int,
        bm25_top_k: int,
    ) -> list[dict[str, Any]]:
        _ = top_k
        _ = dense_top_k
        _ = bm25_top_k
        self.last_query = query
        return [{"chunk_id": "c-1", "text": "section text", "source_key": "cgst", "section_title": "s1"}]


class DummyLLMStrategy:
    def __init__(self, *, rewrite_result: str = "rewritten query") -> None:
        self.rewrite_result = rewrite_result

    async def generate_async(self, *, prompt: str, model: str) -> str:
        _ = model
        if "Rewritten query:" in prompt:
            return self.rewrite_result
        if "Return JSON only with keys: verdict, score, rationale." in prompt:
            return '{"verdict":"supported","score":0.9,"rationale":"grounded"}'
        return "final grounded answer"

    async def generate_stream_async(self, *, prompt: str, model: str) -> AsyncIterator[str]:
        _ = prompt
        _ = model
        yield "final "
        yield "grounded "
        yield "answer"


@pytest.mark.asyncio
async def test_answer_rewrites_query_before_retrieval(monkeypatch) -> None:
    retriever = DummyRetriever()
    service = ChatService(retriever=retriever)

    monkeypatch.setattr(
        "tax_talk.api.services.chat_services.get_llm_strategy_async",
        lambda provider=None: DummyLLMStrategy(rewrite_result="rewritten 54f query"),
    )
    monkeypatch.setattr(settings, "faithfulness_check_enabled", False)

    response = await service.answer(
        ChatRequest(query="section 54f exemption", top_k=3, dense_top_k=6, bm25_top_k=6),
        current_user={"sub": "u1"},
    )

    assert retriever.last_query == "rewritten 54f query"
    assert response.answer == "final grounded answer"
    assert response.faithfulness is None


@pytest.mark.asyncio
async def test_answer_uses_original_query_when_rewrite_empty(monkeypatch) -> None:
    retriever = DummyRetriever()
    service = ChatService(retriever=retriever)

    monkeypatch.setattr(
        "tax_talk.api.services.chat_services.get_llm_strategy_async",
        lambda provider=None: DummyLLMStrategy(rewrite_result="   "),
    )
    monkeypatch.setattr(settings, "faithfulness_check_enabled", False)

    request = ChatRequest(query="original query", top_k=3, dense_top_k=6, bm25_top_k=6)
    _ = await service.answer(request, current_user={"sub": "u1"})

    assert retriever.last_query == "original query"


@pytest.mark.asyncio
async def test_answer_includes_faithfulness_when_enabled(monkeypatch) -> None:
    retriever = DummyRetriever()
    service = ChatService(retriever=retriever)

    monkeypatch.setattr(
        "tax_talk.api.services.chat_services.get_llm_strategy_async",
        lambda provider=None: DummyLLMStrategy(),
    )
    monkeypatch.setattr(settings, "faithfulness_check_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_check_provider", "")
    monkeypatch.setattr(settings, "faithfulness_check_model", "gemini-3.1-flash-lite")

    response = await service.answer(
        ChatRequest(query="section 54f exemption", top_k=3, dense_top_k=6, bm25_top_k=6),
        current_user={"sub": "u1"},
    )

    assert response.faithfulness is not None
    assert response.faithfulness["verdict"] == "supported"


@pytest.mark.asyncio
async def test_stream_emits_faithfulness_terminal_event(monkeypatch) -> None:
    retriever = DummyRetriever()
    service = ChatService(retriever=retriever)

    monkeypatch.setattr(
        "tax_talk.api.services.chat_services.get_llm_strategy_async",
        lambda provider=None: DummyLLMStrategy(),
    )
    monkeypatch.setattr(settings, "faithfulness_check_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_check_provider", "")
    monkeypatch.setattr(settings, "faithfulness_check_model", "gemini-3.1-flash-lite")

    events = [
        event
        async for event in service.stream_answer(
            ChatRequest(query="stream this", top_k=3, dense_top_k=6, bm25_top_k=6),
            current_user={"sub": "u1"},
        )
    ]

    assert events[-1].event == "faithfulness"
    assert events[-1].faithfulness is not None
    assert retriever.last_query == "rewritten query"
