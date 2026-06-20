"""Service layer for chat answers and streaming responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langfuse import observe
from tax_talk.api.schemas.chat import ChatRequest, ChatResponse, ChatStreamEvent
from tax_talk.api.services.prompt_builder import build_chat_prompt
from tax_talk.core.config import settings
from tax_talk.core.runtime import get_llm_strategy_async
from tax_talk.retrieval import HybridRetriever


class ChatService:
    """Coordinates retrieval and generation for chat endpoints."""

    def __init__(
        self, retriever: HybridRetriever | None = None, *, provider: str | None = None
    ) -> None:
        self._retriever = retriever or HybridRetriever()
        self._provider = provider

    @observe(name="api-chat-answer", as_type="span", capture_input=True, capture_output=True)
    async def answer(
        self,
        request: ChatRequest,
        *,
        current_user: dict[str, Any],
    ) -> ChatResponse:
        """Return a full chat answer with retrieval-backed citations.

        Args:
                request: Chat request payload.
                current_user: Authenticated user payload.

        Returns:
                ChatResponse with final answer text and citations.
        """
        _ = current_user
        hits = await self._retriever.retrieve_async(
            request.query,
            top_k=request.top_k,
            dense_top_k=request.dense_top_k,
            bm25_top_k=request.bm25_top_k,
        )
        prompt = build_chat_prompt(query=request.query, hits=hits)
        strategy = get_llm_strategy_async(self._provider)
        answer = await strategy.generate_async(prompt=prompt, model=settings.chat_model)

        return ChatResponse(answer=answer, citations=hits[: request.top_k])

    @observe(name="api-chat-stream", as_type="span", capture_input=True, capture_output=False)
    async def stream_answer(
        self,
        request: ChatRequest,
        *,
        current_user: dict[str, Any],
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream chat answer tokens and a final citations event.

        Args:
                request: Chat request payload.
                current_user: Authenticated user payload.

        Returns:
                AsyncIterator over stream events.
        """
        _ = current_user
        hits = await self._retriever.retrieve_async(
            request.query,
            top_k=request.top_k,
            dense_top_k=request.dense_top_k,
            bm25_top_k=request.bm25_top_k,
        )
        prompt = build_chat_prompt(query=request.query, hits=hits)
        strategy = get_llm_strategy_async(self._provider)

        async for token in strategy.generate_stream_async(prompt=prompt, model=settings.chat_model):
            yield ChatStreamEvent(event="token", text=token)

        yield ChatStreamEvent(event="done", citations=hits[: request.top_k])
