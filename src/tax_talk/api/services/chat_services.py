"""Service layer for chat answers and streaming responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from langfuse import observe
from tax_talk.api.schemas.chat import ChatRequest, ChatResponse, ChatStreamEvent
from tax_talk.api.services.prompt_builder import (
	build_chat_prompt,
	build_faithfulness_prompt,
	build_query_rewrite_prompt,
)
from tax_talk.core.config import settings
from tax_talk.core.runtime import get_llm_strategy_async, get_logger
from tax_talk.retrieval import HybridRetriever

log = get_logger(__name__)


class ChatService:
	"""Coordinates retrieval and generation for chat endpoints."""

	def __init__(self, retriever: HybridRetriever | None = None, *, provider: str | None = None) -> None:
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
		rewritten_query = await self._rewrite_query_async(query=request.query)
		hits = await self._retriever.retrieve_async(
			rewritten_query,
			top_k=request.top_k,
			dense_top_k=request.dense_top_k,
			bm25_top_k=request.bm25_top_k,
		)
		prompt = build_chat_prompt(query=rewritten_query, hits=hits)
		strategy = get_llm_strategy_async(self._provider)
		answer = await strategy.generate_async(prompt=prompt, model=settings.chat_model)
		faithfulness = await self._check_faithfulness_async(
			question=request.query,
			answer=answer,
			hits=hits,
		)

		return ChatResponse(answer=answer, citations=hits[: request.top_k], faithfulness=faithfulness)

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
		rewritten_query = await self._rewrite_query_async(query=request.query)
		hits = await self._retriever.retrieve_async(
			rewritten_query,
			top_k=request.top_k,
			dense_top_k=request.dense_top_k,
			bm25_top_k=request.bm25_top_k,
		)
		prompt = build_chat_prompt(query=rewritten_query, hits=hits)
		strategy = get_llm_strategy_async(self._provider)
		generated_tokens: list[str] = []

		async for token in strategy.generate_stream_async(prompt=prompt, model=settings.chat_model):
			generated_tokens.append(token)
			yield ChatStreamEvent(event="token", text=token)

		yield ChatStreamEvent(event="done", citations=hits[: request.top_k])

		faithfulness = await self._check_faithfulness_async(
			question=request.query,
			answer="".join(generated_tokens).strip(),
			hits=hits,
		)
		if faithfulness is not None:
			yield ChatStreamEvent(event="faithfulness", faithfulness=faithfulness)

	@observe(name="api-query-rewrite", as_type="generation", capture_input=False, capture_output=False)
	async def _rewrite_query_async(self, *, query: str) -> str:
		"""Rewrite user query for retrieval while preserving legal semantics.

		Args:
			query: Original user query.

		Returns:
			Rewritten query, or original query on failure/empty output.
		"""
		prompt = build_query_rewrite_prompt(query=query)
		strategy = get_llm_strategy_async(self._provider)
		try:
			rewritten = await strategy.generate_async(prompt=prompt, model=settings.chat_model)
		except RuntimeError as exc:
			log.warning("Query rewrite failed; using original query. %s", exc)
			return query

		rewritten_query = rewritten.strip() if isinstance(rewritten, str) else ""
		return rewritten_query or query

	@observe(
		name="api-faithfulness-check",
		as_type="generation",
		capture_input=False,
		capture_output=False,
	)
	async def _check_faithfulness_async(
		self,
		*,
		question: str,
		answer: str,
		hits: list[dict[str, Any]],
	) -> dict[str, Any] | None:
		"""Run optional answer-grounding check using retrieved context.

		Args:
			question: User question.
			answer: Generated answer text.
			hits: Retrieved context hits.

		Returns:
			Faithfulness payload or None when disabled/unavailable.
		"""
		if not settings.faithfulness_check_enabled:
			return None

		provider = settings.faithfulness_check_provider.strip() or (self._provider or "")
		model = settings.faithfulness_check_model.strip() or settings.chat_model
		strategy = get_llm_strategy_async(provider or None)
		prompt = build_faithfulness_prompt(question=question, answer=answer, hits=hits)

		try:
			result = await strategy.generate_async(prompt=prompt, model=model)
		except RuntimeError as exc:
			log.warning("Faithfulness check failed; continuing without score. %s", exc)
			return None

		if not isinstance(result, str):
			return None
		try:
			parsed = json.loads(result)
		except json.JSONDecodeError:
			return {
				"verdict": "unknown",
				"score": None,
				"rationale": result.strip()[:300],
			}

		if not isinstance(parsed, dict):
			return None
		return parsed
