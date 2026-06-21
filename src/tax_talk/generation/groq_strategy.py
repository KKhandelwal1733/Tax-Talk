"""Groq strategy implementation via OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langfuse import observe

from tax_talk.generation.llm_provider import LLMStrategy


class GroqLLMStrategy(LLMStrategy):
    """Groq text generation strategy using OpenAI-compatible chat completions."""

    def __init__(self, client: Any, *, async_client: Any | None = None) -> None:
        self._client = client
        self._async_client = async_client

    @observe(name="generation-groq", as_type="generation", capture_input=True, capture_output=True)
    def generate(self, *, prompt: str, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return content if isinstance(content, str) else ""

    @observe(
        name="generation-groq-stream", as_type="generation", capture_input=True, capture_output=True
    )
    def generate_stream(self, *, prompt: str, model: str) -> Iterator[str]:
        """Stream response chunks from Groq.

        Args:
            prompt: Prompt text to send to Groq.
            model: Groq model name.

        Returns:
            Iterator yielding non-empty delta content chunks.
        """
        stream = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            first_choice = choices[0]
            delta = getattr(first_choice, "delta", None)
            content = getattr(delta, "content", "") if delta is not None else ""
            if isinstance(content, str) and content:
                yield content

    @observe(
        name="generation-groq-async", as_type="generation", capture_input=True, capture_output=True
    )
    async def generate_async(self, *, prompt: str, model: str) -> str:
        if self._async_client is None:
            raise RuntimeError("Groq async client is not configured.")

        response = await self._async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return content if isinstance(content, str) else ""

    @observe(
        name="generation-groq-stream-async",
        as_type="generation",
        capture_input=True,
        capture_output=True,
    )
    async def generate_stream_async(self, *, prompt: str, model: str) -> AsyncIterator[str]:
        if self._async_client is None:
            raise RuntimeError("Groq async client is not configured.")

        stream = await self._async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
        )
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            first_choice = choices[0]
            delta = getattr(first_choice, "delta", None)
            content = getattr(delta, "content", "") if delta is not None else ""
            if isinstance(content, str) and content:
                yield content
