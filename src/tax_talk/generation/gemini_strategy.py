"""Gemini strategy implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langfuse import observe

from tax_talk.generation.llm_provider import LLMStrategy


class GeminiLLMStrategy(LLMStrategy):
    """Google Gemini text generation strategy."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._aio_client: Any | None = None

    def _get_aio_client(self) -> Any:
        """Return the SDK async Gemini client accessor if available."""
        if self._aio_client is not None:
            return self._aio_client

        aio_client = getattr(self._client, "aio", None)
        if aio_client is None:
            raise RuntimeError("Gemini async client is unavailable on the configured SDK.")

        self._aio_client = aio_client
        return self._aio_client

    @observe(
        name="generation-gemini", as_type="generation", capture_input=True, capture_output=True
    )
    def generate(self, *, prompt: str, model: str) -> str:
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(response, "text", "")
        return text if isinstance(text, str) else ""

    @observe(
        name="generation-gemini-stream",
        as_type="generation",
        capture_input=True,
        capture_output=True,
    )
    def generate_stream(self, *, prompt: str, model: str) -> Iterator[str]:
        """Stream response chunks from Gemini.

        Args:
            prompt: Prompt text to send to Gemini.
            model: Gemini model name.

        Returns:
            Iterator yielding non-empty text chunks as they arrive.
        """
        response = self._client.models.generate_content_stream(
            model=model,
            contents=[prompt],
        )
        for chunk in response:
            text = getattr(chunk, "text", "")
            if isinstance(text, str) and text:
                yield text

    @observe(
        name="generation-gemini-async",
        as_type="generation",
        capture_input=True,
        capture_output=True,
    )
    async def generate_async(self, *, prompt: str, model: str) -> str:
        aio_client = self._get_aio_client()

        response = await aio_client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(response, "text", "")
        return text if isinstance(text, str) else ""

    @observe(
        name="generation-gemini-stream-async",
        as_type="generation",
        capture_input=True,
        capture_output=True,
    )
    async def generate_stream_async(self, *, prompt: str, model: str) -> AsyncIterator[str]:
        aio_client = self._get_aio_client()

        stream = await aio_client.models.generate_content_stream(
            model=model,
            contents=[prompt],
        )
        async for chunk in stream:
            text = getattr(chunk, "text", "")
            if isinstance(text, str) and text:
                yield text
