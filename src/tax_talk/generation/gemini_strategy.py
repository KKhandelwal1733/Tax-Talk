"""Gemini strategy implementation."""

from __future__ import annotations

from typing import Any

from langfuse import observe

from tax_talk.generation.llm_provider import LLMStrategy


class GeminiLLMStrategy(LLMStrategy):
    """Google Gemini text generation strategy."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @observe(name="generation-gemini", as_type="generation", capture_input=True, capture_output=True)
    def generate(self, *, prompt: str, model: str) -> str:
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(response, "text", "")
        return text if isinstance(text, str) else ""
