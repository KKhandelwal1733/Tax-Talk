"""Groq strategy implementation via OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from langfuse import observe
from tax_talk.generation.llm_provider import LLMStrategy


class GroqLLMStrategy(LLMStrategy):
    """Groq text generation strategy using OpenAI-compatible chat completions."""

    def __init__(self, client: Any) -> None:
        self._client = client

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
