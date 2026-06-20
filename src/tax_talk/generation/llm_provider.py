"""LLM strategy interface for provider-specific text generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator


class LLMStrategy(ABC):
    """Provider-agnostic contract for generating text from a prompt."""

    @abstractmethod
    def generate(self, *, prompt: str, model: str) -> str:
        """Generate response text for the given prompt and model."""

    @abstractmethod
    def generate_stream(self, *, prompt: str, model: str) -> Iterator[str]:
        """Stream response text chunks for the given prompt and model.

        Args:
            prompt: Prompt text to send to the provider.
            model: Provider model name to use.

        Returns:
            Iterator yielding response text chunks in generation order.
        """

    @abstractmethod
    async def generate_async(self, *, prompt: str, model: str) -> str:
        """Generate response text asynchronously for the given prompt and model."""

    @abstractmethod
    async def generate_stream_async(self, *, prompt: str, model: str) -> AsyncIterator[str]:
        """Stream response text chunks asynchronously for the given prompt and model.

        Args:
            prompt: Prompt text to send to the provider.
            model: Provider model name to use.

        Returns:
            AsyncIterator yielding response text chunks in generation order.
        """
