"""LLM strategy interface for provider-specific text generation."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMStrategy(ABC):
    """Provider-agnostic contract for generating text from a prompt."""

    @abstractmethod
    def generate(self, *, prompt: str, model: str) -> str:
        """Generate response text for the given prompt and model."""
