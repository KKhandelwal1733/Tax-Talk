from __future__ import annotations

from types import SimpleNamespace

import pytest

from tax_talk.core import runtime
from tax_talk.core.config import settings
from tax_talk.generation.gemini_strategy import GeminiLLMStrategy
from tax_talk.generation.groq_strategy import GroqLLMStrategy


def _reset_runtime_singletons() -> None:
    runtime._gemini_client = None
    runtime._gemini_otel_instrumented = False
    runtime._groq_client = None
    runtime._groq_async_client = None
    runtime._llm_strategies = {}
    runtime._llm_async_strategies = {}


def test_get_llm_strategy_returns_gemini_singleton(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "contextual_summary_fallback_provider", "gemini")

    class FakeGeminiInstrumentor:
        def instrument(self) -> None:
            return

    class FakeGeminiClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.models = SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(text="generated from gemini"),
                generate_content_stream=lambda **kwargs: [
                    SimpleNamespace(text="generated "),
                    SimpleNamespace(text="from gemini"),
                ],
            )

    monkeypatch.setattr(runtime, "GoogleGenAIInstrumentor", FakeGeminiInstrumentor)
    monkeypatch.setattr(runtime, "genai", SimpleNamespace(Client=FakeGeminiClient))

    strategy1 = runtime.get_llm_strategy("gemini")
    strategy2 = runtime.get_llm_strategy("gemini")

    assert isinstance(strategy1, GeminiLLMStrategy)
    assert strategy1 is strategy2
    assert strategy1.generate(prompt="p", model="m") == "generated from gemini"
    assert "".join(strategy1.generate_stream(prompt="p", model="m")) == "generated from gemini"


def test_get_llm_strategy_returns_groq_singleton(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
    monkeypatch.setattr(settings, "contextual_summary_fallback_provider", "groq")

    class FakeGroqClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key

            def _create(**kwargs):
                if kwargs.get("stream"):
                    return [
                        SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="generated "))]
                        ),
                        SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="from groq"))]
                        ),
                    ]
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="generated from groq"))
                    ]
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    monkeypatch.setattr(runtime, "Groq", FakeGroqClient)

    strategy1 = runtime.get_llm_strategy("groq")
    strategy2 = runtime.get_llm_strategy("groq")

    assert isinstance(strategy1, GroqLLMStrategy)
    assert strategy1 is strategy2
    assert strategy1.generate(prompt="p", model="m") == "generated from groq"
    assert "".join(strategy1.generate_stream(prompt="p", model="m")) == "generated from groq"


def test_get_llm_strategy_rejects_unknown_provider(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "contextual_summary_fallback_provider", "unknown")

    with pytest.raises(RuntimeError, match="Unsupported LLM provider"):
        runtime.get_llm_strategy("unknown")


def test_get_llm_strategy_errors_when_no_provider_keys(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "contextual_summary_fallback_provider", "")

    with pytest.raises(RuntimeError, match="No LLM provider available"):
        runtime.get_llm_strategy(None)


def test_get_llm_strategy_auto_selects_gemini_when_provider_missing(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")

    class FakeGeminiClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.models = SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(text="generated from gemini")
            )

    monkeypatch.setattr(runtime, "genai", SimpleNamespace(Client=FakeGeminiClient))

    strategy = runtime.get_llm_strategy(None)

    assert isinstance(strategy, GeminiLLMStrategy)


def test_get_gemini_client_instruments_once(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")

    instrument_calls: list[str] = []

    class FakeGeminiInstrumentor:
        def instrument(self) -> None:
            instrument_calls.append("instrument")

    class FakeGeminiClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(runtime, "GoogleGenAIInstrumentor", FakeGeminiInstrumentor)
    monkeypatch.setattr(runtime, "genai", SimpleNamespace(Client=FakeGeminiClient))

    client1 = runtime.get_gemini_client()
    client2 = runtime.get_gemini_client()

    assert client1 is client2
    assert instrument_calls == ["instrument"]


@pytest.mark.asyncio
async def test_close_gemini_client_closes_aio_transport(monkeypatch) -> None:
    _reset_runtime_singletons()
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")

    calls: list[str] = []

    class FakeAioClient:
        async def close(self) -> None:
            calls.append("aio_close")

    class FakeGeminiClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.aio = FakeAioClient()

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(runtime, "GoogleGenAIInstrumentor", None)
    monkeypatch.setattr(runtime, "genai", SimpleNamespace(Client=FakeGeminiClient))

    runtime.get_gemini_client()
    await runtime.close_gemini_client()

    assert calls == ["aio_close", "close"]
