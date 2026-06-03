from __future__ import annotations

import pytest

from tax_talk.evals import runner
from tax_talk.evals.runner import _parse_fallback_chain

# ---------------------------------------------------------------------------
# Dummy helpers
# ---------------------------------------------------------------------------


class DummyStrategy:
    """Configurable dummy LLM strategy for testing fallback behaviour."""

    def __init__(self, *, response: str = "answer", raises: Exception | None = None) -> None:
        self.calls: int = 0
        self._response = response
        self._raises = raises

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._response


class DummyLimiter:
    """Dummy rate limiter that records how many times it was called."""

    def __init__(self) -> None:
        self.calls: int = 0

    def wait_for_slot(self) -> None:
        self.calls += 1


# ---------------------------------------------------------------------------
# Phase 2 — Parser tests
# ---------------------------------------------------------------------------


def test_parse_fallback_chain_empty_string_returns_empty_list() -> None:
    assert _parse_fallback_chain("") == []


def test_parse_fallback_chain_whitespace_only_returns_empty_list() -> None:
    assert _parse_fallback_chain("   ") == []


def test_parse_fallback_chain_single_entry() -> None:
    result = _parse_fallback_chain("groq/llama3-70b")
    assert result == [("groq", "llama3-70b")]


def test_parse_fallback_chain_multiple_entries_preserves_order() -> None:
    result = _parse_fallback_chain("groq/llama3-70b,gemini/gemini-2.0-flash-lite")
    assert result == [("groq", "llama3-70b"), ("gemini", "gemini-2.0-flash-lite")]


def test_parse_fallback_chain_strips_whitespace() -> None:
    result = _parse_fallback_chain(" groq / llama3-70b , gemini / gemini-2.0-flash-lite ")
    assert result == [("groq", "llama3-70b"), ("gemini", "gemini-2.0-flash-lite")]


def test_parse_fallback_chain_invalid_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match="expected 'provider/model' format"):
        _parse_fallback_chain("groq-llama3-70b")


def test_parse_fallback_chain_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        _parse_fallback_chain("openai/gpt-4o")


def test_parse_fallback_chain_empty_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Empty model name"):
        _parse_fallback_chain("gemini/")


# ---------------------------------------------------------------------------
# Phase 3 — Fallback execution tests
# ---------------------------------------------------------------------------


def test_generate_answer_primary_succeeds_no_fallback_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = DummyStrategy(response="primary answer")
    fallback = DummyStrategy(response="fallback answer")
    limiter = DummyLimiter()

    monkeypatch.setattr(runner, "_eval_answer_rate_limiter", limiter)
    monkeypatch.setattr(runner, "get_llm_strategy", lambda _prov: primary)

    answer = runner._generate_grounded_answer(
        question="Q",
        contexts=["ctx"],
        provider="gemini",
        model="gemini-3.5-flash",
        fallback_chain=[("groq", "llama3-70b")],
    )

    assert answer == "primary answer"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert limiter.calls == 1


def test_generate_answer_primary_fails_fallback_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = DummyStrategy(raises=RuntimeError("rate limited"))
    fallback = DummyStrategy(response="fallback answer")
    limiter = DummyLimiter()

    def fake_get_strategy(prov: str) -> DummyStrategy:
        return primary if prov == "gemini" else fallback

    monkeypatch.setattr(runner, "_eval_answer_rate_limiter", limiter)
    monkeypatch.setattr(runner, "get_llm_strategy", fake_get_strategy)

    answer = runner._generate_grounded_answer(
        question="Q",
        contexts=["ctx"],
        provider="gemini",
        model="gemini-3.5-flash",
        fallback_chain=[("groq", "llama3-70b")],
    )

    assert answer == "fallback answer"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert limiter.calls == 2  # one per attempt


def test_generate_answer_all_fail_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = DummyStrategy(raises=RuntimeError("unavailable"))
    limiter = DummyLimiter()

    monkeypatch.setattr(runner, "_eval_answer_rate_limiter", limiter)
    monkeypatch.setattr(runner, "get_llm_strategy", lambda _prov: failing)

    with pytest.raises(RuntimeError, match="All 2 eval generation attempt"):
        runner._generate_grounded_answer(
            question="Q",
            contexts=["ctx"],
            provider="gemini",
            model="gemini-3.5-flash",
            fallback_chain=[("groq", "llama3-70b")],
        )

    assert failing.calls == 2
    assert limiter.calls == 2


def test_generate_answer_rate_limiter_called_per_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limiter is invoked once per attempt, not once per call."""
    error = RuntimeError("fail")
    strategies = {
        "gemini": DummyStrategy(raises=error),
        "groq": DummyStrategy(raises=error),
    }
    limiter = DummyLimiter()

    monkeypatch.setattr(runner, "_eval_answer_rate_limiter", limiter)
    monkeypatch.setattr(runner, "get_llm_strategy", lambda prov: strategies[prov])

    with pytest.raises(RuntimeError):
        runner._generate_grounded_answer(
            question="Q",
            contexts=["ctx"],
            provider="gemini",
            model="model-a",
            fallback_chain=[("groq", "model-b")],
        )

    assert limiter.calls == 2
