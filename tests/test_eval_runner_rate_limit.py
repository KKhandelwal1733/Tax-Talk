from __future__ import annotations

from tax_talk.evals import runner


class _FakeLimiter:
    def __init__(self) -> None:
        self.calls = 0

    def wait_for_slot(self) -> None:
        self.calls += 1


class _FakeStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls += 1
        return " ok "


def test_generate_grounded_answer_waits_for_rate_limiter(monkeypatch) -> None:
    fake_limiter = _FakeLimiter()
    fake_strategy = _FakeStrategy()

    monkeypatch.setattr(runner, "_eval_answer_rate_limiter", fake_limiter)
    monkeypatch.setattr(runner, "get_llm_strategy", lambda provider: fake_strategy)

    answer = runner._generate_grounded_answer(
        question="What is GST?",
        contexts=["GST applies to supply."],
        provider="gemini",
        model="gemini-3.5-flash",
    )

    assert fake_limiter.calls == 1
    assert fake_strategy.calls == 1
    assert answer == "ok"
