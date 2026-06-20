from __future__ import annotations

import os
from typing import Any

import pytest

from tax_talk.core import config, runtime
from tax_talk.ingestion import embeddings


class DummyLangfuseClient:
    def __init__(self) -> None:
        self.flush_called = False

    def flush(self) -> None:
        self.flush_called = True

    def __repr__(self) -> str:
        return "<DummyLangfuseClient>"


def dummy_get_client(*args: Any, **kwargs: Any) -> DummyLangfuseClient:
    return DummyLangfuseClient()


@pytest.fixture(autouse=True)
def reset_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Langfuse singleton state is reset for each test."""
    monkeypatch.setattr(runtime, "_langfuse_client", None)


def test_langfuse_settings_are_loaded_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-demo")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-demo")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    test_settings = config.Settings()
    monkeypatch.setattr(runtime, "settings", test_settings)
    monkeypatch.setattr(runtime, "get_client", dummy_get_client)

    client = runtime.get_langfuse_client()

    assert isinstance(client, DummyLangfuseClient)
    assert runtime._langfuse_client is client
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-demo"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-demo"
    assert os.environ["LANGFUSE_HOST"] == "https://cloud.langfuse.com"


def test_langfuse_settings_are_loaded_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    test_settings = config.Settings(
        langfuse_public_key="pk-demo",
        langfuse_secret_key="sk-demo",
        langfuse_host="https://cloud.langfuse.com",
    )
    monkeypatch.setattr(runtime, "settings", test_settings)
    monkeypatch.setattr(runtime, "get_client", dummy_get_client)

    client = runtime.get_langfuse_client()

    assert isinstance(client, DummyLangfuseClient)
    assert runtime._langfuse_client is client
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-demo"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-demo"
    assert os.environ["LANGFUSE_HOST"] == "https://cloud.langfuse.com"


def test_normalize_embedding_inputs_replaces_invalid_unicode() -> None:
    surrogate = "\ud835"
    normalized = embeddings._normalize_embedding_inputs([f"bad{surrogate}text"])

    assert surrogate not in normalized[0]
    assert normalized[0] == "bad text"


class DummyLangfuseClientNoGeneration:
    def __repr__(self) -> str:
        return "<DummyLangfuseClientNoGeneration>"


def test_track_embedding_usage_handles_missing_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embeddings,
        "settings",
        config.Settings(langfuse_public_key="pk-demo", langfuse_secret_key="sk-demo"),
    )
    monkeypatch.setattr(runtime, "get_langfuse_client", lambda: DummyLangfuseClientNoGeneration())

    embeddings._track_embedding_usage(
        operation="texts",
        text_count=1,
        before_stats={},
        after_stats={},
    )


class DummyLangfuseClientWithObservation:
    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []

    def start_as_current_observation(self, as_type: str, name: str):
        owner = self

        class DummyContext:
            def __enter__(self):
                return owner

            def __exit__(self, exc_type, exc, tb):
                return False

        self.current = {"as_type": as_type, "name": name, "updates": []}
        return DummyContext()

    def update(self, **kwargs: object) -> None:
        self.current["updates"].append(kwargs)
        self.observations.append(self.current)

    def __repr__(self) -> str:
        return "<DummyLangfuseClientWithObservation>"


def test_demo_langfuse_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    langfuse = pytest.importorskip("langfuse")

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setattr(langfuse, "get_client", dummy_get_client)
    monkeypatch.setattr(runtime, "_langfuse_client", None)
    monkeypatch.setattr(runtime, "settings", config.Settings())

    @langfuse.observe(name="demo-observer")
    def demo_observed(value: int) -> int:
        return value * 2

    assert demo_observed(3) == 6


def test_flush_langfuse_client_flushes_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyLangfuseClient()
    monkeypatch.setattr(runtime, "_langfuse_client", client)

    runtime.flush_langfuse_client()

    assert client.flush_called is True
