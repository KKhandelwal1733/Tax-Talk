from __future__ import annotations

from tax_talk.ingestion.embedding_strategies.factory import _get_strategy_spec


def test_sentence_transformer_is_canonical_provider() -> None:
    spec = _get_strategy_spec("sentence_transformer")

    assert spec.display_name == "sentence-transformers"


def test_local_provider_is_rejected() -> None:
    try:
        _get_strategy_spec("local")
    except ValueError as exc:
        assert "Unknown EMBEDDING_PROVIDER" in str(exc)
    else:
        raise AssertionError("Expected local provider to be rejected")
