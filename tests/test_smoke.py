"""Smoke tests — these should always pass and break the build if they don't."""

from tax_talk.core.config import settings


def test_config_loads() -> None:
    """Settings should load without error even with empty env."""
    assert settings.env in {"development", "staging", "production", "test"}
    assert settings.max_cost_per_request_usd > 0


def test_imports() -> None:
    """All submodules should be importable."""
    import tax_talk as tax_talk
    import tax_talk.api.main
    import tax_talk.ingestion.run  # noqa: F401
