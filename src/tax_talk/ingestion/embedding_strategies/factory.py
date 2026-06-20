"""Strategy selection and singleton lifecycle for embeddings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.ingestion.embedding_strategies.embedding_strategy import EmbeddingStrategy
from tax_talk.ingestion.embedding_strategies.gemini import GeminiEmbeddingStrategy
from tax_talk.ingestion.embedding_strategies.sentence_transformer import LocalEmbeddingStrategy

# from tax_talk.ingestion.embedding_strategies.voyage import VoyageEmbeddingStrategy

log = get_logger(__name__)

StrategyFactory = Callable[[], EmbeddingStrategy]


@dataclass(frozen=True)
class StrategySpec:
    display_name: str
    factory: StrategyFactory
    model_setting_attr: str
    required_key_attr: str | None = None
    required_key_env_name: str | None = None


_STRATEGY_SPECS: dict[str, StrategySpec] = {
    "sentence_transformer": StrategySpec(
        display_name="sentence-transformers",
        factory=LocalEmbeddingStrategy,
        model_setting_attr="embedding_model_sentence_transformer",
    ),
    "gemini": StrategySpec(
        display_name="Gemini",
        factory=GeminiEmbeddingStrategy,
        model_setting_attr="embedding_model_gemini",
        required_key_attr="gemini_api_key",
        required_key_env_name="GEMINI_API_KEY",
    ),
    #     "voyage": StrategySpec(
    #         display_name="Voyage AI",
    #         factory=VoyageEmbeddingStrategy,
    #         model_setting_attr="embedding_model_voyage",
    #         required_key_attr="voyage_api_key",
    #         required_key_env_name="VOYAGE_API_KEY",
    #     ),
}

_strategy_lock = Lock()
_strategy_cache: EmbeddingStrategy | None = None


def _get_strategy_spec(provider: str) -> StrategySpec:
    spec = _STRATEGY_SPECS.get(provider)
    if spec is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
            "Choose: sentence_transformer | gemini | voyage"
        )
    return spec


def _validate_strategy_requirements(spec: StrategySpec) -> None:
    if spec.required_key_attr is None:
        return

    key_value = getattr(settings, spec.required_key_attr)
    if key_value:
        return

    env_name = spec.required_key_env_name or spec.required_key_attr.upper()
    raise ValueError(f"{env_name} not set in .env - required for {spec.display_name} embeddings.")


def _build_strategy(provider: str) -> EmbeddingStrategy:
    spec = _get_strategy_spec(provider)
    _validate_strategy_requirements(spec)

    model_name = getattr(settings, spec.model_setting_attr)
    log.info("Using %s embedder: %s", spec.display_name, model_name)

    return spec.factory()


def get_embedding_strategy() -> EmbeddingStrategy:
    """Return singleton strategy for configured embedding provider."""
    global _strategy_cache

    if _strategy_cache is not None:
        return _strategy_cache

    with _strategy_lock:
        if _strategy_cache is None:
            provider = settings.embedding_provider.lower().strip()
            _strategy_cache = _build_strategy(provider)

    log.info("Embedder ready. Dimensions: %d", _strategy_cache.dimensions)
    return _strategy_cache
