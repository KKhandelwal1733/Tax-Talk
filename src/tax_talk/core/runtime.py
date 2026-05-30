"""Runtime singletons for shared infrastructure clients and logging."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

try:
    import cohere
except ImportError:  # pragma: no cover
    cohere = None

try:
    import google.genai as genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from langfuse import get_client
except ImportError:  # pragma: no cover
    get_client = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover
    QdrantClient = None

from tax_talk.core.config import settings
from tax_talk.generation.gemini_strategy import GeminiLLMStrategy
from tax_talk.generation.groq_strategy import GroqLLMStrategy
from tax_talk.generation.llm_provider import LLMStrategy

_logging_lock = Lock()
_logging_configured = False

_qdrant_lock = Lock()
_qdrant_client: Any | None = None

_langfuse_lock = Lock()
_langfuse_client: Any | None = None

_cohere_lock = Lock()
_cohere_client: Any | None = None

_gemini_lock = Lock()
_gemini_client: Any | None = None

_groq_lock = Lock()
_groq_client: Any | None = None

_llm_strategy_lock = Lock()
_llm_strategies: dict[str, LLMStrategy] = {}


def configure_logging() -> None:
    """Configure root logging once for the whole process."""
    global _logging_configured
    if _logging_configured:
        return

    with _logging_lock:
        if _logging_configured:
            return

        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=settings.log_level,
                format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            )
        else:
            root_logger.setLevel(settings.log_level)

        _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger after ensuring global logging is configured."""
    configure_logging()
    return logging.getLogger(name)


def get_qdrant_client() -> Any:
    """Return a singleton Qdrant client for the current process."""
    if QdrantClient is None:
        raise RuntimeError(
            "qdrant-client is not installed; install qdrant-client to use Qdrant integration."
        )

    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )

    return _qdrant_client


def get_langfuse_client() -> Any:
    """Return a singleton Langfuse client for the current process."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    with _langfuse_lock:
        if _langfuse_client is None:
            client_kwargs: dict[str, Any] = {}
            if settings.langfuse_public_key:
                client_kwargs["public_key"] = settings.langfuse_public_key
            if settings.langfuse_secret_key:
                client_kwargs["secret_key"] = settings.langfuse_secret_key
            if settings.langfuse_host:
                client_kwargs["host"] = settings.langfuse_host

            if get_client is None:
                raise RuntimeError(
                    "Langfuse is not installed; install the langfuse SDK to enable observability."
                )

            try:
                if client_kwargs:
                    _langfuse_client = get_client(**client_kwargs)
                else:
                    _langfuse_client = get_client()
            except TypeError:
                # Some SDK variants do not accept explicit args in get_client().
                _langfuse_client = get_client()

    return _langfuse_client


def get_cohere_client() -> Any:
    """Return a singleton Cohere client for the current process."""
    global _cohere_client
    if _cohere_client is not None:
        return _cohere_client

    with _cohere_lock:
        if _cohere_client is None:
            if cohere is None:
                raise RuntimeError(
                    "cohere SDK is not installed; install cohere to enable reranking."
                )

            api_key = settings.cohere_api_key or None
            if api_key:
                _cohere_client = cohere.ClientV2(api_key=api_key)
            else:
                _cohere_client = cohere.ClientV2()

    return _cohere_client


def get_gemini_client() -> Any:
    """Return a singleton Gemini client for the current process."""
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed; install google-genai to enable Gemini generation."
        )

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    with _gemini_lock:
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=settings.gemini_api_key)

    return _gemini_client


def get_groq_client() -> Any:
    """Return a singleton Groq client for the current process."""
    if Groq is None:
        raise RuntimeError("groq SDK is not installed; install groq to enable Groq generation.")

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    global _groq_client
    if _groq_client is not None:
        return _groq_client

    with _groq_lock:
        if _groq_client is None:
            _groq_client = Groq(api_key=settings.groq_api_key)

    return _groq_client


def get_llm_strategy(provider: str | None = None) -> LLMStrategy:
    """Return a runtime-owned singleton strategy for the requested provider."""
    log = get_logger(__name__)
    configured_provider = (provider or settings.contextual_summary_fallback_provider or "").strip()
    provider_key = configured_provider.lower()

    if not provider_key:
        if settings.gemini_api_key:
            provider_key = "gemini"
        elif settings.groq_api_key:
            provider_key = "groq"
        else:
            message = "No LLM provider available. Configure GEMINI_API_KEY or GROQ_API_KEY."
            log.error(message)
            raise RuntimeError(message)

    strategy = _llm_strategies.get(provider_key)
    if strategy is not None:
        return strategy

    with _llm_strategy_lock:
        strategy = _llm_strategies.get(provider_key)
        if strategy is not None:
            return strategy

        try:
            if provider_key == "gemini":
                strategy = GeminiLLMStrategy(get_gemini_client())
            elif provider_key == "groq":
                strategy = GroqLLMStrategy(get_groq_client())
            else:
                raise RuntimeError(
                    f"Unsupported LLM provider '{provider_key}'. Use 'gemini' or 'groq'."
                )
        except Exception as exc:
            log.error("LLM provider '%s' is unavailable: %s", provider_key, exc)
            raise

        _llm_strategies[provider_key] = strategy
        return strategy
