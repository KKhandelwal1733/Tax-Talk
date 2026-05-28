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
    from langfuse import get_client
except ImportError:  # pragma: no cover
    get_client = None

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover
    QdrantClient = None

from tax_talk.core.config import settings

_logging_lock = Lock()
_logging_configured = False

_qdrant_lock = Lock()
_qdrant_client: QdrantClient | None = None

_langfuse_lock = Lock()
_langfuse_client: Any | None = None

_cohere_lock = Lock()
_cohere_client: Any | None = None


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


def get_qdrant_client() -> QdrantClient:
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
