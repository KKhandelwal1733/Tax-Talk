"""Tokenization helpers for retrieval pipelines."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text with lowercase unicode word boundaries for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())
