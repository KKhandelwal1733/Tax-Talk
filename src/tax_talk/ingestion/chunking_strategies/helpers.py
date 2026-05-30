
from __future__ import annotations

import re

_HEADING_PATTERN = re.compile(
    r"^(section|sec\.?|rule|chapter|part|article|clause)\b|^\d+[a-zA-Z]?\.\s+",
    flags=re.IGNORECASE,
)


def split_fixed_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """Split text using fixed-size windows with overlap and boundary hints."""
    body = text.strip()
    if not body:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0

    while start < len(body):
        end = start + chunk_size

        if end >= len(body):
            chunks.append((body[start:], start, len(body)))
            break

        split_point = body.rfind("\n\n", start + chunk_size // 2, end)

        if split_point == -1:
            split_point = body.rfind(". ", start + chunk_size // 2, end)
            if split_point != -1:
                split_point += 1

        if split_point == -1:
            split_point = end

        chunks.append((body[start:split_point].strip(), start, split_point))
        start = split_point - overlap

        if start <= 0:
            start = split_point

    return [(c, s, e) for c, s, e in chunks if c.strip()]


def _is_heading_paragraph(paragraph: str) -> bool:
    normalized = paragraph.strip()
    if not normalized:
        return False
    if _HEADING_PATTERN.match(normalized):
        return True
    is_short = len(normalized) <= 120
    has_alpha = any(ch.isalpha() for ch in normalized)
    return is_short and has_alpha and normalized.upper() == normalized


def split_semantic_text(
    text: str,
    *,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> list[tuple[str, int, int]]:
    """Split text by heading/section-aware paragraph grouping."""
    if min_chunk_chars <= 0 or max_chunk_chars <= 0:
        raise ValueError("min_chunk_chars and max_chunk_chars must be > 0")
    if min_chunk_chars > max_chunk_chars:
        raise ValueError("min_chunk_chars cannot be greater than max_chunk_chars")

    body = text.strip()
    if not body:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[tuple[str, int, int]] = []
    active_parts: list[str] = []
    active_start = 0
    active_len = 0
    cursor = 0

    def flush() -> None:
        nonlocal active_parts, active_start, active_len
        if not active_parts:
            return
        payload = "\n\n".join(active_parts).strip()
        if payload:
            chunks.append((payload, active_start, active_start + len(payload)))
        active_parts = []
        active_start = 0
        active_len = 0

    for paragraph in paragraphs:
        para_start = body.find(paragraph, cursor)
        if para_start < 0:
            para_start = cursor
        cursor = para_start + len(paragraph)

        is_heading = _is_heading_paragraph(paragraph)
        would_exceed = active_len > 0 and (active_len + len(paragraph) + 2) > max_chunk_chars

        if active_parts and (would_exceed or (is_heading and active_len >= min_chunk_chars)):
            flush()

        if not active_parts:
            active_start = para_start

        active_parts.append(paragraph)
        active_len += len(paragraph) + (2 if active_len else 0)

    flush()
    return chunks


def prepend_contextual_summary(chunk_text: str, summary: str, *, label: str) -> str:
    """Prefix chunk text with contextual summary using configured label."""
    normalized_summary = summary.strip()
    if not normalized_summary:
        return chunk_text
    resolved_label = label.strip() or "Context"
    return f"{resolved_label}: {normalized_summary}\n\n{chunk_text}"
