"""Contextual summary strategy for metadata-first chunk prefixing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_llm_strategy, get_logger
from tax_talk.ingestion.loader import SourceDocument

log = get_logger(__name__)

_SOURCE_META_PRIORITY = ("content", "verified_text", "note", "use_for")


@dataclass(frozen=True)
class ContextualSummaryResult:
    """Resolved contextual summary text and how it was produced."""

    text: str = ""
    source: str = "none"


def _normalize_whitespace(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _unique_non_empty(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = _normalize_whitespace(part)
        if not normalized:
            continue
        folded = normalized.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        result.append(normalized)
    return result


def _get_nested_mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key, {})
    return value if isinstance(value, dict) else {}


def build_metadata_summary(doc: SourceDocument) -> str:
    """Build a deterministic candidate summary from curated raw metadata."""
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    source_meta = _get_nested_mapping(metadata, "source_meta")
    chunk_metadata = _get_nested_mapping(metadata, "chunk_metadata")
    ingestion_metadata = _get_nested_mapping(metadata, "ingestion_metadata")

    parts: list[str] = []
    for field_name in _SOURCE_META_PRIORITY:
        value = source_meta.get(field_name)
        if isinstance(value, str):
            parts.append(value)

    if not parts:
        act_name = chunk_metadata.get("act_name") or doc.source_key
        doc_type = chunk_metadata.get("doc_type") or "document"
        period = ingestion_metadata.get("applicable_period") or chunk_metadata.get(
            "applicable_period"
        )
        status = ingestion_metadata.get("act_status") or chunk_metadata.get("act_status")

        fallback_parts = [f"{act_name} ({doc_type})"]
        if isinstance(period, str) and period.strip():
            fallback_parts.append(f"Applicable period: {period}.")
        if isinstance(status, str) and status.strip():
            fallback_parts.append(f"Status: {status}.")
        parts.extend(fallback_parts)

    summary = " ".join(_unique_non_empty(parts))
    return _truncate(summary, settings.contextual_summary_max_chars)


def _is_metadata_summary_strong(summary: str) -> bool:
    return len(summary.strip()) >= settings.contextual_summary_metadata_min_chars


def _generate_llm_summary(doc: SourceDocument, cleaned_text: str) -> str:
    truncated_prefix = _truncate(cleaned_text, settings.contextual_summary_document_prefix_chars)
    if not truncated_prefix:
        return ""

    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    chunk_metadata = _get_nested_mapping(metadata, "chunk_metadata")
    ingestion_metadata = _get_nested_mapping(metadata, "ingestion_metadata")

    identity_lines = [
        f"Source key: {doc.source_key}",
        f"Filename: {metadata.get('filename', doc.file_path.name)}",
        f"Act name: {chunk_metadata.get('act_name', '')}",
        f"Document type: {chunk_metadata.get('doc_type', '')}",
        f"Applicable period: {ingestion_metadata.get('applicable_period', '')}",
        f"Act status: {ingestion_metadata.get('act_status', '')}",
    ]
    identity_block = "\n".join(line for line in identity_lines if not line.endswith(": "))
    prompt = (
        "Produce a concise factual summary for contextual chunking of a legal or tax source. "
        "Return 2 to 4 sentences only. Focus on what the source is, what topics it covers, "
        "and any period/status cues. Do not invent facts.\n\n"
        f"{identity_block}\n\n"
        "Document prefix:\n"
        f"{truncated_prefix}"
    )

    try:
        strategy = get_llm_strategy(settings.contextual_summary_fallback_provider)
        text = strategy.generate(
            prompt=prompt,
            model=settings.contextual_summary_fallback_model,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.warning("Contextual summary fallback failed for %s: %s", doc.source_key, exc)
        return ""
    if not isinstance(text, str):
        return ""
    return _truncate(_normalize_whitespace(text), settings.contextual_summary_max_chars)


def build_contextual_summary(doc: SourceDocument, cleaned_text: str) -> ContextualSummaryResult:
    """Resolve the summary to prepend for chunks from a source document."""
    if not settings.contextual_summary_enabled:
        return ContextualSummaryResult()

    metadata_summary = build_metadata_summary(doc)
    if _is_metadata_summary_strong(metadata_summary):
        return ContextualSummaryResult(text=metadata_summary, source="metadata")

    if not settings.contextual_summary_llm_fallback_enabled:
        return ContextualSummaryResult()

    llm_summary = _generate_llm_summary(doc, cleaned_text)
    if llm_summary:
        return ContextualSummaryResult(text=llm_summary, source="llm")

    return ContextualSummaryResult()
