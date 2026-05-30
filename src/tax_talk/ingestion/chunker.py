"""
src/tax_talk/ingestion/chunker.py

Step 2 of the ingestion pipeline:
    Splits SourceDocuments into chunks and maps metadata.

Usage:
    from tax_talk.ingestion.chunker import chunk_documents, Chunk
    chunks = chunk_documents(docs)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.ingestion.chunking_strategies import (
    CHUNKING_STRATEGIES,
    ContextualChunkingStrategy,
    FixedChunkingStrategy,
    SemanticChunkingStrategy,
    get_chunking_strategy,
    resolve_chunking_strategy,
)
from tax_talk.ingestion.chunking_strategies.base import ChunkingStrategy
from tax_talk.ingestion.chunking_strategies.helpers import (
    prepend_contextual_summary,
    split_fixed_text,
    split_semantic_text,
)
from tax_talk.ingestion.contextual_summary import ContextualSummaryResult
from tax_talk.ingestion.loader import SourceDocument
from tax_talk.models.ingestion import ChunkRecord

__all__ = [
    "CHUNKING_STRATEGIES",
    "Chunk",
    "ContextualChunkingStrategy",
    "FixedChunkingStrategy",
    "SemanticChunkingStrategy",
    "chunk_document",
    "chunk_documents",
    "chunk_text",
    "chunk_text_semantic",
    "get_chunking_strategy",
    "read_chunks_jsonl",
    "resolve_chunking_strategy",
    "write_chunks_jsonl",
]

log = get_logger(__name__)

# Target chunk size and overlap in characters.
CHUNK_SIZE_CHARS = settings.chunk_size_chars
CHUNK_OVERLAP_CHARS = settings.chunk_overlap_chars
SEMANTIC_CHUNK_MIN_CHARS = settings.semantic_chunk_min_chars
SEMANTIC_CHUNK_MAX_CHARS = settings.semantic_chunk_max_chars


def _strip_surrogate_codepoints(value: str) -> str:
    """Remove unpaired UTF-16 surrogate code points that break UTF-8 JSON serialization."""
    return re.sub(r"[\ud800-\udfff]", "", value)


@dataclass
class Chunk:
    """A single chunk ready for embedding and Qdrant upsert."""

    chunk_id: str
    text: str
    source_key: str
    filename: str
    applicable_period: str = "unknown"
    act_status: str = "unknown"
    doc_type: str = "act"
    act_name: str = ""
    chapter: str = ""
    section_number_new: str = ""
    section_number_old: str = ""
    section_title: str = ""
    contextual_summary: str = ""
    contextual_summary_source: str = "none"
    chunking_strategy: str = "contextual"
    chunk_index: int = 0
    total_chunks: int = 0
    char_start: int = 0
    char_end: int = 0

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Convert to Qdrant point payload (all metadata except the vector)."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_key": self.source_key,
            "filename": self.filename,
            "applicable_period": self.applicable_period,
            "act_status": self.act_status,
            "doc_type": self.doc_type,
            "act_name": self.act_name,
            "chapter": self.chapter,
            "section_number_new": self.section_number_new,
            "section_number_old": self.section_number_old,
            "section_title": self.section_title,
            "contextual_summary": self.contextual_summary,
            "contextual_summary_source": self.contextual_summary_source,
            "chunking_strategy": self.chunking_strategy,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def _get_nested_mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key, {})
    return value if isinstance(value, dict) else {}


def _resolve_chunk_metadata(doc: SourceDocument) -> dict[str, str]:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    ingestion_metadata = _get_nested_mapping(metadata, "ingestion_metadata")
    chunk_metadata = _get_nested_mapping(metadata, "chunk_metadata")

    return {
        "applicable_period": str(
            ingestion_metadata.get(
                "applicable_period",
                chunk_metadata.get("applicable_period", metadata.get("applicable_period", "unknown")),
            )
            or "unknown"
        ),
        "act_status": str(
            ingestion_metadata.get(
                "act_status",
                chunk_metadata.get("act_status", metadata.get("act_status", "unknown")),
            )
            or "unknown"
        ),
        "doc_type": str(chunk_metadata.get("doc_type", metadata.get("doc_type", "act")) or "act"),
        "act_name": str(chunk_metadata.get("act_name", metadata.get("act_name", "")) or ""),
        "chapter": str(chunk_metadata.get("chapter", metadata.get("chapter", "")) or ""),
        "section_number_new": str(
            chunk_metadata.get("section_number_new", metadata.get("section_number_new", "")) or ""
        ),
        "section_number_old": str(
            chunk_metadata.get("section_number_old", metadata.get("section_number_old", "")) or ""
        ),
        "section_title": str(chunk_metadata.get("section_title", metadata.get("section_title", "")) or ""),
    }


def _prepend_contextual_summary(chunk_text: str, summary: str) -> str:
    """Backward-compatible wrapper around summary prefix helper."""
    return prepend_contextual_summary(
        chunk_text,
        summary,
        label=settings.contextual_summary_prefix_label,
    )


def _make_chunk_id(source_key: str, text: str, chunk_index: int) -> str:
    """Deterministic chunk ID — same input always produces same ID."""
    raw = f"{source_key}::{chunk_index}::{text[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _clean_text(text: str) -> str:
    """Basic cleanup for legal text."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\x00+", "", text)
    text = re.sub(r"‍", "", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, int, int]]:
    """Split text into fixed-size chunks."""
    return split_fixed_text(text, chunk_size=chunk_size, overlap=overlap)


def chunk_text_semantic(
    text: str,
    *,
    min_chunk_chars: int = SEMANTIC_CHUNK_MIN_CHARS,
    max_chunk_chars: int = SEMANTIC_CHUNK_MAX_CHARS,
) -> list[tuple[str, int, int]]:
    """Split text into heading/section-aware semantic chunks."""
    return split_semantic_text(
        text,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )


def chunk_document(doc: SourceDocument, *, chunking_strategy: str | None = None) -> list[Chunk]:
    """Chunk a single SourceDocument into Chunk objects."""
    cleaned = _clean_text(doc.text)
    cleaned = _strip_surrogate_codepoints(cleaned)
    strategy_impl: ChunkingStrategy = get_chunking_strategy(chunking_strategy)

    raw_chunks = strategy_impl.split_text(cleaned)
    resolved_metadata = _resolve_chunk_metadata(doc)
    summary_result: ContextualSummaryResult = strategy_impl.build_summary(doc, cleaned)

    chunks: list[Chunk] = []
    for i, (text, char_start, char_end) in enumerate(raw_chunks):
        final_text = strategy_impl.render_chunk(text, summary_result)
        chunk_id = _make_chunk_id(doc.source_key, final_text, i)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=final_text,
                source_key=doc.source_key,
                filename=doc.metadata.get("filename", ""),
                applicable_period=resolved_metadata["applicable_period"],
                act_status=resolved_metadata["act_status"],
                doc_type=resolved_metadata["doc_type"],
                act_name=resolved_metadata["act_name"],
                chapter=resolved_metadata["chapter"],
                section_number_new=resolved_metadata["section_number_new"],
                section_number_old=resolved_metadata["section_number_old"],
                section_title=resolved_metadata["section_title"],
                contextual_summary=summary_result.text,
                contextual_summary_source=summary_result.source,
                chunking_strategy=strategy_impl.name,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                char_start=char_start,
                char_end=char_end,
            )
        )

    log.info(
        "Chunked %s (%s) with strategy=%s -> %d chunks",
        doc.source_key,
        doc.metadata.get("filename", ""),
        strategy_impl.name,
        len(chunks),
    )
    return chunks


def chunk_documents(docs: list[SourceDocument], *, chunking_strategy: str | None = None) -> list[Chunk]:
    """Chunk all documents. Returns flat list of all chunks."""
    resolved_strategy = resolve_chunking_strategy(chunking_strategy)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunking_strategy=resolved_strategy))
    log.info("Total chunks (%s): %d", resolved_strategy, len(all_chunks))
    return all_chunks


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunks to JSONL with deterministic ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(chunks, key=lambda c: (c.source_key, c.filename, c.chunk_index, c.chunk_id))

    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in ordered:
            record = ChunkRecord.from_chunk(chunk)
            handle.write(record.model_dump_json(ensure_ascii=False) + "\n")


def read_chunks_jsonl(input_path: Path) -> list[Chunk]:
    """Load chunks from a JSONL artifact."""
    chunks: list[Chunk] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = ChunkRecord.model_validate(payload)
            chunks.append(record.to_chunk())
    return chunks


if __name__ == "__main__":
    from tax_talk.ingestion.loader import load_all_sources

    docs = load_all_sources()
    chunks = chunk_documents(docs)
    print(f"\nTotal chunks: {len(chunks)}")
    print("\nSample chunk:")
    if chunks:
        c = chunks[0]
        print(f"  ID:      {c.chunk_id}")
        print(f"  Source:  {c.source_key}")
        print(f"  Period:  {c.applicable_period}")
        print(f"  Status:  {c.act_status}")
        print(f"  Chars:   {c.char_start}-{c.char_end}")
        print(f"  Preview: {c.text[:200]}...")
