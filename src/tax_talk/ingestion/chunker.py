"""
src/tax_talk/ingestion/chunker.py

Step 2 of the ingestion pipeline:
    Splits SourceDocuments into chunks of ~512 tokens with overlap.
    Each chunk carries the full metadata from its source + a chunk_id.

Week 3: naive fixed-size chunking.
Week 4: contextual chunking (prepend LLM-generated section summary per chunk).

Usage:
    from tax_talk.ingestion.chunker import chunk_documents, Chunk
    chunks = chunk_documents(docs)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tax_talk.ingestion.loader import SourceDocument
from tax_talk.core.runtime import get_logger

log = get_logger(__name__)

# Target chunk size and overlap in characters.
# ~4 chars per token is a rough average for English legal text.
# 512 tokens × 4 chars = ~2,048 chars per chunk.
# Overlap helps retrieve answers that straddle a boundary.
CHUNK_SIZE_CHARS = 2_000      # ≈ 512 tokens
CHUNK_OVERLAP_CHARS = 200     # ≈ 50 token overlap


def _strip_surrogate_codepoints(value: str) -> str:
    """Remove unpaired UTF-16 surrogate code points that break UTF-8 JSON serialization."""
    return re.sub(r"[\ud800-\udfff]", "", value)


@dataclass
class Chunk:
    """
    A single chunk ready for embedding and Qdrant upsert.
    chunk_id is deterministic: sha256(source_key + text).
    """
    chunk_id: str
    text: str
    source_key: str
    filename: str
    # Metadata fields — all searchable in Qdrant payload
    applicable_period: str = "unknown"  # "FY 2026-27 onwards" | "AY 2026-27 and earlier" | "all"
    act_status: str = "unknown"         # "current" | "legacy"
    doc_type: str = "act"               # "act" | "rules" | "notification" | "circular"
    act_name: str = ""
    # Optional structural fields (fill in during week 4 contextual chunking)
    chapter: str = ""
    section_number_new: str = ""
    section_number_old: str = ""
    section_title: str = ""
    # Chunk position info
    chunk_index: int = 0
    total_chunks: int = 0
    char_start: int = 0
    char_end: int = 0

    def to_qdrant_payload(self) -> dict:
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
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


class ChunkRecord(BaseModel):
    """Validated on-disk representation for chunk JSONL artifacts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    chunk_id: str = Field(min_length=1)
    text: str
    source_key: str = Field(min_length=1)
    filename: str = ""
    applicable_period: str = "unknown"
    act_status: str = "unknown"
    doc_type: Literal["act", "rules", "notification", "circular"] | str = "act"
    act_name: str = ""
    chapter: str = ""
    section_number_new: str = ""
    section_number_old: str = ""
    section_title: str = ""
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    def to_chunk(self) -> Chunk:
        return Chunk(**self.model_dump())

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkRecord":
        payload = asdict(chunk)
        for key, value in payload.items():
            if isinstance(value, str):
                payload[key] = _strip_surrogate_codepoints(value)
        return cls.model_validate(payload)


def _make_chunk_id(source_key: str, text: str, chunk_index: int) -> str:
    """Deterministic chunk ID — same input always produces same ID."""
    raw = f"{source_key}::{chunk_index}::{text[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _clean_text(text: str) -> str:
    """
    Basic cleanup for Indian legal/statutory text:
    - Collapse excessive whitespace
    - Remove page headers/footers (page numbers, repeated headers)
    - Normalize line endings
    """
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)    # max 3 consecutive newlines
    text = re.sub(r" {2,}", " ", text)           # max 1 space

    # Remove standalone page numbers (e.g. "42", "  42  " on its own line)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Remove common PDF extraction artifacts from Indian govt docs
    text = re.sub(r"\x00+", "", text)    # null bytes
    text = re.sub(r"‍", "", text)        # zero-width joiner
    text = re.sub(r"[\ud800-\udfff]", "", text) 
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, int, int]]:
    """
    Split text into (chunk_text, char_start, char_end) tuples.

    Tries to split on paragraph boundaries (double newlines) when possible,
    falling back to hard character splits.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append((text[start:], start, len(text)))
            break

        # Try to split on a paragraph boundary near the end of the window
        split_point = text.rfind("\n\n", start + chunk_size // 2, end)

        if split_point == -1:
            # Fall back to sentence boundary (". " or ".\n")
            split_point = text.rfind(". ", start + chunk_size // 2, end)
            if split_point != -1:
                split_point += 1  # include the period

        if split_point == -1:
            # Hard split at chunk_size
            split_point = end

        chunks.append((text[start:split_point].strip(), start, split_point))
        start = split_point - overlap  # overlap window

        # Safety: make sure we always advance
        if start <= 0:
            start = split_point

    return [(c, s, e) for c, s, e in chunks if c.strip()]


def chunk_document(doc: SourceDocument) -> list[Chunk]:
    """Chunk a single SourceDocument into Chunk objects."""
    cleaned = _clean_text(doc.text)
    cleaned = _strip_surrogate_codepoints(cleaned)
    raw_chunks = chunk_text(cleaned)

    chunks = []
    for i, (text, char_start, char_end) in enumerate(raw_chunks):
        chunk_id = _make_chunk_id(doc.source_key, text, i)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=text,
            source_key=doc.source_key,
            filename=doc.metadata.get("filename", ""),
            applicable_period=doc.metadata.get("applicable_period", "unknown"),
            act_status=doc.metadata.get("act_status", "unknown"),
            doc_type=doc.metadata.get("doc_type", "act"),
            act_name=doc.metadata.get("act_name", ""),
            chunk_index=i,
            total_chunks=len(raw_chunks),
            char_start=char_start,
            char_end=char_end,
        ))

    log.info(
        "Chunked %s (%s) → %d chunks",
        doc.source_key,
        doc.metadata.get("filename", ""),
        len(chunks),
    )
    return chunks


def chunk_documents(docs: list[SourceDocument]) -> list[Chunk]:
    """Chunk all documents. Returns flat list of all chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    log.info("Total chunks: %d", len(all_chunks))
    return all_chunks


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunks to JSONL with deterministic ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(chunks, key=lambda c: (c.source_key, c.filename, c.chunk_index, c.chunk_id))

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in ordered:
            record = ChunkRecord.from_chunk(chunk)
            f.write(record.model_dump_json(ensure_ascii=False) + "\n")


def read_chunks_jsonl(input_path: Path) -> list[Chunk]:
    """Load chunks from a JSONL artifact."""
    chunks: list[Chunk] = []
    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
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
        print(f"  Chars:   {c.char_start}–{c.char_end}")
        print(f"  Preview: {c.text[:200]}...")
