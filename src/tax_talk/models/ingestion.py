from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class EmbeddingManifest(BaseModel):
    """Validated metadata for persisted embedding artifacts."""

    model_config = ConfigDict(extra="forbid")

    source_key: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    embedding_rows: int = Field(ge=0)
    embedding_dimensions: int = Field(ge=0)
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    qdrant_collection: str = ""
    chunking_strategy: str = ""
    embedding_batch_size: int = Field(ge=1)
    ingest_batch_size: int = Field(ge=1)
    generated_at_unix: float = Field(ge=0)


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
    contextual_summary: str = ""
    contextual_summary_source: Literal["metadata", "llm", "none"] | str = "none"
    chunking_strategy: str = "contextual"
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    def to_chunk(self) -> Chunk:
        return Chunk(**self.model_dump())

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkRecord:
        payload = asdict(chunk)
        for key, value in payload.items():
            if isinstance(value, str):
                payload[key] = _strip_surrogate_codepoints(value)
        return cls.model_validate(payload)
