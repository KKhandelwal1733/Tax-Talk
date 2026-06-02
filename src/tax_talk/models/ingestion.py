from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from tax_talk.ingestion.chunker import Chunk


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
        from tax_talk.ingestion.chunker import Chunk

        return Chunk(**self.model_dump())

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkRecord:
        from tax_talk.ingestion.chunker import _strip_surrogate_codepoints

        payload = asdict(chunk)
        for key, value in payload.items():
            if isinstance(value, str):
                payload[key] = _strip_surrogate_codepoints(value)
        return cls.model_validate(payload)
