from __future__ import annotations

from pathlib import Path

from tax_talk.core.config import settings
from tax_talk.ingestion.chunker import (
    chunk_document,
    read_chunks_jsonl,
    write_chunks_jsonl,
)
from tax_talk.ingestion.loader import SourceDocument


def _build_doc(*, source_meta: dict[str, str] | None = None) -> SourceDocument:
    return SourceDocument(
        source_key="cgst_act_2025",
        file_path=Path("data/raw/cgst_act_2025/source.pdf"),
        text=(
            "Central Goods and Services Tax Act, 2017. "
            "This statute governs levy and collection of GST on intra-State supplies. "
            "It includes definitions, registration, returns, payments, and penalties."
        ),
        metadata={
            "filename": "source.pdf",
            "source_meta": source_meta or {
                "content": (
                    "Central Goods and Services Tax Act, 2017 - core law for levy and "
                    "collection of GST on intra-State supply of goods or services."
                ),
                "use_for": (
                    "Primary CGST statutory text covering sections, definitions, charging "
                    "provisions, input tax credit, registration, returns, payment, and penalties."
                ),
            },
            "ingestion_metadata": {
                "applicable_period": "all",
                "act_status": "current",
            },
            "chunk_metadata": {
                "doc_type": "act",
                "act_name": "CGST Act",
                "chapter": "Preliminary",
                "section_number_new": "1",
                "section_number_old": "",
                "section_title": "Short title, extent and commencement",
            },
        },
    )


def test_chunk_document_uses_nested_metadata(monkeypatch) -> None:
    monkeypatch.setattr(settings, "contextual_summary_enabled", False)

    chunk = chunk_document(_build_doc())[0]

    assert chunk.applicable_period == "all"
    assert chunk.act_status == "current"
    assert chunk.doc_type == "act"
    assert chunk.act_name == "CGST Act"
    assert chunk.chapter == "Preliminary"
    assert chunk.section_number_new == "1"
    assert chunk.section_title == "Short title, extent and commencement"


def test_chunk_document_prefers_source_meta_summary(monkeypatch) -> None:
    monkeypatch.setattr(settings, "contextual_summary_enabled", True)
    monkeypatch.setattr(settings, "contextual_summary_metadata_min_chars", 40)
    monkeypatch.setattr(settings, "contextual_summary_llm_fallback_enabled", True)

    from tax_talk.ingestion import contextual_summary as summary_module

    def fail_if_called(doc: SourceDocument, cleaned_text: str) -> str:
        raise AssertionError("LLM fallback should not be used for strong source_meta")

    monkeypatch.setattr(summary_module, "_generate_llm_summary", fail_if_called)

    chunk = chunk_document(_build_doc())[0]

    assert chunk.contextual_summary_source == "metadata"
    assert chunk.contextual_summary
    assert chunk.text.startswith("Context: ")
    assert "Primary CGST statutory text" in chunk.contextual_summary


def test_chunk_document_falls_back_to_llm_for_weak_metadata(monkeypatch) -> None:
    monkeypatch.setattr(settings, "contextual_summary_enabled", True)
    monkeypatch.setattr(settings, "contextual_summary_metadata_min_chars", 80)
    monkeypatch.setattr(settings, "contextual_summary_llm_fallback_enabled", True)

    from tax_talk.ingestion import contextual_summary as summary_module

    monkeypatch.setattr(
        summary_module,
        "_generate_llm_summary",
        lambda doc, cleaned_text: "Generated summary for GST charging, returns, and penalties.",
    )

    chunk = chunk_document(_build_doc(source_meta={"content": "GST Act"}))[0]

    assert chunk.contextual_summary_source == "llm"
    assert chunk.contextual_summary == "Generated summary for GST charging, returns, and penalties."
    assert chunk.text.startswith("Context: Generated summary for GST charging, returns, and penalties.")


def test_chunk_document_uses_plain_text_when_fallback_fails(monkeypatch) -> None:
    monkeypatch.setattr(settings, "contextual_summary_enabled", True)
    monkeypatch.setattr(settings, "contextual_summary_metadata_min_chars", 80)
    monkeypatch.setattr(settings, "contextual_summary_llm_fallback_enabled", True)

    from tax_talk.ingestion import contextual_summary as summary_module

    monkeypatch.setattr(summary_module, "_generate_llm_summary", lambda doc, cleaned_text: "")

    chunk = chunk_document(_build_doc(source_meta={"content": "GST Act"}))[0]

    assert chunk.contextual_summary_source == "none"
    assert chunk.contextual_summary == ""
    assert not chunk.text.startswith("Context: ")


def test_chunk_record_round_trip_preserves_contextual_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "contextual_summary_enabled", True)
    monkeypatch.setattr(settings, "contextual_summary_metadata_min_chars", 40)
    monkeypatch.setattr(settings, "contextual_summary_llm_fallback_enabled", False)

    chunks = chunk_document(_build_doc())
    output_path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl(chunks, output_path)
    loaded = read_chunks_jsonl(output_path)

    assert loaded[0].contextual_summary == chunks[0].contextual_summary
    assert loaded[0].contextual_summary_source == "metadata"
    assert loaded[0].chunk_id == chunks[0].chunk_id
