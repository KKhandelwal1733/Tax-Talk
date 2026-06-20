from __future__ import annotations

from pathlib import Path

from tax_talk.ingestion import run as ingestion_run


def test_artifact_paths_are_strategy_scoped() -> None:
    chunks_path, embeddings_path, manifest_path = ingestion_run._artifact_paths(
        "fixed", "gst_circulars_cbic"
    )

    expected_base = Path("data/processed/fixed/gst_circulars_cbic")
    assert chunks_path.as_posix().endswith((expected_base / "chunks.jsonl").as_posix())
    assert embeddings_path.as_posix().endswith((expected_base / "embeddings.npy").as_posix())
    assert manifest_path.as_posix().endswith((expected_base / "manifest.json").as_posix())


def test_discover_processed_source_keys_uses_strategy_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed_root = tmp_path / "processed"
    (processed_root / "fixed" / "src_b").mkdir(parents=True)
    (processed_root / "fixed" / "src_a").mkdir(parents=True)
    (processed_root / "semantic" / "src_c").mkdir(parents=True)

    monkeypatch.setattr(ingestion_run, "DATA_PROCESSED", processed_root)

    result = ingestion_run._discover_source_keys(
        sources=None,
        data_type="processed",
        chunking_strategy="fixed",
    )

    assert result == ["src_a", "src_b"]
