"""
src/tax_talk/ingestion/run.py

Main ingestion pipeline entrypoint.
Run with: make ingest
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

from langfuse import observe

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_langfuse_client, get_logger
from tax_talk.ingestion.chunker import (
    Chunk,
    chunk_documents,
    read_chunks_jsonl,
    write_chunks_jsonl,
)
from tax_talk.ingestion.embeddings import (
    EmbeddingManifest,
    embed_texts,
    read_embedding_manifest,
    read_embeddings_npy,
    write_embedding_manifest,
    write_embeddings_npy,
)
from tax_talk.ingestion.loader import DATA_RAW, load_source
from tax_talk.ingestion.qdrant_store import QdrantStore

log = get_logger(__name__)

INGEST_BATCH_SIZE = 128
DATA_PROCESSED = Path(__file__).parent.parent.parent.parent / "data" / "processed"

Stage = Literal["chunk", "embed", "upsert"]
STAGES: tuple[Stage, Stage, Stage] = ("chunk", "embed", "upsert")


def _stage_rank(stage: Stage) -> int:
    return STAGES.index(stage)


def _should_run(stage: Stage, from_stage: Stage, to_stage: Stage) -> bool:
    return _stage_rank(from_stage) <= _stage_rank(stage) <= _stage_rank(to_stage)


def _discover_source_keys(sources: list[str] | None, data_type: str = "raw") -> list[str]:
    if sources:
        return sorted(set(sources))
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / data_type
    if not data_dir.exists():
        return []
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir())


def _artifact_paths(source_key: str) -> tuple[Path, Path, Path]:
    source_dir = DATA_PROCESSED / source_key
    return source_dir / "chunks.jsonl", source_dir / "embeddings.npy", source_dir / "manifest.json"


def _delete_stage_artifacts(source_key: str, from_stage: Stage) -> None:
    chunks_path, embeddings_path, manifest_path = _artifact_paths(source_key)

    if from_stage == "chunk":
        paths = (chunks_path, embeddings_path, manifest_path)
    elif from_stage == "embed":
        paths = (embeddings_path, manifest_path)
    else:
        paths = ()

    for path in paths:
        if path.exists():
            path.unlink()
            log.info("Deleted artifact: %s", path)


def _embedding_model_name() -> str:
    provider = settings.embedding_provider.lower().strip()
    if provider == "gemini":
        return settings.embedding_model_gemini
    if provider == "voyage":
        return settings.embedding_model_voyage
    return settings.embedding_model_local


@observe(name="ingestion-run")
def run_ingestion(
    sources: list[str] | None = None,
    from_stage: Stage = "chunk",
    to_stage: Stage = "upsert",
) -> None:
    lf = get_langfuse_client()

    if _stage_rank(from_stage) > _stage_rank(to_stage):
        raise ValueError(f"Invalid stage range: from_stage={from_stage}, to_stage={to_stage}")

    source_keys = _discover_source_keys(sources,"raw")
    if not source_keys:
        log.error("No source directories found in data/raw/. Run: uv run python scripts/download_corpus.py")
        sys.exit(1)

    log.info("Running ingestion stages %s -> %s for %d source(s)", from_stage, to_stage, len(source_keys))

    store: QdrantStore | None = None
    if _should_run("upsert", from_stage, to_stage):
        log.info("Setting up Qdrant collection")
        store = QdrantStore()
        store.create_collection_if_not_exists()
        existing = store.count(exact=True)
        log.info("Qdrant collection '%s': %d existing points", settings.qdrant_collection, existing)

    total_upserted = 0
    total_chunks = 0
    t_start = time.time()

    chunk_cache: dict[str, list[Chunk]] = {}

    # Pass 1: build and persist all chunks for selected sources before any embedding starts.
    if _should_run("chunk", from_stage, to_stage):
        for source_key in source_keys:
            chunks_path, _, _ = _artifact_paths(source_key)
            chunks_path.parent.mkdir(parents=True, exist_ok=True)

            docs = load_source(source_key)
            if not docs:
                log.warning("Skipping %s: no documents loaded.", source_key)
                continue

            chunks = chunk_documents(docs)
            if not chunks:
                log.warning("Skipping %s: no chunks generated.", source_key)
                continue

            write_chunks_jsonl(chunks, chunks_path)
            chunk_cache[source_key] = chunks
            total_chunks += len(chunks)
            log.info("Wrote %d chunks to %s", len(chunks), chunks_path)

    if _should_run("chunk", from_stage, to_stage):
        active_source_keys = list(chunk_cache.keys())
    else:
        active_source_keys = _discover_source_keys(sources,"processed")
        if not active_source_keys:
            log.error("No processed source directories found in data/processed/. Run with --from-stage chunk to create them.")
            sys.exit(1)

    if not _should_run("embed", from_stage, to_stage) and not _should_run("upsert", from_stage, to_stage):
        active_source_keys = []

    # Pass 2: read chunks from processed artifacts, then embed and optionally upsert.
    for source_key in active_source_keys:
        chunks_path, embeddings_path, manifest_path = _artifact_paths(source_key)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        if source_key in chunk_cache:
            chunks = chunk_cache[source_key]
        else:
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"Missing chunks artifact for {source_key}: {chunks_path}. "
                    f"Run from stage 'chunk' first."
                )
            chunks = read_chunks_jsonl(chunks_path)
            total_chunks += len(chunks)

        if _should_run("embed", from_stage, to_stage):
            vectors: list[list[float]] = []

            for batch_start in range(0, len(chunks), INGEST_BATCH_SIZE):
                batch = chunks[batch_start : batch_start + INGEST_BATCH_SIZE]
                log.info(
                    "  %s embed batch %d-%d of %d",
                    source_key,
                    batch_start + 1,
                    batch_start + len(batch),
                    len(chunks),
                )
                vectors.extend(embed_texts([c.text for c in batch]))

            write_embeddings_npy(vectors, embeddings_path)
            write_embedding_manifest(
                manifest_path,
                EmbeddingManifest(
                    source_key=source_key,
                    chunk_count=len(chunks),
                    embedding_rows=len(vectors),
                    embedding_dimensions=len(vectors[0]) if vectors else 0,
                    embedding_provider=settings.embedding_provider,
                    embedding_model=_embedding_model_name(),
                    embedding_batch_size=settings.embedding_batch_size,
                    ingest_batch_size=INGEST_BATCH_SIZE,
                    generated_at_unix=time.time(),
                ),
            )
            log.info("Wrote embeddings to %s", embeddings_path)
        else:
            if not embeddings_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing embedding artifacts for {source_key}: {embeddings_path} / {manifest_path}. "
                    f"Run from stage 'embed' first."
                )
            vectors = read_embeddings_npy(embeddings_path)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Artifact mismatch for {source_key}: {len(chunks)} chunks vs {len(vectors)} embeddings."
            )

        if vectors and len(vectors[0]) != settings.embedding_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch for {source_key}: "
                f"got {len(vectors[0])}, expected {settings.embedding_dimensions}."
            )

        if _should_run("upsert", from_stage, to_stage):
            if store is None:
                raise RuntimeError("Qdrant store not initialized for upsert stage.")

            if not _should_run("embed", from_stage, to_stage):
                manifest = read_embedding_manifest(manifest_path)
                current_model = _embedding_model_name()
                if manifest.embedding_model and manifest.embedding_model != current_model:
                    log.warning(
                        "Embedding model differs for %s (artifact=%s, current=%s).",
                        source_key,
                        manifest.embedding_model,
                        current_model,
                    )

            n = store.upsert_chunks(chunks, vectors)
            total_upserted += n
            log.info("Upserted %d chunk(s) for %s", n, source_key)

    elapsed = time.time() - t_start
    final_count = store.count(exact=True) if store is not None else 0

    lf.flush()

    log.info("=" * 60)
    log.info("✓ Ingestion complete")
    log.info("  Sources processed:          %d", len(source_keys))
    log.info("  Chunks prepared:            %d", total_chunks)
    log.info("  Chunks upserted this run:   %d", total_upserted)
    if store is not None:
        log.info("  Total in Qdrant now:        %d", final_count)
    else:
        log.info("  Qdrant upsert stage:        skipped")
    log.info("  Time taken:                 %.1fs", elapsed)
    log.info("  Langfuse trace host:        %s", settings.langfuse_host)
    log.info("=" * 60)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", help="Specific source keys to ingest")
    parser.add_argument("--from-stage", choices=STAGES, default="chunk", help="Start stage")
    parser.add_argument("--to-stage", choices=STAGES, default="upsert", help="End stage")
    parser.add_argument("--test", action="store_true", help="Run retrieval test after ingestion")
    args = parser.parse_args()

    run_ingestion(
        sources=args.sources,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )


