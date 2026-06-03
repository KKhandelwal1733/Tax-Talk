"""
src/tax_talk/ingestion/run.py

Main ingestion pipeline entrypoint.
Run with: make ingest
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from langfuse import observe
from tax_talk.core.config import settings
from tax_talk.core.runtime import get_langfuse_client, get_logger
from tax_talk.ingestion.chunker import (
    CHUNKING_STRATEGIES,
    chunk_documents,
    read_chunks_jsonl,
    resolve_chunking_strategy,
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
from tax_talk.ingestion.loader import load_source
from tax_talk.ingestion.qdrant_store import QdrantStore
from tax_talk.models.ingestion import Chunk

log = get_logger(__name__)

INGEST_BATCH_SIZE = 128
DATA_PROCESSED = Path(__file__).parent.parent.parent.parent / "data" / "processed"

Stage = Literal["chunk", "embed", "upsert"]
STAGES: tuple[Stage, Stage, Stage] = ("chunk", "embed", "upsert")


def _stage_rank(stage: Stage) -> int:
    return STAGES.index(stage)


def _should_run(stage: Stage, from_stage: Stage, to_stage: Stage) -> bool:
    return _stage_rank(from_stage) <= _stage_rank(stage) <= _stage_rank(to_stage)


def _discover_source_keys(
    sources: list[str] | None,
    data_type: str = "raw",
    chunking_strategy: str | None = None,
) -> list[str]:
    if sources:
        return sorted(set(sources))

    if data_type == "processed":
        if not chunking_strategy:
            raise ValueError("chunking_strategy is required when discovering processed sources")
        data_dir = DATA_PROCESSED / chunking_strategy
    else:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / data_type

    if not data_dir.exists():
        return []
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir())


def _artifact_paths(chunking_strategy: str, source_key: str) -> tuple[Path, Path, Path]:
    source_dir = DATA_PROCESSED / chunking_strategy / source_key
    return source_dir / "chunks.jsonl", source_dir / "embeddings.npy", source_dir / "manifest.json"


def _delete_stage_artifacts(chunking_strategy: str, source_key: str, from_stage: Stage) -> None:
    chunks_path, embeddings_path, manifest_path = _artifact_paths(chunking_strategy, source_key)

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
    chunking_strategy: str | None = None,
    collection_name: str | None = None,
) -> None:
    lf = get_langfuse_client()
    resolved_strategy = resolve_chunking_strategy(chunking_strategy)
    resolved_collection = (collection_name or settings.qdrant_collection).strip()
    if not resolved_collection:
        raise ValueError("collection_name must not be blank")

    if _stage_rank(from_stage) > _stage_rank(to_stage):
        raise ValueError(f"Invalid stage range: from_stage={from_stage}, to_stage={to_stage}")

    if _should_run("chunk", from_stage, to_stage):
        source_keys = _discover_source_keys(sources, "raw")
        if not source_keys:
            log.error(
                "No source directories found in data/raw/. Run: uv run python scripts/download_corpus.py"
            )
            sys.exit(1)
    else:
        source_keys = _discover_source_keys(sources, "processed", resolved_strategy)
        if not source_keys:
            log.error(
                "No processed source directories found in data/processed/%s/. Run with --from-stage chunk first.",
                resolved_strategy,
            )
            sys.exit(1)

    log.info(
        "Running ingestion stages %s -> %s for %d source(s), strategy=%s, collection=%s",
        from_stage,
        to_stage,
        len(source_keys),
        resolved_strategy,
        resolved_collection,
    )

    store: QdrantStore | None = None
    if _should_run("upsert", from_stage, to_stage):
        log.info("Setting up Qdrant collection")
        store = QdrantStore(collection_name=resolved_collection)
        store.create_collection_if_not_exists()
        existing = store.count(exact=True)
        log.info("Qdrant collection '%s': %d existing points", resolved_collection, existing)

    total_upserted = 0
    total_chunks = 0
    t_start = time.time()

    chunk_cache: dict[str, list[Chunk]] = {}

    # Pass 1: build and persist all chunks for selected sources before any embedding starts.
    if _should_run("chunk", from_stage, to_stage):
        for source_key in source_keys:
            chunks_path, _, _ = _artifact_paths(resolved_strategy, source_key)
            chunks_path.parent.mkdir(parents=True, exist_ok=True)

            docs = load_source(source_key)
            if not docs:
                log.warning("Skipping %s: no documents loaded.", source_key)
                continue

            chunks = chunk_documents(docs, chunking_strategy=resolved_strategy)
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
        active_source_keys = _discover_source_keys(sources, "processed", resolved_strategy)
        if not active_source_keys:
            log.error(
                "No processed source directories found in data/processed/%s/. Run with --from-stage chunk to create them.",
                resolved_strategy,
            )
            sys.exit(1)

    if not _should_run("embed", from_stage, to_stage) and not _should_run(
        "upsert", from_stage, to_stage
    ):
        active_source_keys = []

    should_embed = _should_run("embed", from_stage, to_stage)
    should_upsert = _should_run("upsert", from_stage, to_stage)
    max_workers = 1
    if active_source_keys and (should_embed or should_upsert):
        configured_workers = max(1, settings.ingestion_max_workers)
        if settings.embedding_local_mode.lower().strip() == "hf_inference":
            configured_workers = min(configured_workers, max(1, settings.hf_max_parallel_sources))
        max_workers = min(configured_workers, len(active_source_keys))

    log.info("Embed/upsert worker count: %d", max_workers)

    def process_source(source_key: str) -> tuple[int, int]:
        chunks_path, embeddings_path, manifest_path = _artifact_paths(resolved_strategy, source_key)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        if source_key in chunk_cache:
            chunks = chunk_cache[source_key]
        else:
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"Missing chunks artifact for {source_key}: {chunks_path}. "
                    "Run from stage 'chunk' first with the same --chunking-strategy."
                )
            chunks = read_chunks_jsonl(chunks_path)

        if should_embed:
            vectors: list[list[float]] = []

            for batch_start in range(0, len(chunks), INGEST_BATCH_SIZE):
                batch = chunks[batch_start : batch_start + INGEST_BATCH_SIZE]
                log.info(
                    "  [%s] embed batch %d-%d of %d",
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
                    qdrant_collection=resolved_collection,
                    chunking_strategy=resolved_strategy,
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
                    "Run from stage 'embed' first with the same --chunking-strategy."
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

        upserted = 0
        if should_upsert:
            if store is None:
                raise RuntimeError("Qdrant store not initialized for upsert stage.")

            if not should_embed:
                manifest = read_embedding_manifest(manifest_path)
                current_model = _embedding_model_name()
                if manifest.embedding_model and manifest.embedding_model != current_model:
                    log.warning(
                        "Embedding model differs for %s (artifact=%s, current=%s).",
                        source_key,
                        manifest.embedding_model,
                        current_model,
                    )

            upserted = store.upsert_chunks(chunks, vectors)
            log.info("Upserted %d chunk(s) for %s", upserted, source_key)

        return len(chunks), upserted

    if max_workers == 1:
        for source_key in active_source_keys:
            chunk_count, upserted = process_source(source_key)
            total_chunks += chunk_count
            total_upserted += upserted
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest") as executor:
            futures = {
                executor.submit(process_source, source_key): source_key
                for source_key in active_source_keys
            }
            for future in as_completed(futures):
                source_key = futures[future]
                try:
                    chunk_count, upserted = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Ingestion failed for source={source_key}, strategy={resolved_strategy}"
                    ) from exc
                total_chunks += chunk_count
                total_upserted += upserted

    elapsed = time.time() - t_start
    final_count = store.count(exact=True) if store is not None else 0

    lf.flush()

    log.info("=" * 60)
    log.info("✓ Ingestion complete")
    log.info("  Sources processed:          %d", len(source_keys))
    log.info("  Chunks prepared:            %d", total_chunks)
    log.info("  Chunking strategy:          %s", resolved_strategy)
    log.info("  Qdrant collection:          %s", resolved_collection)
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
    parser.add_argument(
        "--chunking-strategy",
        choices=CHUNKING_STRATEGIES,
        default=settings.chunking_strategy,
        help="Chunking strategy for this ingestion run",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Qdrant collection name for this ingestion run",
    )
    parser.add_argument("--test", action="store_true", help="Run retrieval test after ingestion")
    args = parser.parse_args()

    run_ingestion(
        sources=args.sources,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        chunking_strategy=args.chunking_strategy,
        collection_name=args.collection,
    )
