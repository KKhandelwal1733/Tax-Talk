from __future__ import annotations

import argparse
from pathlib import Path

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_logger
from tax_talk.evals.dataset import load_golden_dataset
from tax_talk.evals.runner import dump_retrieval_rows, run_from_settings, score_from_dump
from tax_talk.retrieval.hybrid import HybridRetriever

log = get_logger(__name__)

_STAGES = ("retrieve", "score", "all")


def main() -> None:
    """CLI entrypoint for evaluation runs.

    Stages:
      retrieve   Run retrieval + generation, write JSONL dump. No RAGAS scoring.
      score      Score an existing JSONL dump with RAGAS. No LLM retrieval.
      all        Run retrieve then score in one shot.

    Omit --stage to run all stages automatically.

    Examples:
      # Full run (no --stage needed)
      python -m tax_talk.evals.run --collection tax_2024

      # Explicit full run
      python -m tax_talk.evals.run --stage all --collection tax_2024

      # Stage 1 only — inspect answers before scoring
      python -m tax_talk.evals.run --stage retrieve --collection tax_2024

      # Stage 2 only — re-score a frozen dump
      python -m tax_talk.evals.run --stage score --dump-path eval_results/retrieval-semantic-tax_2024-....jsonl
    """
    parser = argparse.ArgumentParser(
        description="RAGAS eval CLI — run retrieval, scoring, or both.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__,
    )

    parser.add_argument(
        "--stage",
        choices=_STAGES,
        default=None,
        help="Stage to run: 'retrieve', 'score', or 'all'. Omit to run all stages.",
    )

    # -- retrieve args -------------------------------------------------------
    retrieve_group = parser.add_argument_group("retrieve stage")
    retrieve_group.add_argument(
        "--collection", default=settings.qdrant_collection, help="Qdrant collection name"
    )
    retrieve_group.add_argument(
        "--chunking-strategy", default=settings.chunking_strategy, help="Chunking strategy label"
    )
    retrieve_group.add_argument(
        "--output-dir", default=settings.eval_results_dir, help="Directory for output files"
    )

    retrieve_group.add_argument(
        "--dataset", default=settings.eval_dataset_path, help="Path to golden QA JSONL"
    )
    retrieve_group.add_argument(
        "--provider", default=settings.eval_provider, help="LLM provider for answer generation"
    )
    retrieve_group.add_argument(
        "--model", default=settings.eval_model, help="Model name for generation"
    )
    retrieve_group.add_argument(
        "--top-k", type=int, default=settings.eval_top_k, help="Retriever top_k per query"
    )
    retrieve_group.add_argument(
        "--fallback-chain",
        default=settings.eval_fallback_chain_csv,
        help="Comma-separated fallback pairs e.g. 'groq/llama3-70b,gemini/gemini-2.0-flash-lite'",
    )

    # -- score args ----------------------------------------------------------
    score_group = parser.add_argument_group("score stage")
    score_group.add_argument(
        "--dump-path",
        type=Path,
        default=None,
        help="Path to a retrieval JSONL dump (required when --stage score)",
    )
    score_group.add_argument(
        "--score-output-dir",
        type=Path,
        default=None,
        help="Override output dir for scored results (defaults to --output-dir)",
    )

    args = parser.parse_args()

    # -- resolve stage -------------------------------------------------------
    stage = args.stage or "all"
    output_dir = Path(args.output_dir)
    score_output_dir = args.score_output_dir or output_dir

    # -- validation ----------------------------------------------------------
    if stage == "score" and args.dump_path is None:
        parser.error("--stage score requires --dump-path")

    if stage == "score" and not args.dump_path.exists():
        parser.error(f"dump file not found: {args.dump_path}")

    # -- dispatch ------------------------------------------------------------
    if stage == "retrieve":
        samples = load_golden_dataset(Path(args.dataset))
        retriever = HybridRetriever(collection_name=args.collection)
        dump_path = dump_retrieval_rows(
            samples=samples,
            retriever=retriever,
            provider=args.provider,
            model=args.model,
            collection=args.collection,
            chunking_strategy=args.chunking_strategy,
            top_k=args.top_k,
            output_dir=output_dir,
            fallback_chain=None,
        )
        log.info("Retrieve stage complete. Dump written: %s", dump_path)
        log.info("Inspect answers, then run: --stage score --dump-path %s", dump_path)

    elif stage == "score":
        result_path = score_from_dump(
            dump_path=args.dump_path,
            output_dir=score_output_dir,
        )
        log.info("Score stage complete: %s", result_path)

    else:  # all — explicitly passed or omitted
        result_path = run_from_settings(
            collection=args.collection,
            chunking_strategy=args.chunking_strategy,
            dataset_path=args.dataset,
            provider=args.provider,
            model=args.model,
            top_k=args.top_k,
            output_dir=str(output_dir),
            fallback_chain_csv=args.fallback_chain,
        )
        log.info("Eval completed: %s", result_path)


if __name__ == "__main__":
    main()
