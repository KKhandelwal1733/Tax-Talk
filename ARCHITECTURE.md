from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langfuse import observe
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

from tax_talk.core.config import settings
from tax_talk.core.rate_limit import SlidingWindowRateLimiter
from tax_talk.core.runtime import get_langfuse_client, get_llm_strategy, get_logger
from tax_talk.evals.dataset import GoldenQASample, load_golden_dataset
from tax_talk.retrieval.hybrid import HybridRetriever

_VALID_PROVIDERS: frozenset[str] = frozenset({"gemini", "groq"})

# 4 metrics × 3 samples = 12 calls per batch < 15 RPM limit.
_RAGAS_BATCH_SIZE: int = 1
_RAGAS_BATCH_SLEEP_SECONDS: int = 62

log = get_logger(__name__)
_eval_answer_rate_limiter = SlidingWindowRateLimiter(
    max_calls=settings.eval_llm_rate_limit_calls,
    window_seconds=settings.eval_llm_rate_limit_window_seconds,
)


# ---------------------------------------------------------------------------
# RAGAS evaluator — built lazily on first scoring call
# ---------------------------------------------------------------------------

def _build_ragas_evaluator() -> tuple[Any, Any]:
    """Build RAGAS LLM judge and Google embeddings lazily.

    Uses native google-genai SDK via the runtime Gemini client singleton.
    GoogleEmbeddings auto-configures for the same provider as the LLM.

    Returns:
        Tuple of (evaluator_llm, evaluator_embeddings).
    """
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory

    from tax_talk.core.runtime import get_gemini_client

    gemini_client = get_gemini_client()

    evaluator_llm = llm_factory(
        "gemini-3.1-flash-lite",
        provider="google",
        client=gemini_client,
    )
    evaluator_embeddings = GoogleEmbeddings(
        client=gemini_client,
        model="gemini-embedding-001",
    )
    return evaluator_llm, evaluator_embeddings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_fallback_chain(csv: str) -> list[tuple[str, str]]:
    """Parse a CSV fallback chain string into an ordered list of (provider, model) pairs.

    Args:
        csv: Comma-separated entries in ``provider/model`` format, e.g.
            ``"groq/llama3-70b,gemini/gemini-2.0-flash-lite"``.
            Empty string returns an empty list.

    Returns:
        Ordered list of (provider, model) tuples.

    Raises:
        ValueError: If any entry is malformed or contains an unsupported provider.
    """
    if not csv.strip():
        return []
    pairs: list[tuple[str, str]] = []
    for raw in csv.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "/" not in entry or entry.count("/") != 1:
            raise ValueError(
                f"Invalid fallback chain entry '{entry}': expected 'provider/model' format."
            )
        provider, model = entry.split("/", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}' in fallback chain."
                f" Supported: {sorted(_VALID_PROVIDERS)}"
            )
        if not model:
            raise ValueError(f"Empty model name in fallback chain entry '{entry}'.")
        pairs.append((provider, model))
    return pairs


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    """Per-sample intermediate payload — retrieval output before RAGAS scoring."""

    id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    collection: str
    chunking_strategy: str
    latency_ms: float


def _rows_from_jsonl(path: Path) -> list[EvalRow]:
    """Load EvalRows from a JSONL retrieval dump.

    Args:
        path: Path to a JSONL file written by dump_retrieval_rows().

    Returns:
        List of EvalRow instances.
    """
    rows: list[EvalRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(EvalRow(**json.loads(line)))
    return rows


# ---------------------------------------------------------------------------
# Stage 1 — retrieve + generate, dump to JSONL
# ---------------------------------------------------------------------------

@observe(name="eval-generate-answer", as_type="generation", capture_input=False, capture_output=False)
def _generate_grounded_answer(
    *,
    question: str,
    contexts: list[str],
    provider: str,
    model: str,
    fallback_chain: list[tuple[str, str]] | None = None,
) -> str:
    """Generate a grounded answer from retrieved contexts, with optional provider fallbacks.

    Args:
        question: The user question to answer.
        contexts: Retrieved context passages to ground the answer.
        provider: Primary LLM provider name.
        model: Primary model name.
        fallback_chain: Ordered list of (provider, model) fallback pairs tried if primary fails.

    Returns:
        Generated answer string, stripped of surrounding whitespace.

    Raises:
        RuntimeError: If all provider attempts (primary + fallbacks) fail.
    """
    attempts = [(provider, model)] + (fallback_chain or [])
    joined_context = "\n\n".join(contexts[:8])
    prompt = (
        "Answer the tax question using only the provided context. "
        "If context is insufficient, say so explicitly. Keep answer concise.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{joined_context}"
    )
    failures: list[str] = []
    for idx, (prov, mod) in enumerate(attempts):
        try:
            _eval_answer_rate_limiter.wait_for_slot()
            strategy = get_llm_strategy(prov)
            response = strategy.generate(prompt=prompt, model=mod)
            if idx > 0:
                log.info(
                    "eval-generate-answer: succeeded on fallback attempt %d/%d (%s/%s)",
                    idx + 1,
                    len(attempts),
                    prov,
                    mod,
                )
            return response.strip() if isinstance(response, str) else ""
        except Exception as exc:
            failures.append(f"attempt {idx + 1} ({prov}/{mod}): {exc}")
            log.warning(
                "eval-generate-answer: attempt %d/%d failed (%s/%s): %s",
                idx + 1,
                len(attempts),
                prov,
                mod,
                exc,
            )
    raise RuntimeError(
        f"All {len(attempts)} eval generation attempt(s) failed. "
        + "; ".join(failures)
    )


@observe(name="eval-build-rows", as_type="span", capture_input=False, capture_output=False)
def build_eval_rows(
    *,
    samples: list[GoldenQASample],
    retriever: HybridRetriever,
    provider: str,
    model: str,
    collection: str,
    chunking_strategy: str,
    top_k: int,
    fallback_chain: list[tuple[str, str]] | None = None,
) -> list[EvalRow]:
    """Run retrieval+generation for each sample and return normalized rows.

    Args:
        samples: Golden QA samples to evaluate.
        retriever: Hybrid retriever instance targeting the eval collection.
        provider: Primary LLM provider for answer generation.
        model: Primary model name for generation.
        collection: Qdrant collection name (stored in result metadata).
        chunking_strategy: Strategy label (stored in result metadata).
        top_k: Number of retrieved chunks per question.
        fallback_chain: Optional ordered fallback (provider, model) pairs.

    Returns:
        List of EvalRow instances with answers, contexts, and latencies.
    """
    rows: list[EvalRow] = []
    for sample in samples:
        started = time.perf_counter()
        hits = retriever.retrieve(sample.question, top_k=top_k)
        contexts = [str(hit.get("text", "")).strip() for hit in hits if hit.get("text")]
        answer = _generate_grounded_answer(
            question=sample.question,
            contexts=contexts,
            provider=provider,
            model=model,
            fallback_chain=fallback_chain,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            EvalRow(
                id=sample.id,
                question=sample.question,
                answer=answer,
                contexts=contexts,
                ground_truth=sample.expected_answer,
                collection=collection,
                chunking_strategy=chunking_strategy,
                latency_ms=elapsed_ms,
            )
        )
    return rows


@observe(name="eval-dump-rows", as_type="span", capture_input=False, capture_output=False)
def dump_retrieval_rows(
    *,
    samples: list[GoldenQASample],
    retriever: HybridRetriever,
    provider: str,
    model: str,
    collection: str,
    chunking_strategy: str,
    top_k: int,
    output_dir: Path,
    fallback_chain: list[tuple[str, str]] | None = None,
) -> Path:
    """Run retrieval+generation and persist raw rows to JSONL — no scoring yet.

    Args:
        samples: Golden QA samples to evaluate.
        retriever: Hybrid retriever targeting the eval collection.
        provider: Primary LLM provider for answer generation.
        model: Primary model name.
        collection: Qdrant collection name.
        chunking_strategy: Strategy label for result metadata.
        top_k: Retrieval cutoff per question.
        output_dir: Directory to write the dump file.
        fallback_chain: Optional ordered fallback (provider, model) pairs.

    Returns:
        Path to the written JSONL dump file.
    """
    rows = build_eval_rows(
        samples=samples,
        retriever=retriever,
        provider=provider,
        model=model,
        collection=collection,
        chunking_strategy=chunking_strategy,
        top_k=top_k,
        fallback_chain=fallback_chain,
    )

    stamped = time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / f"retrieval-{chunking_strategy}-{collection}-{stamped}.jsonl"

    with dump_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")

    get_langfuse_client().flush()
    log.info("Wrote retrieval dump (%d rows): %s", len(rows), dump_path)
    return dump_path


# ---------------------------------------------------------------------------
# Stage 2 — score a dump with RAGAS (no LLM retrieval)
# ---------------------------------------------------------------------------

@observe(name="eval-ragas", as_type="span", capture_input=False, capture_output=False)
def compute_ragas_scores(rows: list[EvalRow]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute RAGAS metrics using Gemini as the LLM judge.

    Processes rows in batches to stay within the 15 RPM rate limit.
    Uses class-based metrics with explicit LLM/embeddings binding.
    Only AnswerCorrectness requires embeddings.

    Args:
        rows: Eval rows containing questions, answers, contexts, and ground truths.

    Returns:
        Tuple of (aggregate metric dict, per-row metric list).
    """
    evaluator_llm, evaluator_embeddings = _build_ragas_evaluator()
    run_config = RunConfig(max_workers=1)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    metric_cols = ["faithfulness", "context_precision", "context_recall", "answer_correctness"]
    # metric_cols = ["faithfulness", "answer_correctness"]
    all_scores: dict[str, list[float]] = {col: [] for col in metric_cols}
    all_row_metrics: list[dict[str, Any]] = []
    total_batches = math.ceil(len(rows) / _RAGAS_BATCH_SIZE)

    for batch_idx, start in enumerate(range(0, len(rows), _RAGAS_BATCH_SIZE)):
        batch = rows[start : start + _RAGAS_BATCH_SIZE]

        if batch_idx > 0:
            log.info(
                "Rate limit pause: sleeping %ds before batch %d/%d ...",
                _RAGAS_BATCH_SLEEP_SECONDS,
                batch_idx + 1,
                total_batches,
            )
            time.sleep(_RAGAS_BATCH_SLEEP_SECONDS)

        log.info(
            "Scoring batch %d/%d (%d rows) ...",
            batch_idx + 1,
            total_batches,
            len(batch),
        )

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=row.question,
                    response=row.answer,
                    retrieved_contexts=row.contexts,
                    reference=row.ground_truth,
                )
                for row in batch
            ]
        )

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=run_config,
        )

        frame = result.to_pandas()  # type: ignore[attr-defined]
        for col in metric_cols:
            if col in frame.columns:
                all_scores[col].extend(frame[col].dropna().tolist())
        all_row_metrics.extend(frame.to_dict(orient="records"))

        log.info("Batch %d/%d scored.", batch_idx + 1, total_batches)

    aggregates = {
        col: float(sum(vals) / len(vals))
        for col, vals in all_scores.items()
        if vals
    }
    return aggregates, all_row_metrics


@observe(name="eval-score-dump", as_type="span", capture_input=True, capture_output=False)
def score_from_dump(*, dump_path: Path, output_dir: Path) -> Path:
    """Score a previously written retrieval dump with RAGAS — no LLM retrieval.

    Args:
        dump_path: Path to a JSONL file written by dump_retrieval_rows().
        output_dir: Directory to write the scored result JSON.

    Returns:
        Path to the written result JSON file.
    """
    rows = _rows_from_jsonl(dump_path)
    if not rows:
        raise ValueError(f"No rows found in dump: {dump_path}")

    log.info("Scoring %d rows from dump: %s", len(rows), dump_path)
    aggregates, ragas_rows = compute_ragas_scores(rows)

    first = rows[0]
    stamped = time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"scored-{first.chunking_strategy}-{first.collection}-{stamped}.json"

    latencies = [r.latency_ms for r in rows]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)

    payload = {
        "source_dump": str(dump_path),
        "collection": first.collection,
        "chunking_strategy": first.chunking_strategy,
        "sample_count": len(rows),
        "metrics": {
            **aggregates,
            "latency_p95_ms": sorted_latencies[p95_index],
        },
        "samples": [asdict(row) for row in rows],
        "ragas_rows": ragas_rows,
    }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    get_langfuse_client().flush()
    log.info("Wrote scored results: %s", result_path)
    return result_path


# ---------------------------------------------------------------------------
# Stage 1+2 combined — single-shot entry point
# ---------------------------------------------------------------------------

@observe(name="eval-run", as_type="span", capture_input=True, capture_output=False)
def run_eval(
    *,
    collection: str,
    chunking_strategy: str,
    dataset_path: Path,
    provider: str,
    model: str,
    top_k: int,
    output_dir: Path,
    fallback_chain: list[tuple[str, str]] | None = None,
) -> Path:
    """Run one eval sweep (retrieve + score) and persist artifacts.

    Dump is always written before scoring begins. If scoring fails, the dump
    survives and can be re-scored with score_from_dump().

    Args:
        collection: Qdrant collection to query.
        chunking_strategy: Strategy label used for result metadata.
        dataset_path: JSONL golden dataset path.
        provider: LLM provider used for answer generation.
        model: Model name for generation.
        top_k: Retrieval cutoff per question.
        output_dir: Parent directory for eval result files.
        fallback_chain: Optional ordered fallback (provider, model) pairs.

    Returns:
        Path to the written scored result JSON file.
    """
    samples = load_golden_dataset(dataset_path)
    retriever = HybridRetriever(collection_name=collection)

    dump_path = dump_retrieval_rows(
        samples=samples,
        retriever=retriever,
        provider=provider,
        model=model,
        collection=collection,
        chunking_strategy=chunking_strategy,
        top_k=top_k,
        output_dir=output_dir,
        fallback_chain=fallback_chain,
    )

    return score_from_dump(dump_path=dump_path, output_dir=output_dir)


def run_from_settings(
    *,
    collection: str,
    chunking_strategy: str,
    dataset_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
    output_dir: str | None = None,
    fallback_chain_csv: str | None = None,
) -> Path:
    """Resolve defaults from settings and execute one eval run.

    Args:
        collection: Qdrant collection to evaluate.
        chunking_strategy: Chunking strategy label for result metadata.
        dataset_path: Override path to golden QA JSONL dataset.
        provider: Override LLM provider (default: settings.eval_provider).
        model: Override model name (default: settings.eval_model).
        top_k: Override retriever top_k (default: settings.eval_top_k).
        output_dir: Override results output directory.
        fallback_chain_csv: Override fallback chain CSV (default: settings.eval_fallback_chain_csv).

    Returns:
        Path to the written scored result JSON file.
    """
    raw_csv = (
        fallback_chain_csv
        if fallback_chain_csv is not None
        else settings.eval_fallback_chain_csv
    )
    fallback_chain = _parse_fallback_chain(raw_csv)
    return run_eval(
        collection=collection,
        chunking_strategy=chunking_strategy,
        dataset_path=Path(dataset_path or settings.eval_dataset_path),
        provider=(provider or settings.eval_provider).strip(),
        model=(model or settings.eval_model).strip(),
        top_k=top_k or settings.eval_top_k,
        output_dir=Path(output_dir or settings.eval_results_dir),
        fallback_chain=fallback_chain,
    )