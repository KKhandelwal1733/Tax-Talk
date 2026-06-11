from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

from langfuse import observe
from tax_talk.core.config import settings
from tax_talk.core.rate_limit import SlidingWindowRateLimiter
from tax_talk.core.runtime import get_langfuse_client, get_llm_strategy, get_logger
from tax_talk.evals.dataset import GoldenQASample, load_golden_dataset
from tax_talk.retrieval.hybrid import HybridRetriever

_VALID_PROVIDERS: frozenset[str] = frozenset({"gemini", "groq"})
_RAGAS_BATCH_SIZE: int = 1
_RAGAS_BATCH_SLEEP_SECONDS: int = 62

log = get_logger(__name__)
_eval_answer_rate_limiter = SlidingWindowRateLimiter(
    max_calls=settings.eval_llm_rate_limit_calls,
    window_seconds=settings.eval_llm_rate_limit_window_seconds,
)

_ANSWER_PROMPT_TEMPLATE = (
    "You are an expert Tax AI Assistant specialized in Indian Tax Statutes "
    "(CGST Act 2017, IGST Act 2017, Income-tax Act 1961, and Income-tax Act 2025) "
    "and CBIC circulars, notifications, and GST Council decisions.\n"
    "Your task is to answer the following user question by analyzing and synthesizing the provided tax context chunks.\n\n"
    "CRITICAL CONSTRAINTS & REASONING GUIDELINES:\n"
    "1. SYNTHESIZE: Read the provided context fragments collectively. If the complete answer requires "
    "combining information from multiple chunks (e.g., a general rule in one chunk and an exception or "
    "threshold in another), combine them into a single coherent legal conclusion. Do not treat any single "
    "chunk as the whole answer.\n"
    "2. AMENDMENTS FIRST: Each chunk is labelled with its source, document type, applicable period, and "
    "section reference where available. If a chunk is marked as 'amended' or carries a later applicable "
    "period than another chunk covering the same provision, prefer the amended version and explicitly note "
    "the change with its effective date.\n"
    "3. CIRCULAR vs. STATUTE: Circulars and notifications clarify but do not override statute. Where both "
    "are present, state the statutory basis first and then the clarification from the circular.\n"
    "4. STATUTORY SYNONYMS: Do not return a negative answer due to terminology mismatch. Recognize "
    "equivalent terms across statutes, for example: 'clinical establishment' or 'authorized medical "
    "practitioner' maps to healthcare or hospital context; 'renting of immovable property' maps to "
    "leasing or letting out property; 'consideration' maps to price or payment for supply.\n"
    "5. TEMPORAL FLAGS: If the answer applies only from a specific date, financial year, or notification "
    "effective date, state this explicitly in your response.\n"
    "6. MANDATORY GROUNDING: Rely strictly on the facts present in the provided context. Do not invent "
    "or assume section numbers, notification numbers, rates, thresholds, or dates not present in the text. "
    "If the context is genuinely insufficient to derive a reliable legal conclusion, state clearly: "
    "'The provided context does not specify [specific missing information].'\n\n"
    "OUTPUT FORMAT:\n"
    "- Begin with a direct answer to the question in 1 to 3 sentences.\n"
    "- Follow with supporting detail in bullet points if the answer involves multiple conditions, rates, "
    "thresholds, or exceptions. Use prose for simple single-rule answers.\n"
    "- End with a 'Legal Basis:' line citing the specific sections, notifications, or circulars referenced "
    "in your answer, using the format: Section X, [Act Name] [Year]; Circular No., date.\n"
    "- Keep the total response under 300 words unless the question requires a detailed slab or rate structure.\n\n"
    "Question: {question}\n\n"
    "Context Chunks:\n{context}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_fallback_chain(csv: str) -> list[tuple[str, str]]:
    """Parse ``"provider/model,provider/model,..."`` into a list of (provider, model) pairs."""
    if not csv.strip():
        return []
    pairs: list[tuple[str, str]] = []
    for raw in csv.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "/" not in entry or entry.count("/") != 1:
            raise ValueError(f"Invalid fallback entry '{entry}': expected 'provider/model'.")
        provider, model = entry.split("/", 1)
        provider, model = provider.strip().lower(), model.strip()
        if provider not in _VALID_PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Supported: {sorted(_VALID_PROVIDERS)}")
        if not model:
            raise ValueError(f"Empty model in fallback entry '{entry}'.")
        pairs.append((provider, model))
    return pairs


def _p95(latencies: list[float]) -> float:
    """Return the 95th-percentile value from a list of latencies."""
    s = sorted(latencies)
    idx = min(len(s) - 1, math.ceil(len(s) * 0.95) - 1)
    return s[max(0, idx)]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    """Per-sample payload: retrieval output before RAGAS scoring."""
    id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    collection: str
    chunking_strategy: str
    latency_ms: float


def _rows_from_jsonl(path: Path) -> list[EvalRow]:
    return [
        EvalRow(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# RAGAS evaluator — builds ordered list of (llm, embeddings) candidates
# ---------------------------------------------------------------------------

_ragas_evaluators: list[tuple[Any, Any]] | None = None


def _build_ragas_evaluators() -> list[tuple[Any, Any]]:
    """Build an ordered list of RAGAS (llm, embeddings) candidates from the full attempt chain.

    Each valid LLM judge is paired with the shared Gemini embeddings instance.
    During scoring, if one candidate fails the next is tried automatically.
    """
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from tax_talk.core.runtime import get_gemini_client, get_groq_client

    primary_provider = settings.eval_provider.strip().lower()
    primary_model = settings.eval_model.strip()
    fallback_chain = _parse_fallback_chain(settings.eval_fallback_chain_csv)
    all_attempts = [(primary_provider, primary_model)] + fallback_chain

    # --- Embeddings (always Gemini) ---
    gemini_attempts = [(p, m) for p, m in all_attempts if p == "gemini"]
    if not gemini_attempts:
        raise RuntimeError(
            f"RAGAS evaluator requires at least one 'gemini' entry for embeddings. Got: {all_attempts}"
        )

    embeddings = None
    last_exc: Exception | None = None
    for _, model in gemini_attempts:
        try:
            embeddings = GoogleEmbeddings(client=get_gemini_client(), model="gemini-embedding-001")
            log.info("RAGAS embeddings initialised via gemini/%s", model)
            break
        except Exception as exc:
            last_exc = exc
            log.warning("RAGAS embeddings failed via gemini/%s: %s", model, exc)

    if embeddings is None:
        raise RuntimeError(f"Could not initialise RAGAS embeddings. Last error: {last_exc}")

    # --- LLM judges (one per viable provider entry) ---
    candidates: list[tuple[Any, Any]] = []
    for provider, model in all_attempts:
        try:
            if provider == "gemini":
                llm = llm_factory(model, provider="google", client=get_gemini_client())
            elif provider == "groq":
                llm = llm_factory(model, provider="groq", client=get_groq_client())
            else:
                log.warning("RAGAS evaluator: unsupported provider '%s', skipping.", provider)
                continue
            candidates.append((llm, embeddings))
            log.info("RAGAS LLM judge candidate ready: %s/%s", provider, model)
        except Exception as exc:
            log.warning("RAGAS LLM judge init failed %s/%s: %s", provider, model, exc)

    if not candidates:
        raise RuntimeError(f"No RAGAS LLM judge could be initialised. Tried: {all_attempts}")
    return candidates


def _get_ragas_evaluators() -> list[tuple[Any, Any]]:
    """Return cached evaluator candidate list, building on first call."""
    global _ragas_evaluators
    if _ragas_evaluators is None:
        _ragas_evaluators = _build_ragas_evaluators()
    return _ragas_evaluators


def _reset_ragas_evaluators() -> None:
    """Reset the cache — for testing only."""
    global _ragas_evaluators
    _ragas_evaluators = None


# ---------------------------------------------------------------------------
# Stage 1 — retrieve + generate
# ---------------------------------------------------------------------------


@observe(name="eval-generate-answer", as_type="generation", capture_input=False, capture_output=False)
def _generate_grounded_answer(
    *, question: str, contexts: list[str], provider: str, model: str,
    fallback_chain: list[tuple[str, str]] | None = None,
) -> str:
    """Generate a grounded answer from retrieved contexts, with provider fallbacks."""
    attempts = [(provider, model)] + (fallback_chain or [])
    prompt = _ANSWER_PROMPT_TEMPLATE.format(
        question=question, context="\n\n".join(contexts[:8]),
    )
    failures: list[str] = []
    for idx, (prov, mod) in enumerate(attempts):
        try:
            _eval_answer_rate_limiter.wait_for_slot()
            response = get_llm_strategy(prov).generate(prompt=prompt, model=mod)
            if idx > 0:
                log.info("eval-generate-answer: succeeded on fallback %d/%d (%s/%s)", idx + 1, len(attempts), prov, mod)
            return response.strip() if isinstance(response, str) else ""
        except Exception as exc:
            failures.append(f"attempt {idx + 1} ({prov}/{mod}): {exc}")
            log.warning("eval-generate-answer: attempt %d/%d failed (%s/%s): %s", idx + 1, len(attempts), prov, mod, exc)
    raise RuntimeError(f"All {len(attempts)} eval generation attempt(s) failed. " + "; ".join(failures))


@observe(name="eval-build-rows", as_type="span", capture_input=False, capture_output=False)
def build_eval_rows(
    *, samples: list[GoldenQASample], retriever: HybridRetriever,
    provider: str, model: str, collection: str, chunking_strategy: str,
    top_k: int, fallback_chain: list[tuple[str, str]] | None = None,
) -> list[EvalRow]:
    """Run retrieval+generation for each sample and return EvalRows."""
    rows: list[EvalRow] = []
    for sample in samples:
        started = time.perf_counter()
        hits = retriever.retrieve(sample.question, top_k=top_k)
        contexts = [str(h.get("text", "")).strip() for h in hits if h.get("text")]
        answer = _generate_grounded_answer(
            question=sample.question, contexts=contexts,
            provider=provider, model=model, fallback_chain=fallback_chain,
        )
        rows.append(EvalRow(
            id=sample.id, question=sample.question, answer=answer, contexts=contexts,
            ground_truth=sample.expected_answer, collection=collection,
            chunking_strategy=chunking_strategy, latency_ms=(time.perf_counter() - started) * 1000.0,
        ))
    return rows


@observe(name="eval-dump-rows", as_type="span", capture_input=False, capture_output=False)
def dump_retrieval_rows(
    *, samples: list[GoldenQASample], retriever: HybridRetriever,
    provider: str, model: str, collection: str, chunking_strategy: str,
    top_k: int, output_dir: Path, fallback_chain: list[tuple[str, str]] | None = None,
) -> Path:
    """Run retrieval+generation and persist raw rows to JSONL."""
    rows = build_eval_rows(
        samples=samples, retriever=retriever, provider=provider, model=model,
        collection=collection, chunking_strategy=chunking_strategy,
        top_k=top_k, fallback_chain=fallback_chain,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / f"retrieval-{chunking_strategy}-{collection}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    dump_path.write_text(
        "\n".join(json.dumps(asdict(r)) for r in rows) + "\n", encoding="utf-8",
    )
    get_langfuse_client().flush()
    log.info("Wrote retrieval dump (%d rows): %s", len(rows), dump_path)
    return dump_path


# ---------------------------------------------------------------------------
# Stage 2 — score with RAGAS (evaluator fallback on failure)
# ---------------------------------------------------------------------------


def _build_metrics(llm: Any, embeddings: Any) -> list:
    """Instantiate RAGAS metrics bound to the given llm/embeddings."""
    return [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]


def _score_batch(
    batch: list[EvalRow], evaluators: list[tuple[Any, Any]], run_config: RunConfig,
) -> Any:
    """Score a single batch, falling back through evaluator candidates on failure.

    Raises:
        RuntimeError: If all evaluators fail for this batch.
    """
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r.question, response=r.answer,
            retrieved_contexts=r.contexts, reference=r.ground_truth,
        ) for r in batch
    ])
    last_exc: Exception | None = None
    for idx, (llm, embeddings) in enumerate(evaluators):
        try:
            return evaluate(
                dataset=dataset,
                metrics=_build_metrics(llm, embeddings),
                run_config=run_config,
            )
        except Exception as exc:
            last_exc = exc
            log.warning("RAGAS scoring failed with evaluator %d/%d: %s", idx + 1, len(evaluators), exc)
    raise RuntimeError(f"All {len(evaluators)} RAGAS evaluator(s) failed for batch. Last: {last_exc}")


_FAILED_SENTINEL: dict[str, float | None] = {
    "faithfulness": None, "context_precision": None,
    "context_recall": None, "answer_correctness": None,
}


@observe(name="eval-ragas", as_type="span", capture_input=False, capture_output=False)
def compute_ragas_scores(rows: list[EvalRow]) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    """Compute RAGAS metrics with evaluator fallback and graceful batch failure.

    Returns:
        Tuple of (aggregate metrics, per-row metrics, list of failed sample IDs).
    """
    evaluators = _get_ragas_evaluators()
    run_config = RunConfig(max_workers=1)
    metric_cols = ["faithfulness", "context_precision", "context_recall", "answer_correctness"]
    all_scores: dict[str, list[float]] = {c: [] for c in metric_cols}
    all_row_metrics: list[dict[str, Any]] = []
    failed_sample_ids: list[str] = []
    total_batches = math.ceil(len(rows) / _RAGAS_BATCH_SIZE)

    for batch_idx, start in enumerate(range(0, len(rows), _RAGAS_BATCH_SIZE)):
        if batch_idx > 0:
            log.info("Rate limit pause: sleeping %ds before batch %d/%d ...", _RAGAS_BATCH_SLEEP_SECONDS, batch_idx + 1, total_batches)
            time.sleep(_RAGAS_BATCH_SLEEP_SECONDS)

        batch = rows[start : start + _RAGAS_BATCH_SIZE]
        log.info("Scoring batch %d/%d (%d rows) ...", batch_idx + 1, total_batches, len(batch))

        try:
            result = _score_batch(batch, evaluators, run_config)
            frame = result.to_pandas()
            for col in metric_cols:
                if col in frame.columns:
                    all_scores[col].extend(frame[col].dropna().tolist())
            all_row_metrics.extend(frame.to_dict(orient="records"))
            log.info("Batch %d/%d scored.", batch_idx + 1, total_batches)
        except RuntimeError:
            batch_ids = [r.id for r in batch]
            failed_sample_ids.extend(batch_ids)
            for r in batch:
                all_row_metrics.append({"id": r.id, "status": "scoring_failed", **_FAILED_SENTINEL})
            log.error("Batch %d/%d FAILED — all evaluators exhausted. Skipping samples: %s", batch_idx + 1, total_batches, batch_ids)

    aggregates = {c: sum(v) / len(v) for c, v in all_scores.items() if v}
    if failed_sample_ids:
        log.warning("%d/%d samples failed scoring: %s", len(failed_sample_ids), len(rows), failed_sample_ids)
    return aggregates, all_row_metrics, failed_sample_ids


@observe(name="eval-score-dump", as_type="span", capture_input=True, capture_output=False)
def score_from_dump(*, dump_path: Path, output_dir: Path) -> Path:
    """Score a previously written retrieval dump with RAGAS."""
    rows = _rows_from_jsonl(dump_path)
    if not rows:
        raise ValueError(f"No rows found in dump: {dump_path}")

    log.info("Scoring %d rows from dump: %s", len(rows), dump_path)
    aggregates, ragas_rows, failed_ids = compute_ragas_scores(rows)

    first = rows[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"scored-{first.chunking_strategy}-{first.collection}-{time.strftime('%Y%m%d-%H%M%S')}.json"

    payload = {
        "source_dump": str(dump_path),
        "collection": first.collection,
        "chunking_strategy": first.chunking_strategy,
        "sample_count": len(rows),
        "scored_count": len(rows) - len(failed_ids),
        "failed_count": len(failed_ids),
        "failed_sample_ids": failed_ids,
        "metrics": {**aggregates, "latency_p95_ms": _p95([r.latency_ms for r in rows])},
        "samples": [asdict(r) for r in rows],
        "ragas_rows": ragas_rows,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    get_langfuse_client().flush()
    log.info("Wrote scored results: %s", result_path)
    return result_path


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


@observe(name="eval-run", as_type="span", capture_input=True, capture_output=False)
def run_eval(
    *, collection: str, chunking_strategy: str, dataset_path: Path,
    provider: str, model: str, top_k: int, output_dir: Path,
    fallback_chain: list[tuple[str, str]] | None = None,
) -> Path:
    """Run one eval sweep (retrieve + score) and persist artifacts."""
    samples = load_golden_dataset(dataset_path)
    retriever = HybridRetriever(collection_name=collection)
    dump_path = dump_retrieval_rows(
        samples=samples, retriever=retriever, provider=provider, model=model,
        collection=collection, chunking_strategy=chunking_strategy,
        top_k=top_k, output_dir=output_dir, fallback_chain=fallback_chain,
    )
    return score_from_dump(dump_path=dump_path, output_dir=output_dir)


def run_from_settings(
    *, collection: str, chunking_strategy: str, dataset_path: str | None = None,
    provider: str | None = None, model: str | None = None, top_k: int | None = None,
    output_dir: str | None = None, fallback_chain_csv: str | None = None,
) -> Path:
    """Resolve defaults from settings and execute one eval run."""
    raw_csv = fallback_chain_csv if fallback_chain_csv is not None else settings.eval_fallback_chain_csv
    return run_eval(
        collection=collection, chunking_strategy=chunking_strategy,
        dataset_path=Path(dataset_path or settings.eval_dataset_path),
        provider=(provider or settings.eval_provider).strip(),
        model=(model or settings.eval_model).strip(),
        top_k=top_k or settings.eval_top_k,
        output_dir=Path(output_dir or settings.eval_results_dir),
        fallback_chain=_parse_fallback_chain(raw_csv),
    )