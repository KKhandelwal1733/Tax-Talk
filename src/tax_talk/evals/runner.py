from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from langfuse import observe
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from tax_talk.core.config import settings
from tax_talk.core.runtime import get_langfuse_client, get_llm_strategy, get_logger
from tax_talk.evals.dataset import GoldenQASample, load_golden_dataset
from tax_talk.retrieval.hybrid import HybridRetriever

log = get_logger(__name__)


@dataclass
class EvalRow:
    """Per-sample intermediate payload used by RAGAS and result dumps."""

    id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    collection: str
    chunking_strategy: str
    latency_ms: float


@observe(name="eval-generate-answer", as_type="generation", capture_input=False, capture_output=False)
def _generate_grounded_answer(
    *,
    question: str,
    contexts: list[str],
    provider: str,
    model: str,
) -> str:
    """Generate a grounded answer from retrieved contexts."""
    strategy = get_llm_strategy(provider)
    joined_context = "\n\n".join(contexts[:8])
    prompt = (
        "Answer the tax question using only the provided context. "
        "If context is insufficient, say so explicitly. Keep answer concise.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{joined_context}"
    )
    response = strategy.generate(prompt=prompt, model=model)
    return response.strip() if isinstance(response, str) else ""


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
) -> list[EvalRow]:
    """Run retrieval+generation for each sample and return normalized rows."""
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


@observe(name="eval-ragas", as_type="span", capture_input=False, capture_output=False)
def compute_ragas_scores(rows: list[EvalRow]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute RAGAS metrics and return aggregate and row-level views."""
    dataset = Dataset.from_dict(
        {
            "question": [row.question for row in rows],
            "answer": [row.answer for row in rows],
            "contexts": [row.contexts for row in rows],
            "ground_truth": [row.ground_truth for row in rows],
        }
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
    )
    frame = result.to_pandas()
    metric_cols = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    ]
    aggregates = {col: float(frame[col].mean()) for col in metric_cols if col in frame.columns}
    row_metrics = frame.to_dict(orient="records")
    return aggregates, row_metrics


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
) -> Path:
    """Run one eval sweep and persist local artifacts.

    Args:
        collection: Qdrant collection to query.
        chunking_strategy: Strategy label used for result metadata.
        dataset_path: JSONL golden dataset path.
        provider: LLM provider used for answer generation.
        model: Model name for generation.
        top_k: Retrieval cutoff per question.
        output_dir: Parent directory for eval result files.

    Returns:
        Path to the written result JSON file.
    """
    samples = load_golden_dataset(dataset_path)
    retriever = HybridRetriever(collection_name=collection)

    rows = build_eval_rows(
        samples=samples,
        retriever=retriever,
        provider=provider,
        model=model,
        collection=collection,
        chunking_strategy=chunking_strategy,
        top_k=top_k,
    )
    aggregates, ragas_rows = compute_ragas_scores(rows)

    latencies = [r.latency_ms for r in rows]
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
        latency_p95 = sorted_latencies[p95_index]
    else:
        latency_p95 = 0.0

    stamped = time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"eval-{chunking_strategy}-{collection}-{stamped}.json"

    payload = {
        "collection": collection,
        "chunking_strategy": chunking_strategy,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "dataset_path": str(dataset_path),
        "sample_count": len(rows),
        "metrics": {
            **aggregates,
            "latency_p95_ms": latency_p95,
        },
        "samples": [asdict(row) for row in rows],
        "ragas_rows": ragas_rows,
    }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Ensure traces are flushed so run-level spans and child spans are visible.
    get_langfuse_client().flush()
    log.info("Wrote eval results: %s", result_path)
    return result_path


def run_from_settings(
    *,
    collection: str,
    chunking_strategy: str,
    dataset_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
    output_dir: str | None = None,
) -> Path:
    """Resolve defaults from settings and execute one eval run."""
    return run_eval(
        collection=collection,
        chunking_strategy=chunking_strategy,
        dataset_path=Path(dataset_path or settings.eval_dataset_path),
        provider=(provider or settings.eval_provider).strip(),
        model=(model or settings.eval_model).strip(),
        top_k=top_k or settings.eval_top_k,
        output_dir=Path(output_dir or settings.eval_results_dir),
    )
