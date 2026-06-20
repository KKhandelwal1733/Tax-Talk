# Architecture

> This doc captures the *why* behind every design decision. Update it as you make changes — interviewers will read this and ask about specific choices.

## Current implemented flow

```
User question
    │
    ▼
┌─────────────────────────────────────────────┐
│ Hybrid retrieval                            │
│  • BM25 over chunks  (rank_bm25)            │
│  • Dense over chunks (Qdrant + configured emb)│
│    - default: sentence-transformer (BGE base)│
│    - optional: Gemini embeddings             │
│  • Reciprocal Rank Fusion                    │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ Optional reranker — Cohere Rerank v4         │
│  • default pool: top-30 candidates           │
│  • default reranked return: top-10           │
│  • fail-open to fused ranking on errors      │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ API response                                │
│  • POST /chat returns grounded answer + cites│
│  • POST /chat/stream streams SSE events      │
│    (event/id/data contract)                  │
│  • GET /health/live and /health/ready        │
└─────────────────┬───────────────────────────┘
                  ▼
        Grounded answer + citations

Langfuse tracing is implemented for API + retrieval spans.
```

## Planned flow (not fully implemented yet)

```
[Query rewrite / HyDE] -> [Hybrid retrieval + rerank] -> [Answer generation + citations] -> [Faithfulness judge]
```

Planned stages above are roadmap items and should not be presented as production-complete.

## Decisions log

Each decision below should have: *what* / *why* / *alternatives considered* / *how to revisit*.

### Embeddings: sentence-transformer (default) + Gemini (optional)
- **Current default:** `sentence_transformer` with `BAAI/bge-base-en-v1.5` (768 dims).
- **Why:** no-card path, predictable local execution, good retrieval quality for legal-style text.
- **Optional provider:** Gemini `models/text-embedding-004` when API-based embeddings are preferred.
- **Alternatives considered:** Voyage `voyage-3` and OpenAI embeddings; keep provider consistent between ingestion and query-time retrieval.
- **How to revisit:** when eval set is expanded, compare retrieval quality/cost/latency across providers on the same benchmark set.

### Chunking: fixed-size with metadata-first contextual summaries (implemented)
- **Current approach:** strategy-selectable chunking with collection isolation per strategy.
    - `fixed`: char-based chunking tuned to about 512 tokens (`chunk_size_chars`, `chunk_overlap_chars`)
    - `semantic`: heading/section-aware paragraph grouping (`semantic_chunk_min_chars`, `semantic_chunk_max_chars`)
    - `contextual`: fixed chunking + prepended contextual source summary
- **Model boundary:** canonical `Chunk` schema lives in `src/tax_talk/models/ingestion.py`; `chunker.py` only orchestrates chunk assembly and IO.
- **Processed artifact contract:** hard strategy-scoped layout only: `data/processed/<strategy>/<source_key>/`.
- **Summary policy:** use curated raw `source_meta` first; if it is missing or too weak and fallback is enabled, generate one short LLM summary from a truncated document prefix.
- **Why:** preserves deterministic chunk boundaries while adding document-level context to improve retrieval over legal text.
- **Behavioral safeguard:** if summary generation is unavailable or fails, ingestion falls back to plain chunk text and continues.
- **How to revisit:** compare metadata-only, hybrid fallback, and fully semantic chunking using retrieval eval metrics before changing the default strategy.

### Evaluation pipeline: offline strategy comparison (implemented)
- **Dataset:** `data/eval/golden_qa_v0_30.jsonl` (30 hand-built QA pairs).
- **Runner:** `src/tax_talk/evals/run.py` orchestrates retrieval + grounded generation + RAGAS metrics.
- **Metrics tracked in code:** faithfulness, context precision, context recall, answer relevancy, plus latency p95.
- **Observability:** Langfuse spans on eval run orchestration and sample-level generation (`eval-run`, `eval-build-rows`, `eval-generate-answer`, `eval-ragas`).
- **Why:** keep chunking strategy comparison reproducible by isolating each strategy into its own Qdrant collection.

### Ingestion throughput: source-level parallel embed/upsert for hosted HF mode (implemented)
- **Current approach:** embed/upsert phases can process independent sources with bounded worker threads.
- **Provider guardrails:** in `hf_inference` mode, source worker count and concurrent request count are explicitly capped; HF calls use retry/backoff for 429/5xx.
- **Why:** reduce ingestion wall-clock time while avoiding API throttling storms and preserving deterministic per-source artifact outputs.

### Hybrid retrieval: BM25 + dense + RRF
- **Why:** BM25 catches exact section numbers ("Section 80C", "Rule 42") that dense retrieval misses. Dense catches semantic matches BM25 misses ("rebate on home loan interest"). RRF is parameter-free.
- **Alternatives:** weighted score combination (needs tuning).

### Reranker: Cohere Rerank v4 (optional)
- **Current model:** `rerank-v4.0-pro`.
- **Why:** strong semantic reordering quality on top of fused candidates.
- **Current defaults:** candidate pool 30, reranked return 10.
- **Behavioral safeguard:** fail-open fallback to fused ranking when key is missing or rerank call fails.
- **Alternatives:** BGE reranker (self-hosted) and no-rerank mode.

### Generation and faithfulness checks: partially implemented (chat synthesis now enabled)
- **Current state:** chat synthesis endpoint is implemented (`/chat`, `/chat/stream`) using retrieval-backed prompts and provider strategy runtime.
- **Implemented now:** runtime-owned singleton LLM strategy access for Gemini and Groq via `get_llm_strategy(...)` and async generation access via `get_llm_strategy_async(...)` in `src/tax_talk/core/runtime.py`.
- **Contextual-summary usage:** ingestion LLM fallback routes through the strategy layer instead of direct SDK calls.
- **Groq integration:** uses native Groq SDK client (`from groq import Groq`) rather than OpenAI-compat path.
- **Configured eval defaults today:** `eval_provider=gemini`, `eval_model=gemini-3.5-flash`.
- **How to revisit:** plug answer generation and judge stages into the same strategy/runtime abstraction to avoid provider-specific logic leaking into API handlers.

### API lifespan + shutdown hygiene (implemented)
- **Current behavior:** FastAPI lifespan warms Qdrant on startup and performs graceful shutdown hooks.
- **Shutdown hooks:**
    - flushes Langfuse buffered events (`flush_langfuse_client()`)
    - closes Gemini client transports (`close_gemini_client()`), including async SDK handles when present
- **Why:** prevent trace/event loss and reduce leaked client resources during process stop/reload.

### API streaming contract (implemented)
- **Current behavior:** `POST /chat/stream` uses FastAPI SSE primitives (`EventSourceResponse` + `ServerSentEvent`).
- **Event format:** server emits SSE `event`, `id`, and JSON `data` fields for each token/done frame.
- **Why:** align with FastAPI-native SSE behavior and improve stream client interoperability.

## Open questions

- How to handle multilingual queries (Hindi tax queries)?
- Should case law have its own index with judgment-level metadata?
- Worth fine-tuning a small reranker on hand-labeled pairs?
