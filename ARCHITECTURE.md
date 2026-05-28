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
│  • POST /retrieve returns ranked hit payloads│
│  • GET /health for liveness                  │
└─────────────────┬───────────────────────────┘
                  ▼
            Ranked chunks + scores

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

### Chunking: fixed-size with overlap (implemented)
- **Current approach:** char-based chunking tuned to about 512 tokens (`CHUNK_SIZE_CHARS=2000`, `CHUNK_OVERLAP_CHARS=200`).
- **Why:** simple, deterministic, fast to run, and robust for legal text boundaries with overlap.
- **Alternatives considered:** semantic chunking and contextual chunking with LLM-generated summaries.
- **How to revisit:** run side-by-side retrieval evals (context recall + citation precision) before migrating chunking strategy.

### Hybrid retrieval: BM25 + dense + RRF
- **Why:** BM25 catches exact section numbers ("Section 80C", "Rule 42") that dense retrieval misses. Dense catches semantic matches BM25 misses ("rebate on home loan interest"). RRF is parameter-free.
- **Alternatives:** weighted score combination (needs tuning).

### Reranker: Cohere Rerank v4 (optional)
- **Current model:** `rerank-v4.0-pro`.
- **Why:** strong semantic reordering quality on top of fused candidates.
- **Current defaults:** candidate pool 30, reranked return 10.
- **Behavioral safeguard:** fail-open fallback to fused ranking when key is missing or rerank call fails.
- **Alternatives:** BGE reranker (self-hosted) and no-rerank mode.

### Generation and faithfulness checks: planned
- **Current state:** generation package is a placeholder; no production answer synthesis endpoint yet.
- **Configured model slots today:** `dev_model=gemini-2.0-flash`, `speed_model=groq/llama-3.3-70b-versatile`, `eval_model=gemini-2.0-flash`.
- **How to revisit:** document final generation and judge architecture once generation endpoints and evaluation harness are wired into runtime API.

## Open questions

- How to handle multilingual queries (Hindi tax queries)?
- Should case law have its own index with judgment-level metadata?
- Worth fine-tuning a small reranker on hand-labeled pairs?
