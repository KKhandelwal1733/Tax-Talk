# TAX TALK: GST + Income Tax RAG

Production-grade retrieval-augmented generation over Indian GST and Income Tax statutes, notifications, circulars, and case law.

**Status:** 🚧 In active development — see [project board](https://github.com/KKhandelwal1733/Tax-Talk).

## What this does

Given a natural-language question about Indian indirect or direct taxation, the system retrieves the most relevant sections from the official corpus (CGST Act, IGST Act, Income Tax Act 1961, recent CBIC/CBDT notifications, and landmark case law) and produces a grounded answer with inline citations to the source clauses.

**Example queries it handles:**
- *"Is GST applicable on free samples given to distributors?"*
- *"What's the TDS rate for payments to a non-resident professional for technical services?"*
- *"How does Section 54F exemption work if the new house is purchased outside India?"*

## Why this exists

Indian tax law is interpreted across thousands of pages of statutes, hundreds of yearly notifications, and decades of case law. Most consumer chatbots hallucinate citations confidently. This system is built to *not* hallucinate — every claim links to its source.

## Architecture

```
[Query] → [Query Rewriter] → [Hybrid Retrieval: BM25 + Dense]
                                            ↓
                              [Cohere Rerank (top-k → top-n)]
                                            ↓
                          [Answer Generator with Citations]
                                            ↓
                              [Faithfulness Check (LLM judge)]
                                            ↓
                                  [Response + Sources]

All steps traced in Langfuse. Eval suite runs nightly on 100+ golden Q/A pairs.
```

Architecture diagram: see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | Local `BAAI/bge-m3` / Gemini `text-embedding-004` | No-card path; strong quality with portable setup |
| Vector DB | Qdrant (self-hosted) | Open-source, supports binary quantization |
| Sparse retrieval | BM25 (rank_bm25) | Catches exact section numbers and statute references |
| Fusion | Reciprocal Rank Fusion | Robust combination of sparse + dense |
| Reranker | Cohere Rerank v4 | Optional post-RRF semantic reranking |
| Generation | Gemini 3.5 Flash / Groq Llama 3.3 70B | Fast, no-card friendly model strategy |
| Orchestration | LangChain (retrieval) + Pydantic AI (generation) | Type-safe outputs |
| Observability | Langfuse Cloud | Free tier, captures all traces and costs |
| API | FastAPI + SSE streaming | Production-standard |
| Eval framework | RAGAS + custom LLM-as-judge | Faithfulness, context relevance, citation accuracy |
| Hosting | Azure VM (B1S, Central India) + Streamlit frontend | Free via Azure for Students |

### Embedding execution

`EMBEDDING_PROVIDER=sentence_transformer` now runs only via Hugging Face Inference API and requires `HF_TOKEN`.

`EMBEDDING_PROVIDER=gemini` runs via Gemini embeddings and requires `GEMINI_API_KEY`.

## Data sources (all official, public)

- **GST**: CGST/IGST/UTGST Acts 2017 + CGST Rules + CBIC notifications + circulars, all from [cbic-gst.gov.in/gst-acts.html](https://cbic-gst.gov.in/gst-acts.html) and [gstcouncil.gov.in](https://gstcouncil.gov.in/cgst-circulars). No statutory replacement — 2017 framework still in force.
- **Income Tax (current)**: **Income-tax Act, 2025** (Act 30 of 2025) — effective April 1, 2026, repealed the 1961 Act. 536 sections across 23 chapters. Source: [incometaxindia.gov.in](https://incometaxindia.gov.in/Pages/income-tax-bill-2025.aspx).
- **Income Tax (legacy)**: **Income-tax Act, 1961** as amended by Finance Act 2026 — still applies to FY 2025-26 (AY 2026-27) returns being filed in July 2026, and to all open assessments/appeals for prior years.
- **Cross-reference**: [indiacode.nic.in](https://www.indiacode.nic.in/) for tie-breaking, [egazette.gov.in](https://egazette.gov.in/) for Finance Act and gazette notifications.

> **Transition handling is a project differentiator.** Every chunk in the index is tagged with `applicable_period` ("FY 2026-27 onwards" for 2025 Act, "AY 2026-27 and earlier" for 1961 Act). The retriever can filter by the period the user's question concerns. Naive RAG implementations don't handle this — they conflate the two Acts and produce wrong answers.

Corpus license/copyright: All statutes are public domain. Notifications and circulars are government-issued public documents. Case law citations preserve original source links.

## Evals

See [EVALS.md](./EVALS.md) for the full eval harness, metrics, and current scores.

Current snapshot (will update weekly):
- Faithfulness (LLM-as-judge): 0.9058775374951844
- Context precision: 0.8429292928594947
- Context recall: 0.803030303030303
- Citation accuracy: 0.5305352920780899

## Live demo

[domain-oracle.yourname.me](https://domain-oracle.yourname.me) — Not yet.

## Local development

```bash
# Prerequisites: Python 3.11+, Docker, uv
git clone https://github.com/KKhandelwal1733/Tax-Talk
cd tax-talk

# Install dependencies
uv sync --extra dev

# Bring up infrastructure (Qdrant)
docker compose up -d

# Copy env and add your API keys
cp .env.example .env
# Edit .env: add GEMINI_API_KEY, GROQ_API_KEY, COHERE_API_KEY, LANGFUSE_*

# Run ingestion (chunks + embeds the corpus)
make ingest

# Start API
make serve

# Run evals
make eval
```

### Run the API in Docker

If Qdrant is already running elsewhere, build and run only the API container:

```bash
docker build -t tax-talk-api .

# If Qdrant is reachable from the host machine
docker run --rm -p 8000:8000 --env-file .env \
    -e QDRANT_URL=http://host.docker.internal:6333 \
    tax-talk-api
```

If your Qdrant container is on a shared Docker network, replace `QDRANT_URL` with that container hostname, for example `http://qdrant:6333`.

### API endpoints

- `GET /health/live`: liveness check.
- `GET /health/ready`: readiness check (Qdrant connectivity).
- `POST /chat`: grounded answer with citations.
- `POST /chat/stream`: SSE token stream with terminal citations event.
    - Uses FastAPI SSE response primitives (`EventSourceResponse` / `ServerSentEvent`).
    - Emits sequential SSE `id` values per stream event.

Example request:

```bash
curl -X POST http://localhost:8000/chat \
    -H "authorization: Bearer <SUPABASE_JWT>" \
    -H "content-type: application/json" \
    -d '{
        "query": "section 54F exemption",
        "top_k": 5,
        "dense_top_k": 20,
        "bm25_top_k": 20
    }'
```

The chat endpoints always use the configured `CHAT_MODEL` server-side; request payloads no longer accept a per-request model override.

### Auth notes (Supabase JWT signature verification)

Bearer tokens are verified with **full RSA signature validation** against your Supabase JWKS endpoint. The API requires:

- **Token format:** `Authorization: Bearer <JWT>`
- **Signature verification:** RS256 signature checked against `SUPABASE_URL/.well-known/jwks.json`
- **Expiration check:** `exp` claim is always validated; expired tokens return 401
- **Issuer check:** Always validated against `SUPABASE_JWT_ISSUER`; mismatches return 401
- **Audience check:** Always validated against `SUPABASE_JWT_AUDIENCE`; mismatches return 401

Required environment variables (when Supabase auth is enabled):

- `SUPABASE_URL` — e.g., `https://yourproject.supabase.co` (fetches JWKS from `/.well-known/jwks.json`)
- `SUPABASE_JWT_ISSUER` — **Required** if `SUPABASE_URL` is set; e.g., `https://yourproject.supabase.co`
- `SUPABASE_JWT_AUDIENCE` — **Required** if `SUPABASE_URL` is set; e.g., `authenticated`

If any of these are required but missing or empty, the application will fail to start with a configuration error. This ensures auth is always properly configured.

Example request with a valid Supabase JWT:

```bash
curl -X POST http://localhost:8000/chat \
    -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \
    -H "content-type: application/json" \
    -d '{"query": "section 54F exemption", "top_k": 5}'
```

If `SUPABASE_URL` is unset (empty string), token verification is bypassed and all endpoints are public (no authentication required).

### Observability for retrieval

Retrieval spans are instrumented with Langfuse `@observe` decorators:

- `api-chat`
- `api-chat-stream-route`
- `retrieval-hybrid`
- `retrieval-hybrid-async`
- `retrieval-dense-search`
- `retrieval-dense-search-async`
- `retrieval-bm25-search`
- `retrieval-bm25-load-index`
- `retrieval-rrf-fusion`
- `retrieval-cohere-rerank`

### Optional Cohere rerank settings

Rerank runs after RRF and is enabled by default when a Cohere API key is configured.

- `RERANK_ENABLED` (default `true`)
- `RERANK_MODEL` (default `rerank-v4.0-pro`)
- `RERANK_TOP_K` (default `30`) candidate pool size sent to reranker
- `RERANK_TOP_N` (default `10`) number of reranked items returned from Cohere
- `RERANK_MAX_TOKENS_PER_DOC` (default `4096`)

Fail-open behavior: if Cohere is not configured or rerank fails, retrieval returns the fused RRF ranking.

### Processed-stage ingestion (resumable)

Ingestion now persists intermediate artifacts under a hard strategy-scoped layout:

- `data/processed/<chunking_strategy>/<source_key>/`

Examples:

- `data/processed/fixed/gst_circulars_cbic/chunks.jsonl`
- `data/processed/semantic/it_act_2025_icai/embeddings.npy`
- `data/processed/contextual/igst_act_2025/manifest.json`

Each source folder contains:

- `chunks.jsonl`
- `embeddings.npy`
- `manifest.json`

Legacy `data/processed/<source_key>/` paths are not used.

Chunk generation now also:

- maps nested raw metadata correctly from `ingestion_metadata` and `chunk_metadata`
- prepends contextual source summaries to chunk text when available
- prefers curated `source_meta` first and optionally falls back to an LLM summary over a truncated document prefix
- persists the applied contextual summary and its provenance in chunk artifacts for auditability

Run full processed flow:

```bash
uv run python -m tax_talk.ingestion.run --from-stage chunk --to-stage upsert
```

Resume from embedding stage (reuse existing chunks):

```bash
uv run python -m tax_talk.ingestion.run --from-stage embed --to-stage upsert
```

Run upsert only (reuse cached chunks + embeddings):

```bash
uv run python -m tax_talk.ingestion.run --from-stage upsert --to-stage upsert
```

Limit to selected sources:

```bash
uv run python -m tax_talk.ingestion.run --sources it_act_2025_icai gst_circulars_cbic
```

Run strategy-specific ingestion into separate collections:

```bash
uv run python -m tax_talk.ingestion.run --chunking-strategy fixed --collection gst_income_tax_fixed
uv run python -m tax_talk.ingestion.run --chunking-strategy semantic --collection gst_income_tax_semantic
uv run python -m tax_talk.ingestion.run --chunking-strategy contextual --collection gst_income_tax_contextual
```

Supported chunking strategies:

- `fixed` — fixed-size chunks with overlap
- `semantic` — heading/section-aware paragraph grouping
- `contextual` — fixed chunks with metadata-first contextual summary prefixing

### HF inference parallel ingestion knobs

When `EMBEDDING_PROVIDER=sentence_transformer`, embed/upsert phases run via HF Inference with bounded source-level concurrency.

- `INGESTION_MAX_WORKERS` (default `2`) source worker threads
- `HF_MAX_PARALLEL_SOURCES` (default `2`) hard cap for source workers in HF mode
- `HF_MAX_CONCURRENT_REQUESTS` (default `2`) concurrent HF requests across workers
- `HF_RETRY_MAX_ATTEMPTS` (default `5`) retries for 429/5xx
- `HF_RETRY_INITIAL_DELAY_SECONDS` (default `1.0`) exponential backoff base
- `HF_RETRY_MAX_DELAY_SECONDS` (default `12.0`) backoff ceiling

### Contextual chunking settings

Contextual summaries are applied at ingestion time and reused for all chunks from a source document.

- `CONTEXTUAL_SUMMARY_ENABLED` (default `true`)
- `CONTEXTUAL_SUMMARY_METADATA_MIN_CHARS` (default `120`)
- `CONTEXTUAL_SUMMARY_MAX_CHARS` (default `600`)
- `CONTEXTUAL_SUMMARY_PREFIX_LABEL` (default `Context`)
- `CONTEXTUAL_SUMMARY_LLM_FALLBACK_ENABLED` (default `false`)
- `CONTEXTUAL_SUMMARY_FALLBACK_PROVIDER` (default `gemini`, supported: `gemini`, `groq`)
- `CONTEXTUAL_SUMMARY_FALLBACK_MODEL` (default `gemini-2.0-flash`)
- `CONTEXTUAL_SUMMARY_DOCUMENT_PREFIX_CHARS` (default `12000`)

Provider runtime details:

- LLM fallback uses runtime-owned singleton strategies for Gemini and Groq.
- Groq integration uses the native Python SDK client pattern (`from groq import Groq`) and calls `client.chat.completions.create(...)`.
- If provider setup is invalid or unavailable, fallback is fail-open and ingestion continues with plain chunk text.

Behavior:

- use `source_meta` first when it is informative enough
- if metadata is weak and fallback is enabled, generate one short summary per document from a truncated document prefix
- if fallback is disabled or fails, ingestion continues with plain chunk text

### Offline evals with per-strategy collections

Golden dataset (30 hand-built pairs): `data/eval/golden_qa_v0_30.jsonl`

Run one strategy eval:

```bash
uv run python -m tax_talk.evals.run --collection gst_income_tax_fixed --chunking-strategy fixed
```

Compare all three:

```bash
uv run python -m tax_talk.evals.run --collection gst_income_tax_fixed --chunking-strategy fixed
uv run python -m tax_talk.evals.run --collection gst_income_tax_semantic --chunking-strategy semantic
uv run python -m tax_talk.evals.run --collection gst_income_tax_contextual --chunking-strategy contextual
```

Metrics computed via RAGAS:

- faithfulness
- context precision
- context recall
- answer relevancy

Each run writes local artifacts under `data/eval/results/` and emits Langfuse traces for eval orchestration and sample-level generation.

## Lessons learned (will fill in as I go)

> *Section to be added — interviewers love an honest "what I'd do differently" reflection.*

## License

MIT — see [LICENSE](./LICENSE).
