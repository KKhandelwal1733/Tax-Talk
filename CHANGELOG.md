# Changelog

All notable changes to **domain-oracle** are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) + custom analysis.

## Change Summary

| # | Version | Type | Summary | Date |
|---|---------|------|---------|------|
| 1 | 0.1.0 | MINOR | Initial project scaffold — ingestion pipeline, hybrid retrieval, API stub | 2025-01-01 |
| 2 | 0.1.1 | PATCH | Added .copilot configuration files — rules, optimization, module context, workflow hooks | 2026-05-28 |
| 3 | 0.1.2 | PATCH | Implemented FastAPI endpoints (`/health`, `/retrieve`) and added Langfuse observer tracing for retrieval pipeline | 2026-05-28 |

---

## [0.1.2] — 2026-05-28

### What Changed
- Replaced API stub in `src/tax_talk/api/main.py` with FastAPI app routes:
	- `GET /health`
	- `POST /retrieve`
- Created centralized models package `src/tax_talk/models/` and moved Pydantic models there:
	- `src/tax_talk/models/api.py` (`HealthResponse`, `RetrieveRequest`, `RetrieveResponse`)
	- `src/tax_talk/models/ingestion.py` (`EmbeddingManifest`, `ChunkRecord`)
- Updated imports in API and ingestion modules to use centralized model definitions.
- Added API request/response schemas and input validation for retrieval parameters.
- Added Langfuse `@observe` decorators to retrieval flow in `src/tax_talk/retrieval/hybrid.py`:
	- `retrieval-hybrid`
	- `retrieval-dense-search`
	- `retrieval-bm25-search`
	- `retrieval-bm25-load-index`
	- `retrieval-rrf-fusion`
- Added query embedding observer with relevant type in `src/tax_talk/ingestion/embeddings.py`:
	- `embed-query` with `as_type="embedding"`
- Updated root retrieval observer type in `src/tax_talk/retrieval/hybrid.py`:
	- `retrieval-hybrid` uses `as_type="retriever"`
- Rebalanced Langfuse capture policy to avoid losing trace utility while limiting heavy payload capture:
	- `capture_input=True` on root retrieval entry spans (`api-retrieve`, `retrieval-hybrid`, `embed-query`)
	- `capture_output=True` on root retriever span (`retrieval-hybrid`) for final result visibility
	- `capture_output=False` on retrieval-heavy operations
	- `qdrant-search` now disables input/output capture to avoid storing vectors and large hit payloads
- Added optional Cohere rerank stage after RRF in `src/tax_talk/retrieval/hybrid.py` using Cohere v2 client pattern.
- Added rerank settings in `src/tax_talk/core/config.py` (`rerank_enabled`, `rerank_model`, `rerank_top_k`, `rerank_top_n`, `rerank_max_tokens_per_doc`).
- Added Cohere runtime singleton in `src/tax_talk/core/runtime.py` (`get_cohere_client`).
- Split retrieval into smaller modules under `src/tax_talk/retrieval/`:
	- `dense_search.py`, `bm25_index.py`, `rerank.py`, and helper modules under `helpers/` (`fusion.py`, `tokenization.py`, `filters.py`)
	- `hybrid.py` now orchestrates these modules while preserving public behavior.
- Added retrieval tests for rerank success and fail-open fallback in `tests/test_hybrid_retrieval.py` using monkeypatched dummy Cohere clients.
- Added API tests in `tests/test_api_main.py` using monkeypatched dummy retriever (no live retrieval calls).
- Updated `README.md` with endpoint usage and observability span names.

### Why Changed
- Expose retrieval functionality through a production-ready HTTP interface.
- Add retrieval observability spans for latency/error tracing and debugging.
- Keep docs and testing aligned with implemented API behavior.

### Pros
- API now supports direct health checks and retrieval requests.
- Retrieval execution path is traceable in Langfuse with nested spans.
- Unit tests validate route contracts without external dependencies.

### Cons
- Retrieval endpoint currently returns raw hit payloads (schema may be tightened later).
- No streaming or generation endpoint yet.

---

## [0.1.1] — 2026-05-28

### What Changed
- Created `COPILOT.md` — root project context with architecture, tech stack, commands, conventions
- Created `.copilot/rules.md` — 28 strict guardrails for code generation
- Created `.copilot/optimization.md` — speed/alignment policy
- Created `.copilot/api.md` — API module context
- Created `.copilot/core.md` — core module context
- Created `.copilot/ingestion.md` — ingestion module context
- Created `.copilot/embedding-strategies.md` — embedding strategies context
- Created `.copilot/retrieval.md` — retrieval module context
- Created `.copilot/generation.md` — generation module context
- Created `.copilot/tests.md` — testing conventions
- Created `.copilot/hooks.md` — pre/post workflow hooks with changelog enforcement

### Why Changed
- Maximize coding assistant output quality by providing project-specific context, strict guardrails, and enforced workflows
- Eliminate hallucination, scope creep, and convention drift in AI-generated code

### Pros
- AI assistant now has full project awareness (architecture, patterns, dependencies)
- Strict rules prevent common mistakes (bare except, print(), hardcoded secrets, import *)
- Module context files prevent cross-boundary violations
- Workflow hooks ensure changelog tracking and rule compliance
- All files are committed and team-shared — consistent behavior across developers

### Cons
- Additional files to maintain when project structure changes
- Hooks add slight overhead to each interaction (reading context files)
- Rules may occasionally be too restrictive for exploratory/prototyping work

---

## [0.1.0] — 2025-01-01

### What Changed
- Initial project structure with `src/tax_talk/` package
- Ingestion pipeline: loader, chunker, embeddings, qdrant_store
- Embedding strategies: sentence_transformer, gemini, voyage (Strategy + Factory)
- Hybrid retrieval: BM25 + dense + RRF fusion
- FastAPI stub with health endpoint
- Core: pydantic-settings config, runtime singletons
- Tests: embedding provider, hybrid retrieval, langfuse config, smoke

### Why Changed
- Bootstrap production RAG pipeline for Indian GST + Income Tax domain

### Pros
- Clean modular architecture with clear separation of concerns
- Resumable ingestion pipeline with stage control
- Pluggable embedding providers via ABC + factory
- Thread-safe singletons for shared resources

### Cons
- Generation module is still a stub
- Only ~13 tests; coverage not enforced
- No CI/CD pipeline configured yet
- BM25 index rebuilt on every cold start (no persistence)
