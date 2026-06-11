# Changelog

All notable changes to **tax-talk** are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) + custom analysis.

## Change Summary

| # | Version | Type | Summary | Date |
|---|---------|------|---------|------|
| 1 | 0.1.0 | MINOR | Initial project scaffold — ingestion pipeline, hybrid retrieval, API stub | 2025-01-01 |
| 2 | 0.1.1 | PATCH | Added .copilot configuration files — rules, optimization, module context, workflow hooks | 2026-05-28 |
| 3 | 0.1.2 | PATCH | Implemented FastAPI endpoints (`/health`, `/retrieve`) and added Langfuse observer tracing for retrieval pipeline | 2026-05-28 |
| 4 | 0.1.3 | PATCH | Added runtime-owned LLM strategy singletons (Gemini + Groq), provider-based contextual-summary fallback, and switched Groq client to native SDK | 2026-05-29 |
| 5 | 0.2.0 | MINOR | Added strategy-isolated chunking collections and a new offline eval package with RAGAS + Langfuse tracing | 2026-05-29 |
| 6 | 0.2.1 | PATCH | Refactored chunker strategy selection to explicit strategy-pattern implementations with registry-based resolution | 2026-05-29 |
| 7 | 0.2.2 | PATCH | Moved chunking strategy implementations into dedicated ingestion/chunking_strategies package and kept chunker.py as orchestration | 2026-05-29 |
| 8 | 0.2.3 | PATCH | Consolidated canonical Chunk model under models/ingestion and removed duplicate runtime definition from chunker | 2026-05-30 |
| 9 | 0.2.4 | PATCH | Hard-switched processed artifacts to strategy-scoped folders and added HF-safe source-level parallel embed/upsert | 2026-05-30 |
| 10 | 0.2.5 | PATCH | Removed unused config model-slot fields and aligned architecture docs to active eval settings | 2026-05-30 |
| 11 | 0.2.6 | PATCH | Updated LLM strategy auto-selection to prefer available API keys (Gemini first) when provider is omitted | 2026-05-30 |
| 12 | 0.2.7 | PATCH | Added SlidingWindowRateLimiter + eval rate-limit config fields; wired limiter before every eval LLM call | 2026-06-01 |
| 13 | 0.2.8 | PATCH | Migrated .copilot/*.md to .github/instructions/*.instructions.md; fixed copilot-instructions.md diagnostics | 2026-06-01 |
| 14 | 0.2.9 | MINOR | Added eval fallback chain — eval_fallback_chain_csv config, ordered attempts, --fallback-chain CLI arg, 13 new tests | 2026-06-01 |
| 15 | 0.2.10 | PATCH | Merged lazy RAGAS judge/embeddings builder into eval runner and switched scoring path to EvaluationDataset/SingleTurnSample | 2026-06-01 |

---

## [0.2.10] — 2026-06-01

### What Changed
- `src/tax_talk/evals/runner.py`: merged lazy RAGAS evaluator builder into production runner as `_build_ragas_evaluator()`.
- `src/tax_talk/evals/runner.py`: updated `compute_ragas_scores(...)` to build `EvaluationDataset` with `SingleTurnSample` and call `evaluate(..., llm=evaluator_llm, embeddings=evaluator_embeddings)`.
- `src/tax_talk/evals/runner.py`: kept evaluator-specific imports deferred via `import_module(...)` and keyed to `settings.gemini_api_key` for eval-only initialization.

### Why Changed
- User requested implementing evaluation approach into `runner.py` while keeping project structure intact.

### Pros
- Preserves architecture rule: eval-only heavy clients stay lazily initialized and out of runtime singletons.
- Makes LLM-judge + embedding dependencies explicit at eval execution point.
- Maintains existing fallback/rate-limit retrieval-generation flow unchanged.

### Cons
- Eval now depends on optional runtime availability of Google/RAGAS embedding wrapper packages at execution time.
- RAGAS metric import deprecation warnings still exist and were not changed in this patch.

---

## [0.2.9] — 2026-06-01

### What Changed
- `src/tax_talk/core/config.py`: added `eval_fallback_chain_csv: str = ""` field
- `src/tax_talk/evals/runner.py`: added `_VALID_PROVIDERS`, `_parse_fallback_chain()`, refactored `_generate_grounded_answer` with ordered attempts + fallback loop; added `fallback_chain` kwarg to `build_eval_rows`, `run_eval`, `run_from_settings`
- `src/tax_talk/evals/run.py`: added `--fallback-chain` argparse argument defaulting to `settings.eval_fallback_chain_csv`
- `tests/test_eval_runner_fallback.py` (NEW): 13 tests — 8 parser tests + 4 execution behaviour tests using `DummyStrategy` + `DummyLimiter`

### Why Changed
- Allow eval runs to continue when the primary LLM provider is unavailable or rate-limited by automatically trying configured fallback providers

### Pros
- Zero new dependencies; implemented with stdlib + existing runtime
- Loosely coupled — fallback chain is a pure config value (`provider/model,provider/model`)
- Rate limiter called per attempt, preventing burst usage across all providers
- Parser validates eagerly at startup; bad config fails fast with actionable error messages

### Cons
- Fallback chain shares the same sliding-window rate limiter as the primary; may cause longer waits when multiple fallbacks are needed
- No per-provider rate limiter differentiation yet

---

## [0.2.8] — 2026-06-01

### What Changed
- Migrated all `.copilot/*.md` files to `.github/instructions/*.instructions.md` with correct `applyTo` frontmatter patterns (10 files)
- Fixed `.github/copilot-instructions.md` diagnostics: renumbered rules 1–38 sequentially, clarified rule 6 minimum-diff exception, updated rule 16 to `@observe()` guidance

### Why Changed
- VS Code Copilot now reads instruction files from `.github/instructions/`; `.copilot/` is a legacy location
- Diagnostics tool flagged duplicate numbering and vague rules

### Pros
- Instructions are now auto-applied by pattern match (`applyTo`) without manual attachment
- Clear, unambiguous rule set reduces assistant guessing

### Cons
- None — low-risk change

---

## [0.2.7] — 2026-06-01

### What Changed
- `src/tax_talk/core/rate_limit.py` (NEW): `SlidingWindowRateLimiter(max_calls, window_seconds)` with thread-safe `wait_for_slot()`
- `src/tax_talk/core/config.py`: added `eval_llm_rate_limit_calls: int = 4` and `eval_llm_rate_limit_window_seconds: float = 60.0`
- `src/tax_talk/evals/runner.py`: module-level `_eval_answer_rate_limiter` singleton; `wait_for_slot()` called before every `strategy.generate()` in `_generate_grounded_answer`
- `tests/test_eval_runner_rate_limit.py` (NEW): 2 tests verifying defaults and limiter invocation

### Why Changed
- Primary LLM provider has a 5 RPM limit; without throttling, concurrent eval rows exhaust the quota and cause failures

### Pros
- Generic rate limiter with no eval or provider coupling — reusable across the codebase
- Defaults set to 4 RPM (safe margin below the 5 RPM quota)

### Cons
- Single global limiter shared across all parallel threads; may over-throttle if eval parallelism is added later

---

## [0.2.6] — 2026-05-30

### What Changed
- Updated `get_llm_strategy(provider=None)` in `src/tax_talk/core/runtime.py` to auto-select based on available keys only, with priority:
	- Gemini first if `GEMINI_API_KEY` is present
	- then Groq if `GROQ_API_KEY` is present
- Removed implicit default-provider coupling when `provider` is omitted.
- Added regression test in `tests/test_llm_strategy_runtime.py` to verify Gemini-first auto-selection.

### Why Changed
- Make runtime provider resolution deterministic and key-driven when no explicit provider is supplied.
- Avoid hidden fallback selection behavior when both providers are configured.

### Pros
- Clearer and more predictable default strategy behavior.
- Better alignment with explicit-provider calls already used in eval and contextual-summary flows.

### Cons
- Any previous reliance on implicit fallback-provider selection for `provider=None` is no longer applicable.

---

## [0.2.5] — 2026-05-30

### What Changed
- Removed unused settings from `src/tax_talk/core/config.py`:
	- `dev_model`
	- `speed_model`
	- `eval_judge_model`
- Updated `ARCHITECTURE.md` to replace stale model-slot references with active eval defaults.

### Why Changed
- Eliminate dead configuration keys that were not referenced by runtime code.
- Reduce confusion between active eval settings and unused placeholders.

### Pros
- Cleaner, lower-noise config surface.
- Less ambiguity when tuning eval behavior.

### Cons
- Any external `.env` values using removed keys are now ignored.

---

## [0.2.4] — 2026-05-30

### What Changed
- Hard cutover for processed artifact layout in ingestion:
	- artifacts now resolve only under `data/processed/<chunking_strategy>/<source_key>/`
	- legacy `data/processed/<source_key>/` discovery/read paths are no longer used
- Updated stage-resume resolution (`--from-stage`, `--to-stage`) to enforce strategy-scoped artifact paths.
- Added bounded source-level parallel processing for embed/upsert phases in ingestion runner.
- Added HF inference safety controls:
	- source-level concurrency cap in HF mode
	- global concurrent HF request limiter
	- retry + exponential backoff for transient HF failures (429/5xx)
- Added regression tests for strategy-scoped artifact path resolution.

### Why Changed
- Ensure strict isolation of processed artifacts by chunking strategy.
- Reduce ingestion time for hosted HF embedding mode without destabilizing API usage.

### Pros
- Cleaner and deterministic artifact layout aligned with strategy-based experiments.
- Faster ingestion on multi-source runs with bounded parallelism.
- Better resilience against transient HF throttling/errors.

### Cons
- Existing legacy processed artifacts must be regenerated under strategy folders.
- Added concurrency/retry settings increase runtime configuration surface.

---

## [0.2.3] — 2026-05-30

### What Changed
- Moved the canonical `Chunk` model definition to `src/tax_talk/models/ingestion.py`.
- Updated `src/tax_talk/ingestion/chunker.py` to import and use the shared `Chunk` model instead of defining a duplicate dataclass.
- Updated ingestion call sites (`run.py`, `qdrant_store.py`) to import `Chunk` from the models module.
- Updated `.copilot/ingestion.md` to reflect the canonical model ownership.

### Why Changed
- Remove model duplication and keep a single source of truth for chunk fields and payload serialization.
- Align with repository convention to keep models in the `models/` package.

### Pros
- Lower schema drift risk between runtime chunk handling and persisted artifacts.
- Clearer module boundaries: chunker orchestrates, models define schemas.

### Cons
- Ingestion runtime now depends on the models module for chunk shape.

---

## [0.2.2] — 2026-05-29

### What Changed
- Created new package folder `src/tax_talk/ingestion/chunking_strategies/` and moved strategy logic into dedicated modules:
	- `base.py` (`ChunkingStrategy` contract)
	- `fixed.py` (`FixedChunkingStrategy`)
	- `semantic.py` (`SemanticChunkingStrategy`)
	- `contextual.py` (`ContextualChunkingStrategy`)
	- `registry.py` (`CHUNKING_STRATEGIES`, resolver/factory)
	- `helpers.py` (fixed/semantic split helpers and contextual prefix helper)
	- `__init__.py` package exports
- Updated `src/tax_talk/ingestion/chunker.py` to focus on orchestration and re-export compatibility symbols while delegating strategy behavior to the new package.
- Kept compatibility for existing imports/tests that import strategy classes and resolver APIs from `chunker.py`.

### Why Changed
- User requested moving chunking strategy implementations into a dedicated folder.
- Improve module boundaries so strategy implementations are decoupled from chunk orchestration and artifact IO.

### Pros
- Cleaner package layout and easier navigation for strategy-specific changes.
- Better extensibility for adding new strategies without bloating `chunker.py`.
- Preserved backward compatibility for existing call sites.

### Cons
- More modules/files to track when debugging strategy behavior.

---

## [0.2.1] — 2026-05-29

### What Changed
- Refactored `src/tax_talk/ingestion/chunker.py` to use an explicit strategy-pattern design:
	- added `ChunkingStrategy` abstract contract
	- added concrete `FixedChunkingStrategy`, `SemanticChunkingStrategy`, and `ContextualChunkingStrategy`
	- added registry-backed `get_chunking_strategy(...)` factory
	- moved splitting internals to strategy-friendly helpers (`_split_fixed_text`, `_split_semantic_text`)
	- updated `chunk_document(...)` to delegate splitting/summary/rendering to strategy objects instead of inline `if/else`
- Added strategy-factory tests in `tests/test_chunking_strategies.py`.
- Updated `.copilot/ingestion.md` to document strategy resolver/factory usage.

### Why Changed
- User requested chunker distribution to follow the strategy pattern.
- Reduce branching in orchestration and make strategy extensions safer and clearer.

### Pros
- Cleaner separation of concerns between orchestration and strategy behavior.
- Easier to add new strategies without editing core chunking orchestration flow.
- Testable strategy factory with explicit implementation mapping.

### Cons
- Slightly more indirection compared to direct function branching.

---

## [0.2.0] — 2026-05-29

### What Changed
- Added strategy-aware ingestion controls for collection-isolated indexing:
	- `src/tax_talk/ingestion/run.py` now accepts `--chunking-strategy` and `--collection`
	- `src/tax_talk/ingestion/chunker.py` now supports `fixed`, `semantic`, and `contextual` strategies
	- `src/tax_talk/ingestion/qdrant_store.py` accepts explicit collection names
	- `src/tax_talk/retrieval/hybrid.py` can target a specific collection at construction
- Added strategy metadata persistence in chunk artifacts and payloads:
	- `chunking_strategy` field in `Chunk` payloads and `ChunkRecord`
	- embedding manifest now records `qdrant_collection` and `chunking_strategy`
- Added new eval package under `src/tax_talk/evals/`:
	- `dataset.py` for golden QA JSONL validation
	- `runner.py` for retrieval + grounded generation + RAGAS scoring
	- `run.py` CLI entrypoint for collection/strategy-specific eval runs
- Added hand-built eval dataset seed at `data/eval/golden_qa_v0_30.jsonl`.
- Added tests for strategy resolution and eval dataset loading:
	- `tests/test_chunking_strategies.py`
	- `tests/test_eval_dataset.py`
- Updated docs and assistant context files:
	- `README.md`, `EVALS.md`, `ARCHITECTURE.md`
	- `.copilot/ingestion.md`, `.copilot/retrieval.md`, `.copilot/tests.md`, `.copilot/hooks.md`, new `.copilot/evals.md`

### Why Changed
- Enable fair chunking-strategy experiments by isolating each strategy in its own Qdrant collection.
- Provide an executable offline eval path for 30 hand-built domain QA pairs with RAGAS metrics and Langfuse traces.

### Pros
- Reproducible strategy benchmarking with explicit collection boundaries.
- First-class eval module in `src/` with validated dataset loading and metric generation.
- Better observability for evaluation runs through Langfuse spans.

### Cons
- Additional configuration and CLI options increase operational surface area.
- Initial semantic chunking implementation is rule-based and may need later tuning.

---

## [0.1.3] — 2026-05-29

### What Changed
- Added a generation strategy abstraction for provider-specific LLM calls:
	- `src/tax_talk/generation/llm_provider.py` (`LLMStrategy` contract)
	- `src/tax_talk/generation/gemini_strategy.py`
	- `src/tax_talk/generation/groq_strategy.py`
	- updated `src/tax_talk/generation/__init__.py` exports
- Added Langfuse tracing on provider generation calls:
	- `generation-gemini` span on Gemini `generate()`
	- `generation-groq` span on Groq `generate()`
	- both use `as_type="generation"` with prompt/output capture
- Extended runtime singletons in `src/tax_talk/core/runtime.py`:
	- `get_gemini_client()` singleton
	- `get_groq_client()` singleton
	- `get_llm_strategy(provider)` runtime-owned strategy singleton cache
- Refactored contextual-summary fallback in `src/tax_talk/ingestion/contextual_summary.py` to use runtime provider strategy access instead of direct provider SDK calls.
- Added contextual-summary provider selector in config:
	- `contextual_summary_fallback_provider` in `src/tax_talk/core/config.py`
- Switched Groq integration to native Groq Python SDK usage:
	- replaced OpenAI-compat Groq path with `from groq import Groq`
	- updated runtime Groq client initialization to `Groq(api_key=...)`
	- added dependency `groq>=0.9.0` in `pyproject.toml`
- Added tests for runtime strategy/provider behavior in `tests/test_llm_strategy_runtime.py`.

### Why Changed
- Support project-wide reusable LLM provider calls behind a single runtime abstraction.
- Avoid duplicated provider wiring and per-call client construction.
- Keep Groq integration aligned with official current SDK usage.

### Pros
- LLM provider calls are now centralized, testable, and easy to extend.
- Runtime reuses singleton clients/strategies, reducing repeated initialization overhead.
- Contextual-summary fallback can switch providers through config.

### Cons
- Slightly more runtime indirection for LLM call paths.
- Additional provider configuration surface to maintain.

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
- Updated `ARCHITECTURE.md` to distinguish implemented retrieval/API flow from planned query-rewrite/generation/judge stages, and synced embedding/rerank decisions to current defaults.
- Fixed ingestion chunk metadata mapping to read nested `ingestion_metadata` / `chunk_metadata` values from raw `metadata.json` instead of falling back to `unknown`.
- Added metadata-first contextual chunk summaries with optional LLM fallback over a truncated document prefix, and persisted summary text/provenance in chunk artifacts.
- Added chunker regression tests covering nested metadata mapping, metadata-summary path, LLM fallback path, and artifact round-trips.

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
