# Tax Talk: GST + Income Tax RAG

Production-grade retrieval-augmented generation over Indian GST and Income Tax statutes, notifications, circulars, and case law.

**Status:** 🚧 In active development — see [project board](https://github.com/YOUR_USERNAME/tax_talk-gst/projects).

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
| Embeddings | Azure OpenAI `text-embedding-3-large` | Best-in-class on legal/regulatory text; billed to Azure credit (no card needed) |
| Vector DB | Qdrant (self-hosted) | Open-source, supports binary quantization |
| Sparse retrieval | BM25 (rank_bm25) | Catches exact section numbers and statute references |
| Fusion | Reciprocal Rank Fusion | Robust combination of sparse + dense |
| Reranker | Cohere Rerank v3 | Significant precision lift on hand-labeled set |
| Generation | Azure OpenAI GPT-4o / Gemini 2.0 Flash | GPT-4o for high-stakes, Gemini for dev |
| Orchestration | LangChain (retrieval) + Pydantic AI (generation) | Type-safe outputs |
| Observability | Langfuse Cloud | Free tier, captures all traces and costs |
| API | FastAPI + SSE streaming | Production-standard |
| Eval framework | RAGAS + custom LLM-as-judge | Faithfulness, context relevance, citation accuracy |
| Hosting | Azure VM (B1S, Central India) + Vercel frontend | Free via Azure for Students |

## Data sources (all official, public)

- **GST**: CBIC notifications, circulars at [cbic.gov.in](https://www.cbic.gov.in/)
- **Income Tax**: Acts and notifications at [incometax.gov.in](https://www.incometax.gov.in/)
- **Acts**: CGST Act, IGST Act, UTGST Act, Income Tax Act 1961
- **Case law**: Selected Supreme Court and High Court judgments from indiankanoon.org (sampled, attribution preserved)

Corpus license/copyright: All statutes are public domain. Notifications and circulars are government-issued public documents. Case law citations preserved with original source links.

## Evals

See [EVALS.md](./EVALS.md) for the full eval harness, metrics, and current scores.

Current snapshot (will update weekly):
- Faithfulness (LLM-as-judge): _TBD_
- Context precision: _TBD_
- Context recall: _TBD_
- Citation accuracy: _TBD_
- Avg latency p95: _TBD_
- Avg cost per query: _TBD_

## Costs

See [COSTS.md](./COSTS.md) for token-level cost breakdown and optimization choices.

## Live demo

[tax_talk.yourname.me](https://tax_talk.yourname.me) — coming after week 7.

## Local development

```bash
# Prerequisites: Python 3.11+, Docker, uv
git clone https://github.com/YOUR_USERNAME/tax_talk-gst
cd tax_talk-gst

# Install dependencies
uv sync

# Bring up infrastructure (Qdrant)
docker compose up -d

# Copy env and add your API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, LANGFUSE_*

# Run ingestion (chunks + embeds the corpus)
make ingest

# Start API
make serve

# Run evals
make eval
```

## Roadmap

- [x] Week 3: Basic ingestion pipeline working
- [ ] Week 4: Hybrid retrieval + reranking
- [ ] Week 4: First 30 Q/A eval pairs hand-labeled
- [ ] Week 7: FastAPI + SSE streaming + deploy
- [ ] Week 8: Full eval suite (100+ pairs), 3 experiments documented

## Lessons learned (will fill in as I go)

> *Section to be added — interviewers love an honest "what I'd do differently" reflection.*

## License

MIT — see [LICENSE](./LICENSE).
