"""Centralized configuration via Pydantic Settings.

All runtime config comes from environment variables (loaded from .env in dev).
Import settings anywhere in the app instead of reading os.environ directly.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === LLM API keys (no-card path) ===
    anthropic_api_key: str = Field(default="", description="Claude API key — $5 starter only")
    gemini_api_key: str = Field(
        default="", description="Google Gemini — primary workhorse + embeddings"
    )
    groq_api_key: str = Field(default="", description="Groq — fast inference")
    cohere_api_key: str = Field(default="", description="Cohere — for Rerank")
    voyage_api_key: str = Field(default="", description="Voyage AI — optional, best for legal text")
    hf_token: str = Field(
        default="", description="Hugging Face token for authenticated Hub/Inference requests"
    )

    # === Embeddings (pick ONE provider, must be consistent across ingestion + queries) ===
    # Options:
    #   "sentence_transformer" → sentence-transformers BAAI/bge-base-en-v1.5 (free, no API, recommended)
    #   "gemini"  → text-embedding-004 via Gemini API (free tier, needs GEMINI_API_KEY)
    #   "voyage"  → voyage-3 via Voyage AI (best for legal text, needs VOYAGE_API_KEY)
    embedding_provider: str = "sentence_transformer"
    embedding_model_local: str = "BAAI/bge-base-en-v1.5"  # 270 MB, 768-dim, fast on CPU
    embedding_local_mode: str = "local"  # local | hf_inference
    embedding_model_gemini: str = "models/text-embedding-004"  # 768-dim, free 1500 RPD
    embedding_model_voyage: str = "voyage-3"  # 1024-dim, 50M free tokens
    embedding_batch_size: int = 64  # chunks per embedding batch (tune for memory)
    embedding_dimensions: int = 768  # must match Qdrant collection — change if provider changes

    # === Reranking (optional post-retrieval stage) ===
    rerank_enabled: bool = True
    rerank_model: str = "rerank-v4.0-pro"
    rerank_top_k: int = 30
    rerank_top_n: int = 10
    rerank_max_tokens_per_doc: int = 4096

    # === Contextual chunk summaries ===
    contextual_summary_enabled: bool = True
    contextual_summary_metadata_min_chars: int = 400
    contextual_summary_max_chars: int = 600
    contextual_summary_prefix_label: str = "Context"
    contextual_summary_llm_fallback_enabled: bool = True
    contextual_summary_fallback_provider: str = "gemini"
    contextual_summary_fallback_model: str = "gemini-3.1-flash-lite"
    contextual_summary_document_prefix_chars: int = 12_000

    # === Chunking strategies for ingestion experiments ===
    chunking_strategy: str = "contextual"  # fixed | semantic | contextual
    chunk_size_chars: int = 2_000
    chunk_overlap_chars: int = 200
    semantic_chunk_min_chars: int = 900
    semantic_chunk_max_chars: int = 2_400

    # === Observability ===
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # === Infrastructure ===
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "gst_income_tax"

    # === Eval runner defaults ===
    eval_provider: str = "gemini"
    eval_model: str = "gemini-2.0-flash"
    eval_top_k: int = 8
    eval_dataset_path: str = "data/eval/golden_qa_v0_30.jsonl"
    eval_results_dir: str = "data/eval/results"

    # === Postgres ===
    database_url: str = ""

    # === App config ===
    env: str = "development"
    log_level: str = "INFO"

    # === Model selection (3-tier strategy, no-card path) ===
    dev_model: str = "gemini-2.0-flash"
    speed_model: str = "groq/llama-3.3-70b-versatile"
    eval_judge_model: str = "gemini-2.0-flash"

    # === Cost controls (set on day one, never bypass) ===
    max_cost_per_request_usd: float = 0.10
    max_agent_loops: int = 5

    def model_post_init(self, __context):
        """Export Langfuse settings to os.environ so @observe decorators find them."""
        if self.langfuse_public_key:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_public_key
        if self.langfuse_secret_key:
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_secret_key
        if self.langfuse_host:
            os.environ["LANGFUSE_HOST"] = self.langfuse_host


# Singleton import target
settings = Settings()
