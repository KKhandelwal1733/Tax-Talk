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
    #   "sentence_transformer" → HF Inference API using BAAI/bge-base-en-v1.5 (requires HF_TOKEN)
    #   "gemini"  → text-embedding-004 via Gemini API (free tier, needs GEMINI_API_KEY)
    #   "voyage"  → voyage-3 via Voyage AI (best for legal text, needs VOYAGE_API_KEY)
    embedding_provider: str = "sentence_transformer"
    embedding_model_sentence_transformer: str = "BAAI/bge-m3"
    embedding_model_gemini: str = "models/text-embedding-004"  # 768-dim, free 1500 RPD
    embedding_model_voyage: str = "voyage-3"  # 1024-dim, 50M free tokens
    embedding_batch_size: int = 64  # chunks per embedding batch (tune for memory)
    embedding_dimensions: int = 1024  # must match Qdrant collection — change if provider changes
    ingestion_max_workers: int = 2  # source-level workers for embed/upsert phases
    hf_max_parallel_sources: int = 2  # hard cap for source-level concurrency in hf_inference mode
    hf_max_concurrent_requests: int = 2  # concurrent HF embedding requests across workers
    hf_retry_max_attempts: int = 5
    hf_retry_initial_delay_seconds: float = 1.0
    hf_retry_max_delay_seconds: float = 12.0

    # === Reranking (optional post-retrieval stage) ===
    rerank_enabled: bool = True
    rerank_model: str = "rerank-v4.0-pro"
    rerank_top_k: int = 30
    rerank_top_n: int = 10
    rerank_max_tokens_per_doc: int = 4096

    # === Contextual chunk summaries ===
    contextual_summary_enabled: bool = True
    contextual_summary_metadata_min_chars: int = 250
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
    eval_model: str = "gemini-3.5-flash"
    eval_fallback_chain_csv: str = ""
    eval_top_k: int = 8
    eval_llm_rate_limit_calls: int = 4
    eval_llm_rate_limit_window_seconds: float = 60.0
    eval_dataset_path: str = "data/eval/golden_qa_v0_30.jsonl"
    eval_results_dir: str = "data/eval/results"

    # === Postgres ===
    database_url: str = ""

    # === App config ===
    env: str = "development"
    log_level: str = "INFO"
    
    #=== Supabase ===
    supabase_url: str = ""
    supabase_jwt_issuer: str = ""
    supabase_jwt_audience: str = ""

    # === Cost controls (set on day one, never bypass) ===
    max_cost_per_request_usd: float = 0.10
    max_agent_loops: int = 5

    # chat model
    chat_model: str = "gemini-3.1-flash-lite"

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
