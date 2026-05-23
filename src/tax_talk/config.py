"""Centralized configuration via Pydantic Settings.

All runtime config comes from environment variables (loaded from .env in dev).
Import `settings` anywhere in the app instead of reading os.environ directly.
"""

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
    gemini_api_key: str = Field(default="", description="Google Gemini — primary workhorse")
    groq_api_key: str = Field(default="", description="Groq — fast inference")
    cohere_api_key: str = Field(default="", description="Cohere — for Rerank")

    # === Azure OpenAI (high-stakes layer, billed to Azure credit) ===
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_gpt4o_deployment: str = "gpt-4o"
    azure_openai_gpt4o_mini_deployment: str = "gpt-4o-mini"
    azure_openai_embed_deployment: str = "embed-3-large"

    # === Observability ===
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"

    # === Infrastructure ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "gst_income_tax"

    # === Postgres ===
    database_url: str = ""

    # === App config ===
    env: str = "development"
    log_level: str = "INFO"

    # === Model selection (3-tier strategy, no-card path) ===
    dev_model: str = "gemini-2.0-flash"
    speed_model: str = "groq/llama-3.3-70b-versatile"
    eval_model: str = "azure/gpt-4o"

    # === Cost controls (set on day one, never bypass) ===
    max_cost_per_request_usd: float = 0.10
    max_agent_loops: int = 5


# Singleton import target
settings = Settings()
