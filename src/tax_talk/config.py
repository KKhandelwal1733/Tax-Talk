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

    # === LLM API keys ===
    anthropic_api_key: str = Field(default="", description="Claude API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    cohere_api_key: str = Field(default="", description="Cohere API key (for Rerank)")

    # === Observability ===
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # === Infrastructure ===
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "gst_income_tax"

    # === App config ===
    env: str = "development"
    log_level: str = "INFO"

    # === Model selection (3-tier strategy) ===
    dev_model: str = "gemini-2.0-flash"
    speed_model: str = "groq/llama-3.3-70b-versatile"
    eval_model: str = "claude-sonnet-4-5"

    # === Cost controls (set on day one, never bypass) ===
    max_cost_per_request_usd: float = 0.10
    max_agent_loops: int = 5


# Singleton import target
settings = Settings()
