"""12-factor settings: all runtime config from environment variables.

A single Settings object is the only way config enters the platform — no
scattered os.environ reads, so a deploy is fully described by its env.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASKDESK_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/askdesk"
    # Postgres schema the chunks table lives in. Lets AskDesk share a database
    # without touching anyone else's namespace (e.g. per-team corpora).
    db_schema: str = "public"
    chat_model: str = "anthropic/claude-sonnet-4-6"
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    api_key: str = "change-me"
    max_critic_retries: int = 1
    top_k: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
