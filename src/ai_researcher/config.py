"""Centralized configuration management using pydantic-settings.

All configuration is loaded from environment variables and/or a `.env` file.
Use `get_settings()` to access the singleton settings instance.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is three levels up from this file: src/ai_researcher/config.py → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Keys ---
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for Llama model access.",
    )
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for advanced web search (Optional).",
    )
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace Hub token for authenticated model downloads.",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini models (used as LLM judge).",
    )
    serper_api_key: str | None = Field(
        default=None,
        description="Serper.dev API key for Google Scholar queries.",
    )

    # --- LangSmith Observability ---
    langsmith_tracing_v2: str = Field(
        default="false",
        description="Enable LangSmith tracing (true/false).",
    )
    langsmith_api_key: str | None = Field(
        default=None,
        description="LangSmith API key for telemetry.",
    )
    langsmith_project: str = Field(
        default="ai-researcher",
        description="LangSmith project name.",
    )

    # --- Model Configuration ---
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Name of the model to use (e.g. gemini-2.5-flash, gemma2-9b-it, llama-3.1-8b-instant).",
    )
    model_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the model.",
    )

    # --- Agent Configuration ---
    thread_id: str = Field(
        default="default-thread",
        description="Conversation thread ID for the checkpointer.",
    )

    # --- Checkpointing ---
    checkpoint_backend: str = Field(
        default="sqlite",
        description="Checkpointer backend: 'memory' or 'sqlite'.",
    )
    checkpoint_db_url: str = Field(
        default="output/checkpoints.db",
        description=(
            "Database path for the SQLite checkpointer. "
            "Relative or absolute path to a .db file. "
            "Defaults to 'output/checkpoints.db' for persistence."
        ),
    )

    # --- Tool Configuration ---
    max_arxiv_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of arXiv papers to return per search.",
    )
    pdf_request_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="HTTP request timeout in seconds for PDF downloads.",
    )

    # --- Output ---
    output_dir: Path = Field(
        default=PROJECT_ROOT / "output",
        description="Directory for generated PDF files.",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("groq_api_key")
    @classmethod
    def validate_api_key_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Set it in your .env file or as an environment variable."
            )
        return v.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton Settings instance. Cached after first call."""
    settings = Settings()  # type: ignore[call-arg]

    # Push environment variables into os.environ for LangSmith auto-discovery
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    if settings.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
    if settings.serper_api_key:
        os.environ["SERPER_API_KEY"] = settings.serper_api_key

    if settings.langsmith_tracing_v2.lower() == "true":
        # LangChain natively requires the LANGCHAIN_ prefix for telemetry under the hood.
        # We push both just to be 100% safe.
        os.environ["LANGSMITH_TRACING_V2"] = "true"

        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        if settings.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    return settings
