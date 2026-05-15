"""
logbot/core/config.py
─────────────────────
Single source of truth for every runtime setting in LogBot.

Design decisions (interview-ready talking points):
  • Pydantic BaseSettings  → env vars, .env file, and defaults unified.
  • Field validators       → fail-fast on misconfiguration at startup.
  • Computed properties    → no string-building scattered in downstream code.
  • Nested models          → cohesion; each subsystem owns its settings block.
  • lru_cache singleton    → one parse per process lifetime, safe to import anywhere.
"""

from __future__ import annotations

import socket
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"
    TEST        = "test"


class LogLevel(str, Enum):
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingModel(str, Enum):
    """Supported HuggingFace sentence-transformer models."""
    MINILM    = "sentence-transformers/all-MiniLM-L6-v2"   # fast,  384-dim
    MPNET     = "sentence-transformers/all-mpnet-base-v2"   # accurate, 768-dim
    BGE_SMALL = "BAAI/bge-small-en-v1.5"                   # retrieval-optimised


# ──────────────────────────────────────────────────────────────────────────────
# Nested settings blocks
# ──────────────────────────────────────────────────────────────────────────────

class VectorStoreSettings(BaseSettings):
    """ChromaDB connection + collection config."""
    model_config = SettingsConfigDict(env_prefix="CHROMA_", extra="ignore")

    host:            str = Field("localhost",       description="ChromaDB host")
    port:            int = Field(8000,              description="ChromaDB HTTP port")
    collection_name: str = Field("logbot_logs",     description="Default collection")
    distance_metric: str = Field("cosine",          description="cosine | l2 | ip")
    persist_dir:     str = Field("data/embeddings", description="Local persist path")

    @field_validator("distance_metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        allowed = {"cosine", "l2", "ip"}
        if v not in allowed:
            raise ValueError(f"distance_metric must be one of {allowed}, got '{v}'")
        return v

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class SparkSettings(BaseSettings):
    """PySpark / log-ingestion pipeline config."""
    model_config = SettingsConfigDict(env_prefix="SPARK_", extra="ignore")

    app_name:       str = Field("LogBot-Pipeline",   description="Spark app name")
    master:         str = Field("local[*]",          description="Spark master URL")
    log_level:      str = Field("WARN",              description="Spark internal log level")
    batch_size:     int = Field(1000,                description="Rows per micro-batch")
    max_offsets:    int = Field(10_000,              description="Max Kafka offsets/trigger")
    checkpoint_dir: str = Field("data/checkpoints",  description="Streaming checkpoint path")
    raw_log_path:   str = Field("data/raw",          description="Ingest source directory")
    processed_path: str = Field("data/processed",    description="Processed output directory")

    @field_validator("master")
    @classmethod
    def validate_master(cls, v: str) -> str:
        if not (v.startswith("local") or v.startswith("spark://") or v == "yarn"):
            raise ValueError(f"Invalid Spark master: '{v}'")
        return v


class LLMSettings(BaseSettings):
    """LLM / HuggingFace inference config."""
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider:          str            = Field("huggingface",
                                              description="huggingface | openai | anthropic")
    model_name:        str            = Field("mistralai/Mistral-7B-Instruct-v0.2")
    embedding_model:   EmbeddingModel = Field(EmbeddingModel.MINILM)
    embedding_dim:     int            = Field(384,   description="Must match embedding_model output dim")
    max_new_tokens:    int            = Field(512,   description="Max tokens in LLM response")
    temperature:       float          = Field(0.1,   description="Low temp → deterministic analysis")
    top_p:             float          = Field(0.9)
    device:            str            = Field("cpu", description="cpu | cuda | mps")
    hf_token:          Optional[str]  = Field(None,  description="HuggingFace API token (gated models)")
    openai_api_key:    Optional[str]  = Field(None,  env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str]  = Field(None,  env="ANTHROPIC_API_KEY")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {v}")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding_dim must be positive")
        return v

    @model_validator(mode="after")
    def warn_missing_token(self) -> "LLMSettings":
        """Emit a soft warning if provider needs a key but none is set."""
        if self.provider == "openai" and not self.openai_api_key:
            print("[WARN] LLM provider is 'openai' but OPENAI_API_KEY is not set.")
        if self.provider == "anthropic" and not self.anthropic_api_key:
            print("[WARN] LLM provider is 'anthropic' but ANTHROPIC_API_KEY is not set.")
        return self


class AnomalySettings(BaseSettings):
    """Anomaly detection thresholds and model config."""
    model_config = SettingsConfigDict(env_prefix="ANOMALY_", extra="ignore")

    isolation_forest_contamination: float     = Field(0.05,  description="Expected anomaly fraction")
    isolation_forest_estimators:    int       = Field(100,   description="n_estimators")
    z_score_threshold:              float     = Field(3.0,   description="Z-score cutoff")
    rolling_window_minutes:         int       = Field(5,     description="Rolling stats window")
    min_log_samples:                int       = Field(50,    description="Min samples before scoring")
    severity_levels:                List[str] = Field(
        default=["INFO", "WARNING", "ERROR", "CRITICAL"],
        description="Severity ladder (ascending)"
    )
    alert_on_severity:              List[str] = Field(
        default=["ERROR", "CRITICAL"],
        description="Severities that trigger immediate alerts"
    )

    @field_validator("isolation_forest_contamination")
    @classmethod
    def validate_contamination(cls, v: float) -> float:
        if not 0.0 < v < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {v}")
        return v


class APISettings(BaseSettings):
    """FastAPI server config."""
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host:                      str       = Field("0.0.0.0", description="Bind host")
    port:                      int       = Field(8080,      description="Bind port")
    workers:                   int       = Field(1,         description="Uvicorn worker count")
    reload:                    bool      = Field(False,     description="Hot-reload (dev only)")
    cors_origins:              List[str] = Field(default=["*"])
    request_timeout_seconds:   int       = Field(30)
    max_log_lines_per_request: int       = Field(500, description="Guard against giant payloads")

    @property
    def base_url(self) -> str:
        host = "localhost" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}"


# ──────────────────────────────────────────────────────────────────────────────
# Root settings
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Root configuration object.

    Hierarchy (highest → lowest priority):
      1. Environment variables
      2. .env file (resolved relative to project root)
      3. Field defaults below
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    app_name:    str         = Field("LogBot",             description="Service name")
    app_version: str         = Field("0.1.0",              description="Semantic version")
    environment: Environment = Field(Environment.DEVELOPMENT)
    log_level:   LogLevel    = Field(LogLevel.INFO)
    debug:       bool        = Field(False)

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    log_dir:  Path = Field(default_factory=lambda: Path("logs"))

    # ── Subsystem blocks (nested, each reads its own env-prefix) ─────────────
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    spark:        SparkSettings       = Field(default_factory=SparkSettings)
    llm:          LLMSettings         = Field(default_factory=LLMSettings)
    anomaly:      AnomalySettings     = Field(default_factory=AnomalySettings)
    api:          APISettings         = Field(default_factory=APISettings)

    # ── Runtime metadata (computed, not from env) ─────────────────────────────
    hostname: str = Field(default_factory=socket.gethostname)

    # ── Validators ────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def ensure_directories(self) -> "Settings":
        """Create data/ and logs/ directories if they don't exist."""
        for d in [
            self.data_dir,
            self.log_dir,
            Path(self.spark.raw_log_path),
            Path(self.spark.processed_path),
            Path(self.vector_store.persist_dir),
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def production_safety_checks(self) -> "Settings":
        """Enforce stricter rules in production."""
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("debug=True is not allowed in production")
            if self.api.reload:
                raise ValueError("api.reload=True is not allowed in production")
            if "*" in self.api.cors_origins:
                raise ValueError("Wildcard CORS origin not allowed in production")
        return self

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def service_banner(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  {self.app_name} v{self.app_version}\n"
            f"  env={self.environment.value}  host={self.hostname}\n"
            f"  log_level={self.log_level.value}  debug={self.debug}\n"
            f"  api       → {self.api.base_url}\n"
            f"  chroma    → {self.vector_store.http_url}\n"
            f"  llm       → provider={self.llm.provider}\n"
            f"              model={self.llm.model_name}\n"
            f"{'='*55}"
        )

    def to_safe_dict(self) -> Dict:
        """Return config as dict with secrets redacted — safe for logging."""
        d = self.model_dump()
        for secret in ("hf_token", "openai_api_key", "anthropic_api_key"):
            if d.get("llm", {}).get(secret):
                d["llm"][secret] = "***REDACTED***"
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Usage:
        from logbot.core.config import get_settings
        cfg = get_settings()

    In tests, call get_settings.cache_clear() before patching env vars.
    """
    return Settings()


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  →  python -m logbot.core.config
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    cfg = get_settings()
    print(cfg.service_banner)

    print("\n[safe config dump]")
    print(json.dumps(cfg.to_safe_dict(), indent=2, default=str))

    # Spot-checks
    assert isinstance(cfg.is_production, bool)
    assert isinstance(cfg.is_development, bool)
    assert cfg.vector_store.http_url.startswith("http://")
    assert cfg.api.base_url.startswith("http://")
    assert cfg.anomaly.isolation_forest_contamination == 0.05
    assert cfg.llm.embedding_dim > 0
    print("\n✅  All assertions passed — config is healthy.")
