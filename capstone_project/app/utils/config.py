"""
app/utils/config.py
-------------------
Centralised configuration loaded from .env via python-dotenv.
All other modules import settings from here — no direct os.getenv() calls elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=True)


class Settings(BaseSettings):
    """Application-wide settings validated by Pydantic."""

    # --- OpenRouter ---
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="mistralai/mistral-7b-instruct", env="OPENROUTER_MODEL"
    )

    # --- Paths ---
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"), env="CHROMA_PERSIST_DIR"
    )
    policies_dir: Path = Field(
        default=Path("./data/policies"), env="POLICIES_DIR"
    )
    audit_log_path: Path = Field(
        default=Path("./logs/audit.json"), env="AUDIT_LOG_PATH"
    )

    # --- Retrieval tuning ---
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")

    # Minimum cosine similarity score for a chunk to pass the refusal gate.
    # Range [0.0, 1.0]. Lower values = more permissive (fewer refusals).
    # 0.10 is a safe floor for compliance Q&A with cosine distance.
    # Set SIMILARITY_MIN_SCORE=0.20 in .env to tighten if needed.
    similarity_min_score: float = Field(default=0.10, env="SIMILARITY_MIN_SCORE")

    # Minimum confidence required before flagging for human review.
    confidence_threshold: float = Field(default=0.65, env="CONFIDENCE_THRESHOLD")

    # --- ChromaDB ---
    chroma_collection_name: str = "compliance_policies"

    @field_validator("chroma_persist_dir", "policies_dir", "audit_log_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Resolve relative paths against the project root."""
        p = Path(v)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
