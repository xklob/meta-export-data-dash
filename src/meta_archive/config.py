from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-backed process settings.

    User preferences that belong to an archive live in SQLite instead of this object.
    """

    model_config = SettingsConfigDict(
        env_prefix="META_ARCHIVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    archive_root: Path | None = Field(default=None, alias="META_ARCHIVE_ROOT")
    host: str = Field(default="127.0.0.1", alias="META_ARCHIVE_HOST")
    port: int = Field(default=8765, ge=1, le=65535, alias="META_ARCHIVE_PORT")
    test_mode: bool = Field(default=False, alias="META_ARCHIVE_TEST_MODE")
    embedding_model: str = Field(default="auto", alias="META_ARCHIVE_EMBEDDING_MODEL")
    embedding_batch_size: int = Field(
        default=128, ge=1, le=1024, alias="META_ARCHIVE_EMBEDDING_BATCH_SIZE"
    )
    semantic_candidate_multiplier: int = Field(
        default=8, ge=2, le=50, alias="META_ARCHIVE_SEMANTIC_CANDIDATE_MULTIPLIER"
    )


settings = AppSettings()
