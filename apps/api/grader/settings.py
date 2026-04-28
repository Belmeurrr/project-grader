from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    database_url: str = Field(
        default="postgresql+asyncpg://grader:grader@localhost:5432/grader",
        description="Postgres connection string for SQLAlchemy async engine.",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    celery_broker_url: str = Field(default="redis://localhost:6379/1")

    s3_bucket: str = Field(default="project-grader-dev")
    s3_region: str = Field(default="us-east-1")
    s3_endpoint_url: str | None = Field(
        default=None,
        description="Override for S3 endpoint (used for MinIO in local dev).",
    )
    s3_access_key_id: str | None = Field(default=None)
    s3_secret_access_key: str | None = Field(default=None)
    s3_force_path_style: bool = Field(
        default=False,
        description="MinIO requires path-style URLs; AWS S3 prefers virtual-hosted.",
    )
    s3_presigned_url_ttl_seconds: int = 60 * 15

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    submission_max_image_bytes: int = 25 * 1024 * 1024
    submission_required_shots: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
