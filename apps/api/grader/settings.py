from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
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

    # Path to the manufacturer reference-embeddings .npz produced by
    # ml/data/ingestion/reference_embeddings.embed_references. The
    # embedding-anomaly counterfeit detector looks up authentic
    # exemplars here at inference time. If the file doesn't exist (no
    # references ingested yet, or the path is wrong), the detector
    # gracefully degrades to UNVERIFIED for every submission — the
    # rest of the ensemble (rosette + color) still runs.
    references_embeddings_path: str = Field(
        default="~/manufacturer_refs/reference_embeddings.npz",
        description=(
            "Path to the reference embeddings npz used by the "
            "embedding-anomaly counterfeit detector. Tilde-expanded."
        ),
    )

    # ----- Auth (Clerk) -----
    # The auth dependency supports two modes, switched via `dev_auth_enabled`:
    #
    # * Dev mode (default in env=dev): accepts `Authorization: Dev <clerk_id>`
    #   and auto-creates a User on first sight. Keeps tests hermetic.
    # * Prod mode: verifies an `Authorization: Bearer <jwt>` against Clerk's
    #   JWKS, validates standard claims, and resolves/auto-creates a User
    #   keyed on the JWT `sub`.
    #
    # In dev/staging/prod the default is computed from `env`, but every value
    # is overridable via env var (e.g. `DEV_AUTH_ENABLED=true` in a staging
    # box for debugging). Pydantic-settings loads env vars case-insensitively.
    clerk_jwks_url: str | None = Field(
        default=None,
        description=(
            "URL of the Clerk JWKS endpoint, e.g. "
            "https://<your-instance>.clerk.accounts.dev/.well-known/jwks.json. "
            "Required when `dev_auth_enabled` is False."
        ),
    )
    clerk_issuer: str | None = Field(
        default=None,
        description=(
            "Expected `iss` claim on Clerk JWTs, e.g. "
            "https://<your-instance>.clerk.accounts.dev. Required in prod mode."
        ),
    )
    clerk_audience: str | None = Field(
        default=None,
        description=(
            "Expected `aud` claim. Optional — many Clerk setups omit `aud`. "
            "When set, JWTs without a matching `aud` are rejected."
        ),
    )
    dev_auth_enabled: bool | None = Field(
        default=None,
        description=(
            "When True, accept `Authorization: Dev <token>` (auto-creates User). "
            "When False, require a Clerk `Bearer <jwt>`. If unset, defaults to "
            "True for env=dev and False otherwise."
        ),
    )
    clerk_jwks_cache_ttl_seconds: int = Field(
        default=300,
        description=(
            "TTL for the in-memory JWKS signing-key cache. Clerk rotates keys "
            "rarely, so 5 minutes is a safe default; tune lower if you need "
            "faster propagation after a key rotation."
        ),
    )

    @model_validator(mode="after")
    def _default_dev_auth_from_env(self) -> "Settings":
        # Compute a sane default for `dev_auth_enabled` based on `env` when the
        # operator has not pinned it explicitly. This is the "fail-closed in
        # prod, fail-open in dev" rule.
        if self.dev_auth_enabled is None:
            object.__setattr__(self, "dev_auth_enabled", self.env == "dev")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
