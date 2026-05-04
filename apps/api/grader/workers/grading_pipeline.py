"""Celery task: process a submission through the full grading pipeline.

The task is intentionally a thin wrapper around `pipeline_runner.run_pipeline_sync`.
All orchestration logic lives in the runner so it's testable without a Celery
broker.

Retry policy:
  - Transient infra errors (S3, DB) auto-retry with exponential backoff.
  - Pipeline-level failures (no card detected, quality fail, etc.) are
    persisted as `submission.status = FAILED` and do NOT retry — retrying
    won't change the outcome.
"""

from __future__ import annotations

import logging
import uuid

from celery import shared_task
from celery.exceptions import Reject
from sqlalchemy.ext.asyncio import async_sessionmaker

from grader.db.session import async_session_maker
from grader.workers.celery_app import celery_app
from grader.workers.pipeline_runner import (
    PipelineValidationError,
    run_pipeline_sync,
)
from pipelines.identification import (
    CatalogIndex,
    ImageEmbedder,
    SimpleEmbedder,
    get_embedder,
)

log = logging.getLogger(__name__)


def _build_production_dependencies() -> tuple[CatalogIndex, ImageEmbedder]:
    """Wire the production dependencies for the pipeline.

    Catalog: pgvector-backed, sharing the API session factory.
    Embedder: chosen by env (DinoV2 if GRADER_EMBEDDER=dinov2, else Simple).

    Lives in this module rather than the runner so the runner stays a pure
    function and tests can inject mock implementations.
    """
    from grader.services.catalog_repo import PgVectorCatalogIndex

    catalog = PgVectorCatalogIndex(session_factory=async_session_maker)
    embedder = get_embedder()
    return catalog, embedder


@celery_app.task(
    bind=True,
    name="grader.workers.grading_pipeline.process_submission",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def process_submission(self, submission_id: str) -> dict:
    """Run the grading pipeline for a single submission. Returns a small
    summary dict suitable for the Celery result backend.

    submission_id is a string (uuid serialized) because Celery serializes
    arguments via JSON; UUID isn't JSON-native."""
    try:
        sid = uuid.UUID(submission_id)
    except ValueError as e:
        # Malformed input is a permanent failure; reject without retry.
        raise Reject(reason=f"invalid submission id: {submission_id}", requeue=False) from e

    catalog, embedder = _build_production_dependencies()
    result = run_pipeline_sync(
        submission_id=sid,
        session_factory=async_session_maker,
        catalog=catalog,
        embedder=embedder,
    )

    return {
        "submission_id": str(result.submission_id),
        "status": result.status.value,
        "grades": [str(g.id) for g in result.grades],
        "identified_variant_id": (
            str(result.identified_variant_id)
            if result.identified_variant_id
            else None
        ),
        "failure_reason": result.failure_reason,
    }
