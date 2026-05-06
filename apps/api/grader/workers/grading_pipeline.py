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
from celery.signals import worker_process_init
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

    Note: `DinoV2Embedder` is the expensive bit — it lazily loads torch +
    transformers + the `facebook/dinov2-base` weights (~8-15s cold start)
    and moves the model to GPU. The catalog and embedder are both safe to
    reuse across submissions: the catalog only holds a reference to the
    process-global `async_session_maker`, and the embedder's only mutable
    state is the loaded model (read-only at inference time). They are
    cached per worker process via `worker_process_init` below.
    """
    from grader.services.catalog_repo import PgVectorCatalogIndex

    catalog = PgVectorCatalogIndex(session_factory=async_session_maker)
    embedder = get_embedder()
    return catalog, embedder


# Per-worker-process cache of the heavy dependencies. Populated by the
# `worker_process_init` signal so each Celery prefork child loads torch +
# DINOv2 weights exactly once, not once per task.
#
# IMPORTANT: This is a module-level global on purpose, but Celery's prefork
# pool forks AFTER the master imports the worker module — so each child
# process gets its own fresh copy of `_PRODUCTION_DEPS = None` and then
# the `worker_process_init` signal fires inside that child. The cache is
# therefore process-local even with `--concurrency=N`. We must not
# initialize this in module scope (would either be skipped due to fork
# semantics, or worse, cause CUDA/torch state to be shared across forks
# in unsafe ways).
_PRODUCTION_DEPS: tuple[CatalogIndex, ImageEmbedder] | None = None


@worker_process_init.connect
def _init_production_dependencies(**kwargs: object) -> None:
    """Build the heavy ML dependencies once per worker process.

    Fires after Celery forks each worker child but before any task runs.
    Subsequent tasks read from `_PRODUCTION_DEPS` instead of paying the
    8-15s DINOv2 cold-start tax on every submission.
    """
    global _PRODUCTION_DEPS
    log.info("worker_process_init: building production dependencies")
    _PRODUCTION_DEPS = _build_production_dependencies()
    log.info("worker_process_init: production dependencies ready")


def _get_production_dependencies() -> tuple[CatalogIndex, ImageEmbedder]:
    """Return the cached per-process production deps, building on first
    call if the `worker_process_init` signal hasn't fired (tests, sync
    invocation, eager mode)."""
    global _PRODUCTION_DEPS
    if _PRODUCTION_DEPS is None:
        _PRODUCTION_DEPS = _build_production_dependencies()
    return _PRODUCTION_DEPS


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

    catalog, embedder = _get_production_dependencies()
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
