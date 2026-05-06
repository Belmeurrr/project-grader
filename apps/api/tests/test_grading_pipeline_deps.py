"""Tests for the per-worker-process dependency cache in
`grader.workers.grading_pipeline`.

The Celery task pays an 8-15s tax to load DINOv2 + transformers + GPU
weights on every cold start. We hoist that work to the
`worker_process_init` signal so each worker child loads once and reuses
the deps across all tasks it handles. These tests assert the cache
shape: `_get_production_dependencies()` returns the same instance on
repeated calls, and the underlying `_build_production_dependencies` is
not invoked more than once even across multiple `process_submission`
invocations.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from grader.workers import grading_pipeline


@pytest.fixture(autouse=True)
def _reset_deps_cache():
    """Each test starts with a cold cache. The module-level global persists
    across tests in the same process, which would mask cache bugs."""
    prior = grading_pipeline._PRODUCTION_DEPS
    grading_pipeline._PRODUCTION_DEPS = None
    yield
    grading_pipeline._PRODUCTION_DEPS = prior


def test_get_production_dependencies_caches_on_first_call() -> None:
    """First call builds, subsequent calls return the cached instance.

    This is the test fallback path: when `worker_process_init` doesn't
    fire (tests, eager mode, sync invocation), the lazy build kicks in.
    """
    sentinel = (MagicMock(name="catalog"), MagicMock(name="embedder"))
    with patch.object(
        grading_pipeline,
        "_build_production_dependencies",
        return_value=sentinel,
    ) as build_mock:
        first = grading_pipeline._get_production_dependencies()
        second = grading_pipeline._get_production_dependencies()
        third = grading_pipeline._get_production_dependencies()

    assert first is sentinel
    assert second is sentinel
    assert third is sentinel
    assert first is second is third
    assert build_mock.call_count == 1, (
        f"expected exactly 1 build, got {build_mock.call_count}"
    )


def test_worker_process_init_populates_cache() -> None:
    """The `worker_process_init` signal handler must populate the global
    cache so subsequent `_get_production_dependencies()` calls don't
    rebuild."""
    sentinel = (MagicMock(name="catalog"), MagicMock(name="embedder"))
    with patch.object(
        grading_pipeline,
        "_build_production_dependencies",
        return_value=sentinel,
    ) as build_mock:
        # Simulate Celery firing the signal in a freshly-forked worker.
        grading_pipeline._init_production_dependencies()
        # Now reads should hit the cache.
        deps = grading_pipeline._get_production_dependencies()
        deps_again = grading_pipeline._get_production_dependencies()

    assert deps is sentinel
    assert deps_again is sentinel
    assert build_mock.call_count == 1


def test_process_submission_does_not_rebuild_deps_per_task() -> None:
    """Calling the underlying task body twice should call
    `_build_production_dependencies` at most once. We patch the
    pipeline runner so the task short-circuits without needing a DB or
    real ML stack."""
    sentinel_catalog = MagicMock(name="catalog")
    sentinel_embedder = MagicMock(name="embedder")
    sentinel = (sentinel_catalog, sentinel_embedder)

    fake_result = MagicMock()
    fake_result.submission_id = uuid.uuid4()
    fake_result.status = MagicMock(value="COMPLETED")
    fake_result.grades = []
    fake_result.identified_variant_id = None
    fake_result.failure_reason = None

    with (
        patch.object(
            grading_pipeline,
            "_build_production_dependencies",
            return_value=sentinel,
        ) as build_mock,
        patch.object(
            grading_pipeline,
            "run_pipeline_sync",
            return_value=fake_result,
        ) as runner_mock,
    ):
        sub_id_1 = str(uuid.uuid4())
        sub_id_2 = str(uuid.uuid4())
        sub_id_3 = str(uuid.uuid4())

        # `.run` invokes the task body synchronously without needing a
        # broker. `bind=True` means the task receives `self` as first
        # arg automatically when called via `.run`.
        grading_pipeline.process_submission.run(sub_id_1)
        grading_pipeline.process_submission.run(sub_id_2)
        grading_pipeline.process_submission.run(sub_id_3)

    assert build_mock.call_count == 1, (
        f"expected exactly 1 build across 3 task invocations, got "
        f"{build_mock.call_count}"
    )
    assert runner_mock.call_count == 3
    # Verify each task call received the cached deps, not freshly-built ones.
    for call in runner_mock.call_args_list:
        assert call.kwargs["catalog"] is sentinel_catalog
        assert call.kwargs["embedder"] is sentinel_embedder
