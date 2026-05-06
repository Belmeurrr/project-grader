"""Observability tests — Sentry init, HTTP timer middleware, pipeline-stage logs.

Three minimum-viable additions, three corresponding test groups:

1. Sentry SDK is opt-in: when SENTRY_DSN is unset, sentry_sdk.init must NOT
   be called; when it's set, init runs once with the DSN. We monkeypatch
   sentry_sdk.init to a recording stub and drive the FastAPI lifespan
   manually.

2. The request_timer middleware emits one `http.request` structlog line per
   request, with method/path/status/duration_ms fields. We use
   `structlog.testing.capture_logs` (the canonical structlog test capture)
   inside an httpx ASGI request.

3. Each heavy pipeline stage emits one `pipeline.stage.completed` log line
   with `stage` (from bound contextvars) and `duration_ms`. We mock all
   the heavy collaborators and run the pipeline against a tiny in-memory
   submission, asserting one stage line per heavy stage.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import (
    AuthenticityVerdict,
    ShotKind,
    Submission,
    SubmissionShot,
    SubmissionStatus,
    User,
)
from grader.main import create_app, lifespan
from grader.settings import get_settings

# --------------------------------------------------------------------------
# 1. Sentry init
# --------------------------------------------------------------------------


@pytest.fixture
def reset_settings_cache() -> Iterator[None]:
    """`get_settings` is `@lru_cache`d — every test that twiddles env vars
    must clear it before AND after to avoid leaking the override into other
    tests in the session."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_sentry_init_skipped_when_dsn_unset(
    monkeypatch: pytest.MonkeyPatch, reset_settings_cache: None
) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    calls: list[dict[str, Any]] = []

    def _record(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("grader.main.sentry_sdk.init", _record)

    app = create_app()
    async with lifespan(app):
        pass

    assert calls == []  # no DSN → no init


@pytest.mark.asyncio
async def test_sentry_init_called_once_when_dsn_set(
    monkeypatch: pytest.MonkeyPatch, reset_settings_cache: None
) -> None:
    fake_dsn = "https://abc@o0.ingest.sentry.io/0"
    monkeypatch.setenv("SENTRY_DSN", fake_dsn)
    monkeypatch.setenv("ENV", "staging")
    calls: list[dict[str, Any]] = []

    def _record(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("grader.main.sentry_sdk.init", _record)

    app = create_app()
    async with lifespan(app):
        pass

    assert len(calls) == 1
    assert calls[0]["dsn"] == fake_dsn
    assert calls[0]["environment"] == "staging"
    assert calls[0]["traces_sample_rate"] == 0.1
    # All three integrations are registered (FastAPI / Celery / SQLAlchemy).
    integrations = calls[0]["integrations"]
    integration_class_names = {type(i).__name__ for i in integrations}
    assert {"FastApiIntegration", "CeleryIntegration", "SqlalchemyIntegration"} <= (
        integration_class_names
    )


# --------------------------------------------------------------------------
# 2. HTTP request timer middleware
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_timer_logs_one_line_per_request() -> None:
    """capture_logs intercepts every BoundLogger.info() call within its scope
    and returns a list of event dicts. The middleware must produce exactly
    one `http.request` entry with the four standard fields."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)

    with structlog.testing.capture_logs() as cap:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/healthz")

    assert r.status_code == 200

    http_events = [e for e in cap if e.get("event") == "http.request"]
    assert len(http_events) == 1, f"expected 1 http.request line, got {len(http_events)}: {cap}"

    event = http_events[0]
    assert event["method"] == "GET"
    assert event["path"] == "/healthz"
    assert event["status"] == 200
    assert isinstance(event["duration_ms"], float)
    assert event["duration_ms"] >= 0.0


# --------------------------------------------------------------------------
# 3. Pipeline-stage bound contextvars
# --------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_pipeline_emits_stage_completed_per_heavy_stage(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline must emit one `pipeline.stage.completed` log line per
    heavy stage (detection, identification, counterfeit, grade_compose).
    We mock every heavy collaborator so the test runs without S3, ML
    models, or OpenCV — we're testing the observability wiring, not the
    pipeline business logic."""
    from grader.services import counterfeit, detection, grading, identification
    from grader.workers import pipeline_runner

    # --- Build a minimal Submission with one FRONT_FULL shot ---
    user = User(clerk_id=f"u_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@x")
    db_session.add(user)
    await db_session.flush()

    submission = Submission(user_id=user.id, status=SubmissionStatus.CAPTURING)
    db_session.add(submission)
    await db_session.flush()

    shot = SubmissionShot(
        id=uuid.uuid4(),
        submission_id=submission.id,
        kind=ShotKind.FRONT_FULL,
        s3_key="fake/front.jpg",
        width_px=1000,
        height_px=1400,
        quality_passed=True,
        quality_metadata={},
    )
    db_session.add(shot)
    await db_session.flush()

    # --- Mock every heavy collaborator so no real S3/CV/ML runs ---
    def _fake_detect(*args: Any, **kwargs: Any) -> Any:
        result = MagicMock()
        result.canonical_s3_key = "fake/canonical.jpg"
        return result

    monkeypatch.setattr(detection, "detect_and_dewarp_shot", _fake_detect)

    async def _fake_identify(**kwargs: Any) -> Any:
        outcome = MagicMock()
        outcome.result.chosen = None
        outcome.result.submitted_embedding = None
        return outcome

    monkeypatch.setattr(
        identification, "identify_canonical_for_submission", _fake_identify
    )

    fake_rosette = MagicMock(rosette_score=0.5, confidence=0.5, analyzed_patches=4)
    fake_color = MagicMock(color_score=0.5, confidence=0.5, p95_chroma=0.5)
    fake_embedding = MagicMock(
        embedding_score=0.0, confidence=0.0, n_references=0
    )
    fake_typo = MagicMock(score=0.0, confidence=0.0, levenshtein_distance=None)
    fake_holo = MagicMock(score=0.0, confidence=0.0, flow_ratio=None, holo_mask_fraction=None)

    monkeypatch.setattr(counterfeit, "analyze_rosette", lambda *a, **kw: fake_rosette)
    monkeypatch.setattr(counterfeit, "analyze_color_profile", lambda *a, **kw: fake_color)
    monkeypatch.setattr(counterfeit, "analyze_embedding_anomaly", lambda *a, **kw: fake_embedding)
    monkeypatch.setattr(counterfeit, "analyze_typography_service", lambda *a, **kw: fake_typo)
    monkeypatch.setattr(counterfeit, "analyze_holographic_service", lambda *a, **kw: fake_holo)

    fake_authenticity = MagicMock(
        verdict=AuthenticityVerdict.UNVERIFIED,
        confidence=0.5,
    )
    monkeypatch.setattr(
        counterfeit,
        "persist_authenticity_result",
        AsyncMock(return_value=fake_authenticity),
    )

    fake_centering_result = MagicMock(psa_subgrade=8.0, back_measurement=None)
    fake_edges_result = MagicMock(psa_subgrade=8.5, back_measurement=None, worse_face="front")
    monkeypatch.setattr(grading, "grade_centering", lambda **kw: fake_centering_result)
    monkeypatch.setattr(grading, "grade_edges", lambda **kw: fake_edges_result)

    fake_grade_row = MagicMock(confidence=0.9)
    monkeypatch.setattr(
        grading, "persist_centering_grade", AsyncMock(return_value=fake_grade_row)
    )
    monkeypatch.setattr(
        grading, "persist_edges_grade", AsyncMock(return_value=fake_grade_row)
    )

    # --- Run pipeline under structlog capture ---
    # `capture_logs` clears the configured processor chain, so we pass
    # `merge_contextvars` explicitly — without it, the `stage` and
    # `submission_id` we bind via `bound_contextvars` would not appear
    # in the captured event dicts.
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as cap:
        await pipeline_runner.run_pipeline(
            submission_id=submission.id,
            db=db_session,
            catalog=MagicMock(),
            embedder=MagicMock(),
        )

    stage_events = [e for e in cap if e.get("event") == "pipeline.stage.completed"]
    stages_seen = [e.get("stage") for e in stage_events]

    # All four heavy stages fired exactly once — ordering is fixed by the
    # pipeline code, but the assertion is set-based so a refactor that
    # interleaves them doesn't break the test for the wrong reason.
    assert set(stages_seen) == {
        "detection",
        "identification",
        "counterfeit",
        "grade_compose",
    }, f"unexpected stages: {stages_seen}"
    assert len(stage_events) == 4, f"expected 4 stage events, got {len(stage_events)}"

    # Each event carries a non-negative duration_ms and the bound
    # submission_id (proves the bound_contextvars merge_contextvars
    # processor is wired up).
    for e in stage_events:
        assert isinstance(e["duration_ms"], float)
        assert e["duration_ms"] >= 0.0
        assert e.get("submission_id") == str(submission.id)
