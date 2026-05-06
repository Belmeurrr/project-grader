"""Tests for POST /submissions/{id}/submit.

The endpoint validates pre-flight conditions and enqueues the Celery task.
We don't run a real broker in tests — we patch `process_submission.delay`
to return a fake AsyncResult so the endpoint returns a task_id without
actually dispatching."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from unittest.mock import patch

import boto3
import httpx
import pytest
from moto import mock_aws

from grader.db.models import ShotKind, SubmissionStatus
from grader.services import storage
from grader.settings import get_settings
from tests.fixtures import card_in_scene, encode_jpeg


@pytest.fixture(autouse=True)
def _aws_creds() -> Iterator[None]:
    prior = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    storage.reset_s3_client_cache()
    yield
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    storage.reset_s3_client_cache()


@pytest.fixture
def s3_bucket() -> Iterator[str]:
    with mock_aws():
        bucket = get_settings().s3_bucket
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        storage.reset_s3_client_cache()
        yield bucket


@pytest.fixture(autouse=True)
def _stub_celery_dispatch():
    """Replace `process_submission.delay` with a stub that returns a fake
    AsyncResult so we don't need a running broker."""
    fake_id = "task-" + uuid.uuid4().hex[:12]

    class _FakeResult:
        id = fake_id

    with patch(
        "grader.workers.grading_pipeline.process_submission.delay",
        return_value=_FakeResult(),
    ) as m:
        yield m


async def _create_submission(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    r = await client.post("/submissions", headers=headers, json={})
    assert r.status_code == 201
    return r.json()["id"]


async def _upload_and_register_shot(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    submission_id: str,
    s3_bucket: str,
    kind: str = "front_full",
    image=None,
) -> dict:
    if image is None:
        image = card_in_scene(fill=0.55)
    r = await client.post(
        f"/submissions/{submission_id}/shots/upload-url",
        headers=headers,
        json={"kind": kind, "content_type": "image/jpeg"},
    )
    assert r.status_code == 201, r.text
    upload = r.json()
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=upload["s3_key"], Body=encode_jpeg(image), ContentType="image/jpeg"
    )
    r2 = await client.post(
        f"/submissions/{submission_id}/shots",
        headers=headers,
        json={"shot_id": upload["shot_id"], "s3_key": upload["s3_key"]},
    )
    assert r2.status_code == 201, r2.text
    return r2.json()


# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_requires_auth(
    client: httpx.AsyncClient, s3_bucket: str
) -> None:
    sid = uuid.uuid4()
    r = await client.post(f"/submissions/{sid}/submit")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_submit_404_for_other_users_submission(
    client: httpx.AsyncClient, auth_headers: dict[str, str], s3_bucket: str
) -> None:
    sid = await _create_submission(client, auth_headers)
    other = {"Authorization": "Dev someone-else"}
    r = await client.post(f"/submissions/{sid}/submit", headers=other)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_submit_409_when_required_shots_missing(
    client: httpx.AsyncClient, auth_headers: dict[str, str], s3_bucket: str
) -> None:
    sid = await _create_submission(client, auth_headers)
    r = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    assert r.status_code == 409
    body = r.json()["detail"]
    assert body["reason"] == "missing_required_shots"
    assert "front_full" in body["missing"]


@pytest.mark.asyncio
async def test_submit_409_when_required_shot_quality_failed(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    """If the front shot was uploaded but failed quality, submit must reject."""
    import numpy as np
    sid = await _create_submission(client, auth_headers)
    blank = np.full((1500, 2000, 3), 35, dtype=np.uint8)
    shot = await _upload_and_register_shot(
        client, auth_headers, sid, s3_bucket, image=blank
    )
    assert shot["quality_passed"] is False  # sanity check the precondition

    r = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_submit_202_with_task_id_when_ready(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
    _stub_celery_dispatch,
) -> None:
    sid = await _create_submission(client, auth_headers)
    await _upload_and_register_shot(client, auth_headers, sid, s3_bucket)

    r = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "processing"
    assert body["task_id"] is not None
    _stub_celery_dispatch.assert_called_once_with(sid)


@pytest.mark.asyncio
async def test_submit_idempotent_on_already_processing(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
    _stub_celery_dispatch,
) -> None:
    """Re-submitting an already-processing submission returns its current
    state without re-enqueuing."""
    sid = await _create_submission(client, auth_headers)
    await _upload_and_register_shot(client, auth_headers, sid, s3_bucket)

    first = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    assert first.status_code == 202
    second = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    assert second.status_code == 202
    assert second.json()["status"] == "processing"
    # delay() called only on the first submit, not the idempotent re-submit.
    assert _stub_celery_dispatch.call_count == 1


@pytest.mark.asyncio
async def test_submit_advances_submission_to_processing(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
) -> None:
    sid = await _create_submission(client, auth_headers)
    await _upload_and_register_shot(client, auth_headers, sid, s3_bucket)

    pre = (await client.get(f"/submissions/{sid}", headers=auth_headers)).json()
    assert pre["status"] == "capturing"

    await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    post = (await client.get(f"/submissions/{sid}", headers=auth_headers)).json()
    assert post["status"] == "processing"


@pytest.mark.asyncio
async def test_submit_503_when_broker_unavailable_and_retry_succeeds(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
    db_session,
) -> None:
    """When Celery's broker is unreachable at enqueue time, the endpoint
    must:
      (a) return 503 with a structured detail body so the client retries,
      (b) revert the submission row from PROCESSING -> CAPTURING so the
          early-return guard doesn't strand it forever,
      (c) allow a subsequent retry to succeed normally once the broker
          recovers.
    """
    from grader.db.models import Submission

    sid = await _create_submission(client, auth_headers)
    await _upload_and_register_shot(client, auth_headers, sid, s3_bucket)

    # First attempt: broker is down. delay() raises.
    with patch(
        "grader.workers.grading_pipeline.process_submission.delay",
        side_effect=ConnectionError("broker unreachable"),
    ):
        r = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)

    assert r.status_code == 503
    body = r.json()["detail"]
    assert body["reason"] == "broker_unavailable"
    assert body["retry_after"] == 30

    # Row must have reverted to CAPTURING (not stranded in PROCESSING).
    await db_session.expire_all()
    sub = await db_session.get(Submission, uuid.UUID(sid))
    assert sub.status == SubmissionStatus.CAPTURING

    # Second attempt: broker is back (default autouse stub returns fake result).
    fake_id = "task-recovered-001"

    class _FakeResult:
        id = fake_id

    with patch(
        "grader.workers.grading_pipeline.process_submission.delay",
        return_value=_FakeResult(),
    ) as m:
        r2 = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)

    assert r2.status_code == 202
    body2 = r2.json()
    assert body2["status"] == "processing"
    assert body2["task_id"] == fake_id
    m.assert_called_once_with(sid)


@pytest.mark.asyncio
async def test_submit_rate_limited_at_5_per_minute_per_user(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
    db_session,
    _stub_celery_dispatch,
) -> None:
    """slowapi guards POST /submissions/{id}/submit at 5/min per user.

    The 6th request from the same Clerk user inside a minute must
    return 429. We bypass the idempotency early-return (which would
    short-circuit subsequent calls to 202 without going through the
    limiter on a re-entry path the real attacker doesn't have) by
    forcing the row back to CAPTURING between attempts — that's the
    hot loop a malicious client trying to flood the broker would use.

    Reset the slowapi limiter storage at the start so prior tests'
    counters don't leak in (the limiter is a module-level singleton
    and per-test isolation is not built into slowapi)."""
    from grader.db.models import Submission
    from grader.services.rate_limit import limiter

    limiter.reset()

    sid = await _create_submission(client, auth_headers)
    await _upload_and_register_shot(client, auth_headers, sid, s3_bucket)

    statuses: list[int] = []
    for _ in range(6):
        # Force submission back to CAPTURING so /submit doesn't take
        # the idempotent fast-path on calls 2+.
        sub = await db_session.get(Submission, uuid.UUID(sid))
        sub.status = SubmissionStatus.CAPTURING
        await db_session.commit()

        r = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
        statuses.append(r.status_code)

    # First 5 succeed (202), 6th hits the limiter (429).
    assert statuses[:5] == [202, 202, 202, 202, 202], statuses
    assert statuses[5] == 429, statuses


@pytest.mark.asyncio
async def test_submit_returns_existing_state_for_completed_submission(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    s3_bucket: str,
    db_session,
    _stub_celery_dispatch,
) -> None:
    """Re-submitting a COMPLETED submission returns its terminal state."""
    from grader.db.models import Submission

    sid = await _create_submission(client, auth_headers)
    await _upload_and_register_shot(client, auth_headers, sid, s3_bucket)

    # Force the submission into COMPLETED state to simulate a finished run.
    sub = await db_session.get(Submission, uuid.UUID(sid))
    sub.status = SubmissionStatus.COMPLETED
    await db_session.commit()

    r = await client.post(f"/submissions/{sid}/submit", headers=auth_headers)
    assert r.status_code == 202
    assert r.json()["status"] == "completed"
    assert r.json()["task_id"] is None
    _stub_celery_dispatch.assert_not_called()
