"""End-to-end tests for grader.services.identification.

Walks the full path: a canonical PNG is in S3 (moto), the catalog has the
matching variant, the service runs identification, persists the chosen
variant + confidence on the submission, and writes an audit_log row."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import AuditLog, Submission, SubmissionStatus, User
from grader.services import identification, storage
from grader.settings import get_settings
from pipelines.identification import (
    CardCatalogEntry,
    InMemoryCatalogIndex,
    SimpleEmbedder,
    compute_phash,
)
from tests.fixtures import card_in_scene, synth_card

# All tests use `db_session` to persist the Submission/User/AuditLog rows
# the service writes — needs Postgres.
pytestmark = pytest.mark.requires_postgres


@pytest.fixture(autouse=True)
def _aws_creds_for_moto() -> Iterator[None]:
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


def _put_canonical(bucket: str, submission_id: uuid.UUID, image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    key = f"submissions/{submission_id}/canonical/front_full.png"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/png"
    )
    return key


async def _make_submission(db: AsyncSession) -> Submission:
    user = User(clerk_id=f"u_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@x")
    db.add(user)
    await db.flush()
    sub = Submission(user_id=user.id, status=SubmissionStatus.PROCESSING)
    db.add(sub)
    await db.flush()
    return sub


def _catalog_with_variant(image: np.ndarray, variant_id: str) -> InMemoryCatalogIndex:
    e = SimpleEmbedder()
    catalog = InMemoryCatalogIndex()
    catalog.add(
        CardCatalogEntry(
            variant_id=variant_id,
            name="Black Lotus",
            set_code="LEA",
            card_number="195",
            game="mtg",
            canonical_phash=compute_phash(image),
            canonical_embedding=e.encode(image),
        )
    )
    return catalog


@pytest.mark.asyncio
async def test_identify_persists_chosen_variant_and_confidence(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    canonical = card_in_scene(fill=0.55)
    key = _put_canonical(s3_bucket, sub.id, canonical)

    variant_id = str(uuid.uuid4())
    catalog = _catalog_with_variant(canonical, variant_id)

    outcome = await identification.identify_canonical_for_submission(
        submission_id=sub.id,
        canonical_s3_key=key,
        catalog=catalog,
        embedder=SimpleEmbedder(),
        db=db_session,
    )

    assert outcome.result.identified
    assert outcome.result.chosen.entry.variant_id == variant_id
    await db_session.commit()
    refreshed = await db_session.get(Submission, sub.id)
    assert str(refreshed.identified_variant_id) == variant_id
    assert refreshed.identification_confidence is not None
    assert refreshed.identification_confidence >= 0.95


@pytest.mark.asyncio
async def test_identify_writes_audit_log(s3_bucket: str, db_session: AsyncSession) -> None:
    sub = await _make_submission(db_session)
    canonical = card_in_scene(fill=0.55)
    key = _put_canonical(s3_bucket, sub.id, canonical)
    catalog = _catalog_with_variant(canonical, str(uuid.uuid4()))

    await identification.identify_canonical_for_submission(
        submission_id=sub.id,
        canonical_s3_key=key,
        catalog=catalog,
        embedder=SimpleEmbedder(),
        db=db_session,
    )
    await db_session.commit()

    logs = (
        await db_session.execute(select(AuditLog).where(AuditLog.submission_id == sub.id))
    ).scalars().all()
    assert any(log.action == "identification.completed" for log in logs)
    log = next(log for log in logs if log.action == "identification.completed")
    assert log.payload["identified"] is True
    assert isinstance(log.payload["candidates"], list)


@pytest.mark.asyncio
async def test_identify_no_match_clears_chosen_and_persists_low_confidence(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    query = card_in_scene(fill=0.55)
    key = _put_canonical(s3_bucket, sub.id, query)

    # Catalog has only an unrelated card.
    catalog = _catalog_with_variant(synth_card(image_color=(20, 20, 240)), str(uuid.uuid4()))

    outcome = await identification.identify_canonical_for_submission(
        submission_id=sub.id,
        canonical_s3_key=key,
        catalog=catalog,
        embedder=SimpleEmbedder(),
        db=db_session,
    )
    assert not outcome.result.identified
    await db_session.commit()
    refreshed = await db_session.get(Submission, sub.id)
    assert refreshed.identified_variant_id is None


@pytest.mark.asyncio
async def test_identify_raises_on_missing_canonical(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    with pytest.raises(identification.IdentificationFailedError):
        await identification.identify_canonical_for_submission(
            submission_id=sub.id,
            canonical_s3_key=f"submissions/{sub.id}/canonical/front_full.png",
            catalog=InMemoryCatalogIndex(),
            embedder=SimpleEmbedder(),
            db=db_session,
        )


@pytest.mark.asyncio
async def test_identify_raises_on_corrupt_canonical(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    key = f"submissions/{sub.id}/canonical/front_full.png"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key=key, Body=b"not a png"
    )
    with pytest.raises(identification.IdentificationFailedError, match="decode"):
        await identification.identify_canonical_for_submission(
            submission_id=sub.id,
            canonical_s3_key=key,
            catalog=InMemoryCatalogIndex(),
            embedder=SimpleEmbedder(),
            db=db_session,
        )
