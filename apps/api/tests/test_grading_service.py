"""Tests for grader.services.grading — centering grading + Grade persistence."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import Grade, GradingScheme, Submission, SubmissionStatus, User
from grader.services import grading, storage
from grader.settings import get_settings

_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from tests.fixtures import (  # noqa: E402
    canonical_clean,
    canonical_with_edge_defect,
    synth_card,
)


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


def _put_canonical(bucket: str, key: str, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/png"
    )


# -----------------------------
# grade_centering
# -----------------------------


def test_grade_centering_front_only(s3_bucket: str) -> None:
    key = "test/canonical_front.png"
    _put_canonical(s3_bucket, key, synth_card(40, 40, 40, 40))

    result = grading.grade_centering(front_canonical_s3_key=key)
    assert result.back_measurement is None
    assert abs(result.front_measurement.ratios.left - 50.0) < 1.0
    assert result.psa_subgrade == 10.0
    assert 0.0 < result.confidence <= 1.0


def test_grade_centering_with_back(s3_bucket: str) -> None:
    front_key = "test/front.png"
    back_key = "test/back.png"
    _put_canonical(s3_bucket, front_key, synth_card(40, 40, 40, 40))
    _put_canonical(s3_bucket, back_key, synth_card(40, 40, 40, 40))

    result = grading.grade_centering(
        front_canonical_s3_key=front_key, back_canonical_s3_key=back_key
    )
    assert result.back_measurement is not None
    assert result.psa_subgrade == 10.0


def test_grade_centering_held_back_by_back(s3_bucket: str) -> None:
    """Back centering 92/8 should drop the PSA subgrade below 10."""
    front_key = "test/front.png"
    back_key = "test/back.png"
    _put_canonical(s3_bucket, front_key, synth_card(40, 40, 40, 40))
    # Heavily off-center back: borders 70 vs 6 → ratio about 92/8
    _put_canonical(s3_bucket, back_key, synth_card(70, 6, 40, 40))

    result = grading.grade_centering(front_key, back_key)
    assert result.psa_subgrade < 10.0


def test_grade_centering_raises_on_missing_canonical(s3_bucket: str) -> None:
    with pytest.raises(grading.GradingFailedError, match="decode"):
        grading.grade_centering(front_canonical_s3_key="test/missing.png")


def test_grade_centering_raises_on_corrupt_canonical(s3_bucket: str) -> None:
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=s3_bucket, Key="test/bad.png", Body=b"not a png"
    )
    with pytest.raises(grading.GradingFailedError, match="decode"):
        grading.grade_centering(front_canonical_s3_key="test/bad.png")


# -----------------------------
# compute_psa_final
# -----------------------------


def test_psa_final_returns_none_when_any_missing() -> None:
    assert grading.compute_psa_final(centering=10.0, corners=None, edges=10.0, surface=10.0) is None
    assert grading.compute_psa_final(centering=10.0, corners=10.0, edges=None, surface=10.0) is None
    assert grading.compute_psa_final(centering=10.0, corners=10.0, edges=10.0, surface=None) is None


def test_psa_final_is_minimum() -> None:
    assert grading.compute_psa_final(10.0, 9.0, 8.0, 9.5) == 8.0
    assert grading.compute_psa_final(10.0, 10.0, 10.0, 10.0) == 10.0


def test_psa_final_rounds_to_half_step() -> None:
    assert grading.compute_psa_final(10.0, 8.3, 9.0, 9.0) == 8.5  # 8.3 → 8.5
    assert grading.compute_psa_final(10.0, 8.7, 9.0, 9.0) == 8.5
    assert grading.compute_psa_final(10.0, 8.8, 9.0, 9.0) == 9.0  # 8.8 → 9.0


# -----------------------------
# persist_centering_grade
# -----------------------------


async def _make_submission(db: AsyncSession) -> Submission:
    user = User(clerk_id=f"u_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@x")
    db.add(user)
    await db.flush()
    sub = Submission(user_id=user.id, status=SubmissionStatus.PROCESSING)
    db.add(sub)
    await db.flush()
    return sub


@pytest.mark.asyncio
async def test_persist_centering_inserts_new_grade_row(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    key = "test/front.png"
    _put_canonical(s3_bucket, key, synth_card(40, 40, 40, 40))
    result = grading.grade_centering(front_canonical_s3_key=key)

    grade = await grading.persist_centering_grade(
        submission_id=sub.id, result=result, db=db_session
    )
    assert grade.scheme == GradingScheme.PSA
    assert grade.centering == 10.0
    assert grade.corners is None
    assert grade.edges is None
    assert grade.surface is None
    assert grade.final is None  # partial run — no final until all criteria
    assert grade.front_centering_lr is not None
    assert grade.back_centering_lr is None
    assert grade.model_versions["centering"] == "geometric-v1"


@pytest.mark.asyncio
async def test_persist_centering_updates_existing_grade(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """A second pipeline run for the same scheme must update, not duplicate
    (unique index on submission_id + scheme)."""
    sub = await _make_submission(db_session)
    key = "test/front.png"
    _put_canonical(s3_bucket, key, synth_card(40, 40, 40, 40))
    result1 = grading.grade_centering(front_canonical_s3_key=key)
    g1 = await grading.persist_centering_grade(submission_id=sub.id, result=result1, db=db_session)

    # Re-grade with a worse centering; grade row should update in place.
    _put_canonical(s3_bucket, key, synth_card(60, 20, 40, 40))  # ~75/25 horiz
    result2 = grading.grade_centering(front_canonical_s3_key=key)
    g2 = await grading.persist_centering_grade(submission_id=sub.id, result=result2, db=db_session)

    assert g1.id == g2.id  # same row
    assert g2.centering < g1.centering


@pytest.mark.asyncio
async def test_persist_centering_preserves_other_criteria_when_updating(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """If a future run already populated corners/edges, re-running centering
    must NOT clobber them."""
    sub = await _make_submission(db_session)
    db_session.add(
        Grade(
            submission_id=sub.id,
            scheme=GradingScheme.PSA,
            centering=10.0,
            corners=9.0,
            edges=9.5,
            surface=9.0,
            final=9.0,
            confidence=0.8,
        )
    )
    await db_session.flush()

    key = "test/front.png"
    _put_canonical(s3_bucket, key, synth_card(40, 40, 40, 40))
    result = grading.grade_centering(front_canonical_s3_key=key)
    grade = await grading.persist_centering_grade(submission_id=sub.id, result=result, db=db_session)

    assert grade.corners == 9.0  # untouched
    assert grade.edges == 9.5
    assert grade.surface == 9.0
    # final recomputed using existing corners/edges/surface + new centering
    assert grade.final == 9.0


# -----------------------------
# grade_edges
# -----------------------------


def test_grade_edges_clean_card(s3_bucket: str) -> None:
    key = "test/front_clean.png"
    _put_canonical(s3_bucket, key, canonical_clean())
    result = grading.grade_edges(front_canonical_s3_key=key)
    assert result.back_measurement is None
    assert result.psa_subgrade >= 9.0
    assert result.worse_face == "front"


def test_grade_edges_with_back_takes_worse_face(s3_bucket: str) -> None:
    front_key = "test/front.png"
    back_key = "test/back.png"
    _put_canonical(s3_bucket, front_key, canonical_clean())
    _put_canonical(
        s3_bucket,
        back_key,
        canonical_with_edge_defect(side="top", length_px=300, severity="chip"),
    )
    result = grading.grade_edges(
        front_canonical_s3_key=front_key, back_canonical_s3_key=back_key
    )
    assert result.worse_face == "back"
    assert result.psa_subgrade < result.front_measurement.psa_subgrade


def test_grade_edges_raises_on_missing_canonical(s3_bucket: str) -> None:
    with pytest.raises(grading.GradingFailedError, match="decode"):
        grading.grade_edges(front_canonical_s3_key="test/missing.png")


# -----------------------------
# persist_edges_grade
# -----------------------------


@pytest.mark.asyncio
async def test_persist_edges_requires_existing_centering_row(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    key = "test/front.png"
    _put_canonical(s3_bucket, key, canonical_clean())
    edges_result = grading.grade_edges(front_canonical_s3_key=key)

    with pytest.raises(grading.GradingFailedError, match="before centering"):
        await grading.persist_edges_grade(
            submission_id=sub.id, result=edges_result, db=db_session
        )


@pytest.mark.asyncio
async def test_persist_edges_updates_existing_grade(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    sub = await _make_submission(db_session)
    front = "test/front.png"
    _put_canonical(s3_bucket, front, canonical_clean())

    centering = grading.grade_centering(front_canonical_s3_key=front)
    grade = await grading.persist_centering_grade(
        submission_id=sub.id, result=centering, db=db_session
    )
    assert grade.edges is None  # before edges
    centering_conf = grade.confidence

    edges_result = grading.grade_edges(front_canonical_s3_key=front)
    grade2 = await grading.persist_edges_grade(
        submission_id=sub.id, result=edges_result, db=db_session
    )

    assert grade2.id == grade.id  # same row
    assert grade2.centering == grade.centering  # untouched
    assert grade2.edges is not None
    assert grade2.edges == edges_result.psa_subgrade
    assert grade2.confidence <= centering_conf  # min over criteria
    assert grade2.model_versions["edges"] == "geometric-v1"
    assert grade2.model_versions["edges_worse_face"] == edges_result.worse_face
    # final still None — corners/surface still missing
    assert grade2.final is None


@pytest.mark.asyncio
async def test_persist_edges_recomputes_final_when_complete(
    s3_bucket: str, db_session: AsyncSession
) -> None:
    """If corners + surface were already populated, persisting edges should
    trigger a final-grade computation."""
    sub = await _make_submission(db_session)
    db_session.add(
        Grade(
            submission_id=sub.id,
            scheme=GradingScheme.PSA,
            centering=10.0,
            corners=9.0,
            surface=9.5,
            confidence=0.9,
            front_centering_lr=50.0,
            front_centering_tb=50.0,
        )
    )
    await db_session.flush()

    front = "test/front.png"
    _put_canonical(s3_bucket, front, canonical_clean())
    edges_result = grading.grade_edges(front_canonical_s3_key=front)
    grade = await grading.persist_edges_grade(
        submission_id=sub.id, result=edges_result, db=db_session
    )
    assert grade.final is not None
    assert grade.final == min(10.0, 9.0, edges_result.psa_subgrade, 9.5)
