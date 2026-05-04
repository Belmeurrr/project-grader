"""End-to-end tests for the public GET /cert/{submission_id} endpoint.

The cert page is the artifact a grader user shares publicly, so the
endpoint MUST:
  - require no auth (unlike GET /submissions/{id} which is owner-only)
  - return 404 for in-progress / failed / non-existent submissions
    (don't leak partial state)
  - return 200 with a sanitized payload — no user_id, no S3 keys, no
    audit log, no shot metadata
  - send a Cache-Control header so Next.js ISR + CDN cache reliably
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import (
    AuthenticityResult,
    AuthenticityVerdict,
    CardSet,
    CardVariant,
    Game,
    Grade,
    GradingScheme,
    Submission,
    SubmissionStatus,
    User,
)


async def _make_completed_submission(
    db: AsyncSession,
    *,
    with_authenticity: bool = True,
    with_identification: bool = False,
) -> Submission:
    """Insert a User + Submission + Grade + (optionally) AuthenticityResult
    in the COMPLETED state, with the per-detector dict shape produced
    by the rosette+color ensemble.

    When `with_identification` is True, also insert a CardSet + CardVariant
    and link the submission to the variant via `identified_variant_id` +
    `identification_confidence`. The cert endpoint should then surface
    the variant's name, set code, and card number on the public payload."""
    user = User(
        clerk_id=f"u_{uuid.uuid4().hex[:8]}",
        email=f"{uuid.uuid4().hex[:8]}@x",
    )
    db.add(user)
    await db.flush()

    identified_variant_id: uuid.UUID | None = None
    identification_confidence: float | None = None
    if with_identification:
        card_set = CardSet(
            game=Game.POKEMON,
            code="CRZ",
            name="Crown Zenith",
        )
        db.add(card_set)
        await db.flush()
        variant = CardVariant(
            game=Game.POKEMON,
            set_id=card_set.id,
            card_number="160",
            name="Pikachu V",
        )
        db.add(variant)
        await db.flush()
        identified_variant_id = variant.id
        identification_confidence = 0.93

    submission = Submission(
        user_id=user.id,
        status=SubmissionStatus.COMPLETED,
        completed_at=datetime.now(timezone.utc),
        identified_variant_id=identified_variant_id,
        identification_confidence=identification_confidence,
    )
    db.add(submission)
    await db.flush()

    grade = Grade(
        submission_id=submission.id,
        scheme=GradingScheme.PSA,
        centering=9.0,
        corners=8.5,
        edges=9.5,
        surface=9.0,
        final=9.0,
        confidence=0.85,
    )
    db.add(grade)

    if with_authenticity:
        # Production now writes a 3-detector ensemble: rosette + color +
        # embedding-anomaly. The realistic scenario has embedding-anomaly
        # ABSTAINING (confidence=0, n_references=0) for variants we don't
        # yet have reference exemplars for — it gracefully degrades and
        # the ensemble combines from the two confident detectors. The
        # cert page must render the abstaining detector cleanly without
        # blowing up the layout.
        auth = AuthenticityResult(
            submission_id=submission.id,
            verdict=AuthenticityVerdict.AUTHENTIC,
            confidence=0.92,
            reasons=[
                "halftone rosette pattern detected (rosette_score=0.94, peak_strength=8.10)",
                "high chroma consistent with offset print (color_score=1.00, p95_chroma=72.4)",
                "embedding-anomaly abstained (no_references)",
            ],
            detector_scores={
                "rosette": {
                    "score": 0.94,
                    "verdict": "authentic",
                    "confidence": 1.0,
                    "peak_strength": 8.10,
                    "analyzed_patches": 5,
                },
                "color": {
                    "score": 1.0,
                    "verdict": "authentic",
                    "confidence": 1.0,
                    "p95_chroma": 72.4,
                    "border_stddev": 1.2,
                    "border_white_bgr": [228, 228, 230],
                    "gain_applied": [1.12, 1.12, 1.11],
                },
                "embedding_anomaly": {
                    "score": 0.5,
                    "verdict": "unverified",
                    "confidence": 0.0,
                    "distance_from_centroid": 0.0,
                    "n_references": 0,
                    "manufacturer_profile": "generic",
                    "abstain_reason": "no_references",
                },
            },
            model_versions={
                "rosette": "fft-v1",
                "color": "cielab-chroma-v1",
                "embedding_anomaly": "centroid-cosine-v1",
            },
        )
        db.add(auth)

    await db.flush()
    return submission


@pytest.mark.asyncio
async def test_cert_endpoint_404_for_unknown_id(client: httpx.AsyncClient) -> None:
    """Unknown submission_id → 404 (no auth required to probe)."""
    r = await client.get(f"/cert/{uuid.uuid4()}")
    assert r.status_code == 404
    assert "cert not found" in r.json()["detail"]


@pytest.mark.asyncio
async def test_cert_endpoint_404_for_in_progress_submission(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Submission exists but isn't COMPLETED → opaque 404. Don't surface
    partial / changing state on a public URL."""
    user = User(clerk_id=f"u_{uuid.uuid4().hex[:8]}", email=f"{uuid.uuid4().hex[:8]}@x")
    db_session.add(user)
    await db_session.flush()
    sub = Submission(user_id=user.id, status=SubmissionStatus.PROCESSING)
    db_session.add(sub)
    await db_session.flush()

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_cert_endpoint_no_auth_required(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The cert page is public — no Authorization header should still
    return 200 for a COMPLETED submission."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    # client fixture sends no auth headers
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_cert_endpoint_returns_sanitized_payload(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Headline assertion: the response shape exposes ONLY public-safe
    fields. No user_id, no S3 keys, no audit log."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    body = r.json()

    # Required public fields
    assert body["cert_id"] == str(sub.id)
    assert body["completed_at"] is not None
    assert "grades" in body
    assert "authenticity" in body

    # Forbidden internal fields
    forbidden = {"user_id", "audit_log", "shots", "s3_key", "rejection_reason"}
    assert forbidden.isdisjoint(body.keys()), (
        f"public cert leaked private fields: {forbidden & set(body.keys())}"
    )

    # Grades populated correctly
    grades = body["grades"]
    assert len(grades) == 1
    assert grades[0]["scheme"] == "psa"
    assert grades[0]["final"] == 9.0


@pytest.mark.asyncio
async def test_cert_endpoint_authenticity_per_detector_breakdown(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The per-detector list must surface all three production detectors
    (rosette, color, embedding_anomaly) with each detector's score +
    verdict + forensic metadata. The embedding detector is in its
    realistic abstaining state (no reference exemplars yet for most
    variants); the cert page must still render it cleanly."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    auth = r.json()["authenticity"]
    assert auth is not None
    assert auth["verdict"] == "authentic"

    detectors = {d["detector"]: d for d in auth["detectors"]}
    assert set(detectors) == {"rosette", "color", "embedding_anomaly"}

    assert detectors["rosette"]["score"] == 0.94
    assert detectors["rosette"]["verdict"] == "authentic"
    assert detectors["rosette"]["metadata"]["peak_strength"] == 8.10

    assert detectors["color"]["score"] == 1.0
    assert detectors["color"]["metadata"]["p95_chroma"] == 72.4

    # Abstaining detector — confidence 0, verdict unverified, abstain
    # reason surfaced under metadata so the cert page can show "no
    # reference data available" rather than a misleading 50% score.
    assert detectors["embedding_anomaly"]["verdict"] == "unverified"
    assert detectors["embedding_anomaly"]["confidence"] == 0.0
    assert (
        detectors["embedding_anomaly"]["metadata"]["abstain_reason"]
        == "no_references"
    )

    assert auth["model_versions"]["rosette"] == "fft-v1"
    assert auth["model_versions"]["color"] == "cielab-chroma-v1"
    assert auth["model_versions"]["embedding_anomaly"] == "centroid-cosine-v1"


@pytest.mark.asyncio
async def test_cert_endpoint_handles_no_authenticity(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """A COMPLETED submission can legitimately have no AuthenticityResult
    (counterfeit pipeline soft-failed). The cert page should still load."""
    sub = await _make_completed_submission(db_session, with_authenticity=False)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    assert r.json()["authenticity"] is None


@pytest.mark.asyncio
async def test_cert_endpoint_omits_identified_card_when_not_identified(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """A COMPLETED submission whose identification soft-failed has no
    identified_variant_id on the row. The cert payload should surface
    `identified_card: null` rather than 404 — graded but anonymous is
    a real production state for cards we don't yet have in catalog."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    assert r.json()["identified_card"] is None


@pytest.mark.asyncio
async def test_cert_endpoint_surfaces_identified_card(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """When the identification stage matched a catalog variant, the cert
    payload exposes the variant's name, set code, card number, and the
    identification confidence — sourced from the eager-loaded
    `Submission.identified_variant` + `CardVariant.set` chain."""
    sub = await _make_completed_submission(db_session, with_identification=True)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    card = r.json()["identified_card"]
    assert card is not None
    assert card["name"] == "Pikachu V"
    assert card["set_code"] == "CRZ"
    assert card["card_number"] == "160"
    assert card["confidence"] == 0.93
    assert uuid.UUID(card["variant_id"])  # round-trippable


@pytest.mark.asyncio
async def test_cert_endpoint_renders_partial_grade(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Regression: a Phase-1 Grade row with `final=None` and `corners=None`
    (corners/surface trainers are skeletons — `compute_psa_final` returns
    None whenever any input is missing) MUST NOT cause `GradeOut.model_validate`
    to raise. Before the schema relaxation, this combo 500'd every cert
    page for every real submission until the trainers shipped."""
    user = User(
        clerk_id=f"u_{uuid.uuid4().hex[:8]}",
        email=f"{uuid.uuid4().hex[:8]}@x",
    )
    db_session.add(user)
    await db_session.flush()
    sub = Submission(
        user_id=user.id,
        status=SubmissionStatus.COMPLETED,
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(sub)
    await db_session.flush()
    grade = Grade(
        submission_id=sub.id,
        scheme=GradingScheme.PSA,
        centering=9.0,
        corners=None,
        edges=None,
        surface=None,
        final=None,
        confidence=0.42,
    )
    db_session.add(grade)
    await db_session.flush()

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    body = r.json()
    grades = body["grades"]
    assert len(grades) == 1
    g = grades[0]
    assert g["centering"] == 9.0
    assert g["corners"] is None
    assert g["edges"] is None
    assert g["surface"] is None
    assert g["final"] is None
    assert g["confidence"] == 0.42


@pytest.mark.asyncio
async def test_cert_endpoint_sets_cache_control(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """The response must carry a Cache-Control header so Next.js ISR +
    CDNs cache reliably. Once a submission completes, the payload is
    immutable in practice."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    cc = r.headers.get("Cache-Control", "")
    assert "public" in cc
    assert "max-age" in cc
