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

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import boto3
import httpx
import pytest
from moto import mock_aws
from sqlalchemy.ext.asyncio import AsyncSession

from grader.services import storage
from grader.settings import get_settings

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

# Every test in this module exercises the FastAPI client against a real
# Postgres test DB via the `client`/`db_session` fixtures.
pytestmark = pytest.mark.requires_postgres


@pytest.fixture
def s3_with_canonicals() -> Iterator[str]:
    """Spin up moto S3 with the configured bucket created.

    Tests that exercise the cert endpoint's `images` block use this to
    seed canonical PNGs at the deterministic key paths the endpoint
    expects (``submissions/<id>/canonical/<kind>.png``)."""
    prior = {
        k: os.environ.get(k)
        for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    }
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    storage.reset_s3_client_cache()
    with mock_aws():
        bucket = get_settings().s3_bucket
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        storage.reset_s3_client_cache()
        yield bucket
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    storage.reset_s3_client_cache()


def _seed_canonical(submission_id: uuid.UUID, kind: str) -> str:
    """Write a tiny PNG payload at ``submissions/<id>/canonical/<kind>.png``
    and return the key. The cert endpoint HEADs these keys to decide
    whether to surface a presigned URL."""
    key = f"submissions/{submission_id}/canonical/{kind}.png"
    boto3.client("s3", region_name="us-east-1").put_object(
        Bucket=get_settings().s3_bucket,
        Key=key,
        Body=b"\x89PNG\r\n\x1a\nfake-canonical",
        ContentType="image/png",
    )
    return key


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


@pytest.mark.asyncio
async def test_cert_endpoint_returns_population_stat(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Population / chronology counter: 5 COMPLETED submissions for the
    same variant with grades 9.5/9.0/8.5/8.0/7.5. The cert for the
    9.0-grade submission must report total_graded=5, this_rank=2,
    max_grade=9.5, chronological_index matches insertion order (=2)."""
    user = User(
        clerk_id=f"u_{uuid.uuid4().hex[:8]}",
        email=f"{uuid.uuid4().hex[:8]}@x",
    )
    db_session.add(user)
    await db_session.flush()

    card_set = CardSet(game=Game.POKEMON, code="POP", name="Pop Test Set")
    db_session.add(card_set)
    await db_session.flush()
    variant = CardVariant(
        game=Game.POKEMON,
        set_id=card_set.id,
        card_number="001",
        name="Population Test Card",
    )
    db_session.add(variant)
    await db_session.flush()

    # Insertion order = chronological order via increasing completed_at.
    # Grades intentionally NOT sorted by completed_at so rank-by-grade
    # and chronological_index decouple cleanly.
    base = datetime.now(timezone.utc)
    grades_in_order = [9.5, 9.0, 8.5, 8.0, 7.5]
    submission_ids: list[uuid.UUID] = []
    for i, final in enumerate(grades_in_order):
        sub = Submission(
            user_id=user.id,
            status=SubmissionStatus.COMPLETED,
            completed_at=base + timedelta(minutes=i),
            identified_variant_id=variant.id,
            identification_confidence=0.9,
        )
        db_session.add(sub)
        await db_session.flush()
        grade = Grade(
            submission_id=sub.id,
            scheme=GradingScheme.PSA,
            centering=final,
            corners=final,
            edges=final,
            surface=final,
            final=final,
            confidence=0.85,
        )
        db_session.add(grade)
        submission_ids.append(sub.id)
    await db_session.flush()

    # The 9.0-grade submission was inserted second (index 1).
    target_id = submission_ids[1]
    r = await client.get(f"/cert/{target_id}")
    assert r.status_code == 200, r.text
    pop = r.json()["population"]
    assert pop is not None
    assert pop["total_graded"] == 5
    assert pop["this_rank"] == 2  # 9.5 is rank 1, 9.0 is rank 2
    assert pop["max_grade"] == 9.5
    assert pop["chronological_index"] == 2  # second insertion


@pytest.mark.asyncio
async def test_cert_endpoint_population_null_when_unidentified(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """When the submission has no identified variant (catalog miss), the
    population block is omitted entirely. We don't want to surface
    "1 of 1 graded" — that's noise."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    assert r.json()["population"] is None


@pytest.mark.asyncio
async def test_cert_endpoint_returns_damage_heatmap_regions(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Damage-heatmap MVP: the cert payload includes a ``regions`` array
    derived from the primary Grade row.

    Phase-1 shape (10 entries):
      - 1 centering (whole_card)
      - 4 edges (top / right / bottom / left) — all sharing the
        aggregate edges score until per-side scores get persisted
      - 4 corners (top_left / top_right / bottom_left / bottom_right)
      - 1 surface (whole_card)

    Severity bucketing: centering=9.5/10 → 0.95 → ``minor`` (the
    >0.95 boundary lands in the minor bucket per
    ``REGION_SEVERITY_OK_THRESHOLD``); corners/edges/surface all None
    → ``unknown``."""
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
        centering=9.5,
        corners=None,
        edges=None,
        surface=None,
        final=None,
        confidence=0.4,
    )
    db_session.add(grade)
    await db_session.flush()

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    body = r.json()

    regions = body["regions"]
    assert isinstance(regions, list)
    assert len(regions) == 10  # 1 centering + 4 edges + 4 corners + 1 surface

    by_kind: dict[str, list[dict]] = {}
    for reg in regions:
        by_kind.setdefault(reg["kind"], []).append(reg)

    # Centering — single whole_card entry; 9.5 / 10 = 0.95 → "minor".
    assert len(by_kind["centering"]) == 1
    centering = by_kind["centering"][0]
    assert centering["position"] == "whole_card"
    assert centering["score"] == pytest.approx(0.95)
    assert centering["severity"] == "minor"

    # Edges — 4 entries with the canonical position names; score None
    # today (the Grade row has edges=None) so severity is unknown.
    edge_positions = {e["position"] for e in by_kind["edge"]}
    assert edge_positions == {"top", "right", "bottom", "left"}
    for e in by_kind["edge"]:
        assert e["score"] is None
        assert e["severity"] == "unknown"

    # Corners — 4 placeholders, all unknown.
    corner_positions = {c["position"] for c in by_kind["corner"]}
    assert corner_positions == {
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }
    for c in by_kind["corner"]:
        assert c["score"] is None
        assert c["severity"] == "unknown"

    # Surface — single whole_card placeholder.
    assert len(by_kind["surface"]) == 1
    surface = by_kind["surface"][0]
    assert surface["position"] == "whole_card"
    assert surface["score"] is None
    assert surface["severity"] == "unknown"


@pytest.mark.asyncio
async def test_cert_endpoint_populates_region_reasons(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """DINGS-style itemized rationale: each region carries a `reasons`
    list keyed off (kind, severity).

    Build a Grade row that exercises every severity bucket simultaneously:
      - centering=5.0 → 0.5 → "major" → ["Significant off-center crop"]
      - edges=8.7    → 0.87 → "minor" → ["Minor edge wear"] (per side)
      - corners=10.0 → 1.0  → "ok"    → [] (no defect to flag)
      - surface=None        → "unknown" → ["Analysis pending"]
        (surface trainer hasn't shipped, so unknown surfaces a placeholder
        rationale instead of nothing)."""
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
        centering=5.0,    # → severity=major
        edges=8.7,        # → severity=minor
        corners=10.0,     # → severity=ok
        surface=None,     # → severity=unknown
        final=None,
        confidence=0.5,
    )
    db_session.add(grade)
    await db_session.flush()

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    regions = r.json()["regions"]

    by_kind: dict[str, list[dict]] = {}
    for reg in regions:
        by_kind.setdefault(reg["kind"], []).append(reg)

    # Every region MUST carry a `reasons` list (default []).
    for reg in regions:
        assert "reasons" in reg
        assert isinstance(reg["reasons"], list)

    # Centering: severity=major → at least one reason.
    centering = by_kind["centering"][0]
    assert centering["severity"] == "major"
    assert len(centering["reasons"]) >= 1
    assert "off-center" in centering["reasons"][0].lower()

    # Edges: severity=minor → "Minor edge wear" on every side entry.
    for edge in by_kind["edge"]:
        assert edge["severity"] == "minor"
        assert edge["reasons"] == ["Minor edge wear"]

    # Corners: severity=ok → empty reasons (nothing to flag).
    for corner in by_kind["corner"]:
        assert corner["severity"] == "ok"
        assert corner["reasons"] == []

    # Surface: severity=unknown for a kind whose trainer hasn't shipped
    # → "Analysis pending" placeholder.
    surface = by_kind["surface"][0]
    assert surface["severity"] == "unknown"
    assert surface["reasons"] == ["Analysis pending"]


@pytest.mark.asyncio
async def test_cert_endpoint_clean_grade_yields_empty_reasons(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """Sanity check: a clean card (every shipped grader at OK) yields
    EMPTY reasons across centering + edges, with the unshipped corner +
    surface kinds still showing the "Analysis pending" placeholder for
    severity=unknown. The cert page renders this as the "No defects
    flagged" empty state for the shipped kinds."""
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
        centering=10.0,
        edges=10.0,
        corners=None,
        surface=None,
        final=None,
        confidence=0.9,
    )
    db_session.add(grade)
    await db_session.flush()

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    regions = r.json()["regions"]
    by_kind: dict[str, list[dict]] = {}
    for reg in regions:
        by_kind.setdefault(reg["kind"], []).append(reg)

    assert by_kind["centering"][0]["reasons"] == []
    for edge in by_kind["edge"]:
        assert edge["reasons"] == []


@pytest.mark.asyncio
async def test_cert_endpoint_images_null_when_no_canonicals(
    client: httpx.AsyncClient, db_session: AsyncSession, s3_with_canonicals: str
) -> None:
    """A COMPLETED submission whose detection stage produced no
    canonical images (or whose canonicals were never written) must
    return ``images: null`` rather than an object full of nulls or a
    stale URL."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["images"] is None


@pytest.mark.asyncio
async def test_cert_endpoint_images_presigns_existing_canonicals(
    client: httpx.AsyncClient, db_session: AsyncSession, s3_with_canonicals: str
) -> None:
    """When the front canonical exists in S3 at the deterministic key
    path, the cert payload surfaces a presigned-GET URL for it. The
    URL must be a string starting with http(s) and reference the s3
    object path so it can be served directly to the public cert page."""
    sub = await _make_completed_submission(db_session)
    front_key = _seed_canonical(sub.id, "front_full")

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    body = r.json()

    images = body["images"]
    assert images is not None
    assert isinstance(images["front_canonical_url"], str)
    assert images["front_canonical_url"].startswith("http")
    # The presigned URL embeds the object path verbatim — the test asserts
    # the s3 key appears somewhere in the URL so we know it's bound to
    # the right object (not just any presigned URL).
    assert front_key in images["front_canonical_url"]
    # Optional shots not seeded → still null.
    assert images["front_flash_url"] is None
    assert images["tilt_url"] is None
    assert "expires_at" in images


@pytest.mark.asyncio
async def test_cert_endpoint_images_includes_flash_when_present(
    client: httpx.AsyncClient, db_session: AsyncSession, s3_with_canonicals: str
) -> None:
    """Both front + flash seeded → both URLs surface. This is the pair
    the Card Vision opacity slider crossfades between."""
    sub = await _make_completed_submission(db_session)
    _seed_canonical(sub.id, "front_full")
    _seed_canonical(sub.id, "front_full_flash")

    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200, r.text
    images = r.json()["images"]
    assert isinstance(images["front_canonical_url"], str)
    assert isinstance(images["front_flash_url"], str)
    assert images["front_flash_url"].startswith("http")


@pytest.mark.asyncio
async def test_cert_endpoint_cache_header_pairs_with_presign_ttl(
    client: httpx.AsyncClient, db_session: AsyncSession, s3_with_canonicals: str
) -> None:
    """The ``max-age`` on the cert response must leave headroom against
    the 1-hour presign TTL so a CDN-cached payload doesn't outlive its
    embedded URLs. Pinning the exact value here so future tweaks are
    deliberate, not accidental."""
    sub = await _make_completed_submission(db_session)
    r = await client.get(f"/cert/{sub.id}")
    assert r.status_code == 200
    cc = r.headers["Cache-Control"]
    assert "max-age=2400" in cc
    # SWR was dropped specifically because it would happily serve a
    # stale cert payload (with a stale presigned URL) past max-age.
    assert "stale-while-revalidate" not in cc
