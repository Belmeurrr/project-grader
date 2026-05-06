"""End-to-end route tests for /submissions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import (
    AuthenticityResult,
    AuthenticityVerdict,
    CardSet,
    CardVariant,
    Game,
    Submission,
    SubmissionStatus,
)

# Every test drives the FastAPI client + persists Submission rows via
# the `client`/`db_session` fixtures.
pytestmark = pytest.mark.requires_postgres


@pytest.mark.asyncio
async def test_create_submission_requires_auth(client: httpx.AsyncClient) -> None:
    r = await client.post("/submissions", json={})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_create_submission_rejects_bad_auth_scheme(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/submissions",
        headers={"Authorization": "Bearer not-implemented-in-dev"},
        json={},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_create_submission_succeeds_with_dev_token(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    r = await client.post("/submissions", headers=auth_headers, json={})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "pending"
    assert body["grades"] == []
    assert body["authenticity"] is None
    assert uuid.UUID(body["id"])


@pytest.mark.asyncio
async def test_create_submission_with_game_hint(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    r = await client.post(
        "/submissions",
        headers=auth_headers,
        json={"game_hint": "pokemon"},
    )
    assert r.status_code == 201


@pytest.mark.asyncio
async def test_get_submission_roundtrip(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    created = (await client.post("/submissions", headers=auth_headers, json={})).json()
    sid = created["id"]

    r = await client.get(f"/submissions/{sid}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["id"] == sid


@pytest.mark.asyncio
async def test_get_submission_404_for_other_user(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    created = (await client.post("/submissions", headers=auth_headers, json={})).json()
    sid = created["id"]

    other = {"Authorization": "Dev someone-else"}
    r = await client.get(f"/submissions/{sid}", headers=other)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_submission_404_for_unknown_id(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    sid = str(uuid.uuid4())
    r = await client.get(f"/submissions/{sid}", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_submission_rejects_bad_game_hint(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    r = await client.post(
        "/submissions",
        headers=auth_headers,
        json={"game_hint": "not_a_real_game"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_get_submission_surfaces_identified_card(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
    dev_token: str,
) -> None:
    """GET /submissions/{id} eager-loads `identified_variant` + its set
    so the owner-side payload mirrors the public cert page on the
    identified card field. Without the relationship + selectinload,
    asyncpg can't service the lazy-load and the field stays None
    (which was the bug pre-fix)."""
    created = (await client.post("/submissions", headers=auth_headers, json={})).json()
    sid = uuid.UUID(created["id"])

    # Promote the submission to a state that mimics a post-identification
    # row: link it to a freshly-created CardVariant + CardSet. Going
    # direct to the DB rather than through the pipeline keeps the test
    # contract narrow ("response includes identified_card when the FK is
    # set") and leaves the pipeline integration to its own suite.
    card_set = CardSet(game=Game.POKEMON, code="CRZ", name="Crown Zenith")
    db_session.add(card_set)
    await db_session.flush()
    variant = CardVariant(
        game=Game.POKEMON,
        set_id=card_set.id,
        card_number="160",
        name="Pikachu V",
    )
    db_session.add(variant)
    await db_session.flush()

    submission = await db_session.scalar(
        select(Submission).where(Submission.id == sid)
    )
    assert submission is not None
    submission.identified_variant_id = variant.id
    submission.identification_confidence = 0.93
    submission.status = SubmissionStatus.COMPLETED
    submission.completed_at = datetime.now(timezone.utc)
    await db_session.flush()

    r = await client.get(f"/submissions/{sid}", headers=auth_headers)
    assert r.status_code == 200, r.text
    card = r.json()["identified_card"]
    assert card is not None
    assert card["name"] == "Pikachu V"
    assert card["set_code"] == "CRZ"
    assert card["card_number"] == "160"
    assert card["confidence"] == 0.93


@pytest.mark.asyncio
async def test_get_submission_omits_identified_card_when_not_identified(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    """A freshly-created submission has no identification yet — the
    response should carry `identified_card: null`, not 500."""
    created = (await client.post("/submissions", headers=auth_headers, json={})).json()
    sid = created["id"]
    r = await client.get(f"/submissions/{sid}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["identified_card"] is None


@pytest.mark.asyncio
async def test_get_submission_with_nested_authenticity_does_not_500(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    """Regression for the AuthenticityOut.detector_scores schema-drift
    bug: the persisted shape under `AuthenticityResult.detector_scores`
    is `{<detector>: {<metric>: <value>}}` (nested dicts, not flat
    floats). Pydantic v2 was rejecting this against the old
    `dict[str, float]` annotation, which 500'd `GET /submissions/{id}`
    for any submission that had run through the counterfeit pipeline.

    The fix relaxed the annotation to `dict[str, dict[str, Any]]`. This
    test exercises `_to_out` end-to-end with the exact nested shape
    `persist_authenticity_result` writes and asserts the round-trip
    preserves it (no flattening, no validation error)."""
    created = (await client.post("/submissions", headers=auth_headers, json={})).json()
    sid = uuid.UUID(created["id"])

    nested_scores = {
        "rosette": {
            "score": 0.92,
            "peak_strength": 0.41,
            "analyzed_patches": 4,
            "confidence": 0.85,
            "manufacturer_profile": "pokemon_modern",
            "verdict": "authentic",
        },
        "color": {
            "score": 0.88,
            "p95_chroma": 12.5,
            "border_white_bgr": [240.0, 241.0, 242.0],
            "border_stddev": 1.2,
            "gain_applied": [1.0, 1.0, 1.0],
            "confidence": 0.8,
            "manufacturer_profile": "pokemon_modern",
            "verdict": "authentic",
        },
        "embedding_anomaly": {
            "score": 0.0,
            "distance_from_centroid": None,
            "n_references": 0,
            "confidence": 0.0,
            "manufacturer_profile": None,
            "verdict": "unverified",
            "abstain_reason": "no_references",
        },
        "typography": {
            "score": 0.95,
            "confidence": 0.9,
            "extracted_text": "PIKACHU V",
            "expected_text": "PIKACHU V",
            "levenshtein_distance": 0,
            "manufacturer_profile": "pokemon_modern",
            "verdict": "authentic",
            "abstain_reason": None,
        },
        "holographic": {
            "score": 0.0,
            "confidence": 0.0,
            "flow_ratio": None,
            "holo_mask_fraction": None,
            "manufacturer_profile": None,
            "verdict": "unverified",
            "abstain_reason": "no_tilt_shot",
        },
    }

    db_session.add(
        AuthenticityResult(
            submission_id=sid,
            verdict=AuthenticityVerdict.AUTHENTIC,
            confidence=0.8,
            detector_scores=nested_scores,
            reasons=["rosette_authentic", "color_authentic"],
            model_versions={"rosette": "v1", "color": "v1"},
        )
    )
    await db_session.flush()

    r = await client.get(f"/submissions/{sid}", headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["authenticity"] is not None
    out_scores = body["authenticity"]["detector_scores"]
    # Nested shape preserved verbatim — no flattening, no coercion.
    assert isinstance(out_scores["rosette"], dict)
    assert out_scores["rosette"]["score"] == 0.92
    assert out_scores["rosette"]["verdict"] == "authentic"
    assert out_scores["embedding_anomaly"]["abstain_reason"] == "no_references"
    assert set(out_scores.keys()) == set(nested_scores.keys())
