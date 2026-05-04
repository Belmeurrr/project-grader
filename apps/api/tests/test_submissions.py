"""End-to-end route tests for /submissions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grader.db.models import (
    CardSet,
    CardVariant,
    Game,
    Submission,
    SubmissionStatus,
)


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
