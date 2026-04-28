"""End-to-end route tests for /submissions."""

from __future__ import annotations

import uuid

import httpx
import pytest


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
