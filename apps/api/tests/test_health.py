import httpx
import pytest

from grader import __version__


@pytest.mark.asyncio
async def test_healthz(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


@pytest.mark.asyncio
async def test_readyz(client: httpx.AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
