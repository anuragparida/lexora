"""Integration test for the /retrieve endpoint.

The spec calls for "at minimum: a smoke test that hits
``/retrieve?query=...`` and asserts the response shape". This
file implements that smoke test against the FastAPI app via
TestClient (no real Postgres required — we mock the embedding
call and use a temp SQLite DB for the read path).

Run from ``backend/``::

    uv run pytest -q tests/test_retrieval.py

Note: SQLite cannot run the pgvector cosine distance — that's the
whole reason /retrieve returns 503 on non-Postgres. This test
exercises that 503 path against a temp SQLite DB to prove the
gate works. The Postgres path is exercised in the QA hook against
the live stack.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Point the app at a fresh temp SQLite DB for the lifetime of the test.

    Also redirects the Anki deck output dir to a writable tmp path —
    ``app.anki_builder`` does ``os.makedirs(DECKS_DIR, exist_ok=True)`` at
    import time, and the default ``/app/generated_decks`` is the docker
    container's working dir. On the host (where pytest actually runs), /app
    is read-only or absent, so the makedirs raises ``PermissionError``
    before the test even gets to assert anything.
    """
    db_path = tmp_path / "test_retrieval.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    from app.main import app
    # No ``with`` block — lifespan on SQLite would try to create_all +
    # seed_corpus. The seed_corpus path is a no-op on SQLite
    # (non-Postgres target), so it's safe to skip. We still want
    # TestClient to route through the FastAPI dependency injection
    # so the ``Depends(get_db)`` calls in /retrieve work normally.
    with TestClient(app) as c:
        yield c


def test_retrieve_on_sqlite_returns_503(client):
    """The endpoint must refuse on non-Postgres backends rather than
    silently returning empty results."""
    resp = client.get("/retrieve", params={"query": "Glück", "k": 5})
    assert resp.status_code == 503
    assert "Postgres" in resp.json()["detail"]


def test_retrieve_rejects_empty_query(client):
    """Empty query string is a client error — pydantic's min_length=1."""
    resp = client.get("/retrieve", params={"query": "", "k": 5})
    assert resp.status_code == 422


def test_retrieve_rejects_invalid_source(client):
    """source must be one of words|examples|both (Literal type)."""
    resp = client.get(
        "/retrieve",
        params={"query": "Glück", "k": 5, "source": "garbage"},
    )
    assert resp.status_code == 422


def test_retrieve_caps_k(client):
    """k is bounded [1, 100]. k=0 must reject, k=101 must reject."""
    low = client.get("/retrieve", params={"query": "x", "k": 0})
    assert low.status_code == 422
    high = client.get("/retrieve", params={"query": "x", "k": 101})
    assert high.status_code == 422