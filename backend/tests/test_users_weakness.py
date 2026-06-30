"""Tests for Phase 2.1 — User + WeaknessProfile routes (auth-free).

These tests prove the data layer shipped by card t_6318d0e1:

- ``POST /users`` creates a row with both ``email`` and ``password_hash``
  set, and returns ``UserOut`` (which omits ``password_hash``).
- ``POST /users`` with a duplicate email returns 409.
- ``GET /weakness-profile/{user_id}`` auto-creates an empty default
  profile on first read so the response shape is always stable.
- ``PUT /weakness-profile/{user_id}`` with valid ``axes`` round-trips.
- ``PUT`` with an out-of-range axis value returns 422.
- The migration is idempotent on both Postgres and SQLite.

Runs against a fresh temp SQLite DB (the same fixture pattern as
``test_retrieval.py``). The Postgres migration idempotency is
exercised separately via the live stack in the QA hook, but the
SQLite path here proves the table shape is portable.

Run from ``backend/``::

    uv run pytest -q tests/test_users_weakness.py

The tests are deliberately hermetic — they never hit the live
network, never use real password hashes, and never depend on a
running Postgres. A Phase 2.2 follow-up can add JWT-issued tests
that DO hit the live stack.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Point the app at a fresh temp SQLite DB for the lifetime of the test.

    Same shape as the Phase 1 ``test_retrieval.py`` fixture — redirects
    ``DATABASE_URL`` and the Anki decks dir so ``app.anki_builder``
    doesn't try to ``mkdir /app/generated_decks`` at import time on the
    read-only host filesystem.
    """
    db_path = tmp_path / "test_users_weakness.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """A ``TestClient`` wired to the fresh temp SQLite DB.

    The lifespan startup runs ``Base.metadata.create_all`` (Phase 1
    behavior) which creates the new ``users`` and ``weakness_profiles``
    tables alongside the Phase 1 baseline. The lifespan also calls
    ``bootstrap.seed_corpus()`` which is a no-op on SQLite (the seed
    path only writes to Postgres), so the fixture is safe.
    """
    from app.main import app

    with TestClient(app) as c:
        yield c


def _create_user(client: TestClient, email: str = "ada@example.com",
                 password_hash: str = "pre-hash-not-real") -> dict:
    """Helper: POST a user and return the JSON body."""
    resp = client.post(
        "/users",
        json={"email": email, "password_hash": password_hash},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# POST /users
# ---------------------------------------------------------------------------


def test_post_users_creates_row_with_email_and_hash(client):
    """The success path: row exists with both fields set, and the
    response shape omits ``password_hash``."""
    body = _create_user(client, email="ada@example.com",
                        password_hash="hash-v1-not-real")
    assert body["email"] == "ada@example.com"
    assert "password_hash" not in body, "password_hash must not leak"
    assert "id" in body and isinstance(body["id"], int)
    assert "created_at" in body


def test_post_users_rejects_duplicate_email(client):
    """409 on a second POST with the same email — both the pre-check
    and the IntegrityError fallback must produce the same status."""
    _create_user(client, email="dup@example.com")
    resp = client.post(
        "/users",
        json={"email": "dup@example.com", "password_hash": "h2"},
    )
    assert resp.status_code == 409
    assert "already" in resp.json()["detail"].lower()


def test_post_users_rejects_empty_password_hash(client):
    """Pydantic's ``min_length=1`` on ``password_hash`` rejects empty strings."""
    resp = client.post(
        "/users",
        json={"email": "nohash@example.com", "password_hash": ""},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /users/me — placeholder until Phase 2.2
# ---------------------------------------------------------------------------


def test_get_users_me_returns_501_placeholder(client):
    """The placeholder is intentional: Phase 2.2 replaces it. The
    body must explicitly name the next card so a curl probe surfaces
    the reason rather than a generic 501."""
    resp = client.get("/users/me")
    assert resp.status_code == 501
    body = resp.json()
    assert "Phase 2.2" in body["detail"]
    assert body["card"] == "t_74c3aa1e"


# ---------------------------------------------------------------------------
# GET /weakness-profile/{user_id}
# ---------------------------------------------------------------------------


def test_get_weakness_profile_auto_creates_empty_default(client):
    """First GET creates the profile so the response shape is always
    stable. ``axes`` is a dict, not a JSON string (the dialect-aware
    deserialization is exercised here on the SQLite path)."""
    user = _create_user(client, email="firstget@example.com")
    resp = client.get(f"/weakness-profile/{user['id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == user["id"]
    assert body["axes"] == {}
    assert "id" in body
    assert "updated_at" in body


def test_get_weakness_profile_returns_404_for_unknown_user(client):
    """A non-existent user_id returns 404 — the route guards on user
    existence before touching the profile table."""
    resp = client.get("/weakness-profile/99999")
    assert resp.status_code == 404
    assert "user not found" in resp.json()["detail"].lower()


def test_get_weakness_profile_is_idempotent(client):
    """Two GETs in a row return the same profile (the auto-create path
    must not insert a duplicate row that would violate the FK UNIQUE
    constraint)."""
    user = _create_user(client, email="idem@example.com")
    first = client.get(f"/weakness-profile/{user['id']}").json()
    second = client.get(f"/weakness-profile/{user['id']}").json()
    assert first["id"] == second["id"]


# ---------------------------------------------------------------------------
# PUT /weakness-profile/{user_id}
# ---------------------------------------------------------------------------


def test_put_weakness_profile_round_trip(client):
    """PUT with valid axes persists and the subsequent GET returns
    the same shape."""
    user = _create_user(client, email="put@example.com")
    # Ensure the profile exists (auto-created by GET)
    client.get(f"/weakness-profile/{user['id']}")

    put_resp = client.put(
        f"/weakness-profile/{user['id']}",
        json={"axes": {"verbs": 2, "collocations": 3, "idioms": 1}},
    )
    assert put_resp.status_code == 200
    body = put_resp.json()
    assert body["axes"] == {"verbs": 2, "collocations": 3, "idioms": 1}

    get_resp = client.get(f"/weakness-profile/{user['id']}")
    assert get_resp.json()["axes"] == {"verbs": 2, "collocations": 3, "idioms": 1}


def test_put_weakness_profile_rejects_out_of_range_axis(client):
    """axis value 5 must 422 — the WeaknessProfileUpdate validator
    enforces 0 <= score <= 3."""
    user = _create_user(client, email="oor@example.com")
    resp = client.put(
        f"/weakness-profile/{user['id']}",
        json={"axes": {"verbs": 5}},
    )
    assert resp.status_code == 422
    # The error body mentions the bad axis name so the frontend can
    # highlight it.
    assert "verbs" in resp.text


def test_put_weakness_profile_rejects_negative_axis(client):
    """axis value -1 must 422."""
    user = _create_user(client, email="neg@example.com")
    resp = client.put(
        f"/weakness-profile/{user['id']}",
        json={"axes": {"verbs": -1}},
    )
    assert resp.status_code == 422


def test_put_weakness_profile_rejects_non_int_axis(client):
    """axis value as a string must 422 — type validation."""
    user = _create_user(client, email="str@example.com")
    resp = client.put(
        f"/weakness-profile/{user['id']}",
        json={"axes": {"verbs": "two"}},
    )
    assert resp.status_code == 422


def test_put_weakness_profile_rejects_unknown_user(client):
    """404 for a non-existent user_id."""
    resp = client.put(
        "/weakness-profile/99999",
        json={"axes": {"verbs": 1}},
    )
    assert resp.status_code == 404


def test_put_weakness_profile_allows_empty_axes_reset(client):
    """``axes={}`` is a valid reset — the user can clear their
    declaration without dropping the profile row."""
    user = _create_user(client, email="reset@example.com")
    client.get(f"/weakness-profile/{user['id']}")

    # Set non-empty axes first
    client.put(
        f"/weakness-profile/{user['id']}",
        json={"axes": {"verbs": 3}},
    )
    # Reset
    resp = client.put(
        f"/weakness-profile/{user['id']}",
        json={"axes": {}},
    )
    assert resp.status_code == 200
    assert resp.json()["axes"] == {}


# ---------------------------------------------------------------------------
# Migration idempotency (SQLite path; Postgres is exercised in QA hook)
# ---------------------------------------------------------------------------


def test_alembic_migration_is_idempotent_on_sqlite(tmp_path, monkeypatch):
    """Running the migration twice on a fresh SQLite DB is a no-op
    the second time (exit code 0, no errors). The ``inspect()``
    guard inside the migration's ``upgrade()`` is what makes this
    work — re-running against an already-migrated DB short-circuits
    both ``create_table`` calls.
    """
    db_path = tmp_path / "idempotent.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Anki decks dir is also needed for the app package to import.
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))

    backend_dir = Path(__file__).resolve().parent.parent
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}",
           "LEXORA_DECKS_DIR": str(tmp_path / "decks")}

    # First apply — runs baseline + Phase 1 embeddings + Phase 2.1
    result1 = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result1.returncode == 0, result1.stderr
    # Alembic logs INFO lines to stderr (the logger config in
    # alembic.ini uses a StreamHandler on sys.stderr); merge the two
    # streams so the assertion below finds the revision id.
    assert "a15ec4b9f736" in (result1.stdout + result1.stderr)

    # Second apply — must be a clean no-op
    result2 = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result2.returncode == 0, result2.stderr
    # On idempotent re-run alembic emits no "Running upgrade" lines
    # because the head revision is already current.
    combined = result2.stdout + result2.stderr
    assert "Running upgrade" not in combined


def test_alembic_migration_downgrade_clean_on_sqlite(tmp_path, monkeypatch):
    """The downgrade path drops ``weakness_profiles`` then ``users``
    and is reversible back to the Phase 1 head."""
    db_path = tmp_path / "downgrade.db"
    backend_dir = Path(__file__).resolve().parent.parent
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}",
           "LEXORA_DECKS_DIR": str(tmp_path / "decks")}

    # Upgrade to head
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=60, check=True,
    )
    # Downgrade to Phase 1 head
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "496091d14711"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # Alembic logs the downgrade INFO lines to stderr.
    assert "phase 2: users + weakness_profiles" in (
        result.stdout + result.stderr
    ).lower()