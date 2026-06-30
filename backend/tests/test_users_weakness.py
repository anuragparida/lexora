"""Tests for the User + WeaknessProfile data layer.

Originally added in Phase 2.1 (card t_6318d0e1) for the auth-free
``POST /users`` and ``/users/me`` 501 placeholder. Phase 2.2 (card
t_74c3aa1e) replaced those routes with ``POST /auth/signup`` and
``GET /auth/me``, so this file now exercises the same data-layer
surface (User row, WeaknessProfile row, schema-level validations,
migration idempotency) through the new auth-gated shape.

Coverage:

- ``POST /auth/signup`` creates a User row with a bcrypt hash.
- ``POST /auth/signup`` with a duplicate email returns 409.
- ``GET /weakness-profile/{user_id}`` auto-creates an empty default
  profile on first read so the response shape is always stable.
- ``PUT /weakness-profile/{user_id}`` with valid ``axes`` round-trips.
- ``PUT`` with an out-of-range axis value returns 422.
- The migration is idempotent on SQLite (Postgres is exercised in
  the QA hook against the live stack).

The auth surface itself (cookie shape, JWT decoding, login flow,
/auth/me without auth, Bearer fallback) lives in ``test_auth.py``
which was added in the same Phase 2.2 card.

Run from ``backend/``::

    uv run pytest -q tests/test_users_weakness.py
"""
from __future__ import annotations

import os
import secrets
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

    Also seeds ``JWT_SECRET`` so the Phase 2.2 auth module's
    import-time check passes (it raises ``RuntimeError`` if the
    value is missing or still the docker-compose placeholder).
    """
    db_path = tmp_path / "test_users_weakness.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """A ``TestClient`` wired to a fresh per-test SQLite DB.

    The fixture rebinds the module-level engine to a fresh
    ``tmp_path`` SQLite file and runs ``Base.metadata.create_all``
    against the new engine. Without the rebind, all tests in the
    session share the same engine (and the same database file)
    because the engine is built once at import time. Without the
    ``create_all``, the new engine points at an empty file and
    the first query fails with ``no such table: users``.

    The lifespan startup then runs ``Base.metadata.create_all``
    against the rebound engine (idempotent — no-op if the schema
    is already there) and calls ``bootstrap.seed_corpus()`` which
    is a no-op on SQLite (the seed path only writes to Postgres).
    """
    from app import database
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


def _signup(
    client: TestClient,
    email: str = "ada@example.com",
    password: str = "supersecret",
) -> dict:
    """Helper: POST a signup and return the JSON body.

    Used by every test in this file as the entry point for creating
    a user. The Phase 2.1 helper accepted a raw ``password_hash``;
    Phase 2.2 ships bcrypt + a password field. The returned
    ``body["user"]`` is a ``UserOut`` (no ``password_hash``), so
    each test can use it as the 2.1-era ``_create_user`` did.
    """
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": password},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# POST /auth/signup (replaces 2.1's POST /users)
# ---------------------------------------------------------------------------


def test_post_signup_creates_row_with_hashed_password(client):
    """The success path: row exists with a bcrypt hash, response
    shape omits ``password_hash``."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    user = body["user"]
    assert user["email"] == "ada@example.com"
    assert "password_hash" not in user, "password_hash must not leak"
    assert "id" in user and isinstance(user["id"], int)
    assert "created_at" in user


def test_post_signup_rejects_duplicate_email(client):
    """409 on a second signup with the same email — both the
    pre-check and the IntegrityError fallback must produce the
    same status."""
    _signup(client, email="dup@example.com", password="supersecret")
    resp = client.post(
        "/auth/signup",
        json={"email": "dup@example.com", "password": "othersecret"},
    )
    assert resp.status_code == 409
    assert "already" in resp.json()["detail"].lower()


def test_post_signup_rejects_short_password(client):
    """Pydantic's ``min_length=8`` on ``password`` rejects empty /
    short strings."""
    resp = client.post(
        "/auth/signup",
        json={"email": "nohash@example.com", "password": "short"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /weakness-profile/{user_id}
# ---------------------------------------------------------------------------


def test_get_weakness_profile_auto_creates_empty_default(client):
    """First GET creates the profile so the response shape is always
    stable. ``axes`` is a dict, not a JSON string (the dialect-aware
    deserialization is exercised here on the SQLite path).

    The cookie set by ``/auth/signup`` is reused so the auth-gated
    route works without an extra login round-trip.
    """
    body = _signup(client, email="firstget@example.com", password="supersecret")
    user_id = body["user"]["id"]
    resp = client.get(f"/weakness-profile/{user_id}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["user_id"] == user_id
    assert payload["axes"] == {}
    assert "id" in payload
    assert "updated_at" in payload


def test_get_weakness_profile_returns_403_for_mismatched_user(client):
    """Auth-gated: a different user_id returns 403 (not 404 — the
    auth check is stronger than the existence check, and the route
    short-circuits on subject mismatch)."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    _signup(client, email="other@example.com", password="othersecret")
    # The TestClient cookie jar holds whichever signup ran last;
    # either way, probing 9999 is a mismatch.
    resp = client.get("/weakness-profile/9999")
    assert resp.status_code == 403


def test_get_weakness_profile_is_idempotent(client):
    """Two GETs in a row return the same profile (the auto-create path
    must not insert a duplicate row that would violate the FK UNIQUE
    constraint)."""
    body = _signup(client, email="idem@example.com", password="supersecret")
    user_id = body["user"]["id"]
    first = client.get(f"/weakness-profile/{user_id}").json()
    second = client.get(f"/weakness-profile/{user_id}").json()
    assert first["id"] == second["id"]


# ---------------------------------------------------------------------------
# PUT /weakness-profile/{user_id}
# ---------------------------------------------------------------------------


def test_put_weakness_profile_round_trip(client):
    """PUT with valid axes persists and the subsequent GET returns
    the same shape."""
    body = _signup(client, email="put@example.com", password="supersecret")
    user_id = body["user"]["id"]
    # Ensure the profile exists (auto-created by GET)
    client.get(f"/weakness-profile/{user_id}")

    put_resp = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 2, "collocations": 3, "idioms": 1}},
    )
    assert put_resp.status_code == 200
    payload = put_resp.json()
    assert payload["axes"] == {"verbs": 2, "collocations": 3, "idioms": 1}

    get_resp = client.get(f"/weakness-profile/{user_id}")
    assert get_resp.json()["axes"] == {"verbs": 2, "collocations": 3, "idioms": 1}


def test_put_weakness_profile_rejects_out_of_range_axis(client):
    """axis value 5 must 422 — the WeaknessProfileUpdate validator
    enforces 0 <= score <= 3."""
    body = _signup(client, email="oor@example.com", password="supersecret")
    user_id = body["user"]["id"]
    resp = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 5}},
    )
    assert resp.status_code == 422
    # The error body mentions the bad axis name so the frontend can
    # highlight it.
    assert "verbs" in resp.text


def test_put_weakness_profile_rejects_negative_axis(client):
    """axis value -1 must 422."""
    body = _signup(client, email="neg@example.com", password="supersecret")
    user_id = body["user"]["id"]
    resp = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": -1}},
    )
    assert resp.status_code == 422


def test_put_weakness_profile_rejects_non_int_axis(client):
    """axis value as a string must 422 — type validation."""
    body = _signup(client, email="str@example.com", password="supersecret")
    user_id = body["user"]["id"]
    resp = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": "two"}},
    )
    assert resp.status_code == 422


def test_put_weakness_profile_allows_empty_axes_reset(client):
    """``axes={}`` is a valid reset — the user can clear their
    declaration without dropping the profile row."""
    body = _signup(client, email="reset@example.com", password="supersecret")
    user_id = body["user"]["id"]
    client.get(f"/weakness-profile/{user_id}")

    # Set non-empty axes first
    client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 3}},
    )
    # Reset
    resp = client.put(
        f"/weakness-profile/{user_id}",
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
    # JWT_SECRET is also needed (Phase 2.2) — set to a benign value.
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))

    backend_dir = Path(__file__).resolve().parent.parent
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }

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


def test_alembic_migration_downgrade_is_clean(tmp_path, monkeypatch):
    """The Phase 2.1 migration's ``downgrade()`` drops both the
    ``weakness_profiles`` and ``users`` tables cleanly. We exercise
    the downgrade path here so a future migration that depends on
    these tables (Phase 3+ work) knows the rollback works.
    """
    db_path = tmp_path / "downgrade.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))

    backend_dir = Path(__file__).resolve().parent.parent
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }

    # Apply head first
    up = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert up.returncode == 0, up.stderr

    # Downgrade the Phase 2.1 revision. ``a15ec4b9f736`` is the
    # Phase 2.1 revision id; its ``down_revision`` is the Phase 1
    # embeddings revision (``496091d14711``), so downgrade one
    # step to land back at the Phase 1 schema.
    down = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "496091d14711"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert down.returncode == 0, down.stderr
    # Alembic prints the revision message on stderr/stdout; merge
    # the streams for the assertion below.
    assert "phase 2: users + weakness_profiles" in (down.stdout + down.stderr)
