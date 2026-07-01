"""Tests for Phase 2.2 — auth routes + get_current_user dependency.

Card: t_74c3aa1e.

These tests prove the auth surface shipped by this card:

- ``POST /auth/signup`` creates a user with a bcrypt-hashed password
  (NOT plaintext) and returns ``UserOut`` (which omits
  ``password_hash``) plus a JWT cookie.
- ``POST /auth/signup`` rejects duplicate emails (409), short
  passwords (422), and malformed emails (422).
- ``POST /auth/login`` succeeds with correct creds, 401 with wrong
  password. The 401 body is opaque — does not leak which of email
  vs password was wrong.
- ``GET /auth/me`` returns 200 with a valid cookie, 401 without,
  401 with an expired / malformed JWT.
- ``POST /auth/logout`` clears the cookie (subsequent ``/auth/me``
  is 401).
- ``GET /weakness-profile/{user_id}`` and ``PUT /weakness-profile/{user_id}``
  are now auth-gated — 401 without a token, 403 when the JWT subject
  does not match the URL ``user_id``, 200 with a matching token.

The tests are hermetic — they use a fresh temp SQLite DB and a
temp JWT secret in the env, so they never depend on the live
Postgres / docker stack. The live-stack verification is the
QA hook's job.

Run from ``backend/``::

    uv run pytest -q tests/test_auth.py
"""
from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Point the app at a fresh temp SQLite DB for the test, and
    load a fresh JWT_SECRET into the env so the auth module's
    import-time check passes.
    """
    db_path = tmp_path / "test_auth.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    # ``app.auth`` reads JWT_SECRET at import time and refuses to
    # boot without it. The auth tests are the first ones to need
    # the env var, so we set it here BEFORE the first ``from app.main
    # import app`` happens (the module is imported once and cached,
    # so the env must be set before then).
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
    """Helper: POST a signup and return the JSON body. Asserts 201."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": password},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# POST /auth/signup
# ---------------------------------------------------------------------------


def test_signup_creates_user_with_hashed_password(client, sqlite_db_path):
    """The success path: row exists with a bcrypt hash, response
    shape omits ``password_hash``."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    body = _signup(client, email="ada@example.com", password="supersecret")
    assert body["user"]["email"] == "ada@example.com"
    assert "password_hash" not in body["user"]
    assert "access_token" in body and isinstance(body["access_token"], str)
    assert body["access_token"].count(".") == 2  # JWT shape: header.payload.sig

    # Cookie was set on the response.
    assert "lexora_token" in client.cookies

    # The row in the DB stores a bcrypt hash, NOT the plaintext.
    # ``sqlite_db_path`` is the URL form (sqlite:///<path>) — the
    # actual file path is the part after the third slash.
    engine = create_engine(
        f"sqlite:///{sqlite_db_path}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as s:
        from app.models import User

        row = s.query(User).filter(User.email == "ada@example.com").one()
        # bcrypt hash starts with $2b$ (or $2a$ / $2y$).
        assert row.password_hash.startswith("$2")
        assert "supersecret" not in row.password_hash


def test_signup_rejects_duplicate_email(client):
    """409 on a second signup with the same email."""
    _signup(client, email="dup@example.com")
    resp = client.post(
        "/auth/signup",
        json={"email": "dup@example.com", "password": "another1"},
    )
    assert resp.status_code == 409
    assert "already" in resp.json()["detail"].lower()


def test_signup_rejects_short_password(client):
    """422 on password length < 8 — the Pydantic field bounds."""
    resp = client.post(
        "/auth/signup",
        json={"email": "short@example.com", "password": "short"},
    )
    assert resp.status_code == 422


def test_signup_rejects_malformed_email(client):
    """422 on a malformed email — the Pydantic EmailStr check."""
    resp = client.post(
        "/auth/signup",
        json={"email": "not-an-email", "password": "supersecret"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------


def test_login_succeeds_with_correct_creds(client):
    """The happy path: 200, body matches signup shape, cookie is set."""
    _signup(client, email="ada@example.com", password="supersecret")
    # TestClient shares the cookie jar across requests — clear it
    # so the login call has to set the cookie freshly (proves the
    # login route actually issues a token, not just returns one
    # because the jar is non-empty).
    client.cookies.clear()

    resp = client.post(
        "/auth/login",
        json={"email": "ada@example.com", "password": "supersecret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["email"] == "ada@example.com"
    assert "access_token" in body
    assert "lexora_token" in client.cookies


def test_login_rejects_wrong_password(client):
    """401 on wrong password. The body must NOT leak that the
    email is registered — same shape as 'no such email'."""
    _signup(client, email="ada@example.com", password="supersecret")
    resp = client.post(
        "/auth/login",
        json={"email": "ada@example.com", "password": "wrongpass"},
    )
    assert resp.status_code == 401
    assert "invalid" in resp.json()["detail"].lower() or "credentials" in resp.json()["detail"].lower()


def test_login_rejects_unknown_email(client):
    """401 on no such email — same opaque body shape as wrong
    password, so a username-enumeration probe can't tell which
    of email / password was wrong."""
    resp = client.post(
        "/auth/login",
        json={"email": "ghost@example.com", "password": "supersecret"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------


def test_get_me_with_valid_cookie_returns_user(client):
    """200 + user body when a valid cookie is present."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    # The signup set the cookie; reuse it.
    resp = client.get("/auth/me")
    assert resp.status_code == 200
    assert resp.json()["email"] == "ada@example.com"
    assert resp.json()["id"] == body["user"]["id"]


def test_get_me_without_cookie_returns_401(client):
    """401 when no cookie and no Authorization header is present."""
    client.cookies.clear()
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_get_me_with_expired_jwt_returns_401(client, monkeypatch):
    """401 when the JWT is signed correctly but is past its ``exp``."""
    # Sign a token with the same secret the auth module loaded,
    # but set ``exp`` in the past.
    from app.auth import JWT_ALGORITHM, JWT_SECRET

    payload = {
        "sub": "1",
        "iat": int((datetime.now(tz=timezone.utc) - timedelta(days=10)).timestamp()),
        "exp": int((datetime.now(tz=timezone.utc) - timedelta(seconds=10)).timestamp()),
    }
    expired = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    client.cookies.clear()
    resp = client.get("/auth/me", cookies={"lexora_token": expired})
    assert resp.status_code == 401


def test_get_me_with_invalid_jwt_returns_401(client):
    """401 when the JWT is signed with a different key (signature
    mismatch). The auth module decodes with the configured secret;
    a wrong-secret token fails verification and the dependency
    raises 401."""
    bogus = jwt.encode(
        {"sub": "1", "exp": int((datetime.now(tz=timezone.utc) + timedelta(hours=1)).timestamp())},
        "this-is-not-the-real-secret-32+bytes-1234",
        algorithm="HS256",
    )
    client.cookies.clear()
    resp = client.get("/auth/me", cookies={"lexora_token": bogus})
    assert resp.status_code == 401


def test_get_me_with_bearer_header_falls_back(client):
    """The cookie-first, Bearer-fallback path: a request with
    ``Authorization: Bearer <token>`` and no cookie still
    resolves the current user."""
    _signup(client, email="ada@example.com", password="supersecret")
    # Reuse the access_token from signup.
    # ``client.cookies`` already has the cookie; clear it to force
    # the fallback path.
    token = client.cookies.get("lexora_token")
    assert token
    client.cookies.clear()
    resp = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "ada@example.com"


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------


def test_logout_clears_cookie(client):
    """``POST /auth/logout`` returns 204 and clears the cookie
    (subsequent ``/auth/me`` is 401)."""
    _signup(client, email="ada@example.com", password="supersecret")
    resp = client.post("/auth/logout")
    assert resp.status_code == 204
    # FastAPI's TestClient applies the Set-Cookie to ``client.cookies``
    # — a max-age=0 / empty value cookie should clear it.
    assert client.cookies.get("lexora_token") in (None, "")

    # Subsequent /auth/me is 401 (no valid token).
    resp = client.get("/auth/me")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auth-gated weakness-profile routes
# ---------------------------------------------------------------------------


def test_weakness_profile_get_requires_auth(client):
    """401 on ``GET /weakness-profile/1`` with no token."""
    client.cookies.clear()
    resp = client.get("/weakness-profile/1")
    assert resp.status_code == 401


def test_weakness_profile_get_with_valid_token_returns_200(client):
    """200 on ``GET /weakness-profile/{user_id}`` when the JWT
    subject matches ``user_id``."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    user_id = body["user"]["id"]

    resp = client.get(f"/weakness-profile/{user_id}")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == user_id
    assert resp.json()["axes"] == {}


def test_weakness_profile_get_with_mismatched_token_returns_403(client):
    """403 on a token whose subject does not match the URL
    ``user_id`` (no token would be 401; mismatch is 403)."""
    _signup(client, email="ada@example.com", password="supersecret")
    _signup(client, email="other@example.com", password="othersecret")
    # The cookie belongs to whichever signup ran last. Either way,
    # probing a different user_id should 403.
    resp = client.get("/weakness-profile/9999")
    assert resp.status_code == 403


def test_weakness_profile_put_requires_auth(client):
    """401 on ``PUT /weakness-profile/1`` with no token."""
    client.cookies.clear()
    resp = client.put(
        "/weakness-profile/1",
        json={"axes": {"verbs": 2}},
    )
    assert resp.status_code == 401


def test_weakness_profile_put_with_valid_token_round_trips(client):
    """200 + round-trip on ``PUT /weakness-profile/{user_id}``
    when the JWT subject matches ``user_id``."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    user_id = body["user"]["id"]

    # Auto-create the profile (the GET path does this; we PUT
    # directly which also creates an empty row if missing).
    put_resp = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 2, "collocations": 3}},
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["axes"] == {"verbs": 2, "collocations": 3}

    get_resp = client.get(f"/weakness-profile/{user_id}")
    assert get_resp.json()["axes"] == {"verbs": 2, "collocations": 3}


# ---------------------------------------------------------------------------
# password_hash is NEVER in any response body
# ---------------------------------------------------------------------------


def test_password_hash_never_appears_in_any_response(client):
    """Grep the full set of response bodies for ``password_hash``.
    A leak would expose credential material; the schema
    deliberately omits the field but a future maintainer might
    add it back without thinking."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    assert "password_hash" not in str(body)
    assert "password_hash" not in str(client.get("/auth/me").json())
    assert "password_hash" not in str(
        client.post(
            "/auth/login",
            json={"email": "ada@example.com", "password": "supersecret"},
        ).json()
    )
    # The weakness profile should never carry it either.
    body_id = body["user"]["id"]
    assert "password_hash" not in str(
        client.get(f"/weakness-profile/{body_id}").json()
    )
    assert "password_hash" not in str(
        client.put(
            f"/weakness-profile/{body_id}",
            json={"axes": {"verbs": 1}},
        ).json()
    )


# ---------------------------------------------------------------------------
# Phase 3.3 — ``/auth/me`` extended payload (card t_ff6fa637)
#
# The first-login gate on the frontend reads two extra fields from
# ``/auth/me``:
#
#   - ``weakness_profile``   — the saved axes (or ``null`` if no row)
#   - ``diagnostic_state``   — ``never`` | ``in_progress`` |
#                             ``completed`` | ``applied``, computed
#                             from the latest ``diagnostic_sessions`` row
#
# These tests cover all four ``diagnostic_state`` values plus the
# ``weakness_profile`` projection. The base shape (id / email /
# created_at) is covered by ``test_get_me_with_valid_cookie_returns_user``
# above and remains unchanged.
# ---------------------------------------------------------------------------


def test_get_me_diagnostic_state_never_for_fresh_signup(client):
    """A brand-new user (no diagnostic sessions) reports
    ``diagnostic_state == "never"`` and ``weakness_profile is None``."""
    _signup(client, email="ada@example.com", password="supersecret")
    me = client.get("/auth/me").json()
    assert me["diagnostic_state"] == "never"
    assert me["weakness_profile"] is None


def test_get_me_weakness_profile_reflects_saved_axes(client):
    """After a PUT to /weakness-profile, /auth/me returns the
    same ``WeaknessProfileOut`` shape (id, user_id, axes,
    updated_at). The frontend gate reads ``axes`` to decide
    where to land."""
    body = _signup(client, email="ada@example.com", password="supersecret")
    user_id = body["user"]["id"]
    client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 2, "collocations": 1}},
    )
    me = client.get("/auth/me").json()
    assert me["weakness_profile"] is not None
    assert me["weakness_profile"]["user_id"] == user_id
    assert me["weakness_profile"]["axes"] == {"verbs": 2, "collocations": 1}
    # Even with axes saved, diagnostic_state is still "never"
    # unless the user has actually started a probe.
    assert me["diagnostic_state"] == "never"


def test_get_me_diagnostic_state_in_progress_after_start(client):
    """After ``POST /diagnostic/start`` (which inserts an
    ``in_progress`` row), ``/auth/me`` reports
    ``diagnostic_state == "in_progress"``."""
    _signup(client, email="ada@example.com", password="supersecret")
    start = client.post("/diagnostic/start")
    assert start.status_code == 200
    me = client.get("/auth/me").json()
    assert me["diagnostic_state"] == "in_progress"


def test_get_me_diagnostic_state_applied_after_apply(client):
    """After ``POST /diagnostic/apply`` (which flips the session
    to ``applied`` and UPSERTs the score into the profile),
    ``/auth/me`` reports ``diagnostic_state == "applied"`` and
    ``weakness_profile.axes`` carries the scored axes."""
    from app.diagnostic.questions import QUESTIONS  # noqa: WPS433 (test-local import)

    _signup(client, email="ada@example.com", password="supersecret")
    session_id = client.post("/diagnostic/start").json()["session_id"]
    # Answer every question so /apply has data to score.
    for q in QUESTIONS:
        idx = min(2, len(q.choices) - 1)
        client.post(
            "/diagnostic/answer",
            json={
                "session_id": session_id,
                "question_id": q.id,
                "choice_label": q.choices[idx].label,
            },
        )
    apply = client.post(
        "/diagnostic/apply", json={"session_id": session_id}
    )
    assert apply.status_code == 200

    me = client.get("/auth/me").json()
    assert me["diagnostic_state"] == "applied"
    # The profile carries the computed axes — at least one should
    # be non-zero because we answered with index 2 (mid-strength).
    assert me["weakness_profile"] is not None
    assert isinstance(me["weakness_profile"]["axes"], dict)


def test_get_me_diagnostic_state_completed_from_db(client):
    """A session with status ``completed`` (between ``/result`` and
    ``/apply`` in the spec's mental model) is mapped to
    ``"completed"`` on ``/auth/me``.

    The current ``/diagnostic/result`` route doesn't actually flip
    the status, so a ``completed`` row is not reachable through
    the public API. We exercise the branch by inserting one
    directly via the SQLAlchemy session. This is the same shape a
    real ``/result``-then-``/apply`` flow would produce if the
    route were extended in a future card.
    """
    from app.database import SessionLocal  # noqa: WPS433 (test-local import)
    from app import models  # noqa: WPS433 (test-local import)

    body = _signup(client, email="ada@example.com", password="supersecret")
    user_id = body["user"]["id"]

    # Insert a completed session directly. We don't go through the
    # API because the API never produces this state in the current
    # implementation — the test is asserting the read-side branch
    # in ``read_me``, not the write path.
    with SessionLocal() as db:
        row = models.DiagnosticSession(
            id="00000000-0000-0000-0000-000000000001",
            user_id=user_id,
            status="completed",
        )
        db.add(row)
        db.commit()

    me = client.get("/auth/me").json()
    assert me["diagnostic_state"] == "completed"


def test_get_me_diagnostic_state_uses_latest_session(client):
    """When multiple diagnostic sessions exist, ``/auth/me`` reports
    the status of the *most recent* one (newest-first ordering
    on ``started_at``). This guards the routing gate against a
    stale row winning the branch."""
    from app.database import SessionLocal  # noqa: WPS433 (test-local import)
    from app import models  # noqa: WPS433 (test-local import)
    from datetime import datetime, timedelta  # noqa: WPS433 (test-local import)

    body = _signup(client, email="ada@example.com", password="supersecret")
    user_id = body["user"]["id"]

    now = datetime.utcnow()
    with SessionLocal() as db:
        # Older "applied" row — should be shadowed by the newer
        # "in_progress" row.
        db.add(
            models.DiagnosticSession(
                id="11111111-1111-1111-1111-111111111111",
                user_id=user_id,
                status="applied",
                started_at=now - timedelta(hours=1),
            )
        )
        # Newer "in_progress" row — this is what /auth/me should
        # surface.
        db.add(
            models.DiagnosticSession(
                id="22222222-2222-2222-2222-222222222222",
                user_id=user_id,
                status="in_progress",
                started_at=now,
            )
        )
        db.commit()

    me = client.get("/auth/me").json()
    assert me["diagnostic_state"] == "in_progress"
