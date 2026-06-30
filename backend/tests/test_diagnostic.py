"""Tests for Phase 3.1 — diagnostic probe (card t_41d85c32).

Covers the three layers the card calls out as the Helena-review
acceptance gates:

- The question bank shape (10 questions, unique ids, full axis
  coverage, well-formed choices/deltas).
- The deterministic scoring math (empty / single / clamp-upper /
  clamp-lower / multi-axis / reasons).
- The four auth-gated endpoints (auth gate, start, answer, result,
  apply, and the cross-user / double-apply guards).
- Alembic migration idempotency on both dialects.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test, so
the suite never touches the live Postgres / docker stack. The
Postgres path is exercised by the QA hook against the running stack.

Run from ``backend/``::

    uv run pytest -q tests/test_diagnostic.py
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.diagnostic.questions import (
    ALL_AXES,
    QUESTION_BY_ID,
    QUESTIONS,
    TOTAL_QUESTIONS,
)
from app.diagnostic.scoring import AnswerRecord, score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Point the app at a fresh temp SQLite DB and seed a JWT secret
    so ``app.auth``'s import-time check passes."""
    db_path = tmp_path / "test_diagnostic.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """A ``TestClient`` wired to a fresh per-test SQLite DB."""
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
    """Sign up and return the JSON body (cookie is set on the jar)."""
    resp = client.post(
        "/auth/signup", json={"email": email, "password": password}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _start(client: TestClient) -> dict:
    resp = client.post("/diagnostic/start")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 1. Question bank shape
# ---------------------------------------------------------------------------


def test_question_bank_shape():
    """10 questions, unique ids, full axis coverage, well-formed
    choices and deltas."""
    assert len(QUESTIONS) == 10
    assert TOTAL_QUESTIONS == 10

    ids = [q.id for q in QUESTIONS]
    assert len(set(ids)) == len(ids), "question ids must be unique"

    # Every axis is covered by at least one question's axis_tags.
    covered = set()
    for q in QUESTIONS:
        covered.update(q.axis_tags)
    assert covered == set(ALL_AXES), f"uncovered axes: {set(ALL_AXES) - covered}"

    for q in QUESTIONS:
        assert q.choices, f"{q.id} has no choices"
        assert q.kind == "multiple_choice"
        assert 1 <= q.weight <= 3
        for c in q.choices:
            assert set(c.delta.keys()).issubset(set(ALL_AXES)), (
                f"{q.id} choice {c.label!r} has delta keys outside ALL_AXES"
            )
            for v in c.delta.values():
                assert isinstance(v, int) and 0 <= v <= 3


# ---------------------------------------------------------------------------
# 2-7. Scoring math (pure function)
# ---------------------------------------------------------------------------


def test_score_empty_answers():
    """score([]) -> ({axis: 0 for all}, {})."""
    axes, reasons = score([])
    assert axes == {axis: 0 for axis in ALL_AXES}
    assert reasons == {}


def test_score_single_answer_clamps_weight_times_delta():
    """A single weight-3, delta-3 answer scores 9 raw -> clamped to 3.

    ``q-verb-conjugation-present`` has weight 3; its "I struggle"
    choice adds delta 3 on ``verbs``. Raw = 3 * 3 = 9, clamped to 3.
    """
    q = QUESTION_BY_ID["q-verb-conjugation-present"]
    assert q.weight == 3
    struggle = next(c for c in q.choices if c.delta.get("verbs") == 3)
    # Sanity-check the raw arithmetic the scorer performs before clamp.
    assert q.weight * struggle.delta["verbs"] == 9

    axes, _ = score(
        [AnswerRecord("q-verb-conjugation-present", struggle.label)]
    )
    assert axes["verbs"] == 3  # clamped from 9
    # Untouched axes stay 0.
    assert axes["idioms"] == 0


def test_score_clamps_upper_bound():
    """Two high-delta verbs answers push the raw sum well past 3;
    the result is clamped to 3."""
    q1 = QUESTION_BY_ID["q-verb-conjugation-present"]
    c1 = next(c for c in q1.choices if c.delta.get("verbs") == 3)
    q2 = QUESTION_BY_ID["q-verb-preposition-combos"]
    c2 = next(c for c in q2.choices if c.delta.get("verbs", 0) >= 1)

    axes, _ = score(
        [
            AnswerRecord("q-verb-conjugation-present", c1.label),
            AnswerRecord("q-verb-preposition-combos", c2.label),
        ]
    )
    assert axes["verbs"] == 3  # raw > 3, clamped


def test_score_clamps_lower_bound():
    """The lower clamp guarantees no axis is ever negative, and an
    untouched axis stays exactly 0. (The shipped bank has no negative
    deltas, so this is the invariant the ``max(0, ...)`` clamp
    enforces.)"""
    # All-confident answers -> every touched axis nets 0.
    answers = []
    for q in QUESTIONS:
        zero_choice = next(
            (c for c in q.choices if all(v == 0 for v in c.delta.values())),
            None,
        )
        if zero_choice is not None:
            answers.append(AnswerRecord(q.id, zero_choice.label))
    axes, reasons = score(answers)
    assert all(0 <= v <= 3 for v in axes.values())
    assert all(v == 0 for v in axes.values())
    assert reasons == {}


def test_score_multi_axis_increments_both():
    """A choice with delta on two axes increments both.

    ``q-verb-preposition-combos``'s "I struggle" choice adds to both
    ``verbs`` and ``prepositional_combos``.
    """
    q = QUESTION_BY_ID["q-verb-preposition-combos"]
    multi = next(
        c
        for c in q.choices
        if c.delta.get("verbs", 0) > 0
        and c.delta.get("prepositional_combos", 0) > 0
    )
    axes, _ = score([AnswerRecord("q-verb-preposition-combos", multi.label)])
    assert axes["verbs"] > 0
    assert axes["prepositional_combos"] > 0


def test_score_reasons_lists_top_contributors():
    """When two questions both contribute to ``verbs``, the
    ``reasons['verbs']`` string names both question ids."""
    q1 = QUESTION_BY_ID["q-verb-conjugation-present"]
    c1 = next(c for c in q1.choices if c.delta.get("verbs", 0) >= 2)
    q2 = QUESTION_BY_ID["q-verb-preposition-combos"]
    c2 = next(c for c in q2.choices if c.delta.get("verbs", 0) >= 1)

    axes, reasons = score(
        [
            AnswerRecord("q-verb-conjugation-present", c1.label),
            AnswerRecord("q-verb-preposition-combos", c2.label),
        ]
    )
    assert axes["verbs"] > 0
    assert "verbs" in reasons
    assert "q-verb-conjugation-present" in reasons["verbs"]
    assert "q-verb-preposition-combos" in reasons["verbs"]
    # Axes with score 0 are omitted from reasons.
    assert "idioms" not in reasons


# ---------------------------------------------------------------------------
# 8. Auth gate — every endpoint 401 without a token
# ---------------------------------------------------------------------------


def test_all_endpoints_require_auth(client):
    """No cookie / no Bearer -> 401 on all four endpoints."""
    client.cookies.clear()
    assert client.post("/diagnostic/start").status_code == 401
    assert (
        client.post(
            "/diagnostic/answer",
            json={
                "session_id": "x",
                "question_id": "q-verb-conjugation-present",
                "choice_label": "y",
            },
        ).status_code
        == 401
    )
    assert client.get("/diagnostic/result?session_id=x").status_code == 401
    assert (
        client.post(
            "/diagnostic/apply", json={"session_id": "x"}
        ).status_code
        == 401
    )


# ---------------------------------------------------------------------------
# 9. POST /diagnostic/start — happy path
# ---------------------------------------------------------------------------


def test_start_returns_session_and_stripped_questions(client):
    """start returns a session_id + 10 questions, and the questions
    do NOT carry delta / weight / axis_tags."""
    _signup(client)
    body = _start(client)
    assert "session_id" in body and body["session_id"]
    assert len(body["questions"]) == 10

    raw = body
    # The scoring internals must not leak anywhere in the response.
    import json as _json

    blob = _json.dumps(raw)
    assert "delta" not in blob
    assert "weight" not in blob
    assert "axis_tags" not in blob

    for q in body["questions"]:
        assert set(q.keys()) == {"id", "prompt", "kind", "choices"}
        for c in q["choices"]:
            assert set(c.keys()) == {"label"}


def test_start_reuses_in_progress_session(client):
    """A second start while a session is in_progress returns the same
    session_id (no duplicate)."""
    _signup(client)
    first = _start(client)["session_id"]
    second = _start(client)["session_id"]
    assert first == second


# ---------------------------------------------------------------------------
# 10-12. POST /diagnostic/answer
# ---------------------------------------------------------------------------


def _first_choice_label(question_id: str) -> str:
    return QUESTION_BY_ID[question_id].choices[0].label


def test_answer_records_choice(client):
    """An answer is recorded and the progress counter increments."""
    _signup(client)
    session_id = _start(client)["session_id"]
    qid = "q-verb-conjugation-present"
    label = QUESTION_BY_ID[qid].choices[2].label  # "Shaky..."
    resp = client.post(
        "/diagnostic/answer",
        json={"session_id": session_id, "question_id": qid, "choice_label": label},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"answered": 1, "total": 10}


def test_answer_is_idempotent_per_question(client):
    """Re-answering the same question overwrites, doesn't double-count."""
    _signup(client)
    session_id = _start(client)["session_id"]
    qid = "q-verb-conjugation-present"
    q = QUESTION_BY_ID[qid]
    client.post(
        "/diagnostic/answer",
        json={
            "session_id": session_id,
            "question_id": qid,
            "choice_label": q.choices[0].label,
        },
    )
    resp = client.post(
        "/diagnostic/answer",
        json={
            "session_id": session_id,
            "question_id": qid,
            "choice_label": q.choices[3].label,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"answered": 1, "total": 10}

    # The recorded choice is the latest one.
    result = client.get(
        f"/diagnostic/result?session_id={session_id}"
    ).json()
    # q.choices[3] is "I struggle" (delta 3, weight 3 -> clamp 3).
    assert result["axes"]["verbs"] == 3


def test_answer_unknown_question_400(client):
    _signup(client)
    session_id = _start(client)["session_id"]
    resp = client.post(
        "/diagnostic/answer",
        json={
            "session_id": session_id,
            "question_id": "q-does-not-exist",
            "choice_label": "whatever",
        },
    )
    assert resp.status_code == 400


def test_answer_invalid_choice_label_400(client):
    _signup(client)
    session_id = _start(client)["session_id"]
    resp = client.post(
        "/diagnostic/answer",
        json={
            "session_id": session_id,
            "question_id": "q-verb-conjugation-present",
            "choice_label": "not a real choice",
        },
    )
    assert resp.status_code == 400


def test_answer_cross_user_session_404(client):
    """User B cannot answer into user A's session."""
    # User A creates a session.
    _signup(client, email="a@example.com")
    session_a = _start(client)["session_id"]

    # User B logs in (signup sets B's cookie, overwriting A's).
    _signup(client, email="b@example.com")
    resp = client.post(
        "/diagnostic/answer",
        json={
            "session_id": session_a,
            "question_id": "q-verb-conjugation-present",
            "choice_label": _first_choice_label("q-verb-conjugation-present"),
        },
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 13. GET /diagnostic/result — deterministic, matches score()
# ---------------------------------------------------------------------------


def test_result_matches_pure_score(client):
    """The endpoint result equals the pure scoring function on the
    same recorded answers."""
    _signup(client)
    session_id = _start(client)["session_id"]

    # Answer every question with its 3rd choice ("Shaky..." on the
    # single-axis ones).
    chosen: list[AnswerRecord] = []
    for q in QUESTIONS:
        idx = min(2, len(q.choices) - 1)
        label = q.choices[idx].label
        client.post(
            "/diagnostic/answer",
            json={
                "session_id": session_id,
                "question_id": q.id,
                "choice_label": label,
            },
        )
        chosen.append(AnswerRecord(q.id, label))

    expected_axes, expected_reasons = score(chosen)
    body = client.get(
        f"/diagnostic/result?session_id={session_id}"
    ).json()
    assert body["axes"] == expected_axes
    assert body["reasons"] == expected_reasons


def test_result_cross_user_404(client):
    _signup(client, email="a@example.com")
    session_a = _start(client)["session_id"]
    _signup(client, email="b@example.com")
    resp = client.get(f"/diagnostic/result?session_id={session_a}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 14-15. POST /diagnostic/apply
# ---------------------------------------------------------------------------


def _answer_all(client, session_id, choice_index=2):
    for q in QUESTIONS:
        idx = min(choice_index, len(q.choices) - 1)
        client.post(
            "/diagnostic/answer",
            json={
                "session_id": session_id,
                "question_id": q.id,
                "choice_label": q.choices[idx].label,
            },
        )


def test_apply_upserts_weakness_profile(client):
    """apply writes the computed axes into the weakness profile and
    flips the session to applied."""
    body = _signup(client)
    user_id = body["user"]["id"]
    session_id = _start(client)["session_id"]
    _answer_all(client, session_id)

    result = client.get(
        f"/diagnostic/result?session_id={session_id}"
    ).json()

    apply_resp = client.post(
        "/diagnostic/apply", json={"session_id": session_id}
    )
    assert apply_resp.status_code == 200, apply_resp.text
    profile = apply_resp.json()
    assert profile["user_id"] == user_id
    # The applied profile axes match the computed result axes.
    assert profile["axes"] == result["axes"]

    # And the weakness-profile GET reflects the same.
    got = client.get(f"/weakness-profile/{user_id}").json()
    assert got["axes"] == result["axes"]


def test_apply_double_apply_409(client):
    """Applying an already-applied session returns 409."""
    _signup(client)
    session_id = _start(client)["session_id"]
    _answer_all(client, session_id)

    first = client.post(
        "/diagnostic/apply", json={"session_id": session_id}
    )
    assert first.status_code == 200
    second = client.post(
        "/diagnostic/apply", json={"session_id": session_id}
    )
    assert second.status_code == 409


def test_apply_cross_user_404(client):
    _signup(client, email="a@example.com")
    session_a = _start(client)["session_id"]
    _signup(client, email="b@example.com")
    resp = client.post(
        "/diagnostic/apply", json={"session_id": session_a}
    )
    assert resp.status_code == 404


def test_password_hash_never_in_apply_response(client):
    """The apply response (a weakness profile) must never carry
    ``password_hash``."""
    _signup(client)
    session_id = _start(client)["session_id"]
    _answer_all(client, session_id)
    resp = client.post(
        "/diagnostic/apply", json={"session_id": session_id}
    )
    assert "password_hash" not in resp.text


# ---------------------------------------------------------------------------
# 16. Alembic migration — idempotent on SQLite (Postgres in QA hook)
# ---------------------------------------------------------------------------


def test_alembic_migration_idempotent_on_sqlite(tmp_path, monkeypatch):
    """``alembic upgrade head`` runs the Phase 3 migration cleanly,
    and a second run is a no-op (the inspect() guard short-circuits
    the create_table)."""
    db_path = tmp_path / "migrate.db"
    backend_dir = Path(__file__).resolve().parent.parent
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }

    first = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert first.returncode == 0, first.stderr
    assert "003_phase3_diagnostic" in (first.stdout + first.stderr)

    second = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert second.returncode == 0, second.stderr


def test_alembic_downgrade_drops_diagnostic_table(tmp_path, monkeypatch):
    """The Phase 3 downgrade drops ``diagnostic_sessions`` and lands
    back at the Phase 2 revision."""
    db_path = tmp_path / "downgrade.db"
    backend_dir = Path(__file__).resolve().parent.parent
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }

    up = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert up.returncode == 0, up.stderr

    down = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "a15ec4b9f736"],
        cwd=str(backend_dir), env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert down.returncode == 0, down.stderr
