"""Tests for Phase 9.2 -- GET /exercises/due widening + GET /auth/me due_by_type (card t_e4988202).

Coverage map (mirrors the card body's "Tests" section):

1. Cloze + matching union -- seed two due cards with
   exercise_type='cloze' and exercise_type='matching' (with
   matching's due_date earliest). Across two calls to
   GET /exercises/due (no type filter), return both types via
   cloze (cloze branch) + 204 with the matching headers (non-cloze
   branch).

2. Cloze-only back-compat -- only cloze cards exist; route
   continues to return ClozeDueExerciseOut (Phase 5.6 contract
   preserved). type=cloze explicit filter matches the legacy
   no-param call.

3. /auth/me.due_by_type 4-key shape -- with one due card each of
   cloze/matching/comprehension/idiom (and one extra cloze for
   variety), the payload reports each count correctly; missing
   types get zero.

4. No-regression empty -- no due cards, no fresh words ->
   GET /exercises/due returns 204 (same as Phase 5.6).

5. type query param filter -- ?type=cloze restricts the pick
   to cloze rows even when matching/comprehension/idiom are also
   due.

Phase 9.2 ships as the read-side widening; the schema column
fsrs_cards.exercise_type is added by Phase 9.1 (card t_0bfdb7ed,
in-flight in parallel). Each fixture in this file performs an
ALTER TABLE fsrs_cards ADD COLUMN exercise_type TEXT NOT NULL
DEFAULT 'cloze' against the test SQLite DB so the route's
inspect()-based column-presence probe returns True, mirroring
the post-9.1 schema. Without the column, the tests would exercise
the legacy all-cloze fallback path (test #2 covers that
explicitly).

Hermetic: fresh temp SQLite DB + JWT secret per test; the
OpenRouter chat-completions call is stubbed via
app.cloze._openai_client so no network is touched. Mirrors the
test_due.py fixture pattern.

Run from backend/::

    uv run pytest -q tests/test_due_queue_widened.py
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Fixtures -- mirror the per-test SQLite + JWT secret pattern from
# test_due.py. Per-test reconfigure of app.database so each test gets its
# own clean DB.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so app.auth's import-time
    check passes. Mirrors test_auth.py and test_due.py.
    """
    db_path = tmp_path / "test_due_widened.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """TestClient wired to a fresh per-test SQLite DB."""
    from app import database, models  # noqa: F401 -- register tables
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB.

    Tests that need direct row inserts (e.g. seeding an FsrsCard to
    test the FSRS-driven branch) request this fixture; tests that
    only exercise the route layer request 'client' instead.
    """
    from sqlalchemy.orm import sessionmaker

    from app import database, models  # noqa: F401 -- see test_due.py docstring

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        yield session


@pytest.fixture
def fsrs_with_type_column(db_session):
    """Ensure fsrs_cards.exercise_type exists in the test DB.

    Phase 9.1 (card t_0bfdb7ed) adds the column to ``app.models``;
    the model-driven ``Base.metadata.create_all`` then creates it
    automatically. When 9.2's tests run in isolation (or before
    9.1 lands on main), the column is missing, so this fixture
    does an idempotent ``ALTER TABLE`` only if ``PRAGMA
    table_info(fsrs_cards)`` does not already list it.

    The route's reflection probe (``inspect(engine).get_columns``)
    needs the column present for tests #1, #3, #5. Tests #2 + the
    legacy fallback test do not request this fixture -- they
    exercise the no-column branch explicitly.
    """
    rows = db_session.execute(text("PRAGMA table_info(fsrs_cards)")).fetchall()
    column_names = {row[1] for row in rows}  # PRAGMA returns (cid, name, ...)
    if "exercise_type" not in column_names:
        db_session.execute(
            text(
                "ALTER TABLE fsrs_cards ADD COLUMN exercise_type "
                "TEXT NOT NULL DEFAULT 'cloze'"
            )
        )
        db_session.commit()
    return db_session


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """Sign up a user; the cookie is set as a side-effect on client."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _seed_word(
    session,
    *,
    word: str,
    word_type: str,
    example_de: str = "Der Hund schläft.",
) -> int:
    """Insert one Word row + a single Example; return word_id."""
    from app import models

    row = models.Word(
        word=word, word_type=word_type, frequency="5", is_complete=True
    )
    session.add(row)
    session.flush()
    session.add(models.Example(word_id=row.id, german=example_de, english=""))
    session.commit()
    return row.id


def _seed_fsrs_card(
    session,
    *,
    word_id: int,
    due_date: datetime,
    state: int = 1,
    reps: int = 0,
    lapses: int = 0,
    difficulty: float | None = None,
    stability: float | None = None,
    exercise_type: str | None = None,
) -> int:
    """Insert one FsrsCard row at the given due_date; return card id.

    When exercise_type is provided and the schema has the column
    (test fixture fsrs_with_type_column), the column is set via raw
    SQL after the SQLAlchemy insert -- the model on main (pre-9.1)
    does not know about the column yet, so we cannot go through
    the ORM mapped attribute.
    """
    from app import models

    row = models.FsrsCard(
        word_id=word_id,
        difficulty=difficulty,
        stability=stability,
        retrievability=None,
        due_date=due_date,
        last_review=due_date,
        reps=reps,
        lapses=lapses,
        state=state,
        elapsed_days=0,
        scheduled_days=0,
    )
    session.add(row)
    session.commit()

    if exercise_type is not None:
        session.execute(
            text(
                "UPDATE fsrs_cards SET exercise_type = :etype "
                "WHERE id = :id"
            ),
            {"etype": exercise_type, "id": row.id},
        )
        session.commit()
    return row.id


# ---------------------------------------------------------------------------
# Stub LLM client -- same shape as test_due.py / test_cloze.py.
# ---------------------------------------------------------------------------


def _stub_openai_client(payload: str) -> OpenAI:
    """Build a stub OpenAI client that returns payload as the assistant
    message content.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-due-widened-stub-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "qwen/qwen3-235b-a22b-2507",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": payload},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 12,
                    "total_tokens": 42,
                },
            },
        )

    return OpenAI(
        api_key="test-key-not-real",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _cloze_payload_for(word_id: int) -> str:
    """A valid ClozeExercise JSON payload keyed to word_id."""
    distractors = [w for w in (word_id + 1, word_id + 2, word_id + 3)]
    return json.dumps(
        {
            "sentence_with_blank": "Der ___ schläft.",
            "answer_word_id": word_id,
            "distractors": distractors,
            "difficulty": "easy",
            "rationale": "Stub rationale.",
            "prompt_template_version": "cloze-v1",
        },
        ensure_ascii=False,
    )


def _stub_cloze_for(monkeypatch, word_id: int) -> None:
    """Patch app.cloze._openai_client to a stub yielding a valid
    ClozeExercise for word_id. Sets the env key so the lazy client
    construction does not bail out.
    """
    payload = _cloze_payload_for(word_id)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _stub_openai_client(payload),
    )


# ===========================================================================
# 1. /exercises/due returns both cloze and non-cloze due cards (over two
#    sequential calls).
# ===========================================================================


def test_due_returns_both_cloze_and_matching_over_two_calls(
    client, db_session, fsrs_with_type_column, monkeypatch
) -> None:
    """Cloze + matching both seeded -> route returns both shapes across calls.

    Card 1: cloze due earliest. type=cloze picks it -> 200 + cloze shape.
    Card 2: matching due later. type=matching picks it -> 204 + headers.

    We exercise the union surface by issuing two filter calls. This
    avoids relying on FSRS-state-mutation between calls (the GET
    does not consume the row -- only POST /exercises/grade does).
    """
    _signup(client)

    cloze_word = _seed_word(db_session, word="Hund", word_type="Noun")
    matching_word = _seed_word(db_session, word="Katze", word_type="Noun")

    now = datetime.utcnow()
    _seed_fsrs_card(
        db_session,
        word_id=cloze_word,
        due_date=now - timedelta(minutes=10),
        state=1,
        exercise_type="cloze",
    )
    matching_id = _seed_fsrs_card(
        db_session,
        word_id=matching_word,
        due_date=now - timedelta(minutes=5),
        state=1,
        exercise_type="matching",
    )

    _stub_cloze_for(monkeypatch, word_id=cloze_word)

    # Call #1 -- type=cloze filter, picks the cloze row.
    resp1 = client.get("/exercises/due?type=cloze")
    assert resp1.status_code == 200, resp1.text
    body1 = resp1.json()
    assert body1["exercise_type"] == "cloze"
    assert body1["answer_word_id"] == cloze_word
    assert body1["due_from_fsrs"] is True

    # Call #2 -- type=matching filter, picks the matching row.
    resp2 = client.get("/exercises/due?type=matching")
    assert resp2.status_code == 204, resp2.text
    assert resp2.headers.get("X-Due-Exercise-Type") == "matching"
    assert resp2.headers.get("X-Due-Card-Id") == str(matching_id)
    assert resp2.headers.get("X-Due-Word-Id") == str(matching_word)




def test_due_cloze_only_legacy_schema_returns_cloze(
    client, db_session, monkeypatch
) -> None:
    """Back-compat preserved: cloze-only seed -> route returns cloze.

    No fsrs_with_type_column fixture here -- this test exercises
    the pre-9.1 (legacy) schema path where every fsrs_cards row
    is implicitly cloze by Phase 5.6 contract. The route's
    inspect() probe returns False for the missing
    exercise_type column; _pick_due_fsrs_card applies no type
    filter, and the legacy cloze-only behaviour is preserved
    end-to-end.
    """
    _signup(client)

    cloze_word = _seed_word(db_session, word="Hund", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=cloze_word,
        due_date=datetime.utcnow() - timedelta(minutes=5),
        state=1,
    )

    _stub_cloze_for(monkeypatch, word_id=cloze_word)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # ClozeDueExerciseOut fields preserved.
    assert body["exercise_type"] == "cloze"
    assert body["answer_word_id"] == cloze_word
    assert body["prompt_template_version"] == "cloze-v1"
    assert body["due_from_fsrs"] is True
    # No non-cloze headers -- legacy schema has nothing to surface.
    assert "X-Due-Exercise-Type" not in resp.headers


def test_due_type_cloze_filter_explicit(
    client, db_session, fsrs_with_type_column, monkeypatch
) -> None:
    """?type=cloze filters even when matching is also due.

    The matching card has the earlier due_date and would normally
    be picked first. The type=cloze filter restricts the SQL
    query to cloze rows only, so the cloze card is picked.
    """
    _signup(client)

    cloze_word = _seed_word(db_session, word="Hund", word_type="Noun")
    matching_word = _seed_word(db_session, word="Katze", word_type="Noun")

    now = datetime.utcnow()
    _seed_fsrs_card(
        db_session,
        word_id=matching_word,
        due_date=now - timedelta(minutes=10),
        state=1,
        exercise_type="matching",
    )
    _seed_fsrs_card(
        db_session,
        word_id=cloze_word,
        due_date=now - timedelta(minutes=5),
        state=1,
        exercise_type="cloze",
    )

    _stub_cloze_for(monkeypatch, word_id=cloze_word)

    resp = client.get("/exercises/due?type=cloze")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["exercise_type"] == "cloze"
    assert body["answer_word_id"] == cloze_word


# ===========================================================================
# 3. /auth/me.due_by_type -- 5-key dict, all counts correct.
#
# Phase 10.6 (card t_da43cc23) widens the closure dict from 4 to
# 5 keys additively. The Pydantic shape stays ``Dict[str, int]``
# (not a closed Literal-tuple), so the wire contract is
# non-breaking — only the implicit "always 4 keys" convention
# shifts to "always 5 keys." phrase_match joins as the 5th
# FSRS-graded exercise type (Phase 10.1 schema, 10.2 Literal
# widening, 10.3 endpoint, 10.5 frontend page).
# ===========================================================================


def test_auth_me_due_by_type_5_keys_with_each_count(
    client, db_session, fsrs_with_type_column
) -> None:
    """due_by_type reports correct counts per exercise type.

    Seeds 2 cloze, 1 matching, 0 comprehension (deliberately
    absent), 1 idiom due. The 5-key dict reports each count
    correctly; the comprehension branch verifies the zero-default
    for missing types. The phrase_match branch verifies the
    zero-default for the Phase-10.1 schema where no phrase_match
    card has been seeded yet.
    """
    _signup(client)

    cloze1 = _seed_word(db_session, word="Hund", word_type="Noun")
    cloze2 = _seed_word(db_session, word="Katze", word_type="Noun")
    matching1 = _seed_word(db_session, word="Maus", word_type="Noun")
    idiom1 = _seed_word(db_session, word="Vogel", word_type="Noun")

    now = datetime.utcnow()
    _seed_fsrs_card(
        db_session, word_id=cloze1, due_date=now - timedelta(minutes=5),
        state=1, exercise_type="cloze",
    )
    _seed_fsrs_card(
        db_session, word_id=cloze2, due_date=now - timedelta(minutes=5),
        state=1, exercise_type="cloze",
    )
    _seed_fsrs_card(
        db_session, word_id=matching1, due_date=now - timedelta(minutes=5),
        state=1, exercise_type="matching",
    )
    _seed_fsrs_card(
        db_session, word_id=idiom1, due_date=now - timedelta(minutes=5),
        state=1, exercise_type="idiom",
    )

    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "due_by_type" in body, body
    counts = body["due_by_type"]
    assert set(counts.keys()) == {
        "cloze", "matching", "comprehension", "idiom", "phrase_match",
    }, counts
    assert counts["cloze"] == 2, counts
    assert counts["matching"] == 1, counts
    assert counts["comprehension"] == 0, counts
    assert counts["idiom"] == 1, counts
    assert counts["phrase_match"] == 0, counts


def test_auth_me_due_by_type_all_zero_when_no_due_cards(
    client, db_session, fsrs_with_type_column
) -> None:
    """due_by_type is all-zeros when there are no due cards.

    Seeds a future-due card so the fsrs_cards table is not empty
    (the dict must reflect 'zero due', not 'no rows'). The SQL
    filter due_date <= now() excludes future cards; every key
    (including the Phase-10.6 ``phrase_match`` bucket) resolves
    to 0.
    """
    _signup(client)
    w = _seed_word(db_session, word="Hund", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=w,
        due_date=datetime.utcnow() + timedelta(days=10),
        state=1,
        exercise_type="cloze",
    )

    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    counts = body["due_by_type"]
    assert counts == {
        "cloze": 0,
        "matching": 0,
        "comprehension": 0,
        "idiom": 0,
        "phrase_match": 0,
    }, counts


def test_auth_me_due_by_type_legacy_schema_falls_back_to_cloze(
    client, db_session
) -> None:
    """Pre-9.1 schema: due_by_type.cloze is the global due-card count.

    No fsrs_with_type_column fixture here -- the test exercises
    the legacy code path where every fsrs_cards row is
    implicitly cloze by Phase 5.6 contract. The route's inspect()
    probe returns False for the missing column; the fallback
    path buckets all due rows under cloze and leaves matching
    / comprehension / idiom / phrase_match at zero.
    """
    _signup(client)

    w1 = _seed_word(db_session, word="Hund", word_type="Noun")
    w2 = _seed_word(db_session, word="Katze", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=w1,
        due_date=datetime.utcnow() - timedelta(minutes=5),
        state=1,
    )
    _seed_fsrs_card(
        db_session,
        word_id=w2,
        due_date=datetime.utcnow() - timedelta(minutes=5),
        state=1,
    )

    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    counts = body["due_by_type"]
    assert counts == {
        "cloze": 2,
        "matching": 0,
        "comprehension": 0,
        "idiom": 0,
        "phrase_match": 0,
    }, counts


# ===========================================================================
# 4. No-regression empty path.
# ===========================================================================


def test_due_returns_204_when_no_cards_and_no_words(client) -> None:
    """No cards + no corpus words -> 204 (Phase 5.6 contract preserved)."""
    _signup(client)
    resp = client.get("/exercises/due")
    assert resp.status_code == 204
    # No matching headers -- the empty branch is distinct from
    # the 'non-cloze due card' branch which carries the headers.
    assert "X-Due-Exercise-Type" not in resp.headers


def test_due_returns_204_when_no_due_cards_with_type_filter(
    client, db_session, fsrs_with_type_column
) -> None:
    """type=cloze with no due cloze cards -> 204 (no false positive)."""
    _signup(client)
    matching_word = _seed_word(db_session, word="Katze", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=matching_word,
        due_date=datetime.utcnow() - timedelta(minutes=5),
        state=1,
        exercise_type="matching",
    )

    resp = client.get("/exercises/due?type=cloze")
    assert resp.status_code == 204
    assert "X-Due-Exercise-Type" not in resp.headers


def test_due_filter_narrows_to_matching_branch(
    client, db_session, fsrs_with_type_column
) -> None:
    """?type=matching picks the matching card and surfaces it via 204 headers.

    Complements test_due_type_cloze_filter_explicit -- proves the
    same narrowing works for the non-cloze branches too. Callers
    that want only matching can ask for it explicitly.
    """
    _signup(client)
    matching_word = _seed_word(db_session, word="Katze", word_type="Noun")
    cloze_word = _seed_word(db_session, word="Hund", word_type="Noun")

    now = datetime.utcnow()
    matching_id = _seed_fsrs_card(
        db_session,
        word_id=matching_word,
        due_date=now - timedelta(minutes=5),
        state=1,
        exercise_type="matching",
    )
    _seed_fsrs_card(
        db_session,
        word_id=cloze_word,
        due_date=now - timedelta(minutes=10),
        state=1,
        exercise_type="cloze",
    )

    resp = client.get("/exercises/due?type=matching")
    assert resp.status_code == 204, resp.text
    assert resp.headers.get("X-Due-Exercise-Type") == "matching"
    assert resp.headers.get("X-Due-Card-Id") == str(matching_id)
