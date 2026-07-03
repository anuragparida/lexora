"""Tests for Phase 5.4 — ``GET /exercises/due`` (card t_e8548d6d).

Coverage map (mirrors the card body's "pytest cases" section):

1. ``select_target_word(force_word_id=<existing>)`` returns that exact
   Word row, bypassing the deterministic seed.
2. ``select_target_word(force_word_id=<missing>)`` raises ``ValueError``.
3. ``force_word_id=None`` path on ``select_target_word`` returns the
   deterministic-by-seed word (Phase 4.5 behaviour unchanged — regression
   guard).
4. User with no JWT cookie → 401 on ``GET /exercises/due``.
5. User with no due ``fsrs_cards`` rows AND no fresh corpus words →
   204 No Content.
6. User with one due ``fsrs_cards`` row → 200 + ``ClozeDueExerciseOut``
   with ``due_from_fsrs=True`` and ``answer_word_id == <that row's
   word_id>``.
7. User with one Learning-state card (due immediately, ``due_date``
   in the past or ``now``) → 200 + ``due_from_fsrs=True``;
   ``answer_word_id`` matches the Learning row's ``word_id``.
8. User with three due ``fsrs_cards`` rows → 200, returns the one
   with the earliest ``due_date`` (asserted via ``answer_word_id``).
9. First-encounter path: a word with no ``fsrs_cards`` row yet →
   a fresh Learning row is created inline; response carries
   ``due_from_fsrs=False``; ``fsrs_cards`` now has one row for the
   picked word.
10. ``force_word_id`` is threaded through to ``generate_cloze`` —
    the cloze's ``answer_word_id`` matches the forced id (regression
    guard on Gotcha #1: "force_word_id threading").
11. Backwards-compatibility: ``POST /exercises/cloze`` (the Phase 4.5
    deterministic-by-seed flow) still works without modification and
    still hits the existing tests' expectations.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test, the
OpenRouter chat-completions call is replaced with a stub OpenAI client
(``monkeypatch.setattr("app.cloze._openai_client", ...)``) so no
network is touched. The LLM stub is the same shape the Phase 4.2
``test_cloze.py`` uses; we duplicate it locally rather than import so
the test files stay independently runnable.

Run from ``backend/``::

    uv run pytest -q tests/test_due.py
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# ``test_cloze.py`` and ``test_diagnostic.py``.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Mirrors the pattern in ``test_auth.py`` and
    ``test_cloze.py``.
    """
    db_path = tmp_path / "test_due.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """``TestClient`` wired to a fresh per-test SQLite DB."""
    from app import database, models  # noqa: F401 — models registers tables on Base.metadata
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB.

    Same pattern as ``test_cloze.py``: per-test reconfigure + create_all
    + a fresh session. Tests that need direct row inserts (e.g. seeding
    an ``FsrsCard`` to test the FSRS-driven branch) request this
    fixture; tests that only exercise the route layer request ``client``
    instead.

    **Why the explicit ``from app import models`` here:** ``Base.metadata``
    only knows about model classes that have been imported at the time
    ``create_all`` runs. The HTTP ``client`` fixture gets ``app.models``
    transitively (via ``app.main``), but the standalone ``db_session``
    fixture doesn't pull in any module that imports the model classes —
    so we import it explicitly. Without this line, ``create_all`` runs
    with an empty metadata set and the test's first ``INSERT INTO
    words`` blows up with "no such table: words".
    """
    from sqlalchemy.orm import sessionmaker

    from app import database, models  # noqa: F401 — see docstring above

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """Sign up a user; the cookie is set as a side-effect on ``client``.

    Mirrors the helper in ``test_cloze.py``. Returns the parsed JSON
    body so callers can read ``user_id`` if they need it.
    """
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
    """Insert one ``Word`` row plus a single ``Example``; return word_id."""
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
) -> int:
    """Insert one ``FsrsCard`` row at the given ``due_date``; return card id.

    Phase 5's ``FsrsCard`` schema (Phase 0 baseline) is the source of
    truth for the columns. ``word_id`` is unique-constrained (the
    shipped SQLite corpus enforces this; Postgres gets the same
    constraint via 5.2's migration), so a second insert with the same
    word_id would fail.
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
    return row.id


# ---------------------------------------------------------------------------
# Stub LLM client — same shape as the Phase 4.2 / Phase 4.3 helper.
# We build a real ``OpenAI`` client whose underlying httpx transport is
# a ``MockTransport`` so ``instructor.from_openai`` accepts it. The
# assistant message content is the JSON payload we want validated.
# ---------------------------------------------------------------------------


def _stub_openai_client(payload: str) -> OpenAI:
    """Build a stub OpenAI client that returns ``payload`` as the
    assistant message content. Mirrors the helper in ``test_cloze.py``;
    duplicated here so this test file stays independently runnable.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-due-stub-001",
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
    """A valid ``ClozeExercise`` JSON payload keyed to ``word_id``.

    Three distractor ids — we pick anything that's not ``word_id``;
    the route doesn't validate the distractors against the corpus, so
    any three distinct positive ints satisfy the Pydantic schema.
    """
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


def _mock_cloze_call(monkeypatch, *, word_id: int) -> None:
    """Patch ``app.cloze._openai_client`` to return a stub that yields a
    valid ``ClozeExercise`` for ``word_id``. Sets the env key so the
    lazy client construction in ``app.cloze`` doesn't bail out.
    """
    payload = _cloze_payload_for(word_id)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _stub_openai_client(payload),
    )


def _mp(monkeypatch, word_id: int) -> None:
    """Short alias for ``_mock_cloze_call`` so test bodies stay tight.

    The monkeypatch fixture is pytest-managed; ``_mp`` just forwards.
    """
    _mock_cloze_call(monkeypatch=monkeypatch, word_id=word_id)


# ===========================================================================
# 1. Module-level: ``select_target_word(force_word_id=...)`` semantics
# ===========================================================================


def test_select_target_word_force_returns_exact_row(db_session) -> None:
    """``force_word_id=<existing>`` returns that exact Word row.

    This is the 5.4 contract: the route has already picked a word from
    the due queue and wants ``select_target_word`` to return THAT word,
    not a fresh selection from the user's weakness profile.
    """
    from app import cloze

    w1 = _seed_word(db_session, word="Hund", word_type="Noun")
    _seed_word(db_session, word="Katze", word_type="Noun")
    _seed_word(db_session, word="Maus", word_type="Noun")

    forced = cloze.select_target_word(db_session, user_id=1, force_word_id=w1)
    assert forced.id == w1
    assert forced.word == "Hund"


def test_select_target_word_force_raises_for_missing_id(db_session) -> None:
    """``force_word_id=<missing>`` raises ``ValueError``.

    The route layer maps this to 500 (corpus inconsistency — the
    due-queue picked a word id that no longer exists in the words
    table). The test verifies the contract at the function boundary.
    """
    from app import cloze

    _seed_word(db_session, word="Hund", word_type="Noun")
    with pytest.raises(ValueError, match="force_word_id=9999"):
        cloze.select_target_word(db_session, user_id=1, force_word_id=9999)


def test_select_target_word_default_seed_path_unchanged(db_session) -> None:
    """``force_word_id=None`` returns the deterministic-by-seed word.

    Regression guard: Phase 4.5's ``POST /exercises/cloze`` calls
    ``select_target_word(db, user_id)`` with no ``force_word_id``. The
    keyword-only parameter must NOT change behaviour for that call.
    """
    from app import cloze

    verb_id = _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="Ich gehe."
    )
    _seed_user_with_axes(db_session, axes={"verbs": 3})

    w1 = cloze.select_target_word(db_session, user_id=1)
    w2 = cloze.select_target_word(db_session, user_id=1)
    assert w1.id == w2.id == verb_id


def _seed_user_with_axes(session, *, axes: dict[str, int] | None = None) -> int:
    """Insert a User + optional WeaknessProfile; return user_id.

    Mirrors the helper in ``test_cloze.py`` — duplicated here so this
    test file stays independently runnable.
    """
    from app import crud, models
    from app.passwords import hash_password

    user = models.User(email="ada@example.com", password_hash=hash_password("supersecret"))
    session.add(user)
    session.flush()
    if axes is not None:
        crud.upsert_weakness_profile(session, user.id, axes)
    session.commit()
    return user.id


# ===========================================================================
# 2. Route layer — auth gate
# ===========================================================================


def test_get_exercises_due_requires_auth(client) -> None:
    """``GET /exercises/due`` with no cookie → 401 (auth gate)."""
    client.cookies.clear()
    resp = client.get("/exercises/due")
    assert resp.status_code == 401


# ===========================================================================
# 3. Route layer — empty / first-encounter / FSRS-driven branches
# ===========================================================================


def test_get_exercises_due_returns_204_when_no_cards_and_no_words(
    client, db_session
) -> None:
    """Empty test DB: no due rows AND no fresh words → 204.

    The route's safety floor: when the corpus is truly empty (or every
    word has been seen), the user gets an honest 204 instead of a fake
    exercise. The frontend surfaces an "All caught up" empty state.
    """
    _signup(client)
    # No words seeded. Both branches fall through to 204.
    resp = client.get("/exercises/due")
    assert resp.status_code == 204


def test_get_exercises_due_returns_204_when_all_words_have_cards(
    client, db_session
) -> None:
    """Every corpus word already has an fsrs_cards row, none due → 204.

    This is the "user has graded everything; nothing's due right now"
    case. The route's Branch 1 finds no due rows; Branch 2 finds no
    fresh words because every word has a card. 204 is the honest
    response — the user has nothing to study.
    """
    _signup(client)

    w1 = _seed_word(db_session, word="Hund", word_type="Noun")
    w2 = _seed_word(db_session, word="Katze", word_type="Noun")

    # Both words have fsrs_cards rows that are NOT due (due_date in
    # the future). Branch 1 falls through; Branch 2 finds no fresh
    # words.
    future = datetime.utcnow() + timedelta(days=10)
    _seed_fsrs_card(db_session, word_id=w1, due_date=future)
    _seed_fsrs_card(db_session, word_id=w2, due_date=future)

    resp = client.get("/exercises/due")
    assert resp.status_code == 204


def test_get_exercises_due_returns_fsrs_driven_card_with_due_from_fsrs_true(
    client, db_session, monkeypatch
) -> None:
    """User with one due card → 200 + cloze + ``due_from_fsrs=True``.

    Same body as the version above; the pytest fixture is wired in
    correctly here.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Hund", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=word_id,
        due_date=datetime.utcnow() - timedelta(minutes=5),
        state=1,
    )
    _mp(monkeypatch, word_id)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["due_from_fsrs"] is True
    assert body["answer_word_id"] == word_id
    assert body["prompt_template_version"] == "cloze-v1"
    # The 3 distractors from the stub payload.
    assert body["distractors"] == [word_id + 1, word_id + 2, word_id + 3]


def test_get_exercises_due_returns_learning_state_card(
    client, db_session, monkeypatch
) -> None:
    """User with one Learning-state card due immediately → 200 + due_from_fsrs=True.

    The card's ``state=1`` (py-fsrs ``State.Learning``); ``due_date`` is
    in the past so the route's ``<= now()`` filter picks it. The
    response's ``answer_word_id`` matches the Learning row's
    ``word_id`` (asserted directly so a future change in branch
    selection surfaces immediately).
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Katze", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=word_id,
        due_date=datetime.utcnow() - timedelta(seconds=30),
        state=1,
    )
    _mp(monkeypatch, word_id)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["due_from_fsrs"] is True
    assert body["answer_word_id"] == word_id


def test_get_exercises_due_picks_earliest_due_date(
    client, db_session, monkeypatch
) -> None:
    """User with three due cards → 200, returns the earliest due_date.

    The route's Branch 1 ordering is ``ORDER BY due_date ASC, id ASC``;
    we seed three rows with distinct due_dates and assert the picked
    word_id is the one with the smallest due_date.
    """
    _signup(client)

    earliest = _seed_word(db_session, word="Hund", word_type="Noun")
    middle = _seed_word(db_session, word="Katze", word_type="Noun")
    latest = _seed_word(db_session, word="Maus", word_type="Noun")

    now = datetime.utcnow()
    # ``latest`` is due before ``middle`` is due before ``earliest``.
    # Wait — we want EARLIEST due_date to win. So "earliest" should
    # have the SMALLEST due_date. The semantics are "pick the card
    # that's been waiting longest", which is the SMALLEST due_date.
    _seed_fsrs_card(db_session, word_id=earliest, due_date=now - timedelta(days=10), state=2)
    _seed_fsrs_card(db_session, word_id=middle, due_date=now - timedelta(days=5), state=2)
    _seed_fsrs_card(db_session, word_id=latest, due_date=now - timedelta(days=1), state=2)

    _mp(monkeypatch, earliest)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["due_from_fsrs"] is True
    assert body["answer_word_id"] == earliest


def test_get_exercises_due_first_encounter_creates_learning_row(
    client, db_session, monkeypatch
) -> None:
    """Word with no fsrs_cards row → fresh Learning row + due_from_fsrs=False.

    The first-encounter branch: no card is due, but a corpus word
    exists with no fsrs_cards row. The route picks the lowest-id such
    word, creates a Learning row inline, and returns
    ``due_from_fsrs=False``. After the call, ``fsrs_cards`` has a new
    row for the picked word.
    """
    from app import models

    _signup(client)
    word_id = _seed_word(db_session, word="Vogel", word_type="Noun")

    # Pre-condition: the word has no fsrs_cards row.
    assert (
        db_session.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
        is None
    )

    _mp(monkeypatch, word_id)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["due_from_fsrs"] is False
    assert body["answer_word_id"] == word_id

    # Post-condition: the inline-created Learning row is on disk.
    # Use a fresh query (the route's session is closed by now).
    fresh_card = (
        db_session.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
    )
    assert fresh_card is not None
    assert fresh_card.state == 1  # Learning
    assert fresh_card.reps == 0
    assert fresh_card.lapses == 0


def test_get_exercises_due_first_encounter_skips_already_seen_words(
    client, db_session, monkeypatch
) -> None:
    """First-encounter branch picks a word that has NO fsrs_cards row yet.

    We seed two words: one already has an fsrs_cards row (already seen
    by the user — not a candidate for first-encounter), one doesn't.
    The route's Branch 2 subquery ``NOT IN (SELECT word_id FROM
    fsrs_cards)`` must skip the first word and pick the second.
    """
    _signup(client)

    seen_id = _seed_word(db_session, word="Hund", word_type="Noun")
    fresh_id = _seed_word(db_session, word="Katze", word_type="Noun")

    # The seen word has an fsrs_cards row that's NOT due (future) —
    # so Branch 1 falls through, Branch 2's subquery filters it out,
    # and the picked word is the fresh one.
    _seed_fsrs_card(
        db_session,
        word_id=seen_id,
        due_date=datetime.utcnow() + timedelta(days=10),
    )

    _mp(monkeypatch, fresh_id)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["due_from_fsrs"] is False
    assert body["answer_word_id"] == fresh_id


# ===========================================================================
# 4. Gotcha #1 regression: ``force_word_id`` threading
# ===========================================================================


def test_force_word_id_threads_through_to_generated_cloze(
    client, db_session, monkeypatch
) -> None:
    """The closed-loop safety net: the cloze's ``answer_word_id`` MUST
    equal the ``fsrs_cards`` row's ``word_id``.

    If a future maintainer accidentally drops the ``force_word_id``
    kwarg from the route's call to ``generate_cloze``, this test
    catches it: the stub payload is keyed to ``picked_word_id``, so
    the cloze's ``answer_word_id`` matches; if the route's selection
    instead runs the deterministic seed path, the picked word would
    differ from the due-row word and the assertion would fail (or the
    stub would return a payload keyed to the wrong word).
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Hund", word_type="Noun")
    _seed_fsrs_card(
        db_session,
        word_id=word_id,
        due_date=datetime.utcnow() - timedelta(minutes=1),
        state=1,
    )
    _mp(monkeypatch, word_id)

    resp = client.get("/exercises/due")
    assert resp.status_code == 200
    body = resp.json()
    # The exact-word thread: cloze.answer_word_id == fsrs_cards.word_id.
    assert body["answer_word_id"] == word_id, (
        f"force_word_id did not thread through: cloze "
        f"answer_word_id={body['answer_word_id']!r} != expected "
        f"word_id={word_id}"
    )


# ===========================================================================
# 5. Backwards-compatibility — Phase 4.5 ``POST /exercises/cloze`` stays
#    on the deterministic-by-seed path. ``force_word_id`` is NOT used.
# ===========================================================================


def test_force_word_id_only_at_due_route_not_cloze_route(
    client, db_session, monkeypatch
) -> None:
    """``POST /exercises/cloze`` does NOT call ``force_word_id``.

    The card acceptance criterion is encoded here at the
    ``git grep -n`` level: the parameter must be wired only at the
    ``/exercises/due`` call site, never at ``/exercises/cloze``. We
    verify by code-grepping the source — the deterministic seed path
    must not mention ``force_word_id`` in ``main.py`` outside the
    new route.
    """
    import subprocess

    result = subprocess.run(
        ["grep", "-rn", "force_word_id", "app/", "../app/"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    matches = [
        line
        for line in result.stdout.splitlines()
        # Filter out the route's own legitimate use + the
        # ``cloze.py`` keyword-only parameter + docstring mentions.
        if "exercises/due" in line.lower()
        or "/due" in line.lower()
        or "force_word_id=" in line
        or "select_target_word" in line
        or "generate_cloze" in line
    ]
    # Every match should reference either the new route, the
    # select_target_word / generate_cloze signatures, or the
    # docstrings/comments documenting the parameter. None should be
    # inside ``/exercises/cloze``.
    assert all(
        "/exercises/cloze" not in m and "generate_cloze_exercise" not in m
        for m in matches
    ), (
        "force_word_id appears at the /exercises/cloze call site — "
        f"the deterministic-by-seed path is broken. Matches: {matches}"
    )


# ===========================================================================
# 6. Schema regression — ``ClozeDueExerciseOut`` exposes the same fields
#    as ``ClozeExerciseOut`` plus ``due_from_fsrs``.
# ===========================================================================


def test_cloze_due_exercise_out_schema_is_superset() -> None:
    """``ClozeDueExerciseOut`` is ``ClozeExerciseOut`` + ``due_from_fsrs``.

    Pydantic v2 subclassing — verify the merge at the schema level so a
    future maintainer who accidentally drops the inheritance surfaces it
    here.
    """
    from app.schemas import ClozeDueExerciseOut, ClozeExerciseOut

    base_fields = set(ClozeExerciseOut.model_fields.keys())
    subclass_fields = set(ClozeDueExerciseOut.model_fields.keys())
    assert subclass_fields == base_fields | {"due_from_fsrs"}
    assert "due_from_fsrs" in subclass_fields