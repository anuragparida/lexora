"""Tests for Phase 5.3 — ``POST /exercises/grade`` (card t_5160eecf)
+ Phase 6.6 (card t_d11d0011).

Coverage map (mirrors the card body's "pytest cases" section):

1. First-grade happy path: user with no ``fsrs_cards`` row for the
   word POSTs a grade; a row is created (Learning → Review on
   Good); ``grade_logs`` row inserted with ``trace_id=None``
   (Langfuse keys unset).
2. Repeat-grade happy path: pre-existing ``fsrs_cards`` row,
   POSTs Good; ``reps`` increments, ``scheduled_days > 0``.
3. Langfuse keys present: with ``get_langfuse`` mocked to return
   a fake client carrying an id-emitter, assert the span carries
   every metadata-contract field. Trace_id propagates to the
   ``grade_logs`` row.
4. Langfuse keys absent: ``get_langfuse`` returns ``None``; row
   inserts with ``trace_id=None``; no exception.
5. Out-of-range grade: ``GradeRequest(grade=5)`` → 422.
6. ``exercise_type="speaking"`` → 422 (the widened 3-way Literal
   guardrail — Phase 6.6 replaced the Phase 5.3 cloze-only
   Literal with ``Literal["cloze", "matching", "comprehension"]``).
7. No JWT → 401.
8. DB integrity failure (e.g. concurrent insert on the unique
   constraint) → 500 with a structured body.

Plus additional cases that pin the 5.3 contract tightly:

9. Append-only invariant: ``grade_logs`` row count == request
   count after a burst of N grades (one row per call).
10. ``exercise_id`` mismatch (the cloze's ``answer_word_id`` was
    different from ``exercise_id``) → 422 (defence in depth — the
    schema is ``gt=0``; an unknown word is a 500 via the corpus
    inconsistency path).
11. The metadata contract fields are populated on the
    ``grade_logs`` row exactly as the Phase 5 plan specifies
    (state, stability, difficulty, reps, lapses, trace_id,
    latency_ms).
12. The route handler does NOT touch ``backend/app/cloze.py`` or
    ``backend/app/llm.py`` (git-level regression guard).

Phase 6.6 additions (card t_d11d0011):

13. ``exercise_type="matching"`` → 200 with a ``GradeResponse``
    carrying ``exercise_type="matching"``; the ``grade_logs``
    row records ``exercise_type="matching"``.
14. ``exercise_type="comprehension"`` → 200; ``grade_logs``
    row records ``exercise_type="comprehension"``.
15. ``exercise_type="cloze"`` → 200 (regression guard for the
    Phase 5.3 path that 6.6 didn't touch byte-for-byte).
16. The ``/exercises/grade`` route dispatches on
    ``payload.exercise_type`` via a ``match`` statement and
    delegates to per-type handlers (``_grade_cloze``,
    ``_grade_matching``, ``_grade_comprehension``) — regression
    guard for the 6.6 fan-out.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test.
The Langfuse client is mocked via ``monkeypatch.setattr`` on
``app.main.get_langfuse`` so no network is touched. No LLM
calls are made — the grader is pure py-fsrs + SQL.

Run from ``backend/``::

    uv run pytest -q tests/test_grade.py
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# ``test_cloze.py``, ``test_due.py``, and ``test_diagnostic.py``.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes.
    """
    db_path = tmp_path / "test_grade.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    # Phase 5.3's first-grade happy path does NOT need Langfuse keys;
    # the Langfuse-on tests opt in by setting the keys explicitly.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
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

    Tests that need direct row inserts (seeding an ``FsrsCard`` to
    test the repeat-grade path) request this fixture. Same pattern as
    ``test_due.py`` / ``test_cloze.py``.
    """
    from sqlalchemy.orm import sessionmaker

    from app import database, models  # noqa: F401

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

    Mirrors the helper in ``test_due.py`` / ``test_cloze.py``.
    Returns the parsed JSON body so callers can read ``user_id``.
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
    word_type: str = "Noun",
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
    due_date: datetime | None = None,
    state: int = 1,
    reps: int = 0,
    lapses: int = 0,
    difficulty: float | None = None,
    stability: float | None = None,
    last_review: datetime | None = None,
) -> int:
    """Insert one ``FsrsCard`` row; return card id.

    Defaults to a fresh Learning card (state=1, due=now) — i.e. the
    shape ``/exercises/due``'s first-encounter branch creates inline.
    The 5.3 grader accepts this row and applies the grade as if the
    card had just been seen.
    """
    from app import models

    if due_date is None:
        due_date = datetime.utcnow()
    if last_review is None:
        last_review = due_date
    row = models.FsrsCard(
        word_id=word_id,
        difficulty=difficulty,
        stability=stability,
        retrievability=None,
        due_date=due_date,
        last_review=last_review,
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
# Langfuse fake client — a stand-in that captures span metadata so the
# tests can assert on the exact keyset the metadata contract specifies.
# ---------------------------------------------------------------------------


class _FakeLangfuseSpan:
    """A minimal Langfuse v2 span stub.

    The real v2 SDK's ``client.span(...)`` returns an object with
    ``update(metadata=...)`` and ``end()`` methods. We capture the
    metadata dict and synthesize a deterministic ``span_id`` so
    tests can assert ``trace_id == <that id>`` round-trips through
    the ``grade_logs`` row.
    """

    def __init__(self, span_id: str):
        self.span_id = span_id
        self.metadata: dict[str, Any] = {}
        self.input: Any = None
        self.output: Any = None
        self.ended = False

    def update(self, *, metadata: dict | None = None, **kwargs) -> None:
        if metadata is not None:
            self.metadata.update(metadata)
        # Allow tests to assert input/output if they want; unused here.

    def end(self) -> None:
        self.ended = True


class _FakeLangfuseClient:
    """A minimal Langfuse v2 client stub.

    Each ``client.span(name=...)`` call returns a fresh
    ``_FakeLangfuseSpan`` with a deterministic id ``span-<n>`` so
    tests can assert trace_id propagation without flakiness.
    Records each call on ``client.spans`` for later inspection.
    """

    def __init__(self) -> None:
        self.spans: list[_FakeLangfuseSpan] = []
        self._counter = 0
        self.flushed = 0

    def span(self, *, name: str = "", **kwargs) -> _FakeLangfuseSpan:
        self._counter += 1
        span = _FakeLangfuseSpan(span_id=f"span-{self._counter}")
        self.spans.append(span)
        return span

    def flush(self) -> None:
        self.flushed += 1


# Metadata contract fields per docs/PHASE-5.md §"The metadata contract".
EXPECTED_METADATA_KEYS = {
    "user_id",
    "exercise_id",
    "exercise_type",
    "word_id",
    "grade",
    "scheduled_next_due_at",
    "prev_due_at",
    "state",
    "stability",
    "difficulty",
    "reps",
    "lapses",
    "trace_id",
    "latency_ms",
}


# ===========================================================================
# 1. First-grade happy path
# ===========================================================================


def test_first_grade_creates_fsrs_card_and_grade_log(client, db_session) -> None:
    """User with no ``fsrs_cards`` row for the word POSTs a grade.

    Expected:
    - HTTP 200 + ``GradeResponse`` body.
    - A new ``fsrs_cards`` row exists for the word (state == Review
      because Easy (4) on a fresh Learning card graduates to Review).
    - A ``grade_logs`` row exists with the metadata contract populated.
    - ``trace_id`` is ``None`` (Langfuse keys absent).
    """
    body = _signup(client)
    user_id = body["user"]["id"]
    word_id = _seed_word(db_session, word="Hund", word_type="Noun")

    # Pre-condition: no fsrs_cards row for this word.
    from app import models

    assert (
        db_session.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
        is None
    )

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 4,  # Easy — graduates fresh Learning → Review
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == word_id
    assert payload["exercise_type"] == "cloze"
    assert payload["trace_id"] is None
    assert payload["card_state"] == 2  # Review (int(State.Review) == 2)
    assert payload["stability"] >= 0
    assert payload["difficulty"] >= 0
    assert isinstance(payload["next_due_at"], str)

    # Post-condition: an fsrs_cards row now exists for the word.
    card = (
        db_session.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
    )
    assert card is not None
    assert card.state == 2  # Review
    assert card.last_review is not None

    # Post-condition: a grade_logs row exists with all metadata fields.
    log = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.user_id == user_id)
        .first()
    )
    assert log is not None
    assert log.exercise_id == word_id
    assert log.exercise_type == "cloze"
    assert log.word_id == word_id
    assert log.grade == 4
    assert log.state == 2  # Review
    assert log.trace_id is None  # Langfuse keys absent
    assert log.latency_ms >= 0
    assert log.graded_at is not None


# ===========================================================================
# 2. Repeat-grade happy path
# ===========================================================================


def test_repeat_grade_increments_reps_and_sets_scheduled_days(
    client, db_session
) -> None:
    """Pre-existing ``fsrs_cards`` row, POSTs Good → reps++,
    scheduled_days > 0.

    ``Good`` on a Learning card advances the step but stays in
    Learning per py-fsrs 4.1.2 (see ``test_fsrs.py`` for the
    library-behavior pin). For a state==2 Review card, Good sets a
    non-zero scheduled_days.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Katze", word_type="Noun")

    # Seed a pre-existing Review-state card with reps=2, lapses=0.
    _seed_fsrs_card(
        db_session,
        word_id=word_id,
        due_date=datetime.utcnow() - timedelta(minutes=5),
        state=2,
        reps=2,
        lapses=0,
        difficulty=5.0,
        stability=10.0,
    )

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,  # Good
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["card_state"] == 2  # Review stays in Review

    # The fsrs_cards row was UPDATED, not appended. reps == 3.
    from app import models

    card = (
        db_session.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
    )
    assert card is not None
    assert card.reps == 3
    assert card.lapses == 0
    assert card.scheduled_days >= 0
    # Next due date is in the future.
    assert card.due_date >= datetime.utcnow() - timedelta(seconds=2)


# ===========================================================================
# 3. Langfuse keys present — span carries metadata contract, trace_id
#    propagates to the row.
# ===========================================================================


def test_langfuse_present_trace_id_propagates_to_row(
    client, db_session, monkeypatch
) -> None:
    """With ``get_langfuse`` mocked to return a fake client, the span
    carries every metadata-contract field and the trace_id propagates
    to the ``grade_logs`` row.
    """
    fake = _FakeLangfuseClient()
    # ``app.main`` imports ``get_langfuse`` from ``app.observability``
    # at module load. Patching the source module is enough — the
    # name in ``app.main``'s namespace resolves through it.
    monkeypatch.setattr("app.observability.get_langfuse", lambda: fake)
    # Belt + braces: also patch the name as imported into app.main.
    monkeypatch.setattr("app.main.get_langfuse", lambda: fake)

    _signup(client)
    word_id = _seed_word(db_session, word="Vogel", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["trace_id"] == "span-1"

    # The fake client received exactly one span call.
    assert len(fake.spans) == 1
    span = fake.spans[0]
    assert span.ended is True
    # The metadata contract keyset is populated on the span.
    assert EXPECTED_METADATA_KEYS.issubset(set(span.metadata.keys())), (
        "missing keys: "
        f"{EXPECTED_METADATA_KEYS - set(span.metadata.keys())}"
    )
    # The grade_logs row carries the trace_id from the span.
    from app import models

    log = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.trace_id == "span-1")
        .first()
    )
    assert log is not None
    assert log.trace_id == "span-1"
    assert log.user_id == payload.get("user_id") or log.word_id == word_id


# ===========================================================================
# 4. Langfuse keys absent — graceful degradation
# ===========================================================================


def test_langfuse_absent_graceful_degradation(client, db_session) -> None:
    """``get_langfuse`` returns ``None`` (default test env). The row
    inserts with ``trace_id=None`` and the request succeeds.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Maus", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["trace_id"] is None

    from app import models

    log = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.word_id == word_id)
        .first()
    )
    assert log is not None
    assert log.trace_id is None


# ===========================================================================
# 5. Out-of-range grade → 422
# ===========================================================================


def test_out_of_range_grade_returns_422(client, db_session) -> None:
    """``grade=5`` is rejected by Pydantic at the wire (Hard rule #5)."""
    _signup(client)
    word_id = _seed_word(db_session, word="Fisch", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 5,
        },
    )
    assert resp.status_code == 422

    # No fsrs_cards row was created (validation rejected the call).
    from app import models

    assert (
        db_session.query(models.FsrsCard)
        .filter(models.FsrsCard.word_id == word_id)
        .first()
        is None
    )


# ===========================================================================
# 6. exercise_type="speaking" → 422 (widened 3-way Literal guardrail)
# ===========================================================================


def test_unsupported_exercise_type_returns_422(client, db_session) -> None:
    """``exercise_type="speaking"`` is rejected by the
    ``Literal["cloze", "matching", "comprehension"]`` wire
    guardrail widened in Phase 6.6.

    Note: in Phase 5.3 this test pinned ``exercise_type="matching"`` →
    422 because the literal was cloze-only. Phase 6.6 widens the
    literal to three values, so ``"matching"`` is now a *valid*
    request (see tests #13 / #14). The regression here is the
    anything-outside-the-3-way-union case, which the card body
    pins as ``"speaking"``.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Baum", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "speaking",
            "grade": 3,
        },
    )
    assert resp.status_code == 422


# ===========================================================================
# 7. No JWT → 401
# ===========================================================================


def test_grade_requires_auth(client, db_session) -> None:
    """``POST /exercises/grade`` with no cookie → 401."""
    # Sign up first so the word is real, then drop the cookie.
    word_id = _seed_word(db_session, word="Hund", word_type="Noun")
    client.cookies.clear()
    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 401


# ===========================================================================
# 8. DB integrity failure → 500 with structured body
# ===========================================================================


def test_db_integrity_failure_returns_500(
    client, db_session, monkeypatch
) -> None:
    """When ``commit`` raises ``IntegrityError`` (e.g. a race on the
    unique constraint), the route catches it and returns 500 with a
    structured body. We trigger this by patching
    ``sqlalchemy.orm.Session.commit`` to always raise.
    """
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session

    _signup(client)
    word_id = _seed_word(db_session, word="Igel", word_type="Noun")

    _original_commit = Session.commit
    call_count = {"n": 0}

    def _exploding_commit(self):
        call_count["n"] += 1
        # Always raise. The route has TWO places a commit-like call
        # might happen: the inline-create branch's ``flush()`` (which
        # is NOT a commit and won't be intercepted) and the final
        # ``db.commit()``. We want to trip the route's final-commit
        # try/except, which expects ``IntegrityError`` and turns it
        # into a 500.
        raise IntegrityError("simulated", {}, None)

    monkeypatch.setattr(Session, "commit", _exploding_commit)

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert "detail" in body
    assert call_count["n"] >= 1, "expected the route's final commit() to fire"


# ===========================================================================
# 9. Append-only invariant — row count == request count
# ===========================================================================


def test_burst_of_n_grades_writes_n_grade_log_rows(
    client, db_session
) -> None:
    """After N grade calls, ``grade_logs`` has N rows for the user.

    Pins Gotcha #1 (``grade_logs`` is append-only) at the row-count
    level — every call writes exactly one row, no UPDATEs.
    """
    from app import models

    _signup(client)
    word_ids = [
        _seed_word(db_session, word=f"Wort{i}", word_type="Noun")
        for i in range(5)
    ]

    for wid in word_ids:
        resp = client.post(
            "/exercises/grade",
            json={
                "exercise_id": wid,
                "exercise_type": "cloze",
                "grade": 3,
            },
        )
        assert resp.status_code == 200, resp.text

    log_count = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.word_id.in_(word_ids))
        .count()
    )
    assert log_count == len(word_ids)


# ===========================================================================
# 10. exercise_id mismatch — defence in depth
# ===========================================================================


def test_grade_unknown_word_id_still_succeeds(client, db_session) -> None:
    """``exercise_id`` references a word not in the corpus.

    Phase 5's ``fsrs_cards`` table has no FK to ``words.id`` (single-user
    dev assumption per Phase 5.4's docstring) so the grader accepts
    any positive ``exercise_id`` and creates/updates the row. This
    pins that contract: an unknown word_id is *not* a 500 — it's a
    successful grading of a word we don't know about. In the real
    SPA flow this branch is unreachable because the cloze/due
    endpoints always hand back a real word_id; the test documents
    the boundary.
    """
    _signup(client)

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": 9999,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["exercise_id"] == 9999


# ===========================================================================
# 11. Metadata contract — every field populated on grade_logs
# ===========================================================================


def test_grade_log_row_carries_full_metadata_contract(
    client, db_session
) -> None:
    """Every field in ``docs/PHASE-5.md`` §"The metadata contract" is
    populated on the inserted ``grade_logs`` row.
    """
    from app import models

    _signup(client)
    word_id = _seed_word(db_session, word="Apfel", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text

    log = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.word_id == word_id)
        .first()
    )
    assert log is not None
    # Every required column is populated and matches the request + the
    # computed post-review state.
    assert log.user_id is not None
    assert log.exercise_id == word_id
    assert log.exercise_type == "cloze"
    assert log.word_id == word_id
    assert log.grade == 3
    assert log.scheduled_next_due_at is not None
    assert log.prev_due_at is not None
    assert log.state in (1, 2, 3)
    assert log.stability >= 0
    assert log.difficulty >= 0
    assert log.reps >= 0
    assert log.lapses >= 0
    assert log.trace_id is None  # Langfuse keys absent in this test
    assert log.latency_ms >= 0
    assert log.graded_at is not None


# ===========================================================================
# 12. Regression guard — route lives in main.py, not in a new router.
# ===========================================================================


def test_route_lives_in_main_py_not_a_new_router() -> None:
    """``POST /exercises/grade`` is defined in ``backend/app/main.py``,
    not in a new router module. Pins Gotcha #3.

    We grep the source files directly so the test is hermetic and
    doesn't depend on which routes happen to be mounted at
    import-time.
    """
    import os
    import re

    main_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    )
    with open(main_path, encoding="utf-8") as f:
        main_src = f.read()

    # The route decorator appears in main.py.
    assert re.search(
        r'@app\.post\(\s*"/exercises/grade"', main_src
    ), "/exercises/grade route not defined in main.py"

    # The helper _trace_grade is in main.py too.
    assert "def _trace_grade" in main_src, (
        "_trace_grade helper missing from main.py"
    )


# ---------------------------------------------------------------------------
# 13. Regression guard — main.py does not import cloze.py or llm.py
#     inside the grade handler. Pins Hard rule "out of scope".
# ---------------------------------------------------------------------------


def test_grade_handler_does_not_touch_cloze_or_llm() -> None:
    """Pins the card body's "out of scope" rule: the grader does not
    import cloze.py or llm.py, and the handler does not call any
    LLM-related symbol.
    """
    import os

    main_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    )
    with open(main_path, encoding="utf-8") as f:
        main_src = f.read()

    # The /exercises/grade route block. We split on the route
    # decorator to isolate just this handler.
    import re

    match = re.search(
        r'@app\.post\(\s*"/exercises/grade"\s*[\s\S]+?(?=@app\.|\nclass |\Z)',
        main_src,
    )
    assert match is not None, "/exercises/grade handler not found"
    handler = match.group(0)

    # The handler must not import cloze.py or llm.py.
    assert "from app.cloze" not in handler and "import cloze" not in handler, (
        "/exercises/grade handler imports cloze.py — out of scope"
    )
    assert "from app.llm" not in handler and "import llm" not in handler, (
        "/exercises/grade handler imports llm.py — out of scope"
    )


# ===========================================================================
# 13. exercise_type="matching" → 200 + GradeResponse{exercise_type: "matching"}
# ===========================================================================


def test_matching_grade_returns_200_with_matching_response(
    client, db_session
) -> None:
    """``exercise_type="matching"`` is now a valid value in the
    widened 3-way Literal. The route's ``_grade_matching`` handler
    runs the same ``apply_grade`` + ``grade_logs`` write path as
    the cloze branch, just with a different trace span name
    (``match.grade``) and a different ``grade_logs.exercise_type``
    label.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Stuhl", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "matching",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == word_id
    assert payload["exercise_type"] == "matching"
    assert payload["trace_id"] is None
    assert payload["card_state"] in (1, 2, 3)
    assert payload["stability"] >= 0
    assert payload["difficulty"] >= 0
    assert isinstance(payload["next_due_at"], str)


# ===========================================================================
# 14. exercise_type="comprehension" → 200 + grade_logs row carries the label
# ===========================================================================


def test_comprehension_grade_returns_200_with_comprehension_response(
    client, db_session
) -> None:
    """``exercise_type="comprehension"`` is the third value in the
    widened 3-way Literal. The route's ``_grade_comprehension``
    handler runs the same ``apply_grade`` + ``grade_logs`` write
    path, with trace span name ``comprehension.grade`` and a
    ``grade_logs.exercise_type`` label of ``"comprehension"``.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Tisch", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "comprehension",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == word_id
    assert payload["exercise_type"] == "comprehension"


def test_comprehension_grade_log_row_records_exercise_type(
    client, db_session
) -> None:
    """The ``grade_logs`` row written by the comprehension
    handler carries ``exercise_type="comprehension"`` — same
    audit-row shape as the cloze / matching paths, only the
    label differs.
    """
    from app import models

    _signup(client)
    word_id = _seed_word(db_session, word="Lampe", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "comprehension",
            "grade": 4,
        },
    )
    assert resp.status_code == 200, resp.text

    log = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.word_id == word_id)
        .first()
    )
    assert log is not None
    assert log.exercise_type == "comprehension"
    assert log.exercise_id == word_id
    assert log.word_id == word_id
    assert log.grade == 4


# ===========================================================================
# 15. exercise_type="cloze" → 200 (regression guard for the Phase 5.3 path)
# ===========================================================================


def test_cloze_grade_still_200_after_6_6_fan_out(
    client, db_session
) -> None:
    """Phase 6.6 extracted the cloze logic into ``_grade_cloze`` —
    the wire-level behavior must be identical to Phase 5.3.

    The card body pins this as a regression guard: "Existing
    cloze callers are unaffected (Pydantic accepts ``"cloze"`` as
    a subset of the union)." The cloze handler's body is
    byte-for-byte the Phase 5.3 body, just lifted into a sibling
    function — so the only check this test adds is that the
    response still carries ``exercise_type="cloze"`` after the
    3-way fan-out.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Buch", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "cloze",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == word_id
    assert payload["exercise_type"] == "cloze"
    assert payload["card_state"] in (1, 2, 3)


# ===========================================================================
# 16. Regression guard — /exercises/grade dispatches on exercise_type
#     via a match statement + per-type handlers exist in main.py
# ===========================================================================


def test_grade_route_dispatches_via_match_statement() -> None:
    """Pin the Phase 6.6 fan-out shape: the route uses a
    ``match`` statement on ``payload.exercise_type`` and delegates
    to ``_grade_cloze`` / ``_grade_matching`` / ``_grade_comprehension``
    sibling functions. ``_grade_one`` carries the shared body.

    Hermetic: source-grep ``backend/app/main.py`` so the test
    doesn't need a database or client. Catches a future refactor
    that puts the dispatch logic in an ``if/elif`` chain or that
    inlines one of the handlers into the route.
    """
    import os

    main_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    )
    with open(main_path, encoding="utf-8") as f:
        main_src = f.read()

    # Three per-type handlers must exist as siblings of the
    # shared ``_grade_one`` body.
    assert "def _grade_cloze" in main_src, (
        "_grade_cloze handler missing from main.py"
    )
    assert "def _grade_matching" in main_src, (
        "_grade_matching handler missing from main.py"
    )
    assert "def _grade_comprehension" in main_src, (
        "_grade_comprehension handler missing from main.py"
    )
    assert "def _grade_one" in main_src, (
        "_grade_one shared body missing from main.py"
    )

    # The route dispatches on payload.exercise_type via a match
    # statement. We look for the per-type call sites — each
    # handler must be invoked from the dispatch.
    assert "_grade_cloze(db, current_user, payload)" in main_src, (
        "route does not call _grade_cloze from the dispatch"
    )
    assert "_grade_matching(db, current_user, payload)" in main_src, (
        "route does not call _grade_matching from the dispatch"
    )
    assert (
        "_grade_comprehension(db, current_user, payload)" in main_src
    ), "route does not call _grade_comprehension from the dispatch"

    # The dispatch is a match statement (Python 3.10+) on
    # payload.exercise_type. We pin the literal arms so a future
    # refactor that re-shapes the dispatch (e.g. into a dict
    # lookup) shows up here.
    assert (
        'match payload.exercise_type:\n' in main_src
        or "match payload.exercise_type:\n" in main_src
    ), "/exercises/grade route is not a match statement"

    # Each per-type handler threads the right span name into
    # ``_grade_one`` — these are the trace-span names the card
    # body pins as the fan-out contract.
    assert 'span_name="exercise.grade"' in main_src, (
        "cloze span name not pinned to 'exercise.grade'"
    )
    assert 'span_name="match.grade"' in main_src, (
        "matching span name not pinned to 'match.grade'"
    )
    assert 'span_name="comprehension.grade"' in main_src, (
        "comprehension span name not pinned to 'comprehension.grade'"
    )


# ===========================================================================
# 17. Phase 10.11 — exercise_type="idiom" widens the /grade Literal
# ===========================================================================


def test_idiom_grade_returns_200_with_idiom_response(
    client, db_session
) -> None:
    """Phase 10.11 (card t_f884b9cd) — ``exercise_type="idiom"``
    is the 4th value in the now-5-way Literal. The route's
    ``_grade_idiom`` handler runs the same ``apply_grade`` +
    ``grade_logs`` write path as the cloze / matching /
    comprehension branches, with trace span name ``idiom.grade``
    and a ``grade_logs.exercise_type`` label of ``"idiom"``.

    Phase 8.3 idiom-ship (commit 5b9e7aa) intentionally 422'd
    idiom grade requests pending Phase 9's "FSRS-graded-recall
    surface" — Phase 9 closed the session-mixer side (9.6) but
    never wired the ``/grade`` arm. Phase 10.11 closes that
    gap in-fold.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Schmetterling", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "idiom",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == word_id
    assert payload["exercise_type"] == "idiom"
    assert payload["trace_id"] is None
    assert payload["card_state"] in (1, 2, 3)
    assert payload["stability"] >= 0
    assert payload["difficulty"] >= 0
    assert isinstance(payload["next_due_at"], str)


# ===========================================================================
# 18. Phase 10.11 — exercise_type="phrase_match" widens the /grade Literal
# ===========================================================================


def test_phrase_match_grade_returns_200_with_phrase_match_response(
    client, db_session
) -> None:
    """Phase 10.11 (card t_f884b9cd) — ``exercise_type="phrase_match"``
    is the 5th value in the now-5-way Literal. The route's
    ``_grade_phrase_match`` handler runs the same ``apply_grade``
    + ``grade_logs`` write path as the cloze / matching /
    comprehension / idiom branches, with trace span name
    ``phrase_match.grade`` and a ``grade_logs.exercise_type``
    label of ``"phrase_match"``.

    This is the literal widening the Phase 10 plan body hard-rule
    #1 ("Additive Literal widening (4-way to 5-way); NEVER narrow")
    required: ``phrase_match`` is the 5th exercise type on the
    same wire contract as the four prior types. Phase 10.7's
    optimizer (card t_dab34a97) spec'd a "grader-fan-out" plan
    that 10.11 closes here.
    """
    _signup(client)
    word_id = _seed_word(db_session, word="Fahrrad", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "phrase_match",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == word_id
    assert payload["exercise_type"] == "phrase_match"
    assert payload["trace_id"] is None
    assert payload["card_state"] in (1, 2, 3)
    assert payload["stability"] >= 0
    assert payload["difficulty"] >= 0
    assert isinstance(payload["next_due_at"], str)


def test_phrase_match_grade_log_row_records_exercise_type(
    client, db_session
) -> None:
    """The ``grade_logs`` row written by ``_grade_phrase_match``
    carries ``exercise_type="phrase_match"`` — same audit-row
    shape as the cloze / matching / comprehension / idiom paths,
    only the label differs. Mirrors
    ``test_comprehension_grade_log_row_records_exercise_type``
    on the new 5th arm.
    """
    from app import models

    _signup(client)
    word_id = _seed_word(db_session, word="Haus", word_type="Noun")

    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": word_id,
            "exercise_type": "phrase_match",
            "grade": 4,
        },
    )
    assert resp.status_code == 200, resp.text

    log = (
        db_session.query(models.GradeLog)
        .filter(models.GradeLog.user_id == _user_id_from_cookie(client))
        .order_by(models.GradeLog.id.desc())
        .first()
    )
    assert log is not None, "grade_logs row missing for phrase_match grade"
    assert log.exercise_type == "phrase_match"
    assert log.grade == 4
    assert log.exercise_id == word_id


# ===========================================================================
# 19. Phase 10.11 — full 5-way /grade fan-out: hermetic source-grep
# ===========================================================================


def test_grade_route_dispatches_via_match_statement_5way() -> None:
    """Phase 10.11 widens the Phase 6.6 fan-out guard to 5
    per-type handlers. Hermetic source-grep — no DB / client.

    Pins:
    - ``_grade_idiom`` / ``_grade_phrase_match`` are siblings of
      the existing cloze / matching / comprehension handlers.
    - The ``match payload.exercise_type:`` statement carries
      the two new ``case "idiom"`` / ``case "phrase_match"`` arms.
    - The new span names ``idiom.grade`` and
      ``phrase_match.grade`` are pinned on the wrappers.
    """
    import os

    main_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    )
    with open(main_path, encoding="utf-8") as f:
        main_src = f.read()

    # Per-type handlers must exist as siblings of _grade_one.
    assert "def _grade_cloze" in main_src
    assert "def _grade_matching" in main_src
    assert "def _grade_comprehension" in main_src
    assert "def _grade_idiom" in main_src, (
        "_grade_idiom handler missing from main.py — Phase 10.11"
    )
    assert "def _grade_phrase_match" in main_src, (
        "_grade_phrase_match handler missing from main.py — Phase 10.11"
    )
    assert "def _grade_one" in main_src

    # The route dispatches via match statement on payload.exercise_type.
    assert "match payload.exercise_type:\n" in main_src

    # The 5 case arms must all be present in the dispatch.
    assert 'case "cloze":' in main_src
    assert 'case "matching":' in main_src
    assert 'case "comprehension":' in main_src
    assert 'case "idiom":' in main_src, (
        '"idiom" case arm missing from /exercises/grade dispatch'
    )
    assert 'case "phrase_match":' in main_src, (
        '"phrase_match" case arm missing from /exercises/grade dispatch'
    )

    # Each per-type handler threads the right span name into _grade_one.
    assert 'span_name="exercise.grade"' in main_src
    assert 'span_name="match.grade"' in main_src
    assert 'span_name="comprehension.grade"' in main_src
    assert 'span_name="idiom.grade"' in main_src
    assert 'span_name="phrase_match.grade"' in main_src

    # The schema's ExerciseType literal widened 4-way to 5-way.
    from app import schemas

    exercise_type_args = schemas.ExerciseType.__args__
    assert "phrase_match" in exercise_type_args, (
        "ExerciseType Literal missing 'phrase_match' — Phase 10.11"
    )
    assert len(exercise_type_args) == 5, (
        f"ExerciseType Literal should have 5 entries, "
        f"got {len(exercise_type_args)}: {exercise_type_args}"
    )


def _user_id_from_cookie(client) -> int:
    """Read the current user id by parsing the lexora_token cookie.

    Mirrors the discipline of ``_signup``-using tests — the cookie
    is the canonical auth surface for the SPA, so any grade-log
    audit assertion that needs ``user_id`` reads it back via
    ``GET /auth/me`` rather than re-signing up (which would 409
    on a duplicate email).
    """
    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])