"""Tests for Phase 9.7 -- 4-type study-session mixing (card t_587fa0ae).

This file ships two layers of coverage that the prior phases laid
the groundwork for but never closed the loop on:

1. **The end-to-end mixing flow** (the "Phase 9.6 mixer" path).
   Authenticate a user, seed one due card per exercise type
   (cloze, matching, comprehension, idiom), pick each via
   ``GET /exercises/due?type=<that-type>``, grade it via
   ``POST /exercises/grade`` with the matching ``exercise_type``
   literal, and verify the FSRS state machine advances (post-grade
   ``due_date`` moves into the future for that card).
   The route's pre-Phase-9 contract deliberately returns 422 for
   ``exercise_type="idiom"`` grade requests (the comment on
   ``grade_exercise`` documents this; FSRS-graded-recall for idiom
   is a Phase 9.1+ follow-up). The mixing flow therefore exercises
   the idiom card via the read-side (GET /exercises/due surfaces it
   with the X-Due-* headers; the user sees it on the union
   landing) but cannot grade it -- the idiom card stays in the
   queue while the other three are graded and rotated off. This
   matches the wire reality today; the test pins the contract so
   a future Phase 9.X that flips the idiom-grade switch has a
   tight signal to update.

2. **The two Phase 9.2 contract holes that the route-layer tests
   intentionally didn't cover.** ``test_due_queue_widened.py``
   pins the per-type filter, the 204 / header shape, the legacy
   pre-9.1 fallback, and the ``due_by_type`` 4-key payload.
   What it never exercised end-to-end is:

   a. **All-types filter (no filter / ``type=any``).** With cards
      of multiple types due, the route must still pick the
      earliest-due row regardless of type -- the cloze card wins
      when it's earliest, and the matching / comprehension / idiom
      card wins when it's earliest. The pure cloze or pure
      non-cloze cases from the prior file don't exercise this.

   b. **All-cards-due / "every type is due".** The widening to a
      4-key ``due_by_type`` dict with non-zero values across all
      four buckets. The Phase 9.6 mixer's first-login gate
      depends on this payload being real for all four keys; this
      file locks the realistic all-four-types-due seed.

   c. **The 422 on idiom grade.** Not a route-layer test in the
      Phase 5.3 / 6.6 sense (that's ``test_grade.py``'s job for
      the three gradeable types) -- but the contract that idiom
      is *intentionally* not a gradeable type today must lock the
      pre-Phase-9.1 idiom branch off so the mix doesn't fire a
      500 by accident. Pin it here alongside the mixer story so
      the next person to flip the idiom-grade switch knows
      exactly which test must move.

Hermetic: fresh per-test SQLite DB + JWT secret so ``app.auth``'s
import-time check passes. The OpenRouter chat-completions call is
stubbed via ``monkeypatch.setattr("app.cloze._openai_client", ...)``
because the cloze branch of ``/exercises/due`` is the only path
in this file that calls the LLM (the other three types render
via the 204+headers shape and never reach ``generate_cloze``).
Mirrors the per-test fixture pattern from ``test_due.py`` /
``test_due_queue_widened.py`` / ``test_grade.py``.

**Schema posture.** This file is written against the SAME schema
posture that ``test_due_queue_widened.py`` assumes: Phase 9.1
(card t_0bfdb7ed) hasn't folded ``fsrs_cards.exercise_type`` into
the SQLAlchemy mapped column on main yet, so each test seeds the
column via raw ``ALTER TABLE`` against the test SQLite DB. The
``_seed_fsrs_card`` helper sets ``exercise_type`` via raw SQL
after the INSERT to mirror the post-9.1 column posture without
depending on the Phase 9.1 model change. Without the column, the
route's ``inspect()`` probe returns False for ``exercise_type``
and the union filter is dropped -- every fsrs_cards row is
implicitly cloze by Phase 5.6 contract, which would silently
collapse our 4-type mix into 1-type cloze. The ``fsrs_with_type_column``
fixture is the single editing point that keeps this invariant.

Run from ``backend/``::

    uv run pytest -q tests/test_session_mixing.py
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Fixtures -- mirror the per-test SQLite + JWT secret pattern from
# test_due.py / test_grade.py / test_due_queue_widened.py. Per-test
# reconfigure of app.database so each test gets its own clean DB.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so app.auth's import-time
    check passes.
    """
    db_path = tmp_path / "test_session_mixing.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    # Phase 5.3 / 6.6 grade path: no Langfuse keys -> trace_id=None
    # on the grade_logs rows. Same convention as test_grade.py.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """``TestClient`` wired to a fresh per-test SQLite DB."""
    from app import database, models  # noqa: F401 -- register tables
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB.

    Tests that seed ``FsrsCard`` rows directly (the 4-type session
    mix) request this fixture.
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
    """Ensure ``fsrs_cards.exercise_type`` exists in the test DB.

    Phase 9.1 (card t_0bfdb7ed) adds the column to ``app.models``;
    the model-driven ``Base.metadata.create_all`` then creates it
    automatically. When 9.7's tests run in isolation (or before
    9.1 lands on main), the column is missing, so this fixture
    does an idempotent ``ALTER TABLE`` only if ``PRAGMA
    table_info(fsrs_cards)`` does not already list it.

    Every test in this file requests this fixture: the 4-type
    session mix is meaningless without the union column.
    """
    rows = db_session.execute(text("PRAGMA table_info(fsrs_cards)")).fetchall()
    column_names = {row[1] for row in rows}
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
    """Sign up a user; the JWT cookie is set as a side-effect.

    The password literal here is intentionally a non-secret-looking
    ``*** so the harness doesn't redact it (the harness replaces
    strings that *look* secret with literal ``*** placeholder bytes
    at write time, which then fails the 8-char ``min_length`` check
    on ``/auth/signup``). A neutral 14-char string avoids both
    pitfalls while staying hermetic per test (it never has to
    match a real account -- JWT cookies are issued in-memory).
    """
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "xxxplain-pw-123"},
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


def _seed_due_card(
    session,
    *,
    word_id: int,
    exercise_type: str,
    due_date: datetime,
    state: int = 1,
    reps: int = 0,
    lapses: int = 0,
    difficulty: float | None = None,
    stability: float | None = None,
) -> int:
    """Insert one ``FsrsCard`` row with the supplied ``exercise_type``.

    The SQLAlchemy model does not declare ``exercise_type`` (Phase 9.1
    hasn't landed on main), so the column is set via raw SQL after
    the INSERT -- the same workaround the per-type fixtures in
    ``test_due_queue_widened.py`` use.
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

    session.execute(
        text(
            "UPDATE fsrs_cards SET exercise_type = :etype WHERE id = :id"
        ),
        {"etype": exercise_type, "id": row.id},
    )
    session.commit()
    return row.id


def _seed_four_type_due_queue(session, *, base_now: datetime | None = None):
    """Seed the canonical 4-type due queue.

    One due card per exercise type -- cloze, matching,
    comprehension, idiom -- each on its own word. ``base_now``
    is the moment from which the relative ``due_date`` offsets
    are computed; defaulting to ``datetime.utcnow()`` makes the
    fixture hermetic but lets each call advance the wall-clock
    for deterministic ordering.

    Returns ``dict[exercise_type, word_id]``.
    """
    now = base_now or datetime.utcnow()
    cards = {}
    cards["cloze"] = _seed_word(session, word="Hund", word_type="Noun")
    cards["matching"] = _seed_word(session, word="Katze", word_type="Noun")
    cards["comprehension"] = _seed_word(
        session, word="Maus", word_type="Noun"
    )
    cards["idiom"] = _seed_word(session, word="Vogel", word_type="Noun")

    # Each type gets a distinct due_date so the route's
    # ``ORDER BY due_date ASC, id ASC`` ordering is unambiguous.
    # Idiom is the latest -- so after grading cloze (earliest),
    # matching, comprehension, the no-filter call surfaces idiom
    # last. Cloze is the earliest so it wins any no-filter tie.
    _seed_due_card(
        session,
        word_id=cards["cloze"],
        exercise_type="cloze",
        due_date=now - timedelta(minutes=20),
    )
    _seed_due_card(
        session,
        word_id=cards["matching"],
        exercise_type="matching",
        due_date=now - timedelta(minutes=15),
    )
    _seed_due_card(
        session,
        word_id=cards["comprehension"],
        exercise_type="comprehension",
        due_date=now - timedelta(minutes=10),
    )
    _seed_due_card(
        session,
        word_id=cards["idiom"],
        exercise_type="idiom",
        due_date=now - timedelta(minutes=5),
    )
    return cards


# ---------------------------------------------------------------------------
# Stub LLM client -- the cloze branch of /exercises/due is the only path
# in this file that calls the LLM (the other three types render via the
# 204+headers shape). Stub it via app.cloze._openai_client so the cloze
# pick doesn't reach OpenRouter.
# ---------------------------------------------------------------------------


def _stub_openai_client(payload: str) -> OpenAI:
    """Build a stub OpenAI client that returns ``payload`` as the
    assistant message content.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-session-mix-stub-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "qwen/qwen3-235b-a22b-2507",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": payload,
                        },
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
    """Patch ``app.cloze._openai_client`` to a stub yielding a valid
    ``ClozeExercise`` for ``word_id``.
    """
    payload = _cloze_payload_for(word_id)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _stub_openai_client(payload),
    )


# ===========================================================================
# Test 1 -- The 4-type due queue shows up in /auth/me.due_by_type with all
#           four buckets = 1.
#
# #2b "all-cards-due" from the card scope. The dict's keys + per-key
# counts are the Phase 9.6 mixer gate's input -- the route handler
# layer keeps this payload stable.
# ===========================================================================


def test_auth_me_due_by_type_all_four_buckets_present(
    client, db_session, fsrs_with_type_column
) -> None:
    """Seeding one card per type populates every bucket in ``due_by_type``.

    Phase 10.6 (card t_da43cc23) widens the closure dict from 4
    to 5 keys additively — ``phrase_match`` joins as the 5th
    bucket with the same zero-default behavior as the 4 prior
    types when no card is seeded for it. The test exercises the
    "all 4 prior buckets + 1 zero-default phrase_match" shape.
    """
    _signup(client)
    _seed_four_type_due_queue(db_session)

    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    counts = body["due_by_type"]

    assert set(counts.keys()) == {
        "cloze", "matching", "comprehension", "idiom", "phrase_match",
    }, counts
    assert counts == {
        "cloze": 1,
        "matching": 1,
        "comprehension": 1,
        "idiom": 1,
        "phrase_match": 0,
    }, counts


# ===========================================================================
# Test 2 -- Single-type filter surfaces each branch shape (cloze 200+body,
#           matching/comprehension/idiom 204+headers) without crossing over.
#
# This is the per-type filter contract exercised through a single seed,
# rather than spread across 3 tests. The 9.7 mix perspective is "after
# seeding 4 cards, I can ask for any one by type and get exactly that
# type back".
# ===========================================================================


def test_each_type_filter_surfaces_the_correct_branch_shape(
    client, db_session, fsrs_with_type_column, monkeypatch
) -> None:
    """``?type=<type>`` picks that card and renders the right shape.

    Cloze -> 200 + ``ClozeDueExerciseOut`` body (the cloze branch
    calls ``generate_cloze``).

    Matching / comprehension / idiom -> 204 + ``X-Due-*`` headers
    (the non-cloze union branch never reaches ``generate_cloze``).

    Hermetic via the cloze stub; the non-cloze branches don't
    need any LLM stubbing.
    """
    _signup(client)
    cards = _seed_four_type_due_queue(db_session)
    _stub_cloze_for(monkeypatch, word_id=cards["cloze"])

    # Cloze branch: 200 + body.
    cloze = client.get("/exercises/due?type=cloze")
    assert cloze.status_code == 200, cloze.text
    body = cloze.json()
    assert body["exercise_type"] == "cloze"
    assert body["answer_word_id"] == cards["cloze"]
    assert body["due_from_fsrs"] is True
    # No non-cloze headers on the cloze path.
    assert "X-Due-Exercise-Type" not in cloze.headers

    # Matching branch: 204 + headers.
    matching = client.get("/exercises/due?type=matching")
    assert matching.status_code == 204, matching.text
    assert matching.headers.get("X-Due-Exercise-Type") == "matching"
    assert matching.headers.get("X-Due-Word-Id") == str(cards["matching"])

    # Comprehension branch: 204 + headers.
    comprehension = client.get("/exercises/due?type=comprehension")
    assert comprehension.status_code == 204, comprehension.text
    assert (
        comprehension.headers.get("X-Due-Exercise-Type") == "comprehension"
    )
    assert (
        comprehension.headers.get("X-Due-Word-Id")
        == str(cards["comprehension"])
    )

    # Idiom branch: 204 + headers.
    idiom = client.get("/exercises/due?type=idiom")
    assert idiom.status_code == 204, idiom.text
    assert idiom.headers.get("X-Due-Exercise-Type") == "idiom"
    assert idiom.headers.get("X-Due-Word-Id") == str(cards["idiom"])


# ===========================================================================
# Test 3 -- The 4-card session E2E.
#
# Grade cloze + matching + comprehension in order. After each grade,
# the next ``GET /exercises/due?type=<that-type>`` returns 204 (the
# card is no longer due). After all three are graded, the no-filter
# call surfaces the only remaining due card (idiom) via the 204+headers
# branch -- proving the union surface ("all-types filter") picks the
# idiom card after the cloze / matching / comprehension types have all
# been rotated off.
#
# The idiom card is intentionally NOT graded: ``POST /exercises/grade``
# with ``exercise_type="idiom"`` is a 422 (the route comment is explicit
# about this -- FSRS-graded-recall for idiom is a Phase 9.1+ follow-up).
# The mix thus ends with ``due_by_type == {idiom: 1, others: 0}``.
# ===========================================================================


def test_full_4_type_session_mix_grades_three_leaves_idiom_pending(
    client, db_session, fsrs_with_type_column, monkeypatch
) -> None:
    """End-to-end study session: pick + grade 3 of 4 cards, idiom
    remains in the queue.

    Steps (each is a real HTTP call against TestClient):

    1. Sign up.
    2. Seed the 4-type queue.
    3. ``GET /auth/me`` -- ``due_by_type`` shows all four at 1.
    4. Pick + grade cloze. Verify the next ``?type=cloze`` is 204.
    5. Pick + grade matching. Verify the next ``?type=matching``
       is 204.
    6. Pick + grade comprehension. Verify the next
       ``?type=comprehension`` is 204.
    7. ``GET /exercises/due`` (no filter) -- idiom is the only
       remaining due card, so the no-filter union surface surfaces
       it via 204 + ``X-Due-Exercise-Type=idiom``. This is the
       "all-types filter" assertion from the card scope -- the
       union surface picks idiom when it's the only row left in
       the queue regardless of type.
    8. ``GET /auth/me`` -- ``due_by_type`` shows only idiom=1.
    """
    _signup(client)
    cards = _seed_four_type_due_queue(db_session)
    _stub_cloze_for(monkeypatch, word_id=cards["cloze"])

    # --- Step 3: /auth/me baseline ------------------------------
    me_initial = client.get("/auth/me")
    assert me_initial.status_code == 200, me_initial.text
    assert me_initial.json()["due_by_type"] == {
        "cloze": 1,
        "matching": 1,
        "comprehension": 1,
        "idiom": 1,
        "phrase_match": 0,
    }

    # --- Step 4: pick + grade cloze -----------------------------
    cloze_pick = client.get("/exercises/due?type=cloze")
    assert cloze_pick.status_code == 200, cloze_pick.text
    assert cloze_pick.json()["answer_word_id"] == cards["cloze"]

    cloze_grade = client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["cloze"],
            "exercise_type": "cloze",
            "grade": 3,  # "good"
        },
    )
    assert cloze_grade.status_code == 200, cloze_grade.text
    cloze_resp = cloze_grade.json()
    assert cloze_resp["exercise_type"] == "cloze"
    assert cloze_resp["exercise_id"] == cards["cloze"]
    assert cloze_resp["card_state"] in {1, 2, 3}
    # The next due must be in the future (FSRS state-machine
    # contract: "good" pushes the card past its current due_date).
    # FastAPI serialises datetime -> ISO string; parse for the
    # comparison.
    next_due_raw = cloze_resp["next_due_at"]
    next_due = datetime.fromisoformat(
        next_due_raw.replace("Z", "+00:00")
        if "T" in next_due_raw
        else next_due_raw
    )
    # Drop tzinfo (DB stores naive UTC) so we can compare against
    # ``datetime.utcnow()`` which is also naive.
    if next_due.tzinfo is not None:
        next_due = next_due.replace(tzinfo=None)
    assert next_due > datetime.utcnow() - timedelta(minutes=1), (
        f"cloze grade did not advance due_date: {next_due_raw!r}"
    )

    # The cloze branch is gone -- verify by re-asking for cloze.
    cloze_after = client.get("/exercises/due?type=cloze")
    assert cloze_after.status_code == 204, cloze_after.text
    assert "X-Due-Exercise-Type" not in cloze_after.headers

    # --- Step 5: pick + grade matching --------------------------
    matching_pick = client.get("/exercises/due?type=matching")
    assert matching_pick.status_code == 204, matching_pick.text
    assert matching_pick.headers["X-Due-Word-Id"] == str(
        cards["matching"]
    )

    matching_grade = client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["matching"],
            "exercise_type": "matching",
            "grade": 3,
        },
    )
    assert matching_grade.status_code == 200, matching_grade.text
    matching_resp = matching_grade.json()
    assert matching_resp["exercise_type"] == "matching"
    assert matching_resp["exercise_id"] == cards["matching"]
    matching_next_raw = matching_resp["next_due_at"]
    matching_next = datetime.fromisoformat(matching_next_raw)
    if matching_next.tzinfo is not None:
        matching_next = matching_next.replace(tzinfo=None)
    assert matching_next > datetime.utcnow() - timedelta(minutes=1)

    matching_after = client.get("/exercises/due?type=matching")
    assert matching_after.status_code == 204, matching_after.text
    assert "X-Due-Exercise-Type" not in matching_after.headers

    # --- Step 6: pick + grade comprehension ----------------------
    comp_pick = client.get("/exercises/due?type=comprehension")
    assert comp_pick.status_code == 204, comp_pick.text
    assert comp_pick.headers["X-Due-Word-Id"] == str(
        cards["comprehension"]
    )

    comp_grade = client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["comprehension"],
            "exercise_type": "comprehension",
            "grade": 3,
        },
    )
    assert comp_grade.status_code == 200, comp_grade.text
    comp_resp = comp_grade.json()
    assert comp_resp["exercise_type"] == "comprehension"
    assert comp_resp["exercise_id"] == cards["comprehension"]
    comp_next_raw = comp_resp["next_due_at"]
    comp_next = datetime.fromisoformat(comp_next_raw)
    if comp_next.tzinfo is not None:
        comp_next = comp_next.replace(tzinfo=None)
    assert comp_next > datetime.utcnow() - timedelta(minutes=1)

    comp_after = client.get("/exercises/due?type=comprehension")
    assert comp_after.status_code == 204, comp_after.text
    assert "X-Due-Exercise-Type" not in comp_after.headers

    # --- Step 7: no-filter union surface picks idiom -----------
    # After the three gradeable cards rotated off, idiom is the
    # only remaining due row. The no-filter branch must surface it
    # -- this is the "all-types filter" union behaviour card #2a
    # requires, exercised here with realistic state.
    any_pick = client.get("/exercises/due")
    assert any_pick.status_code == 204, any_pick.text
    assert any_pick.headers["X-Due-Exercise-Type"] == "idiom"
    assert any_pick.headers["X-Due-Word-Id"] == str(cards["idiom"])
    assert any_pick.headers["X-Due-Card-Id"].isdigit()

    # --- Step 8: /auth/me final state ----------------------------
    me_final = client.get("/auth/me")
    assert me_final.status_code == 200, me_final.text
    # Idiom is the only card still due. The mix sees exactly the
    # icon the Phase 9.6 mixer was built to dispatch.
    assert me_final.json()["due_by_type"] == {
        "cloze": 0,
        "matching": 0,
        "comprehension": 0,
        "idiom": 1,
        "phrase_match": 0,
    }


# ===========================================================================
# Test 4 -- ``POST /exercises/grade`` with ``exercise_type="idiom"`` is a 422.
#
# The route comment is explicit: idiom is intentionally not a gradeable
# type today (FSRS-graded-recall for idiom is a Phase 9.1+ follow-up; the
# match arms cover cloze / matching / comprehension only). The test pins
# this contract so the next person to flip the idiom-grade switch knows
# exactly which assertion must move from "422" to "200 + GradeResponse".
# ===========================================================================


def test_idiom_grade_now_200_after_phase_10_11_widening(
    client, db_session, fsrs_with_type_column
) -> None:
    """``exercise_type="idiom"`` on ``POST /exercises/grade`` returns
    a 200 (the idiom-grade branch is wired in Phase 10.11,
    card t_f884b9cd).

    Phase 10.11 closes the long-standing 422 that the Phase 8.3
    idiom-ship commit (5b9e7aa) intentionally left in place pending
    "Phase 9 adds the FSRS-graded-recall surface". Phase 9 closed
    the read side (Phase 9.6 mixer widening) but never wired the
    ``/grade`` arm. Phase 10.11 widens ``ExerciseType`` 4→5 and
    adds a ``case "idiom"`` arm that routes through ``_grade_idiom``
    (a thin wrapper around ``_grade_one``, mirroring
    ``_grade_matching`` / ``_grade_comprehension`` / ``_grade_phrase_match``).

    The mixer's 5-type flow surfaces idiom via the read side
    (X-Due-* headers) AND now grades it via the write side —
    same ``GradeResponse`` shape as the other four types.
    """
    _signup(client)
    cards = _seed_four_type_due_queue(db_session)

    # Sanity: the idiom card IS in the queue and discoverable via
    # the read-side contract.
    idiom_pick = client.get("/exercises/due?type=idiom")
    assert idiom_pick.status_code == 204, idiom_pick.text
    assert idiom_pick.headers["X-Due-Word-Id"] == str(cards["idiom"])

    # Phase 10.11 wires the idiom-grade branch — the contract is
    # a 200 with the same ``GradeResponse`` shape as cloze /
    # matching / comprehension / phrase_match. ``_grade_idiom``
    # is a thin wrapper around ``_grade_one``; the only per-type
    # difference is the trace span name (``idiom.grade``) and
    # the ``grade_logs.exercise_type`` label (``"idiom"``).
    resp = client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["idiom"],
            "exercise_type": "idiom",
            "grade": 3,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["graded"] is True
    assert payload["exercise_id"] == cards["idiom"]
    assert payload["exercise_type"] == "idiom"


# ===========================================================================
# Test 5 -- After grading all three gradeable cards, the no-filter
#           ``GET /exercises/due`` returns 204 (genuinely empty queue).
#
# Distinct from test #3 step 7 (which leaves idiom in the queue).
# This test grades the three gradeable types and never seeds an
# idiom card -- so the queue becomes truly empty and the no-filter
# route must return 204 with NO X-Due-* headers (the genuinely-empty
# branch, not the non-cloze branch).
# ===========================================================================


def test_no_filter_returns_truly_empty_after_all_gradeable_cards_off(
    client, db_session, fsrs_with_type_column, monkeypatch
) -> None:
    """No-filter GET /exercises/due on a drained queue -> 204, no
    headers.

    Drain = grade cloze + matching + comprehension. No idiom card
    was seeded. The route's genuinely-empty branch fires (Branch 2's
    no-fresh-words fallback), distinct from the non-cloze 204
    branch which carries X-Due-* headers. The two 204 codes are
    indistinguishable by status alone -- the header absence is the
    differentiator, and pinning it here keeps the wire shape tight.
    """
    from app import models  # noqa: F401 -- register tables

    _signup(client)
    cards = _seed_four_type_due_queue(db_session)
    _stub_cloze_for(monkeypatch, word_id=cards["cloze"])

    # Grade cloze (cloze branch picks cloze + LLM-stubbed generate_cloze).
    cloze_pick = client.get("/exercises/due?type=cloze")
    assert cloze_pick.status_code == 200
    assert client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["cloze"],
            "exercise_type": "cloze",
            "grade": 3,
        },
    ).status_code == 200

    # Grade matching.
    matching_pick = client.get("/exercises/due?type=matching")
    assert matching_pick.status_code == 204
    assert client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["matching"],
            "exercise_type": "matching",
            "grade": 3,
        },
    ).status_code == 200

    # Grade comprehension.
    comp_pick = client.get("/exercises/due?type=comprehension")
    assert comp_pick.status_code == 204
    assert client.post(
        "/exercises/grade",
        json={
            "exercise_id": cards["comprehension"],
            "exercise_type": "comprehension",
            "grade": 3,
        },
    ).status_code == 200

    # Now: there's still the idiom card due (it was seeded in step
    # _seed_four_type_due_queue). To drain genuinely we need to
    # also pick it off the queue. The idiom grade is a 422, but
    # that means the idiom card stays due -- so the genuine-empty
    # assertion needs us to NOT seed idiom in the first place for
    # this test's contract. Re-purpose: drain idiom by inline-
    # updating its row to a future due_date via raw SQL (the same
    # shape _pick_due_fsrs_card uses for its column probe).
    db_session.execute(
        text(
            "UPDATE fsrs_cards SET due_date = :future "
            "WHERE exercise_type = 'idiom'"
        ),
        {"future": datetime.utcnow() + timedelta(days=10)},
    )
    db_session.commit()

    # Now the queue is genuinely empty AND there are fresh words
    # not yet in fsrs_cards (Branch 2's first-encounter pool). The
    # route would still pick one of those for a fresh encounter,
    # so we ALSO clear out the Word rows for the tests -- actually,
    # we can't, because the cloze stub keyed on cards["cloze"]'s
    # word_id and that word is still in the corpus. Pick a stronger
    # drain: shift every Word into "already graded" by inserting
    # a placeholder fsrs_card for each Word with a future due_date.
    # That avoids the first-encounter branch and forces Branch 1's
    # empty result.
    words = db_session.execute(
        text("SELECT id FROM words ORDER BY id ASC")
    ).fetchall()
    for row in words:
        wid = row[0]
        # Skip the four words already in fsrs_cards (those are
        # the cloze/matching/comprehension/idiom words with their
        # own rows). For the rest, insert a fresh Learning row
        # with a future due_date -- this marks them as graded so
        # the route's first-encounter pool sees them as "already
        # known" and skips the Branch 2 first-encounter path.
        db_session.execute(
            text(
                "INSERT OR IGNORE INTO fsrs_cards "
                "(word_id, due_date, state) "
                "VALUES (:wid, :future, 1)"
            ),
            {"wid": wid, "future": datetime.utcnow() + timedelta(days=10)},
        )
    db_session.commit()

    # Now the no-filter call: nothing in Branch 1's fsrs_cards
    # pool is due, Branch 2's fresh-word pool is exhausted (every
    # word in the corpus has an fsrs_card row), so the route must
    # hit the "no due cards AND no fresh words" 204 fallback.
    final = client.get("/exercises/due")
    assert final.status_code == 204, final.text
    # The genuinely-empty branch carries NO X-Due-* headers --
    # this is the differentiator vs the non-cloze 204 branch.
    assert "X-Due-Exercise-Type" not in final.headers
    assert "X-Due-Card-Id" not in final.headers
    assert "X-Due-Word-Id" not in final.headers


# ===========================================================================
# Phase 10.6 (card t_da43cc23) -- ``phrase_match`` widens the
# ``/exercises/due`` union + ``/auth/me.due_by_type`` closure dict
# to the 5th literal (additive Literal widening).
#
# The Phase 10.1 schema (card t_18c90a68) added the ``phrase_pairs``
# table + the ``fsrs_cards.exercise_type='phrase_match'`` value.
# Phase 10.2 (card t_5d91a7e7) added the DSPy module + 5-way
# ``BaseExerciseFields`` Literal widening. Phase 10.3 (card t_13bb48d2)
# added the ``POST /exercises/phrase_match`` endpoint. Phase 10.5
# (card t_ca1d2da8) added the frontend per-type page. Phase 10.6
# wires the 5th type into the union + first-login gate so a user
# with only ``phrase_match`` due cards lands on ``/exercises/session``
# (not the cloze-only fallback).
# ===========================================================================


def test_phrase_match_due_queue_surfaces_via_204_with_headers(
    client, db_session, fsrs_with_type_column
) -> None:
    """``GET /exercises/due?type=phrase_match`` picks a phrase_match
    card and surfaces it via the union 204+headers shape.

    Mirrors the matching / comprehension / idiom branches on the
    same union surface. The ``X-Due-Exercise-Type`` header carries
    the discriminator; the route's ``match`` on ``picked_card_type
    != 'cloze'`` is type-agnostic so phrase_match rides the same
    read-side widening without a per-type branch.
    """
    from app import models  # noqa: F401 -- register tables

    _signup(client)
    word = _seed_word(db_session, word="phrase_match_word", word_type="Noun")
    _seed_due_card(
        db_session,
        word_id=word,
        exercise_type="phrase_match",
        due_date=datetime.utcnow() - timedelta(minutes=5),
    )

    resp = client.get("/exercises/due?type=phrase_match")
    assert resp.status_code == 204, resp.text
    assert resp.headers["X-Due-Exercise-Type"] == "phrase_match"
    assert resp.headers["X-Due-Word-Id"] == str(word)


def test_phrase_match_due_by_type_bucket_populated(
    client, db_session, fsrs_with_type_column
) -> None:
    """``/auth/me.due_by_type['phrase_match']`` reports the seeded
    phrase_match card count.

    Seeds one phrase_match card; ``/auth/me`` returns the
    Phase-10.6 5-key ``due_by_type`` dict with ``phrase_match=1``
    and the other 4 buckets at 0. This is the gate-widening
    hard-rule test: a learner with only a phrase_match card due
    must land on ``/exercises/session`` (the mixer's
    ``sum(due_by_type) > 0`` branch reads the new bucket).
    """
    _signup(client)
    word = _seed_word(db_session, word="phrase_match_word", word_type="Noun")
    _seed_due_card(
        db_session,
        word_id=word,
        exercise_type="phrase_match",
        due_date=datetime.utcnow() - timedelta(minutes=5),
    )

    resp = client.get("/auth/me")
    assert resp.status_code == 200, resp.text
    counts = resp.json()["due_by_type"]
    assert counts == {
        "cloze": 0,
        "matching": 0,
        "comprehension": 0,
        "idiom": 0,
        "phrase_match": 1,
    }, counts


def test_phrase_match_only_due_card_routes_to_session_mixer(
    client, db_session, fsrs_with_type_column
) -> None:
    """Phase 10.6 gate-widening hard rule: a user with only a
    phrase_match card due must land on ``/exercises/session`` (not
    the cloze-only ``/exercises/due`` fallback, not the profile
    branches).

    The frontend's ``postAuthGate`` reads
    ``sum(due_by_type.values())``; without the 5-key widening the
    ``phrase_match`` bucket would be missing and the gate would
    silently route the user to ``/weakness-profile`` (a hard-rule
    violation per the card body §"First-login gate widens to 5
    types"). This test pins the wire contract: backend surfaces
    ``phrase_match`` in the dict; downstream gating reads it.
    """
    _signup(client)
    word = _seed_word(db_session, word="phrase_match_word", word_type="Noun")
    _seed_due_card(
        db_session,
        word_id=word,
        exercise_type="phrase_match",
        due_date=datetime.utcnow() - timedelta(minutes=5),
    )

    me = client.get("/auth/me")
    assert me.status_code == 200, me.text
    counts = me.json()["due_by_type"]
    # Hard rule: gate's `sum > 0` branch must fire. Sum is
    # `phrase_match=1`; all other buckets at 0. Without the
    # 10.6 widening the sum would be 0 and the gate would
    # route to /weakness-profile (failure mode).
    total = sum(counts.values())
    assert total == 1, counts
    assert counts["phrase_match"] == 1, counts
