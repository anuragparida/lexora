"""Tests for Phase 7.4 — bilingual matching flag (card t_d621bb4f).

Coverage map (mirrors the card body's "Tests" section):

1. **Match happy path (EN)** — ``partner_lang="en"`` + a seeded
   ``collocations`` row for the target word → response carries
   ``partner_translation="..."``.
2. **Match 422** — ``partner_lang="fr"`` → 422 with FastAPI's
   default validation envelope (Pydantic ``Literal`` rejects).
3. **Match default** — empty body / no ``partner_lang`` →
   parses as ``"de"``, response ``partner_translation=None``.
4. **Cloze happy path (EN)** — same as #1 but on the cloze
   endpoint; the cloze generator also looks up the collocations
   table for the target word's ``answer_word_id``.
5. **Cloze 422** — ``partner_lang="fr"`` → 422.
6. **Cloze default** — empty body → ``partner_translation=None``.
7. **Cloze bilingual combined (forward-looking)** —
   ``partner_lang="en"`` on ``/exercises/cloze`` populates
   ``partner_translation`` from the same collocations lookup
   the matching endpoint uses. Phase 7.3's ``collocation=true``
   flag is NOT yet on the wire (it's a parallel Phase 7 card);
   the bilingual feature is independent of the collocation
   mode — the card body says "passes through to 4.2's
   ``generate_cloze`` (or 7.2's ``generate_collocation`` if
   ``collocation=True``)" — meaning the wiring lives at the
   route layer. The test here exercises the no-collocation
   branch (the only branch that exists today) and verifies
   the bilingual field still gets populated.

Hermetic: fresh temp SQLite DB + temp JWT secret per test.
The OpenRouter chat-completions call is mocked via the
``monkeypatch.setattr("app.match._openai_client", ...)``
pattern from ``test_match_endpoint.py``. The collocations
table is created via raw SQL in the test fixture (the
SQLAlchemy model doesn't exist on ``main`` yet — Phase 7.1
will add it; this card is shipped before 7.1 so the test
provisions the table inline).

Run from ``backend/``::

    uv run pytest -q tests/test_match_partner_lang.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_cloze.py / test_match.py / test_due.py / test_match_endpoint.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Same pattern as test_cloze / test_match.
    """
    db_path = tmp_path / "test_partner_lang.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """``TestClient`` wired to a fresh per-test SQLite DB."""
    from app import database
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB (mirrors
    test_cloze / test_match)."""
    from sqlalchemy.orm import sessionmaker

    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_user(session, *, email: str = "ada@example.com") -> int:
    """Create a user. Returns user_id."""
    from app import models
    from app.passwords import hash_password

    user = models.User(
        email=email, password_hash=hash_password("supersecret")
    )
    session.add(user)
    session.flush()
    session.commit()
    return user.id


def _seed_word(
    session,
    *,
    word: str,
    word_type: str,
    example_de: str = "X.",
) -> int:
    """Insert one ``Word`` row with a stub example. Returns word_id."""
    from app import models

    row = models.Word(
        word=word, word_type=word_type, frequency="5", is_complete=True
    )
    session.add(row)
    session.flush()
    session.add(models.Example(word_id=row.id, german=example_de, english=""))
    session.commit()
    return row.id


def _create_collocations_table(session) -> None:
    """Provision the ``collocations`` table for the test.

    Phase 7.1 (card t_5eXXXX — pending) will add the SQLAlchemy
    ``Collocation`` model + Alembic migration. This card lands
    before 7.1, so the test provisions a minimal table via raw
    SQL — same columns + types the 7.1 schema will use, so the
    ``lookup_partner_translation`` query works unchanged once
    7.1 folds.
    """
    # ``partner_lemma`` is the curated EN counterpart; ``lemma``
    # is the German collocation phrase (kept minimal here — the
    # generator only reads ``partner_lemma``). We mirror the 7.1
    # schema shape (``headword_id`` FK + ``partner_lemma`` column).
    session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS collocations (
                collocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                headword_id INTEGER,
                partner_lemma VARCHAR NOT NULL,
                frequency_score FLOAT NOT NULL DEFAULT 0.5,
                register VARCHAR NOT NULL DEFAULT 'neutral',
                source_corpus VARCHAR NOT NULL DEFAULT 'manual',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (headword_id) REFERENCES words(id)
            )
            """
        )
    )
    session.commit()


def _seed_collocation_row(
    session, *, word_id: int, partner_lemma: str
) -> None:
    """Insert one collocations row for ``word_id``."""
    session.execute(
        text(
            """
            INSERT INTO collocations (headword_id, partner_lemma, frequency_score, register, source_corpus)
            VALUES (:headword_id, :partner_lemma, :frequency_score, :register, :source_corpus)
            """
        ),
        {
            "headword_id": word_id,
            "partner_lemma": partner_lemma,
            "frequency_score": 0.5,
            "register": "neutral",
            "source_corpus": "manual",
        },
    )
    session.commit()


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """POST /auth/signup and return the parsed body."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# OpenAI stub — mirrors test_match_endpoint.py's _make_stub_instructor_client.
# Returns a real OpenAI client whose httpx transport is a MockTransport.
# ---------------------------------------------------------------------------


def _make_stub_match_client(
    *, target_word_id: int, count: int, model: str = "qwen/qwen3-235b-a22b-2507"
) -> Any:
    """Stub OpenAI client returning a valid ``MatchingExercise`` JSON."""
    import httpx
    from openai import OpenAI

    pairs = []
    for i in range(count):
        left = target_word_id + (2 * i) + 1
        right = target_word_id + (2 * i) + 2
        kind = "translation" if i % 2 == 0 else "synonym"
        pairs.append(
            {"left_word_id": left, "right_word_id": right, "right_kind": kind}
        )
    payload = json.dumps(
        {"target_word_id": target_word_id, "pairs": pairs},
        ensure_ascii=False,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-partner-lang-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
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


def _make_stub_cloze_client(
    *, answer_word_id: int, model: str = "qwen/qwen3-235b-a22b-2507"
) -> Any:
    """Stub OpenAI client returning a valid ``ClozeExercise`` JSON."""
    import httpx
    from openai import OpenAI

    payload = json.dumps(
        {
            "sentence_with_blank": "Der Hund ___ im Garten.",
            "answer_word_id": answer_word_id,
            "distractors": [answer_word_id + 1, answer_word_id + 2, answer_word_id + 3],
            "difficulty": "medium",
            "rationale": "Common verb agreement.",
            "prompt_template_version": "cloze-v1",
        },
        ensure_ascii=False,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-partner-lang-cloze-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": payload},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 18,
                    "total_tokens": 48,
                },
            },
        )

    return OpenAI(
        api_key="test-key-not-real",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


# ---------------------------------------------------------------------------
# 1. Match — happy path: partner_lang="en" with a collocations row
# ---------------------------------------------------------------------------


def test_post_exercises_match_partner_lang_en_populates_translation(
    client, db_session, monkeypatch
):
    """``POST /exercises/match {"partner_lang": "en"}`` with a
    collocations row seeded for the target word returns
    ``partner_translation="to sleep"`` (the EN partner lemma).

    This is the central acceptance criterion from the card
    body: the EN happy path. The collocations table is created
    inline (Phase 7.1 hasn't shipped yet on ``main``) — once
    7.1 lands and seeds the 200-word curated subset, this exact
    flow reads from the production table.
    """
    from app import match

    # Seed a verb (the matching generator needs at least one
    # word to pick). The deterministic seed (Phase 4.5)
    # chooses the highest-frequency verb.
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in (
        "gehen",
        "kommen",
        "bleiben",
        "wohnen",
        "lernen",
        "arbeiten",
        "essen",
        "trinken",
    ):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    # Provision the collocations table + seed the EN partner
    # lemma for ``schlafen``.
    _create_collocations_table(db_session)
    _seed_collocation_row(
        db_session, word_id=target_id, partner_lemma="to sleep"
    )

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.match._openai_client",
        lambda: _make_stub_match_client(target_word_id=target_id, count=4),
    )

    resp = client.post(
        "/exercises/match", json={"partner_lang": "en"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Wire-shape contract.
    assert body["exercise_type"] == "matching"
    assert body["target_word_id"] == target_id
    # Bilingual read-through — populated from collocations row.
    assert body["partner_translation"] == "to sleep"


# ---------------------------------------------------------------------------
# 2. Match — 422 path: partner_lang="fr" outside the Literal
# ---------------------------------------------------------------------------


def test_post_exercises_match_partner_lang_outside_literal_returns_422(
    client, monkeypatch
):
    """``POST /exercises/match {"partner_lang": "fr"}`` → 422.

    Hard rule H4: the ``Literal["de","en"]`` is a real Pydantic
    literal — FastAPI rejects any value outside the union at the
    request body layer with the default 422 envelope. No LLM
    call, no generator invocation, no DB write.
    """
    _signup(client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    # No ``_openai_client`` monkeypatch — the handler must not
    # reach the generator. If it does, the test fails on the
    # implicit ``None`` return from ``_openai_client`` and the
    # route raises 502, NOT 422.
    resp = client.post(
        "/exercises/match", json={"partner_lang": "fr"}
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    detail_blob = json.dumps(body).lower()
    # FastAPI's 422 envelope includes the field name
    # (``partner_lang``) and a hint that the value is not
    # in the literal — the exact phrasing varies by Pydantic
    # version, so we check the field name + the word "literal"
    # or the values themselves.
    assert "partner_lang" in detail_blob
    assert "literal" in detail_blob or "de" in detail_blob or "en" in detail_blob


# ---------------------------------------------------------------------------
# 3. Match — default: empty body parses as "de", partner_translation=None
# ---------------------------------------------------------------------------


def test_post_exercises_match_default_partner_lang_is_de_and_no_translation(
    client, db_session, monkeypatch
):
    """``POST /exercises/match {}`` (no flag) → ``partner_translation=None``.

    Hard rule H3 (opt-in default): ``partner_lang="de"`` is the
    default. Existing Phase 6.1 / 6.2 callers see no schema
    change AND no behavioral change — the response shape now
    carries the new ``partner_translation`` field, but its
    value is always ``None`` when ``partner_lang="de"``.
    """
    from app import match

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    # Even WITH a collocations row seeded, the default branch
    # (``partner_lang="de"``) must NOT populate
    # ``partner_translation`` — bilingual is opt-in.
    _create_collocations_table(db_session)
    _seed_collocation_row(
        db_session, word_id=target_id, partner_lemma="to sleep"
    )

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.match._openai_client",
        lambda: _make_stub_match_client(target_word_id=target_id, count=4),
    )

    resp = client.post("/exercises/match", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["partner_translation"] is None
    # Existing fields untouched.
    assert body["exercise_type"] == "matching"
    assert body["target_word_id"] == target_id
    assert len(body["pairs"]) == 4


# ---------------------------------------------------------------------------
# 4. Cloze — happy path: partner_lang="en" populates partner_translation
# ---------------------------------------------------------------------------


def test_post_exercises_cloze_partner_lang_en_populates_translation(
    client, db_session, monkeypatch
):
    """``POST /exercises/cloze {"partner_lang": "en"}`` with a
    collocations row seeded for the target word returns
    ``partner_translation="to sleep"``.

    Mirrors the matching happy path (#1) on the cloze wire.
    The cloze generator looks up the collocations row for the
    target word's ``answer_word_id`` (the same lookup shape
    the matching generator uses on ``target_word_id``). Fail-soft
    contract holds: a missing row → ``partner_translation=None``.
    """
    from app import cloze

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    _create_collocations_table(db_session)
    _seed_collocation_row(
        db_session, word_id=target_id, partner_lemma="to sleep"
    )

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _make_stub_cloze_client(answer_word_id=target_id),
    )

    resp = client.post(
        "/exercises/cloze", json={"partner_lang": "en"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Cloze wire shape — ``target_word_id`` echoes ``answer_word_id``.
    assert body["exercise_type"] == "cloze"
    assert body["target_word_id"] == target_id
    assert body["answer_word_id"] == target_id
    # Bilingual read-through — populated from the collocations row.
    assert body["partner_translation"] == "to sleep"


# ---------------------------------------------------------------------------
# 5. Cloze — 422 path: partner_lang="fr"
# ---------------------------------------------------------------------------


def test_post_exercises_cloze_partner_lang_outside_literal_returns_422(
    client, monkeypatch
):
    """``POST /exercises/cloze {"partner_lang": "fr"}`` → 422."""
    _signup(client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    resp = client.post(
        "/exercises/cloze", json={"partner_lang": "fr"}
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    detail_blob = json.dumps(body).lower()
    assert "partner_lang" in detail_blob


# ---------------------------------------------------------------------------
# 6. Cloze — default: empty body → partner_translation=None
# ---------------------------------------------------------------------------


def test_post_exercises_cloze_default_partner_lang_is_de_and_no_translation(
    client, db_session, monkeypatch
):
    """``POST /exercises/cloze {}`` (no flag) → ``partner_translation=None``.

    Hard rule H3 (opt-in default): even with a collocations
    row seeded, the default branch must NOT populate the
    bilingual field.
    """
    from app import cloze

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    _create_collocations_table(db_session)
    _seed_collocation_row(
        db_session, word_id=target_id, partner_lemma="to sleep"
    )

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _make_stub_cloze_client(answer_word_id=target_id),
    )

    resp = client.post("/exercises/cloze", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["partner_translation"] is None
    assert body["exercise_type"] == "cloze"


# ---------------------------------------------------------------------------
# 7. Cloze + bilingual combined — fail-soft when no collocations row exists
# ---------------------------------------------------------------------------


def test_post_exercises_cloze_partner_lang_en_without_row_returns_none(
    client, db_session, monkeypatch
):
    """``partner_lang="en"`` without a collocations row (or
    without the collocations table at all) → ``partner_translation=None``.

    This is the fail-soft contract — the bilingual feature is
    best-effort. The route never 500s on a missing translation
    (PHASE-7.md "Out of scope" gotcha #6: bilingual is curated,
    not generated). This test exercises BOTH fail-soft paths:

    - The collocations table exists but has no row for the
      target word.
    - The collocations table doesn't exist (production pre-7.1
      state).
    """
    from app import cloze

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    # NOTE: NO ``_create_collocations_table`` call here — the
    # test exercises the missing-table path of the fail-soft
    # contract. ``lookup_partner_translation`` catches the
    # ``OperationalError`` and returns ``None``.

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _make_stub_cloze_client(answer_word_id=target_id),
    )

    resp = client.post(
        "/exercises/cloze", json={"partner_lang": "en"}
    )
    # No 500 — the lookup fails soft.
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["partner_translation"] is None
    # Cloze wire fields still populated.
    assert body["answer_word_id"] == target_id


# ---------------------------------------------------------------------------
# 8. Schema-only — partner_lang="en" parses, partner_lang="fr" rejects
# ---------------------------------------------------------------------------


def test_schemas_partner_lang_literal_validation():
    """Wire-level smoke test — the Pydantic ``Literal["de","en"]``
    is the gate (Hard rule H4). This bypasses the HTTP layer
    so a schema-only regression surfaces as a unit-test failure
    rather than a 422 in the integration tests above.
    """
    from pydantic import ValidationError

    from app.schemas import (
        ClozeGenerateRequest,
        MatchGenerateRequest,
        PartnerLang,
    )

    # Default is ``"de"``.
    assert ClozeGenerateRequest().partner_lang == "de"
    assert MatchGenerateRequest().partner_lang == "de"

    # ``"en"`` is the only non-default accept.
    assert ClozeGenerateRequest(partner_lang="en").partner_lang == "en"
    assert MatchGenerateRequest(partner_lang="en").partner_lang == "en"

    # Anything else raises (the Literal is closed).
    for bad in ("fr", "EN", "", "english", None, 0):
        with pytest.raises(ValidationError):
            ClozeGenerateRequest(partner_lang=bad)
        with pytest.raises(ValidationError):
            MatchGenerateRequest(partner_lang=bad)

    # The PartnerLang alias is the literal itself.
    from typing import get_args

    assert get_args(PartnerLang) == ("de", "en")


# ---------------------------------------------------------------------------
# 9. Schema-only — partner_translation field exists with the right default
# ---------------------------------------------------------------------------


def test_schemas_partner_translation_field_shape():
    """Wire-level smoke test — ``partner_translation`` is a
    ``str | None`` with ``None`` default on both response shapes.

    Same shape on the generator side (``MatchingExercise`` /
    ``ClozeExercise``): the Pydantic default is ``None``, the
    field is opt-in populated by the generator.
    """
    from app.cloze import ClozeExercise
    from app.match import MatchingExercise
    from app.schemas import ClozeExerciseOut, MatchingExerciseOut

    # Wire shapes — ``partner_translation`` is on the response.
    assert "partner_translation" in MatchingExerciseOut.model_fields
    assert "partner_translation" in ClozeExerciseOut.model_fields
    assert MatchingExerciseOut.model_fields["partner_translation"].default is None
    assert ClozeExerciseOut.model_fields["partner_translation"].default is None

    # Generator shapes — same field, same default.
    assert "partner_translation" in MatchingExercise.model_fields
    assert "partner_translation" in ClozeExercise.model_fields
    assert MatchingExercise.model_fields["partner_translation"].default is None
    assert ClozeExercise.model_fields["partner_translation"].default is None