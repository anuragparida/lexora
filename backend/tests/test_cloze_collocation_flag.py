"""Tests for Phase 7.3 — ``/exercises/cloze`` collocation opt-in flag.

Coverage map (mirrors the card body's "Tests" section):

1. **Default ``collocation=False`` produces a Phase 6.1 wire shape
   (byte-for-byte; verified by hashing the prompt bytes).** The
   acceptance gate for Hard rule H10 — the collocation=False path
   must render the same user prompt as the Phase 6.1 cloze path.
2. **``collocation=True`` produces a ``partner_lemma`` field;
   ``partner_lemma`` is ``None`` when ``collocation=False``.** The
   route's discriminator echo + the collocation-cloze branch's
   partner word.
3. **Pydantic literal validation: ``collocation`` only accepts bool
   (string ``"true"`` → 422).** The wire guardrail — Pydantic v2
   rejects any non-bool at the schema layer.
4. **Existing ``/exercises/cloze`` callers with no ``collocation``
   field see no schema change (default value parses).** The
   ``{}`` / empty-body path → 200 + the Phase 6.1 cloze shape
   (plus the wrapper discriminator echoes).

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test, the
OpenRouter chat-completions call is replaced with ``respx`` mocks
so no network is touched. Mirrors the Phase 4.2 / 6.1 / 6.2
patterns (test_diagnostic.py / test_cloze.py / test_collocation.py).

Hard rules enforced:

- **H3** (opt-in default): ``collocation=False`` is the default. No
  flag → default → no behavioural change. Tests 1 + 4 assert this.
- **H10** (existing callers byte-for-byte unchanged): the no-flag
  branch produces a rendered prompt identical to Phase 6.1. Test
  1 asserts this with a SHA-256 hash of the captured prompt bytes
  (the captured payload is the OpenAI ``messages`` array sent on
  the wire — the same shape Phase 6.1 captured).
- **H5** (Pydantic v2 validated input/output): ``collocation`` has
  a real Pydantic type, not a comment. Test 3 asserts this on the
  request side; the wrapper's response-side discriminator is
  asserted in test 2.
- **H8** (lexora board): no ``default`` board work — all work in
  this file stays on the lexora board's worktree.

Run from ``backend/``::

    uv run pytest -q tests/test_cloze_collocation_flag.py
"""
from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret + TestClient
# pattern from test_cloze.py / test_diagnostic.py / test_collocation.py.
#
# Note: a test that uses BOTH ``db_session`` AND ``client`` fixtures
# would double-reconfigure the engine (one fixture per use), which
# routes around the seeded rows. We follow the existing test
# convention: a single ``client_and_session`` fixture yields both a
# TestClient (for the route call) and a SQLAlchemy session (for
# seeding) bound to the same fresh SQLite file.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Point the app at a fresh temp SQLite DB and seed a JWT secret."""
    db_path = tmp_path / "test_cloze_collocation_flag.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    # Disable Langfuse side-effects so the trace hook is a no-op
    # (mirrors test_cloze.py::test_trace_cloze_is_silent_when_keys_missing).
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    return str(db_path)


@pytest.fixture
def client_and_session(sqlite_db_path):
    """Yield ``(client, session)`` against the same fresh SQLite DB.

    A single ``reconfigure_for_test`` call binds the engine; the
    TestClient exercises the route through that engine, and the
    SQLAlchemy session seeds rows into the same database. Mirrors
    the Phase 4.2 / 6.1 / 7.2 fixture pattern.
    """
    from sqlalchemy.orm import sessionmaker

    from app import database, collocation, models  # noqa: F401 — pull in collocation model
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    SessionLocal = sessionmaker(bind=database.engine)
    with TestClient(app) as c:
        with SessionLocal() as session:
            yield c, session


# ---------------------------------------------------------------------------
# Seed helpers — mirror test_cloze.py + test_collocation.py shapes.
# ---------------------------------------------------------------------------


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """Sign up via ``/auth/signup`` and return the JSON body (cookie is set)."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _seed_user_with_axes(
    session,
    *,
    email: str = "ada@example.com",
    axes: dict[str, int] | None = None,
) -> int:
    """Insert a User with an optional WeaknessProfile and return user_id."""
    from app import crud, models
    from app.passwords import hash_password

    user = models.User(
        email=email, password_hash=hash_password("supersecret")
    )
    session.add(user)
    session.flush()
    if axes is not None:
        crud.upsert_weakness_profile(session, user.id, axes)
    session.commit()
    return user.id


def _seed_word(
    session,
    *,
    word: str,
    word_type: str,
    example_de: str = "X schläft.",
    translations: str = "",
    target_word_id: int | None = None,
) -> int:
    """Insert one ``Word`` row plus a single ``Example`` and return word_id."""
    from app import models

    row = models.Word(
        id=target_word_id,
        word=word,
        word_type=word_type,
        frequency="5",
        is_complete=True,
        translations=translations,
    )
    session.add(row)
    session.flush()
    session.add(models.Example(word_id=row.id, german=example_de, english=""))
    session.commit()
    return row.id


def _seed_collocation(
    session,
    *,
    target_word_id: int,
    partner_lemma: str = "treffen",
    partner_register: str = "neutral",
    source_corpus: str = "dwds",
) -> int:
    """Insert one ``Collocation`` row (Phase 7.1 mirror) and return id."""
    from app.collocation import Collocation

    row = Collocation(
        target_word_id=target_word_id,
        partner_lemma=partner_lemma,
        partner_register=partner_register,
        source_corpus=source_corpus,
    )
    session.add(row)
    session.flush()
    session.commit()
    return row.id


# ---------------------------------------------------------------------------
# OpenAI stub — mirrors test_cloze.py's _openai_cloze_response shape.
# The instructor client parses the assistant message JSON and validates
# against the Pydantic schema, so a real ``choices[0].message.content``
# carrying valid JSON is what reaches the validator.
# ---------------------------------------------------------------------------


def _openai_cloze_response(
    *,
    sentence: str = "Der ___ schläft.",
    answer_id: int = 1,
    distractors: list[int] | None = None,
    difficulty: str = "easy",
    rationale: str = "Obvious copula test.",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
) -> dict:
    """Fake OpenAI chat-completions body for the standard cloze branch."""
    if distractors is None:
        distractors = [2, 3, 4]
    return {
        "id": "gen-cloze-7-3-test-001",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen/qwen3-235b-a22b-2507",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "sentence_with_blank": sentence,
                            "answer_word_id": answer_id,
                            "distractors": distractors,
                            "difficulty": difficulty,
                            "rationale": rationale,
                            "prompt_template_version": "cloze-v1",
                        }
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _openai_collocation_response(
    *,
    prompt: str = "Er ___ eine wichtige Entscheidung.",
    target_word_id: int = 1,
    partner_lemma: str = "treffen",
    partner_register: str = "neutral",
    source_corpus: str = "dwds",
    rationale: str = "Stub rationale.",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
) -> dict:
    """Fake OpenAI chat-completions body for the collocation branch."""
    return {
        "id": "gen-collocation-7-3-test-001",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen/qwen3-235b-a22b-2507",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "prompt": prompt,
                            "target_word_id": target_word_id,
                            "partner_lemma": partner_lemma,
                            "partner_register": partner_register,
                            "source_corpus": source_corpus,
                            "rationale": rationale,
                            "prompt_template_version": (
                                "collocation-cloze-v1"
                            ),
                        }
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — Hard rule H10: default collocation=False produces a
# Phase-6.1-identical prompt. We capture the OpenAI request the
# route makes via respx, normalise the ``messages`` payload, and
# hash it. The hash is compared with a value recorded by the
# corresponding Phase 6.1 run (``test_cloze.py`` tests hash the
# same way).
# ---------------------------------------------------------------------------


@respx.mock
def test_collocation_false_default_prompt_byte_for_byte_identical(
    client_and_session, monkeypatch
):
    """Phase 7.3 — H10: ``collocation=False`` (the default) produces
    a rendered user prompt hash-identical to the Phase 6.1 cloze
    path. No ``app.cloze.*`` prompt template mutation, no retrieval
    call, no RAG injection — the same call shape that the
    Phase 6.1 ``POST /exercises/cloze`` made.

    We do NOT compare against a pre-recorded hash from a separate
    Phase 6.1 run (that would couple the test to a separate
    snapshot file). Instead, we assert the SHA-256 of the captured
    ``messages`` array matches the SHA-256 of the same shape
    re-rendered by ``app.cloze.generate_cloze`` directly — i.e. we
    assert ``route through endpoint`` ≡ ``generator direct call``
    for the no-flag path. This is the strongest in-process
    byte-for-byte check for the opt-in default.
    """
    from app import cloze as _cloze_mod

    client, db_session = client_and_session

    # Seed corpus rows (the route picks the target word from the
    # words table + the user's weakness profile's axes).
    verb_id = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de="Der Hund schläft.",
    )
    _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="X geht."
    )
    _seed_word(
        db_session, word="kommen", word_type="Verb", example_de="X kommt."
    )
    _seed_word(
        db_session, word="bleiben", word_type="Verb", example_de="X bleibt."
    )

    # Sign up via the route — this creates the User row + sets the
    # JWT cookie. The seeded weakness profile is attached to
    # ``current_user.id`` post-signup via ``PUT
    # /weakness-profile/{user_id}``.
    _signup(client)
    # Find current_user.id from the signup response — mirrors
    # test_diagnostic.py's pattern of using the response payload.
    # We can't use _seed_user_with_axes here because that would
    # double-create the user (signup race).
    me = client.get("/auth/me").json()
    user_id = me["id"]
    assert me["email"] == "ada@example.com"

    # Stash the user id we'll use for the direct generator call
    # after the route runs. The signed-in user owns the seeded
    # words + axes — the route and the direct generator both
    # read from the same DB state.
    route_user_id = user_id

    # Attach a deterministic ``verbs: 3`` weakness profile to
    # the signed-in user. PUT under the JWT cookie carries the
    # auth — the same shape Phase 4.5 callers used.
    resp_axes = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 3}},
    )
    assert resp_axes.status_code == 200, resp_axes.text

    # Capture the OpenAI request body the route makes.
    captured_payload: dict[str, Any] = {}

    def _capture(request):
        captured_payload["json"] = json.loads(request.content.decode())
        return Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=verb_id,
            ),
        )

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=_capture
    )

    # Hit the endpoint with the default request (no
    # ``collocation`` field — should parse to ``collocation=False``).
    resp = client.post("/exercises/cloze", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Discriminator echoes are present (Phase 7.3 wire surface).
    assert body["collocation"] is False
    assert body["partner_lemma"] is None
    # The standard cloze fields are present and match the stub.
    assert body["sentence_with_blank"] == "Der ___ schläft."
    assert body["answer_word_id"] == verb_id
    assert body["distractors"] == [2, 3, 4]
    assert body["exercise_type"] == "cloze"
    route_messages = captured_payload["json"]["messages"]
    route_prompt_hash = _hash_messages(route_messages)

    # Direct ``generate_cloze`` call with the same params (using
    # the same signed-in user_id so the seed scheme picks the
    # same target word).
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=verb_id,
            ),
        )
    )

    direct_capture: dict[str, Any] = {}

    def _capture_direct(request):
        direct_capture["json"] = json.loads(request.content.decode())
        return Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=verb_id,
            ),
        )

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=_capture_direct
    )

    direct = _cloze_mod.generate_cloze(
        db_session, route_user_id, enable_rag=False
    )
    assert direct is not None  # exercised
    direct_messages = direct_capture["json"]["messages"]
    direct_prompt_hash = _hash_messages(direct_messages)

    # H10 — the opt-in default produces the same prompt bytes as a
    # direct generate_cloze call (with enable_rag=False).
    assert route_prompt_hash == direct_prompt_hash, (
        "Phase 7.3 H10 violation: collocation=False route prompt "
        "differs from a direct generate_cloze prompt. Review "
        "app/main.py for hidden prompt-template mutations on the "
        "no-flag branch."
    )


# ---------------------------------------------------------------------------
# Test 2 — collocation=True → partner_lemma populated; collocation=False
# → partner_lemma is None.
# ---------------------------------------------------------------------------


@respx.mock
def test_collocation_true_populates_partner_lemma(
    client_and_session, monkeypatch
):
    """Phase 7.3 — ``collocation=True`` routes through Phase 7.2's
    ``generate_collocation``; the response carries a populated
    ``partner_lemma`` field. The standard cloze fields are
    absent on this branch (the SPA keys on the discriminator).
    """
    client, db_session = client_and_session

    verb_id = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de="Der Hund schläft.",
    )
    # Seed a Collocation row for the target word — without this,
    # ``generate_collocation`` would raise ``ValueError`` on the
    # collocation-row lookup and the route would surface a 500.
    _seed_collocation(
        db_session,
        target_word_id=verb_id,
        partner_lemma="treffen",
        partner_register="neutral",
        source_corpus="dwds",
    )
    _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="X geht."
    )
    _seed_word(
        db_session, word="kommen", word_type="Verb", example_de="X kommt."
    )
    _seed_word(
        db_session, word="bleiben", word_type="Verb", example_de="X bleibt."
    )

    _signup(client)
    user_id = client.get("/auth/me").json()["id"]
    resp_axes = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 3}},
    )
    assert resp_axes.status_code == 200, resp_axes.text

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_collocation_response(
                prompt="Er ___ eine wichtige Entscheidung.",
                target_word_id=verb_id,
                partner_lemma="treffen",
                partner_register="neutral",
                source_corpus="dwds",
            ),
        )
    )

    resp = client.post(
        "/exercises/cloze",
        json={"collocation": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Discriminator echoes.
    assert body["collocation"] is True
    assert body["partner_lemma"] == "treffen"

    # Collocation-specific fields are present.
    assert body["prompt"] == "Er ___ eine wichtige Entscheidung."
    assert body["partner_register"] == "neutral"
    assert body["source_corpus"] == "dwds"
    assert body["exercise_type"] == "cloze"  # PHASE-7 gotcha #5
    assert body["prompt_template_version"] == "collocation-cloze-v1"

    # Standard cloze fields are NOT present on this branch (the
    # discriminated union).
    assert "sentence_with_blank" not in body
    assert "distractors" not in body


@respx.mock
def test_collocation_false_partner_lemma_is_none(
    client_and_session, monkeypatch
):
    """Phase 7.3 — ``collocation=False`` (the default) populates
    ``partner_lemma`` with ``None`` (not the string ``"None"``, not
    absent). The wrapper always serialises the discriminator
    fields; the standard cloze fields stay populated.
    """
    client, db_session = client_and_session

    verb_id = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de="Der Hund schläft.",
    )
    _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="X geht."
    )
    _seed_word(
        db_session, word="kommen", word_type="Verb", example_de="X kommt."
    )
    _seed_word(
        db_session, word="bleiben", word_type="Verb", example_de="X bleibt."
    )

    _signup(client)
    user_id = client.get("/auth/me").json()["id"]
    resp_axes = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 3}},
    )
    assert resp_axes.status_code == 200, resp_axes.text

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=verb_id,
            ),
        )
    )

    # Explicit collocation=False (not the default — we want to
    # exercise the explicit branch too).
    resp = client.post(
        "/exercises/cloze",
        json={"collocation": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Discriminator echoes.
    assert body["collocation"] is False
    assert body["partner_lemma"] is None  # H10 — None, not absent

    # Standard cloze fields are present.
    assert body["sentence_with_blank"] == "Der ___ schläft."
    assert body["exercise_type"] == "cloze"

    # Collocation-specific fields are NOT present on this branch.
    assert "prompt" not in body or body.get("prompt") is None


# ---------------------------------------------------------------------------
# Test 3 — Pydantic literal validation: collocation only accepts
# bool. String "true" → 422.
# ---------------------------------------------------------------------------


def test_collocation_non_bool_returns_422(client_and_session):
    """Phase 7.3 — H5: ``collocation`` has a real Pydantic type, not
    a comment. FastAPI returns 422 on a non-bool value, with the
    body carrying the standard Pydantic validation error envelope.
    """
    client, _ = client_and_session

    _signup(client)

    # String ``"true"`` — Pydantic v2 strict-bool rejects this.
    resp_string = client.post(
        "/exercises/cloze",
        json={"collocation": "true"},
    )
    assert resp_string.status_code == 422
    err = resp_string.json()
    assert "detail" in err  # FastAPI's standard 422 envelope shape

    # Integer ``1`` — same rejection.
    resp_int = client.post(
        "/exercises/cloze",
        json={"collocation": 1},
    )
    assert resp_int.status_code == 422

    # Float ``1.0`` — same rejection.
    resp_float = client.post(
        "/exercises/cloze",
        json={"collocation": 1.0},
    )
    assert resp_float.status_code == 422

    # List — same rejection.
    resp_list = client.post(
        "/exercises/cloze",
        json={"collocation": [True]},
    )
    assert resp_list.status_code == 422


# ---------------------------------------------------------------------------
# Test 4 — empty body / no collocation field / default value parses;
# round-trip to a Phase 6.1-equivalent cloze response.
# ---------------------------------------------------------------------------


@respx.mock
def test_empty_body_collocation_default_parses_to_false(
    client_and_session, monkeypatch
):
    """Phase 7.3 — H3: ``{}`` (no body, no fields) parses to
    ``ClozeGenerateRequest(enable_rag=False, collocation=False)``.
    The route produces the standard cloze shape — a Phase 4.2 /
    6.1 caller posting the empty body sees no schema change at
    the wire surface (modulo the new ``collocation`` + ``partner_lemma``
    echo fields the SPA can ignore).
    """
    client, db_session = client_and_session

    verb_id = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de="Der Hund schläft.",
    )
    _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="X geht."
    )
    _seed_word(
        db_session, word="kommen", word_type="Verb", example_de="X kommt."
    )
    _seed_word(
        db_session, word="bleiben", word_type="Verb", example_de="X bleibt."
    )

    _signup(client)
    user_id = client.get("/auth/me").json()["id"]
    resp_axes = client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 3}},
    )
    assert resp_axes.status_code == 200, resp_axes.text

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=verb_id,
            ),
        )
    )

    # Empty body — FastAPI parses to ``ClozeGenerateRequest()``
    # defaults.
    resp_empty = client.post("/exercises/cloze", json={})
    assert resp_empty.status_code == 200, resp_empty.text
    body_empty = resp_empty.json()
    assert body_empty["collocation"] is False
    assert body_empty["partner_lemma"] is None
    assert body_empty["sentence_with_blank"] == "Der ___ schläft."
    assert body_empty["exercise_type"] == "cloze"
    assert body_empty["enable_rag"] is False

    # Body with only ``enable_rag`` — also parses; the
    # ``collocation`` field defaults to ``False``.
    resp_eo = client.post(
        "/exercises/cloze",
        json={"enable_rag": False},
    )
    assert resp_eo.status_code == 200, resp_eo.text
    body_eo = resp_eo.json()
    assert body_eo["collocation"] is False
    assert body_eo["partner_lemma"] is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_messages(messages: list[dict[str, Any]]) -> str:
    """SHA-256 of the OpenAI ``messages`` payload, key-order normalised.

    Pydantic v2 serialises dicts in insertion order on Python 3.12,
    so two captures from the same code path produce identical
    bytes. We JSON-dump with ``sort_keys=True`` to dodge any
    list-of-dicts key ordering differences (the messages list can
    have user/assistant/system roles — each is a dict-of-strings).
    """
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
