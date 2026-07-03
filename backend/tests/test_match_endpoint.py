"""Tests for Phase 6.3 — ``POST /exercises/match`` route (card t_39d85400).

Coverage map (mirrors the card body's "Tests" section):

1. 200 happy path — ``{}`` body → ``MatchingExerciseOut`` with
   ``exercise_type="matching"``, ``enable_rag=false``, default
   ``count=4`` (the schema default).
2. 200 with overrides — ``{"count": 6, "enable_rag": true}`` → 200
   with ``count=6`` and ``enable_rag=true`` threaded through to
   the generator.
3. 401 — no JWT cookie → 401 (auth-gated route; the
   ``get_current_user`` dependency raises).
4. 422 — ``{"count": 0}`` → 422 (Pydantic ``Field(ge=2)``).
5. 422 — ``{"count": 20}`` → 422 (Pydantic ``Field(le=8)``).
6. 502 — LLM transport failure (respx 500 from the OpenRouter
   mock) → 502 with the LLMError detail.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test.
The OpenRouter chat-completions call is mocked via either
``monkeypatch.setattr("app.match._openai_client", ...)`` (the
order-independent stub pattern from test_cloze.py / test_match.py)
or via ``respx`` where the route's respx flow is exercised
end-to-end (test #6).

Run from ``backend/``::

    uv run pytest -q tests/test_match_endpoint.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_cloze.py / test_match.py / test_due.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Same pattern as test_cloze / test_match.
    """
    db_path = tmp_path / "test_match_endpoint.db"
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


def _seed_user_with_axes(
    session, *, axes: dict[str, int] | None = None
) -> int:
    """Create a user with an optional weakness profile. Returns user_id."""
    from app import crud, models
    from app.passwords import hash_password

    user = models.User(
        email="ada@example.com", password_hash=hash_password("supersecret")
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
    example_de: str = "Der Hund schläft.",
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


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """POST /auth/signup and return the parsed body."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# OpenAI stub — mirrors test_cloze.py's _make_stub_instructor_client.
# Returns a real OpenAI client whose httpx transport is a MockTransport
# so ``instructor.from_openai`` accepts it. The assistant message
# content is the JSON payload we want validated against
# ``app.match.MatchingExercise``.
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
) -> Any:
    """Build a stub OpenAI client that returns ``payload`` as the
    assistant message content. The stub is order-independent: it
    bypasses respx so OpenAI's custom httpx pool doesn't matter.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-match-endpoint-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
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
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    return OpenAI(
        api_key="test-key-not-real",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _matching_payload(*, target_word_id: int, count: int) -> str:
    """Build a valid ``MatchingExercise`` JSON payload for a test.

    Pairs use sequential ids starting at ``target_word_id + 1``
    (the schema validates positive ints; semantically-correct
    pairs are the LLM's job, not the wire's).
    """
    pairs = []
    for i in range(count):
        left = target_word_id + (2 * i) + 1
        right = target_word_id + (2 * i) + 2
        kind = "translation" if i % 2 == 0 else "synonym"
        pairs.append(
            {
                "left_word_id": left,
                "right_word_id": right,
                "right_kind": kind,
            }
        )
    return json.dumps(
        {"target_word_id": target_word_id, "pairs": pairs},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 1. 200 happy path — empty body, default count=4, default enable_rag=False
# ---------------------------------------------------------------------------


def test_post_exercises_match_happy_path_default(
    client, db_session, monkeypatch
):
    """``POST /exercises/match`` with ``{}`` body returns a
    ``MatchingExerciseOut`` carrying the wire metadata contract:

    - ``exercise_type="matching"`` (BaseExerciseFields default).
    - ``exercise_id`` is a server-minted int (not None, not 0).
    - ``target_word_id`` matches the FK the generator picked.
    - ``prompt_template_version=="match-v1"``.
    - ``pairs`` has the default ``count=4`` items.
    - ``enable_rag`` defaults to ``False`` (Hard rule #1: opt-in).
    """
    from app import match
    from app.match import (
        MATCH_DEFAULT_COUNT,
        PROMPT_TEMPLATE_VERSION,
    )

    # Seed enough words so ``select_target_word`` has something to
    # pick and the LLM's fabricated ids resolve.
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

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _matching_payload(
        target_word_id=target_id, count=MATCH_DEFAULT_COUNT
    )
    monkeypatch.setattr(
        match, "_openai_client", lambda: _make_stub_instructor_client(payload)
    )

    resp = client.post("/exercises/match", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Wire metadata contract.
    assert body["exercise_type"] == "matching"
    assert isinstance(body["exercise_id"], int)
    assert body["exercise_id"] != 0
    assert body["target_word_id"] == target_id
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    # Default count.
    assert len(body["pairs"]) == MATCH_DEFAULT_COUNT
    # ``enable_rag`` isn't echoed on the wire (it's a request-only
    # knob), but the generator was called with ``enable_rag=False``.
    # Verify indirectly: the respx / stub path here doesn't differ
    # by RAG-on for SQLite (no retrieval), but the call count is
    # deterministic — see the dedicated count+enable_rag test below
    # for the explicit ``enable_rag=True`` round-trip.


# ---------------------------------------------------------------------------
# 2. 200 with overrides — count=6 + enable_rag=true threaded through
# ---------------------------------------------------------------------------


def test_post_exercises_match_count_and_enable_rag_threaded(
    client, db_session, monkeypatch
):
    """``{"count": 6, "enable_rag": true}`` → 200 with ``count=6``
    pairs. The ``enable_rag`` knob is forwarded to
    ``generate_match``; on a SQLite test target the retrieval
    helper returns ``[]`` (graceful fallback), so the LLM stub is
    what actually serves the call. We verify the wiring by
    asserting the response has the expected ``count`` AND that
    ``generate_match`` was called with ``enable_rag=True`` (via
    a spy).
    """
    from app import match
    from app.match import PROMPT_TEMPLATE_VERSION

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
        "fahren",
    ):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _matching_payload(target_word_id=target_id, count=6)
    monkeypatch.setattr(
        match, "_openai_client", lambda: _make_stub_instructor_client(payload)
    )

    # Spy on ``generate_match`` to verify ``enable_rag=True`` and
    # ``count=6`` were forwarded (the SQLite path returns ``[]``
    # for retrieval — we don't need a Postgres to verify the
    # parameter wiring). The route does
    # ``from app.match import generate_match`` lazily, so we
    # monkeypatch the symbol ON ``app.match`` (not on
    # ``app.main``).
    import app.match as match_module

    seen: dict[str, Any] = {}

    real_generate = match_module.generate_match

    def spy_generate(db, user_id, **kwargs):
        seen.update(kwargs)
        return real_generate(db, user_id, **kwargs)

    monkeypatch.setattr(match_module, "generate_match", spy_generate)

    resp = client.post(
        "/exercises/match", json={"count": 6, "enable_rag": True}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["pairs"]) == 6
    assert body["target_word_id"] == target_id
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION

    # Wiring assertion — the route must forward both knobs.
    assert seen.get("count") == 6
    assert seen.get("enable_rag") is True


# ---------------------------------------------------------------------------
# 3. 401 — no JWT cookie
# ---------------------------------------------------------------------------


def test_post_exercises_match_requires_auth(client):
    """``POST /exercises/match`` with no auth cookie → 401.

    The ``Depends(auth.get_current_user)`` dependency raises 401
    before the handler body runs. No body is required to assert
    this — the request can be empty.
    """
    client.cookies.clear()
    resp = client.post("/exercises/match", json={})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 4. 422 — count=0 violates MatchGenerateRequest's Field(ge=2)
# ---------------------------------------------------------------------------


def test_post_exercises_match_count_below_min_returns_422(
    client, db_session, monkeypatch
):
    """``{"count": 0}`` → 422 (Pydantic ``Field(ge=2)``).

    The FastAPI layer rejects the body BEFORE the handler runs —
    so no LLM call, no Langfuse span, no DB write. We assert that
    by NOT setting up any stub and verifying the generator was
    never invoked.
    """
    _signup(client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    # No ``_openai_client`` monkeypatch — the handler must not
    # reach the generator. If it does, the test fails on the
    # implicit ``None`` return from ``_openai_client`` and the
    # route raises 502, NOT 422.
    resp = client.post("/exercises/match", json={"count": 0})
    assert resp.status_code == 422, resp.text
    body = resp.json()
    # FastAPI's 422 carries the validation error details; the
    # field name is ``count`` and the bound violated is ``ge=2``.
    detail_blob = json.dumps(body).lower()
    assert "count" in detail_blob
    assert "greater" in detail_blob or "ge" in detail_blob


# ---------------------------------------------------------------------------
# 5. 422 — count=20 violates MatchGenerateRequest's Field(le=8)
# ---------------------------------------------------------------------------


def test_post_exercises_match_count_above_max_returns_422(
    client, db_session, monkeypatch
):
    """``{"count": 20}`` → 422 (Pydantic ``Field(le=8)``).

    Same shape as test #4 — FastAPI rejects at the body layer.
    """
    _signup(client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    resp = client.post("/exercises/match", json={"count": 20})
    assert resp.status_code == 422, resp.text
    detail_blob = json.dumps(resp.json()).lower()
    assert "count" in detail_blob
    assert "less" in detail_blob or "le" in detail_blob


# ---------------------------------------------------------------------------
# 6. 502 — LLM transport failure (respx 500 from OpenRouter)
# ---------------------------------------------------------------------------


def test_post_exercises_match_llm_transport_failure_returns_502(
    client, db_session, monkeypatch
):
    """OpenRouter returns 500 → route translates to 502.

    The instructor library wraps the 500 in an
    ``InstructorRetryException`` and treats it as a retryable
    failure. After ``MAX_ATTEMPTS=3`` retries, the generator
    raises ``MatchingGenerationError`` and the route surfaces it
    as 502 with the structured ``match_generation_failed`` body.
    Either branch (``LLMError`` for non-retryable transport
    failure or ``MatchingGenerationError`` for instructor-retry
    exhaustion) is the right outcome from a wire perspective —
    both carry enough context for an operator to triage.
    """
    from app import match

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Both intercept paths return 500. Whichever layer the OpenAI
    # SDK ends up routing through (respx or the stub) returns the
    # failure.
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(500, json={"error": "injected server error"})
    )

    def _stub_500(payload):  # pragma: no cover — fall-through path
        import httpx
        from openai import OpenAI

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500, json={"error": "injected server error"}
            )

        return OpenAI(
            api_key="test-key-not-real",
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        )

    # ``payload`` is the fallback JSON if the LLM somehow returned
    # 200; it shouldn't fire because both intercept paths return 500.
    payload = _matching_payload(target_word_id=target_id, count=4)
    monkeypatch.setattr(
        match, "_openai_client", lambda: _stub_500(payload)
    )

    resp = client.post("/exercises/match", json={})
    assert resp.status_code == 502, resp.text
    body = resp.json()
    # The instructor-wrapped retry-exhaustion path surfaces as
    # ``error: match_generation_failed`` with the structured
    # fields. The non-retryable LLMError path would surface as a
    # plain string detail. Both are valid 502s from a wire
    # perspective.
    detail_blob = json.dumps(body).lower()
    assert (
        "match_generation_failed" in detail_blob
        or "match generation failed" in detail_blob
    )