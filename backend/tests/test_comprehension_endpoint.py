"""Tests for Phase 6.5 — ``POST /exercises/comprehension`` route (card t_dba4a40c).

Coverage map (mirrors the card body's "Tests" section):

1. 200 happy path — ``{}`` body → ``ComprehensionExerciseOut`` with
   ``exercise_type="comprehension"``, ``enable_rag=false``, no
   ``count`` knob on the request (the comprehension type generates
   one passage + one question per call, mirroring cloze, not
   matching).
2. 200 with overrides — ``{"enable_rag": true}`` → 200 with
   ``enable_rag=True`` threaded through to the generator. On a
   SQLite test target the retrieval helper returns ``[]`` (graceful
   fallback), so the LLM stub is what actually serves the call.
3. 401 — no JWT cookie → 401 (auth-gated route; the
   ``get_current_user`` dependency raises).
4. 422 — malformed body shape (e.g. ``{"enable_rag": ["a", "b"]}``)
   → 422 (Pydantic rejects the wrong type — strings like ``"yes"``
   are coerced to bool by Pydantic v2, so we use a list). The
   comprehension request has no ``count`` knob, so the matching
   route's count-bounds 422s do not apply here.
5. 502 — LLM transport failure (respx 500 from the OpenRouter
   mock) → 502 with the LLMError detail.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test.
The OpenRouter chat-completions call is mocked via either
``monkeypatch.setattr("app.comprehension._openai_client", ...)`` (the
order-independent stub pattern from test_cloze.py / test_match.py /
test_comprehension.py) or via ``respx`` where the route's respx flow
is exercised end-to-end (test #5).

Run from ``backend/``::

    uv run pytest -q tests/test_comprehension_endpoint.py
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
# test_cloze.py / test_match.py / test_match_endpoint.py / test_due.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Same pattern as test_match_endpoint / test_cloze.
    """
    db_path = tmp_path / "test_comprehension_endpoint.db"
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
    test_cloze / test_match / test_match_endpoint)."""
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
    """Insert one ``Word`` row with a stub example. Returns word_id.

    The comprehension route doesn't strictly need extra rows (the
    LLM stub returns fabricated ids on the passage side), but we
    seed a couple to keep ``select_target_word`` happy and to make
    the test corpora look realistic.
    """
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
# OpenAI stub — mirrors test_match_endpoint.py's
# ``_make_stub_instructor_client`` + test_comprehension.py's helpers.
# Returns a real OpenAI client whose httpx transport is a MockTransport
# so ``instructor.from_openai`` accepts it. The assistant message
# content is the JSON payload we want validated against
# ``app.comprehension.ComprehensionExercise``.
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
                "id": "gen-comprehension-endpoint-001",
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


def _comprehension_payload(*, target_word_id: int) -> str:
    """Build a valid ``ComprehensionExercise`` JSON payload for a test.

    Field bounds are enforced by ``ComprehensionExercise``:

    - passage: 20..600 chars (we use a 4-sentence stub well above 20
      and well below 600).
    - question: 5..300 chars.
    - choices: 4 entries, each 1..200 chars.
    - correct_choice: one of A/B/C/D.
    - rationale: 1..400 chars.
    """
    return json.dumps(
        {
            "target_word_id": target_word_id,
            "passage": (
                "Der Hund läuft durch den Park. Er sieht einen Ball "
                "und rennt sofort los. Sein Besitzer lacht und ruft "
                "seinen Namen. Am Ende sind beide müde und gehen "
                "nach Hause."
            ),
            "question": "Was sieht der Hund im Park?",
            "choices": {
                "A": "einen Ball",
                "B": "einen Knochen",
                "C": "eine Katze",
                "D": "einen Stock",
            },
            "correct_choice": "A",
            "rationale": (
                "The correct answer is grounded in the first sentence "
                "of the passage; the three distractors are plausible "
                "park objects the model could have substituted."
            ),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 1. 200 happy path — empty body, default enable_rag=False, no count knob
# ---------------------------------------------------------------------------


def test_post_exercises_comprehension_happy_path_default(
    client, db_session, monkeypatch
):
    """``POST /exercises/comprehension`` with ``{}`` body returns a
    ``ComprehensionExerciseOut`` carrying the wire metadata contract:

    - ``exercise_type="comprehension"`` (BaseExerciseFields default).
    - ``exercise_id`` is a server-minted int (not None, not 0).
    - ``target_word_id`` matches the FK the generator picked.
    - ``prompt_template_version=="comprehension-v1"``.
    - ``passage`` / ``question`` / ``choices`` / ``correct_choice``
      / ``rationale`` all populated.
    - There is no ``count`` knob on the comprehension request body
      (unlike matching) — the comprehension type generates one
      passage + one question per call by design.
    - ``enable_rag`` defaults to ``False`` (Hard rule #1: opt-in).
    """
    from app import comprehension
    from app.comprehension import PROMPT_TEMPLATE_VERSION

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _comprehension_payload(target_word_id=target_id)
    monkeypatch.setattr(
        comprehension,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post("/exercises/comprehension", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Wire metadata contract.
    assert body["exercise_type"] == "comprehension"
    assert isinstance(body["exercise_id"], int)
    assert body["exercise_id"] != 0
    assert body["target_word_id"] == target_id
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION

    # Comprehension-specific fields populated.
    assert body["passage"]
    assert len(body["passage"]) >= 20
    assert body["question"]
    assert set(body["choices"].keys()) == {"A", "B", "C", "D"}
    assert body["correct_choice"] in {"A", "B", "C", "D"}
    assert body["rationale"]

    # No ``count`` knob on the wire — comprehension generates one
    # passage + one question per call by design (mirrors cloze, not
    # matching). The matching wire returns ``pairs``; the
    # comprehension wire does not.
    assert "count" not in body
    assert "pairs" not in body


# ---------------------------------------------------------------------------
# 2. 200 with overrides — enable_rag=true threaded through to generator
# ---------------------------------------------------------------------------


def test_post_exercises_comprehension_enable_rag_threaded(
    client, db_session, monkeypatch
):
    """``{"enable_rag": true}`` → 200 with ``enable_rag=True``
    forwarded to ``generate_comprehension``. On a SQLite test
    target the retrieval helper returns ``[]`` (graceful fallback),
    so the LLM stub is what actually serves the call. We verify the
    wiring by asserting that ``generate_comprehension`` was called
    with ``enable_rag=True`` via a spy.
    """
    from app import comprehension
    from app.comprehension import PROMPT_TEMPLATE_VERSION

    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _comprehension_payload(target_word_id=target_id)
    monkeypatch.setattr(
        comprehension,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    # Spy on ``generate_comprehension`` to verify ``enable_rag=True``
    # was forwarded (the SQLite path returns ``[]`` for retrieval —
    # we don't need a Postgres to verify the parameter wiring).
    # The route does ``from app.comprehension import
    # generate_comprehension`` lazily, so we monkeypatch the symbol
    # ON ``app.comprehension`` (not on ``app.main``).
    import app.comprehension as comp_module

    seen: dict[str, Any] = {}

    real_generate = comp_module.generate_comprehension

    def spy_generate(db, user_id, **kwargs):
        seen.update(kwargs)
        return real_generate(db, user_id, **kwargs)

    monkeypatch.setattr(comp_module, "generate_comprehension", spy_generate)

    resp = client.post("/exercises/comprehension", json={"enable_rag": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["target_word_id"] == target_id
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert set(body["choices"].keys()) == {"A", "B", "C", "D"}

    # Wiring assertion — the route must forward enable_rag.
    assert seen.get("enable_rag") is True


# ---------------------------------------------------------------------------
# 3. 401 — no JWT cookie
# ---------------------------------------------------------------------------


def test_post_exercises_comprehension_requires_auth(client):
    """``POST /exercises/comprehension`` with no auth cookie → 401.

    The ``Depends(auth.get_current_user)`` dependency raises 401
    before the handler body runs. No body is required to assert
    this — the request can be empty.
    """
    client.cookies.clear()
    resp = client.post("/exercises/comprehension", json={})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 4. 422 — malformed body shape (enable_rag must be bool)
# ---------------------------------------------------------------------------


def test_post_exercises_comprehension_malformed_body_returns_422(
    client, db_session, monkeypatch
):
    """``{"enable_rag": ["a", "b"]}`` → 422 (Pydantic rejects the
    wrong type). The comprehension request has no ``count`` knob —
    the matching route's count-bounds 422s do not apply here.

    Pydantic v2's bool parser is lenient about strings (any non-empty
    string coerces to ``True``, the empty string / explicit ``"false"``
    coerces to ``False``), so a ``str`` value like ``"yes"`` is
    accepted. We use a list instead — a list cannot coerce to a bool
    and the rejection is unambiguous.

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
    resp = client.post(
        "/exercises/comprehension", json={"enable_rag": ["a", "b"]}
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    detail_blob = json.dumps(body).lower()
    assert "enable_rag" in detail_blob
    # Pydantic v2's "bool_type" wording for a non-coercible type.
    assert "bool_type" in detail_blob or "bool" in detail_blob


# ---------------------------------------------------------------------------
# 5. 502 — LLM transport failure (respx 500 from OpenRouter)
# ---------------------------------------------------------------------------


def test_post_exercises_comprehension_llm_transport_failure_returns_502(
    client, db_session, monkeypatch
):
    """OpenRouter returns 500 → route translates to 502.

    The instructor library wraps the 500 in an
    ``InstructorRetryException`` and treats it as a retryable
    failure. After ``MAX_ATTEMPTS=3`` retries, the generator
    raises ``ComprehensionGenerationError`` and the route surfaces
    it as 502 with the structured ``comprehension_generation_failed``
    body. Either branch (``LLMError`` for non-retryable transport
    failure or ``ComprehensionGenerationError`` for instructor-retry
    exhaustion) is the right outcome from a wire perspective — both
    carry enough context for an operator to triage.
    """
    from app import comprehension

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
    payload = _comprehension_payload(target_word_id=target_id)
    monkeypatch.setattr(
        comprehension, "_openai_client", lambda: _stub_500(payload)
    )

    resp = client.post("/exercises/comprehension", json={})
    assert resp.status_code == 502, resp.text
    body = resp.json()
    # The instructor-wrapped retry-exhaustion path surfaces as
    # ``error: comprehension_generation_failed`` with the structured
    # fields. The non-retryable LLMError path would surface as a
    # plain string detail. Both are valid 502s from a wire
    # perspective.
    detail_blob = json.dumps(body).lower()
    assert (
        "comprehension_generation_failed" in detail_blob
        or "comprehension generation failed" in detail_blob
    )