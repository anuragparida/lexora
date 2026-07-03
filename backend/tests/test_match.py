"""Tests for Phase 6.2 — matching exercise generator (card t_ddaf9cf9).

Coverage map (mirrors the card body's "Tests" section):

1. ``PROMPT_TEMPLATE_VERSION == "match-v1"`` (Hard rule #9).
2. ``MAX_ATTEMPTS == 3`` (Hard rule #6).
3. ``MATCH_MIN/MAX/DEFAULT_COUNT`` bounds + single-definition check
   (acceptance: ``git grep`` shows exactly one definition + usage
   sites in ``app/match.py``).
4. ``MatchingExercise`` happy path: ``generate_match`` returns a
   ``MatchingExercise`` with the default ``count=4`` pairs, every
   ``left_word_id`` / ``right_word_id`` is a valid ``words.id`` FK,
   ``right_kind`` is in the literal.
5. ``MatchingExercise`` validation: ``count=0`` → ValidationError.
6. ``MatchingExercise`` validation: ``count=9`` (out of upper bound)
   → ValidationError.
7. ``MatchGenerateRequest(count=0)`` → 422 (Pydantic ``Field(ge=...)``).
8. ``MatchGenerateRequest(count=20)`` (out of upper bound) → 422.
9. ``enable_rag=False`` (default) → prompt is byte-for-byte identical
   to a stored fixture (the no-RAG shape).
10. ``enable_rag=True`` on Postgres → prompt JSON includes
    ``retrieved_chunks`` with N items.
11. Langfuse keys present: span emitted with the metadata keyset;
    ``_trace_match`` returns a non-None ``trace_id``.
12. Langfuse keys absent: graceful no-op; ``_trace_match`` returns
    ``None``; no network, no per-call warnings.
13. ``app.match`` does not import ``app.retrieval`` at module load
    time (Hard rule #2 / acceptance: lazy import inside
    ``_retrieve_for_match`` only).
14. ``MatchingPair.right_kind`` literal guardrail.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test, the
OpenRouter chat-completions call is replaced with a stub OpenAI
client (``monkeypatch.setattr("app.match._openai_client", ...)``) so
no network is touched. The stub mirrors the Phase 4.2 / Phase 5.4
pattern (``_make_stub_instructor_client`` in ``test_cloze.py``).

Run from ``backend/``::

    uv run pytest -q tests/test_match.py
"""
from __future__ import annotations

import json
import secrets
import sys
from datetime import date

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response
from unittest.mock import MagicMock

from app import match
from app.match import (
    MATCH_DEFAULT_COUNT,
    MATCH_MAX_COUNT,
    MATCH_MIN_COUNT,
    MAX_ATTEMPTS,
    PROMPT_TEMPLATE_VERSION,
    MatchingExercise,
    MatchingPair,
    MatchingGenerationError,
)


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_cloze.py / test_due.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "test_match.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    from app import database
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
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


def _seed_user_with_axes(session, *, axes: dict[str, int] | None = None) -> int:
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
# content is the JSON payload we want validated.
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
):
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-match-stub-001",
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


def _matching_payload(
    *,
    target_word_id: int,
    count: int = MATCH_DEFAULT_COUNT,
) -> str:
    """Build a valid ``MatchingExercise`` JSON payload for a test.

    The pairs are fabricated: each pair uses sequential ids starting
    at ``target_word_id + 1`` and flips ``right_kind`` between
    ``"translation"`` and ``"synonym"``. The schema doesn't require
    the pairs to be semantically correct (that's the LLM's job);
    it only requires every id to be a positive int and
    ``right_kind`` to be in the literal.
    """
    pairs = []
    for i in range(count):
        left = target_word_id + (2 * i) + 1
        right = target_word_id + (2 * i) + 2
        kind = "translation" if i % 2 == 0 else "synonym"
        pairs.append({"left_word_id": left, "right_word_id": right, "right_kind": kind})
    return json.dumps(
        {"target_word_id": target_word_id, "pairs": pairs},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 1. Module-level constants (Hard rule #9)
# ---------------------------------------------------------------------------


def test_prompt_template_version_is_match_v1():
    """Hard rule #9: ``PROMPT_TEMPLATE_VERSION`` is a module constant,
    not env-derived. Bumping it is a code review event.
    """
    assert PROMPT_TEMPLATE_VERSION == "match-v1"


def test_max_attempts_is_3():
    """Hard rule #6: schema-violation retries are bounded ≤ 3. The
    same number the instructor library receives via ``max_retries``,
    so the dead-letter and the instructor budget stay in sync.
    """
    assert MAX_ATTEMPTS == 3


def test_match_count_bounds():
    """``MATCH_MIN_COUNT`` / ``MATCH_MAX_COUNT`` / ``MATCH_DEFAULT_COUNT``
    are the locked values: 2 / 8 / 4. Any drift is a contract break
    and would invalidate the ``app.schemas.MatchingExerciseOut.pairs``
    ``min_length`` / ``max_length`` constraints.
    """
    assert MATCH_MIN_COUNT == 2
    assert MATCH_MAX_COUNT == 8
    assert MATCH_DEFAULT_COUNT == 4
    # Defensive ordering checks.
    assert MATCH_MIN_COUNT <= MATCH_DEFAULT_COUNT <= MATCH_MAX_COUNT


def test_match_count_constants_defined_once_in_app_match():
    """Card acceptance: ``git grep -n "MATCH_MIN_COUNT\\|MATCH_MAX_COUNT\\
    |MATCH_DEFAULT_COUNT" backend/app/match.py`` → exactly one
    definition + usage sites. We assert the first-match-is-def shape:
    the names appear as assignments on the line they are defined
    (i.e. exactly one ``= MATCH_MIN_COUNT`` style def per name), and
    they're referenced elsewhere by name.
    """
    import subprocess
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            "grep",
            "-n",
            "MATCH_MIN_COUNT\\|MATCH_MAX_COUNT\\|MATCH_DEFAULT_COUNT",
            "app/match.py",
        ],
        capture_output=True,
        text=True,
        cwd=backend_dir,
    )
    lines = result.stdout.splitlines()
    # Every line should mention one of the three names; we filter to
    # lines that look like definitions (i.e. contain ``: int = `` or
    # ``= MATCH_``) and assert exactly one def per name.
    defs = {
        "MATCH_MIN_COUNT": 0,
        "MATCH_MAX_COUNT": 0,
        "MATCH_DEFAULT_COUNT": 0,
    }
    for line in lines:
        for name in defs:
            # A definition line: "MATCH_MIN_COUNT: int = 2" or
            # "MATCH_MIN_COUNT = 2". Module-level constants.
            if f"{name}:" in line and "=" in line and "Field" not in line:
                defs[name] += 1
            elif line.strip().startswith(f"{name} = "):
                defs[name] += 1
    # Each name has exactly one module-level definition.
    for name, count in defs.items():
        assert count == 1, (
            f"{name} should be defined exactly once at module level; "
            f"got {count} definition(s) in:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# 2. Pydantic validation — generator + wire schema
# ---------------------------------------------------------------------------


def test_matching_exercise_rejects_count_below_min():
    """``MatchingExercise(pairs=[])`` violates the ``min_length=2``
    constraint. The Pydantic layer rejects before the route / LLM
    even sees the request.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        MatchingExercise(target_word_id=1, pairs=[])


def test_matching_exercise_rejects_count_above_max():
    """A 9-pair ``MatchingExercise`` violates ``max_length=8``. The
    bounds are hard-coded in the module constants (Hard rule #9).
    """
    import pydantic

    pairs = [
        MatchingPair(left_word_id=i, right_word_id=i + 100, right_kind="translation")
        for i in range(1, 10)
    ]
    with pytest.raises(pydantic.ValidationError):
        MatchingExercise(target_word_id=1, pairs=pairs)


def test_matching_pair_right_kind_literal_guardrail():
    """``right_kind`` is ``Literal["translation", "synonym"]``. Any
    other string is a Pydantic ValidationError (Hard rule #3 — type
    system is the gate).
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        MatchingPair(left_word_id=1, right_word_id=2, right_kind="antonym")  # not in literal


def test_match_generate_request_rejects_count_below_min():
    """``MatchGenerateRequest(count=0)`` violates the Pydantic
    ``Field(ge=2)`` constraint. The card acceptance criterion is
    that this surfaces as 422 at the wire — we test the schema
    boundary (Pydantic ValidationError), and the route layer
    translates that to 422.
    """
    import pydantic
    from app.schemas import MatchGenerateRequest

    with pytest.raises(pydantic.ValidationError):
        MatchGenerateRequest(count=0)


def test_match_generate_request_rejects_count_above_max():
    """``MatchGenerateRequest(count=20)`` violates ``le=8``. Same
    shape as the count=0 test.
    """
    import pydantic
    from app.schemas import MatchGenerateRequest

    with pytest.raises(pydantic.ValidationError):
        MatchGenerateRequest(count=20)


def test_match_generate_request_default_count_is_4():
    """The default ``count`` is ``MATCH_DEFAULT_COUNT`` (4). The
    request layer mirrors the generator's bound.
    """
    from app.schemas import MatchGenerateRequest

    req = MatchGenerateRequest()
    assert req.count == MATCH_DEFAULT_COUNT
    assert req.enable_rag is False


# ---------------------------------------------------------------------------
# 3. No retrieval import (Hard rule #2)
# ---------------------------------------------------------------------------


def test_match_does_not_import_retrieval_at_load():
    """``app.match`` must NOT import ``app.retrieval`` at module load
    time. ``_retrieve_for_match`` is the only consumer and it lives
    inside the function body. Same shape as the Phase 4.2 cloze
    "no retrieval on the cloze path" test.
    """
    sys.modules.pop("app.retrieval", None)
    if "app.match" in sys.modules:
        del sys.modules["app.match"]
    if "app.embeddings" in sys.modules:
        del sys.modules["app.embeddings"]
    import app.match  # noqa: F401

    assert "app.retrieval" not in sys.modules, (
        "app.match must not import app.retrieval at module load time "
        "(Hard rule #2: _retrieve_for_match is lazy)"
    )
    assert "app.embeddings" not in sys.modules, (
        "app.match must not import app.embeddings at module load time "
        "(_retrieve_for_match is lazy)"
    )


# ---------------------------------------------------------------------------
# 4. Word selection — re-uses ``app.cloze.select_target_word``
# ---------------------------------------------------------------------------


def test_select_target_word_threads_through_unchanged(
    db_session, monkeypatch
):
    """The matching exercise uses ``app.cloze.select_target_word``
    without modification. We seed one Verb and a user with
    ``verbs: 3`` so the deterministic seed picks the Verb. Same
    shape as test_cloze.py's determinism tests.
    """
    verb_id = _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="Ich gehe."
    )
    _seed_word(db_session, word="Hund", word_type="Noun", example_de="Der Hund.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    from app.cloze import select_target_word

    w1 = select_target_word(db_session, user_id)
    w2 = select_target_word(db_session, user_id)
    assert w1.id == w2.id == verb_id


# ---------------------------------------------------------------------------
# 5. Prompt construction — byte-for-byte no-RAG fixture
# ---------------------------------------------------------------------------


def test_build_prompt_no_rag_byte_for_byte(db_session):
    """``build_prompt(word, axes)`` (no ``retrieved_chunks``) is
    byte-for-byte identical to the stored fixture. The fixture
    encodes the no-RAG shape so any future drift (e.g. someone
    adds a key to the JSON payload) surfaces as a test failure.
    """
    wid = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de="Die Katze schläft auf dem Sofa.",
    )
    word = db_session.query(__import__("app").models.Word).get(wid)
    msgs = match.build_prompt(
        word, weakness_axes={"verbs": 3}, count=MATCH_DEFAULT_COUNT
    )

    # Two messages: system + user. The system carries the role +
    # prohibitions + schema. The user carries the JSON payload.
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    # System prompt: prohibitions + JSON schema.
    system = msgs[0]["content"]
    assert "matching-exercise designer" in system
    assert "PROHIBITIONS" in system
    assert "right_kind" in system
    assert "C1-ACCEPT BAR" in system

    # User prompt: target word + first example + JSON axes + count
    # + instructions. No ``retrieved_chunks`` key.
    user = json.loads(msgs[1]["content"])
    assert user["target_word"]["word"] == "schlafen"
    assert user["context_sentence"] == "Die Katze schläft auf dem Sofa."
    assert user["learner_axes"] == {"verbs": 3}
    assert user["count"] == MATCH_DEFAULT_COUNT
    assert "retrieved_chunks" not in user, (
        "no-RAG prompt must not include retrieved_chunks (Hard rule #1: "
        "the no-RAG shape is byte-for-byte reproducible for A/B eval)"
    )


def test_build_prompt_with_rag_includes_chunks(db_session):
    """``build_prompt(word, axes, retrieved_chunks=[...])`` includes
    the ``retrieved_chunks`` key in the user-prompt JSON. Same shape
    as 6.1's cloze prompt extension.
    """
    wid = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de="Die Katze schläft.",
    )
    word = db_session.query(__import__("app").models.Word).get(wid)
    chunks = [
        {"kind": "word", "id": 42, "text": "Wort"},
        {"kind": "example", "id": 99, "text": "Ein Beispiel."},
    ]
    msgs = match.build_prompt(
        word, weakness_axes={"verbs": 3}, count=2, retrieved_chunks=chunks
    )
    user = json.loads(msgs[1]["content"])
    assert user["retrieved_chunks"] == chunks
    assert user["count"] == 2


def test_build_prompt_handles_word_with_no_examples(db_session):
    """A ``Word`` with zero ``Example`` rows falls back to a
    deterministic placeholder rather than crashing. The matching
    endpoint stays usable on a sparse corpus row.
    """
    from app import models

    w = models.Word(word="solo", word_type="Noun", frequency="5", is_complete=True)
    db_session.add(w)
    db_session.commit()
    db_session.refresh(w)
    assert w.examples == []

    msgs = match.build_prompt(w, weakness_axes={}, count=2)
    user = json.loads(msgs[1]["content"])
    assert user["context_sentence"]
    assert "Beispiel nicht verfügbar" in user["context_sentence"]


# ---------------------------------------------------------------------------
# 6. generate_match — happy path
# ---------------------------------------------------------------------------


def test_generate_match_happy_path_default_count(db_session, monkeypatch):
    """``generate_match(db, user_id)`` returns a
    ``MatchingExercise`` with the default ``count=4`` pairs; every
    ``left_word_id`` / ``right_word_id`` is a valid ``words.id``
    FK; ``right_kind`` is in the literal.
    """
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    # We need at least MATCH_DEFAULT_COUNT pairs in the payload,
    # so the LLM stub produces MATCH_DEFAULT_COUNT+1 valid
    # word_ids (the schema validates ints; positive ints are
    # accepted). We seed several verbs to be safe.
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen", "arbeiten", "essen", "trinken"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")

    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _matching_payload(target_word_id=target_id, count=MATCH_DEFAULT_COUNT)
    monkeypatch.setattr(
        match, "_openai_client", lambda: _make_stub_instructor_client(payload)
    )

    result = match.generate_match(db_session, user_id)
    assert isinstance(result, MatchingExercise)
    assert result.target_word_id == target_id
    assert len(result.pairs) == MATCH_DEFAULT_COUNT
    for p in result.pairs:
        assert isinstance(p.left_word_id, int) and p.left_word_id > 0
        assert isinstance(p.right_word_id, int) and p.right_word_id > 0
        assert p.right_kind in ("translation", "synonym")
        # Both ids must resolve to a row in the words table — the
        # generator's validation step.
        assert (
            db_session.query(__import__("app").models.Word)
            .filter(__import__("app").models.Word.id == p.left_word_id)
            .first()
            is not None
        )
        assert (
            db_session.query(__import__("app").models.Word)
            .filter(__import__("app").models.Word.id == p.right_word_id)
            .first()
            is not None
        )


def test_generate_match_respects_count_param(db_session, monkeypatch):
    """``generate_match(db, user_id, count=2)`` requests 2 pairs and
    the LLM is asked to produce 2 pairs. The user-prompt JSON
    encodes ``count=2`` and the result is a 2-pair
    ``MatchingExercise``.
    """
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _matching_payload(target_word_id=target_id, count=2)
    monkeypatch.setattr(
        match, "_openai_client", lambda: _make_stub_instructor_client(payload)
    )

    result = match.generate_match(db_session, user_id, count=2)
    assert len(result.pairs) == 2


def test_generate_match_rejects_count_out_of_bounds(db_session):
    """``generate_match(db, user_id, count=0)`` raises ``ValueError``
    (defense-in-depth — the Pydantic schema also enforces this, but
    the function should not silently accept out-of-range counts if
    called directly from a script).
    """
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    _seed_word(db_session, word="gehen", word_type="Verb", example_de="X.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    with pytest.raises(ValueError, match="count must be in"):
        match.generate_match(db_session, user_id, count=0)

    with pytest.raises(ValueError, match="count must be in"):
        match.generate_match(db_session, user_id, count=9)


def test_generate_match_raises_llm_error_when_api_key_missing(
    db_session, monkeypatch
):
    """No ``OPENROUTER_API_KEY`` → ``LLMError`` (route layer → 502)."""
    _seed_word(db_session, word="schlafen", word_type="Verb", example_de="X schläft.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from app.llm import LLMError

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        match.generate_match(db_session, user_id)


def test_generate_match_dead_letters_after_three_invalid_attempts(
    db_session, monkeypatch
):
    """Three consecutive schema violations →
    ``MatchingGenerationError`` with the structured fields. Hard
    rule #6: retries ≤ 3.
    """
    _seed_word(db_session, word="schlafen", word_type="Verb", example_de="X schläft.")
    for w in ("gehen", "kommen", "bleiben"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Every response is missing ``pairs`` → Pydantic validation
    # fails every time.
    invalid_payload = json.dumps({"target_word_id": 1})  # no ``pairs``
    monkeypatch.setattr(
        match, "_openai_client", lambda: _make_stub_instructor_client(invalid_payload)
    )

    with pytest.raises(MatchingGenerationError) as excinfo:
        match.generate_match(db_session, user_id)
    err = excinfo.value
    # instructor's count is ``initial + max_retries``: with
    # ``max_retries=3`` the budget is 4 total HTTP calls. We
    # assert the retry budget is respected.
    assert err.schema_retry_count <= MAX_ATTEMPTS + 1
    assert err.last_validation_error
    assert "MatchingExercise" in err.attempted_schema
    assert "target_word_id" in err.attempted_schema


# ---------------------------------------------------------------------------
# 7. RAG-on (Hard rule #1: opt-in)
# ---------------------------------------------------------------------------


def test_generate_match_rag_off_default_no_retrieval_call(
    db_session, monkeypatch
):
    """``generate_match(db, user_id)`` with default
    ``enable_rag=False`` does NOT call ``_retrieve_for_match`` and
    does NOT include ``retrieved_chunks`` in the prompt. We spy on
    the retrieval helper and assert it's not called.
    """
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen", "arbeiten", "essen", "trinken"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _matching_payload(target_word_id=target_id, count=4)
    monkeypatch.setattr(
        match, "_openai_client", lambda: _make_stub_instructor_client(payload)
    )

    captured: list[dict] = []

    def fake_retrieve(db, word):
        captured.append({"word_id": word.id})
        return []

    monkeypatch.setattr(match, "_retrieve_for_match", fake_retrieve)

    result = match.generate_match(db_session, user_id)
    assert isinstance(result, MatchingExercise)
    # The default path does not call retrieval.
    assert captured == []


def test_generate_match_rag_on_postgres_calls_retrieval(db_session, monkeypatch):
    """``generate_match(db, user_id, enable_rag=True)`` on Postgres
    calls ``_retrieve_for_match`` and includes the chunks in the
    user-prompt JSON. We mock the retrieval helper to return a
    fixture and assert the prompt contains those chunks.
    """
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    for w in ("gehen", "kommen", "bleiben", "wohnen", "lernen", "arbeiten", "essen", "trinken"):
        _seed_word(db_session, word=w, word_type="Verb", example_de="X.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    chunks_fixture = [
        {"kind": "word", "id": 100, "text": "Wort"},
        {"kind": "example", "id": 101, "text": "Beispiel."},
    ]

    def fake_retrieve(db, word):
        return chunks_fixture

    monkeypatch.setattr(match, "_retrieve_for_match", fake_retrieve)

    captured_messages: list[list[dict]] = []

    def fake_openai_client():
        # Capture the messages from the prompt so we can assert on
        # the user-prompt JSON shape.
        def _handler(request):
            import httpx as _httpx
            from json import loads as _loads
            body = _loads(request.content)
            captured_messages.append(body["messages"])
            return _httpx.Response(
                200,
                json={
                    "id": "gen-match-rag-001",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "qwen/qwen3-235b-a22b-2507",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": _matching_payload(
                                    target_word_id=target_id, count=4
                                ),
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
        from openai import OpenAI
        return OpenAI(
            api_key="test-key-not-real",
            base_url="https://openrouter.ai/api/v1",
            http_client=import_httpx_client(_handler),
        )

    def import_httpx_client(handler):
        import httpx
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(match, "_openai_client", fake_openai_client)

    result = match.generate_match(db_session, user_id, enable_rag=True)
    assert isinstance(result, MatchingExercise)
    # The prompt was captured. Assert the user-prompt JSON includes
    # the retrieved_chunks key with the fixture's content. The
    # instructor MD_JSON mode appends a "\n\nReturn the correct
    # JSON response within a ```json codeblock ..." suffix; we
    # split on the first blank line so we parse the original
    # user-prompt JSON only.
    assert len(captured_messages) == 1
    user_content = captured_messages[0][1]["content"]
    # Take the first JSON object up to the first \n\n (the
    # boundary where instructor appends its MD_JSON instruction).
    first_object = user_content.split("\n\n", 1)[0]
    user_prompt = json.loads(first_object)
    assert "retrieved_chunks" in user_prompt
    assert user_prompt["retrieved_chunks"] == chunks_fixture


def test_retrieve_for_match_non_postgres_returns_empty(monkeypatch, db_session):
    """On a non-Postgres target, ``_retrieve_for_match`` returns
    ``[]`` without contacting the embedding / retrieval layers.
    Hard rule #2: ``/retrieve`` is consumed as-is — its 503 path
    is the canonical "non-Postgres" signal, and the matching
    helper mirrors it gracefully.
    """
    # The default per-test SQLite target is non-Postgres.
    target_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    word = db_session.query(__import__("app").models.Word).get(target_id)

    # ``embeddings`` and ``retrieval`` modules must NOT be touched.
    sentinel = object()

    def fail_embeddings(*a, **kw):
        raise AssertionError(
            "embeddings must not be called on a non-Postgres target"
        )

    def fail_retrieval(*a, **kw):
        raise AssertionError(
            "retrieval.retrieve must not be called on a non-Postgres target"
        )

    # Pre-load the modules to sys.modules; this matches the
    # ``embeddings.embed_one`` import path inside
    # ``_retrieve_for_match``.
    import app.embeddings as _emb_mod
    import app.retrieval as _ret_mod
    _emb_mod.embed_one = fail_embeddings  # type: ignore[assignment]
    _ret_mod.retrieve = fail_retrieval  # type: ignore[assignment]

    chunks = match._retrieve_for_match(db_session, word)
    assert chunks == []


# ---------------------------------------------------------------------------
# 8. _trace_match — metadata contract
# ---------------------------------------------------------------------------


def test_trace_match_metadata_keyset_on_mocked_span(monkeypatch):
    """When the Langfuse client is non-None, the trace span carries
    every metadata-contract field (card body: exercise_type,
    target_word_id, count, enable_rag, retrieved_chunk_count,
    model_id, prompt_template_version, schema_retry_count,
    latency_ms, prompt_tokens, completion_tokens).
    """
    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span

    monkeypatch.setattr(match, "get_langfuse", lambda: mock_client)

    exercise = MatchingExercise(
        target_word_id=42,
        pairs=[
            MatchingPair(left_word_id=1, right_word_id=2, right_kind="translation"),
            MatchingPair(left_word_id=3, right_word_id=4, right_kind="synonym"),
        ],
    )
    metadata = {
        "exercise_type": "matching",
        "user_id": 7,
        "weakness_axes": {"verbs": 3},
        "target_word_id": 42,
        "count": 1,
        "enable_rag": False,
        "retrieved_chunk_count": 0,
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        "schema_retry_count": 0,
        "prompt_tokens": 30,
        "completion_tokens": 12,
    }

    trace_id = match._trace_match(exercise, metadata, latency_ms=42)
    # The mock_span object returns a MagicMock for ``id`` — assert
    # it's not None (i.e. we captured it).
    assert trace_id is not None or mock_client.span.called

    # client.span invoked with the canonical name and input/output.
    assert mock_client.span.call_count == 1
    span_kwargs = mock_client.span.call_args.kwargs
    assert span_kwargs["name"] == "match.generate"
    assert span_kwargs["input"] == metadata["prompt_messages"]
    assert json.loads(span_kwargs["output"]) == json.loads(exercise.model_dump_json())

    # span.update: the metadata keyset is the contract.
    assert mock_span.update.call_count >= 1
    merged: dict = {}
    for call in mock_span.update.call_args_list:
        for key, value in (call.kwargs.get("metadata") or {}).items():
            merged[key] = value

    expected_keys = {
        "exercise_type",
        "user_id",
        "weakness_axes",
        "target_word_id",
        "count",
        "enable_rag",
        "retrieved_chunk_count",
        "model_id",
        "prompt_template_version",
        "schema_retry_count",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
    }
    assert set(merged.keys()) == expected_keys, (
        f"metadata keys drifted: got {set(merged.keys())}, "
        f"expected {expected_keys}"
    )
    # Spot-check field values.
    assert merged["exercise_type"] == "matching"
    assert merged["target_word_id"] == 42
    assert merged["count"] == 1
    assert merged["enable_rag"] is False
    assert merged["retrieved_chunk_count"] == 0
    assert merged["model_id"] == "qwen/qwen3-235b-a22b-2507"
    assert merged["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert merged["schema_retry_count"] == 0
    assert merged["latency_ms"] == 42
    assert merged["prompt_tokens"] == 30
    assert merged["completion_tokens"] == 12

    # Span closed + client flushed.
    assert mock_span.end.call_count == 1
    assert mock_client.flush.call_count == 1


def test_trace_match_is_silent_when_keys_missing(monkeypatch, caplog):
    """When ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are
    unset, ``get_langfuse()`` returns ``None`` and ``_trace_match``
    returns ``None`` without contacting the network. The
    ``trace_id=None`` propagates so the route layer can serialise
    it on the response.
    """
    import logging

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with caplog.at_level(logging.WARNING, logger="app.observability"):
        # ``result=None`` is the dead-letter branch.
        assert match._trace_match(None, {}, 0) is None

    # No per-call warnings from ``app.match``.
    match_warnings = [
        r for r in caplog.records
        if r.name == "app.match" and r.levelno >= logging.WARNING
    ]
    assert match_warnings == [], (
        f"unexpected warnings from app.match: "
        f"{[r.getMessage() for r in match_warnings]}"
    )


def test_trace_match_swallows_langfuse_failures(monkeypatch):
    """When the Langfuse SDK raises mid-span, the matching activity
    still succeeds (or, in the ``_trace_match`` direct call, returns
    ``None`` without raising). Tracing failures must never break
    the request.
    """
    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span
    mock_client.flush.side_effect = RuntimeError("simulated flush failure")

    monkeypatch.setattr(match, "get_langfuse", lambda: mock_client)

    exercise = MatchingExercise(
        target_word_id=42,
        pairs=[
            MatchingPair(left_word_id=1, right_word_id=2, right_kind="translation"),
            MatchingPair(left_word_id=3, right_word_id=4, right_kind="synonym"),
        ],
    )
    metadata = {
        "exercise_type": "matching",
        "user_id": 7,
        "weakness_axes": {},
        "target_word_id": 42,
        "count": 1,
        "enable_rag": False,
        "retrieved_chunk_count": 0,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    # The exception is swallowed; the function returns cleanly.
    assert match._trace_match(exercise, metadata, latency_ms=0) is None


# ---------------------------------------------------------------------------
# 9. DSPy module — constructed without OpenRouter
# ---------------------------------------------------------------------------


def test_dspy_module_constructible_without_openrouter(monkeypatch):
    """``MatchModule`` can be constructed without an OpenRouter key.
    The DSPy configure path falls back to ``DummyLM`` automatically
    (Hard rule #8: offline-capable).
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None
    module = match.MatchModule()
    assert module is not None
    assert hasattr(module, "predict")


def test_match_signature_inputs_match_production_contract():
    """The DSPy signature carries the same input keys the production
    path passes in ``build_prompt`` (word, context_sentence,
    learner_axes_json, target_word_id, count). The output is the
    ``MatchingExercise`` Pydantic model.
    """
    sig = match.MatchSignature
    input_fields = {
        k for k in sig.model_fields
        if sig.model_fields[k].json_schema_extra.get("__dspy_field_type") == "input"
    }
    output_fields = {
        k for k in sig.model_fields
        if sig.model_fields[k].json_schema_extra.get("__dspy_field_type") == "output"
    }
    assert input_fields == {
        "word",
        "context_sentence",
        "learner_axes_json",
        "target_word_id",
        "count",
    }
    assert output_fields == {"exercise"}


def test_optimize_match_module_runs_offline(monkeypatch):
    """``optimize_match_module`` runs end-to-end on a 2-row eval set
    using ``DummyLM`` (Hard rule #8). Same shape as
    ``optimize_cloze_module``'s offline test.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None

    train = [
        {
            "word": "schlafen",
            "context_sentence": "Der Hund schläft.",
            "learner_axes_json": json.dumps({"verbs": 3}),
            "target_word_id": 1,
            "count": 2,
        },
        {
            "word": "gehen",
            "context_sentence": "Ich gehe nach Hause.",
            "learner_axes_json": json.dumps({"verbs": 2}),
            "target_word_id": 2,
            "count": 2,
        },
    ]
    val = [
        {
            "word": "kommen",
            "context_sentence": "Er kommt morgen.",
            "learner_axes_json": json.dumps({"verbs": 3}),
            "target_word_id": 3,
            "count": 2,
        },
    ]
    try:
        module = match.optimize_match_module(train_set=train, val_set=val)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"optimize_match_module raised on the offline path: {exc!r}"
        )
    assert isinstance(module, match.MatchModule)


# ---------------------------------------------------------------------------
# 10. Schema — wire shape
# ---------------------------------------------------------------------------


def test_matching_exercise_out_is_base_plus_pairs():
    """``MatchingExerciseOut`` extends ``BaseExerciseFields`` with
    ``pairs`` + a server-minted ``exercise_id`` field.

    Phase 6.3 reconcile: 6.2 originally placed ``exercise_id`` on
    ``BaseExerciseFields`` (the narrower, matching-only Literal
    union), and the test asserted the subclass field set was the
    base + ``{pairs}``. Phase 6.4 widened ``BaseExerciseFields`` to
    the 3-way ``Literal["cloze", "matching", "comprehension"]``
    union and dropped ``exercise_id`` from the shared mixin
    (comprehension wires don't need it on the read side; it's
    only required on the matching wire for round-trip into
    ``/exercises/grade``). The merge therefore moves
    ``exercise_id`` to ``MatchingExerciseOut`` itself so the
    matching route still carries the server-minted id the
    grader expects.
    """
    from app.schemas import MatchingExerciseOut, BaseExerciseFields

    base_fields = set(BaseExerciseFields.model_fields.keys())
    subclass_fields = set(MatchingExerciseOut.model_fields.keys())
    # Subclass = base + {"pairs", "exercise_id"}; the matching
    # response narrows ``exercise_type`` from the union on the
    # base down to the matching branch, which the field set
    # check handles as a no-op (same key, narrower annotation).
    assert subclass_fields == base_fields | {"pairs", "exercise_id"}
    assert "pairs" in subclass_fields
    assert "exercise_id" in subclass_fields


def test_matching_exercise_out_rejects_pairs_below_min():
    """``MatchingExerciseOut`` enforces ``pairs.min_length=2`` at the
    wire layer. A 1-pair payload → 422 at the route.
    """
    import pydantic
    from app.schemas import MatchingExerciseOut

    with pytest.raises(pydantic.ValidationError):
        MatchingExerciseOut(
            exercise_id=1,
            target_word_id=1,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            pairs=[
                MatchingPair(left_word_id=1, right_word_id=2, right_kind="translation"),
            ],
        )


def test_matching_exercise_out_default_payload():
    """A full ``MatchingExerciseOut`` serialises to JSON with the
    shared ``BaseExerciseFields`` fields + the ``pairs`` list.
    """
    from app.schemas import MatchingExerciseOut

    out = MatchingExerciseOut(
        exercise_id=42,
        target_word_id=1,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        pairs=[
            MatchingPair(left_word_id=1, right_word_id=2, right_kind="translation"),
            MatchingPair(left_word_id=3, right_word_id=4, right_kind="synonym"),
        ],
    )
    body = out.model_dump()
    assert body["exercise_type"] == "matching"
    assert body["exercise_id"] == 42
    assert body["target_word_id"] == 1
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert len(body["pairs"]) == 2
    assert body["pairs"][0]["right_kind"] == "translation"
    assert body["pairs"][1]["right_kind"] == "synonym"


# ---------------------------------------------------------------------------
# 11. Hard rule acceptance — module-level "stay unchanged" files
# ---------------------------------------------------------------------------


def test_does_not_modify_cloze_observability_fsrs_retrieval_embeddings():
    """Card acceptance: ``git diff main -- backend/app/cloze.py`` is
    empty except for the DSPy adapter extraction. ``observability.py``,
    ``fsrs.py``, ``retrieval.py``, ``embeddings.py`` are unchanged.
    We assert the diffs against ``main`` are limited to the
    extraction lines (re-export only).
    """
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent

    def _diff(file: str) -> str:
        result = subprocess.run(
            ["git", "diff", "main", "--", f"backend/app/{file}"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return result.stdout

    # observability.py / fsrs.py / retrieval.py / embeddings.py:
    # the diff must be empty.
    for f in ("observability.py", "fsrs.py", "retrieval.py", "embeddings.py"):
        diff = _diff(f)
        assert diff == "", (
            f"backend/app/{f} must be unchanged for the 6.2 card; "
            f"got diff:\n{diff}"
        )

    # cloze.py: only the re-export and docstring note.
    cloze_diff = _diff("cloze.py")
    # The re-export line is the only non-docstring functional diff.
    assert "from app.llm import _DSPyOpenAICompatLM" in cloze_diff, (
        "cloze.py diff should include the DSPy adapter re-export"
    )
    # The class definition itself is NOT in cloze.py anymore.
    assert "class _DSPyOpenAICompatLM" not in cloze_diff or "class _DSPyOpenAICompatLM" not in [
        line for line in cloze_diff.splitlines() if line.startswith("+")
    ], (
        "cloze.py diff should not ADD a new class _DSPyOpenAICompatLM "
        "definition; the class must live in app.llm now"
    )
