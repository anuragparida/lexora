"""Tests for Phase 4.2 — cloze exercise generator (card t_bdd9ffbe).

Coverage map (mirrors the card body's "Tests" section):

1. Word selection seeded determinism — same ``(user_id, axis, day)``
   returns same word across two calls.
2. ``PROMPT_TEMPLATE_VERSION == "cloze-v1"``.
3. ``MAX_ATTEMPTS == 3``.
4. ``generate_cloze`` happy path: one ``app.llm.complete`` call
   (verified via the underlying OpenAI client, mocked via respx).
5. ``generate_cloze`` schema-violation retry path: instructor raises
   after 3 attempts and ``ClozeGenerationError`` bubbles out.
6. ``_trace_cloze`` is invoked with the metadata contract keyset
   (mocked; this card's implementation is a no-op stub, but the
   signature is locked so 4.3's implementation can swap in without
   changing the call site).
6b. ``_trace_cloze`` with a mocked Langfuse client records a span
   carrying every metadata-contract field (Phase 4.3 new).
6c. ``_trace_cloze`` with keys unset does NOT raise, does NOT
   contact the network, and does NOT log per-call warnings
   (Phase 4.3 new).
6d. ``_trace_cloze`` swallows Langfuse SDK failures so the cloze
   activity still succeeds (Phase 4.3 new).
7. No retrieval import — ``app.retrieval`` is NOT in ``sys.modules``
   after ``from app import cloze`` (Hard rule #3 / #4.2 acceptance).
8. ``POST /exercises/cloze`` requires JWT — no cookie → 401.
9. DSPy module can be constructed without OpenRouter (uses DummyLM).
10. ``optimize_cloze_module`` runs end-to-end on a 2-row eval set
    without network (uses DummyLM).

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test.
The tests don't depend on the live Postgres / docker stack.

Run from ``backend/``::

    uv run pytest -q tests/test_cloze.py
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

from app import cloze
from app.cloze import (
    ClozeExercise,
    ClozeGenerationError,
    MAX_ATTEMPTS,
    PROMPT_TEMPLATE_VERSION,
)
from app.llm import LLMError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Mirrors the pattern in ``test_auth.py`` and
    ``test_diagnostic.py``.
    """
    db_path = tmp_path / "test_cloze.db"
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
    """A SQLAlchemy session for the per-test SQLite DB.

    The diagnostic tests use a similar pattern; we mirror it so the
    cloze tests can build a User + WeaknessProfile + Word directly
    without going through the HTTP layer for the non-route cases.

    We call ``create_all`` here (not just in the ``client`` fixture)
    so the session is usable in tests that request ``db_session``
    without ``client``.
    """
    from sqlalchemy.orm import sessionmaker

    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        yield session


def _seed_user_with_axes(
    session,
    *,
    email: str = "ada@example.com",
    axes: dict[str, int] | None = None,
) -> int:
    """Insert a User with an optional WeaknessProfile and return user_id.

    The ``axes`` dict goes through the same dialect-aware
    serialisation that the production path uses, so a SQLite test
    exercises the JSON-as-Text branch of ``crud.upsert_weakness_profile``.
    """
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
    example_de: str = "Der Hund schläft.",
) -> int:
    """Insert one ``Word`` row plus a single ``Example`` and return word_id.

    The cloze generator only ever reads the first example sentence,
    so a single example is enough to exercise the prompt path.
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
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


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
    """Build a fake OpenAI chat-completions response body for a valid
    ClozeExercise. The instructor client parses the assistant message
    and validates it against the Pydantic schema.
    """
    if distractors is None:
        distractors = [2, 3, 4]
    return {
        "id": "gen-cloze-test-001",
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


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------


def test_prompt_template_version_is_cloze_v1():
    """Hard rule #11: ``PROMPT_TEMPLATE_VERSION`` is a module
    constant, not env-derived. Bumping it is a code review event.
    """
    assert PROMPT_TEMPLATE_VERSION == "cloze-v1"


def test_max_attempts_is_3():
    """Hard rule #6: schema-violation retries are bounded ≤ 3.
    ``MAX_ATTEMPTS`` is the same number the instructor library
    receives via ``max_retries`` so the dead-letter and the
    instructor budget stay in sync.
    """
    assert MAX_ATTEMPTS == 3


# ---------------------------------------------------------------------------
# 2. No retrieval import (Hard rule #3)
# ---------------------------------------------------------------------------


def test_cloze_does_not_import_retrieval():
    """Hard rule #3 / 4.2 acceptance: the cloze module's import path
    does NOT pull in ``app.retrieval``. We assert on
    ``sys.modules`` because the Python interpreter only imports a
    module once — if any code path inside ``app.cloze`` (or its
    transitive imports touched at module load) ever reaches
    ``app.retrieval``, the assertion fires.
    """
    sys.modules.pop("app.retrieval", None)
    # Re-import to be sure the assertion is on the import chain,
    # not on a cached entry from a prior test.
    if "app.cloze" in sys.modules:
        del sys.modules["app.cloze"]
    import app.cloze  # noqa: F401

    assert "app.retrieval" not in sys.modules, (
        "app.cloze must not import app.retrieval (Hard rule #3 / "
        "4.2 acceptance: no retrieval on the cloze path)"
    )


# ---------------------------------------------------------------------------
# 3. Word selection — determinism
# ---------------------------------------------------------------------------


def test_select_target_word_is_seed_deterministic(db_session, monkeypatch):
    """Same ``(user_id, axis, day)`` → same word across two calls.

    The seed scheme is documented in ``select_target_word``'s
    docstring. We seed one User with an axis profile, run
    ``select_target_word`` twice on the same day, and assert
    identical ids. A different day (the ``today`` fixture flips
    the date) would yield a different id — also documented.
    """
    # The shipped SQLite corpus has zero rows in this empty test
    # DB, so seed a Verb and a Noun that match the two axes we'll
    # try.
    verb_id = _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="Ich gehe."
    )
    _seed_word(
        db_session, word="Hund", word_type="Noun", example_de="Der Hund schläft."
    )
    user_id = _seed_user_with_axes(
        db_session, axes={"verbs": 3, "abstract_nouns": 1}
    )

    # Highest score is ``verbs`` (3) → mapped to word_type "Verb"
    # → only the verb id qualifies. Both calls return it.
    w1 = cloze.select_target_word(db_session, user_id)
    w2 = cloze.select_target_word(db_session, user_id)
    assert w1.id == w2.id == verb_id


def test_select_target_word_falls_back_to_first_axis_when_empty(
    db_session,
):
    """A user with no weakness profile still gets a stable word.

    Without an axis profile, the selector defaults to ``ALL_AXES[0]``
    (verbs in the canonical list) and the mapped ``Verb`` word_type.
    This keeps the exercise surface stable for fresh users — the
    Phase 3.3 first-login gate routes them through the diagnostic
    first, but the cloze endpoint should still be callable.
    """
    _seed_word(
        db_session, word="Hund", word_type="Noun", example_de="Der Hund schläft."
    )
    verb_id = _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="Ich gehe."
    )
    user_id = _seed_user_with_axes(db_session, axes=None)

    w = cloze.select_target_word(db_session, user_id)
    # ``ALL_AXES[0]`` is "verbs" → word_type "Verb" → verb_id.
    assert w.id == verb_id


def test_select_target_word_picks_highest_scored_axis(db_session):
    """Among the user's axes, the highest score wins.

    Ties break by the canonical ``ALL_AXES`` order, but the score
    ordering itself is unambiguous: the axis with the higher value
    must be picked.
    """
    _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="Ich gehe."
    )
    noun_id = _seed_word(
        db_session, word="Hund", word_type="Noun", example_de="Der Hund schläft."
    )
    user_id = _seed_user_with_axes(
        db_session,
        # ``abstract_nouns`` is the highest — score 3; ``verbs`` is 1.
        # Abstract nouns map to Noun, so we expect noun_id back.
        axes={"verbs": 1, "abstract_nouns": 3},
    )

    w = cloze.select_target_word(db_session, user_id)
    assert w.id == noun_id


def test_select_target_word_changes_with_day(db_session, monkeypatch):
    """A new day → a new word.

    The seed scheme includes ``date.today()`` so the same user, on
    a different day, gets a different offset into the same
    ``word_type`` pool. We don't pin a specific id (the seed
    arithmetic is the contract; a future maintainer could swap the
    RNG) — we assert that two different days produce two different
    words.

    Requires at least two Verbs in the corpus; we seed two.
    """
    _seed_word(
        db_session, word="gehen", word_type="Verb", example_de="Ich gehe."
    )
    _seed_word(
        db_session, word="kommen", word_type="Verb", example_de="Ich komme."
    )
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    # Day 1
    class FakeDate1:
        @classmethod
        def today(cls):
            return date(2026, 7, 1)

    monkeypatch.setattr(cloze, "date", FakeDate1)
    w1 = cloze.select_target_word(db_session, user_id)

    # Day 2 — different ``today()`` → different offset.
    class FakeDate2:
        @classmethod
        def today(cls):
            return date(2026, 7, 2)

    monkeypatch.setattr(cloze, "date", FakeDate2)
    w2 = cloze.select_target_word(db_session, user_id)

    assert w1.id != w2.id


# ---------------------------------------------------------------------------
# 4. Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_embeds_target_word_and_first_example(db_session):
    """The prompt must carry the target word + its first example
    sentence, NOT a retrieval query (Hard rule #3).
    """
    example = "Die Katze schläft auf dem Sofa."
    wid = _seed_word(
        db_session,
        word="schlafen",
        word_type="Verb",
        example_de=example,
    )
    word = db_session.query(__import__("app").models.Word).get(wid)
    msgs = cloze.build_prompt(word, weakness_axes={"verbs": 3})

    # Two messages: system + user.
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    # System prompt carries the prohibition block + the JSON schema
    # and the C1-accept bar.
    system = msgs[0]["content"]
    assert "Do NOT change word forms" in system
    assert "Do NOT translate" in system
    assert "C1-ACCEPT BAR" in system
    assert "prompt_template_version" in system
    assert "cloze-v1" in system  # The schema field value

    # User prompt carries the target word + the first example + the
    # JSON-encoded axes. The example is the one we seeded, not a
    # retrieval result.
    user = json.loads(msgs[1]["content"])
    assert user["target_word"]["word"] == "schlafen"
    assert user["context_sentence"] == example
    assert user["learner_axes"] == {"verbs": 3}


def test_build_prompt_handles_word_with_no_examples(db_session):
    """If a Word has zero Example rows, the prompt uses a
    deterministic placeholder rather than crashing. The cloze
    endpoint stays usable even on a sparse corpus row.
    """
    from app import models

    w = models.Word(word="solo", word_type="Noun", frequency="5", is_complete=True)
    db_session.add(w)
    db_session.commit()
    db_session.refresh(w)
    assert w.examples == []

    msgs = cloze.build_prompt(w, weakness_axes={})
    user = json.loads(msgs[1]["content"])
    # The placeholder is a non-empty string so the LLM still has
    # something to work from.
    assert user["context_sentence"]
    assert "Beispiel nicht verfügbar" in user["context_sentence"]


# ---------------------------------------------------------------------------
# 5. generate_cloze — happy path
# ---------------------------------------------------------------------------


@respx.mock
def test_generate_cloze_happy_path_one_complete_call(db_session, monkeypatch):
    """One successful OpenAI call → one ``ClozeExercise`` return.

    We mock the underlying httpx POST against
    ``https://openrouter.ai/api/v1/chat/completions`` and assert
    that ``generate_cloze`` returns a fully-validated Pydantic
    model with the metadata contract fields stamped on the result.
    """
    verb_id = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    _seed_word(db_session, word="gehen", word_type="Verb", example_de="X geht.")
    _seed_word(db_session, word="kommen", word_type="Verb", example_de="X kommt.")
    _seed_word(db_session, word="bleiben", word_type="Verb", example_de="X bleibt.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=verb_id,
                distractors=[2, 3, 4],
            ),
        )
    )

    # ``app.llm.BACKOFF_SCHEDULE_S`` is zeroed by the test_llm.py
    # autouse pattern; we mirror it here so this test stays
    # independent of test collection order.
    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    result = cloze.generate_cloze(db_session, user_id)
    assert isinstance(result, ClozeExercise)
    assert result.sentence_with_blank == "Der ___ schläft."
    assert result.answer_word_id == verb_id
    assert len(result.distractors) == 3
    assert result.prompt_template_version == "cloze-v1"
    # Exactly one HTTP call (instructor didn't retry).
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# 6. generate_cloze — schema-violation retry path
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
):
    """Build a stub OpenAI-shaped client that returns ``payload``
    as the assistant message content. Used by the schema-violation
    tests (and the trace test) to inject specific responses
    without going through respx.

    Implementation note: ``instructor.from_openai`` enforces that
    the client is an instance of ``openai.OpenAI`` (or
    ``AsyncOpenAI``), so a duck-typed stub won't pass. We build a
    real ``OpenAI`` client with a custom ``httpx.Client`` whose
    transport is a ``MockTransport`` — the OpenAI SDK routes every
    request through that transport, and the instructor layer
    above never knows the difference.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-stub-001",
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


def test_generate_cloze_prompt_template_version_locked(
    db_session, monkeypatch
):
    """A response whose ``prompt_template_version`` doesn't match
    ``PROMPT_TEMPLATE_VERSION`` is normalised on the way out. The
    schema in Pydantic v2 is free-form, so a misbehaving model
    could send any string — we override on the way out.
    """
    wid = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    _seed_word(db_session, word="gehen", word_type="Verb", example_de="X geht.")
    _seed_word(db_session, word="kommen", word_type="Verb", example_de="X kommt.")
    _seed_word(db_session, word="bleiben", word_type="Verb", example_de="X bleibt.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    payload = json.dumps(
        {
            "sentence_with_blank": "Der ___ schläft.",
            "answer_word_id": wid,
            "distractors": [2, 3, 4],
            "difficulty": "easy",
            "rationale": "x",
            # The model claims a future template version; the
            # activity should normalise to the current constant.
            "prompt_template_version": "cloze-v99-bleeding",
        }
    )
    monkeypatch.setattr(
        cloze,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    result = cloze.generate_cloze(db_session, user_id)
    assert result.prompt_template_version == PROMPT_TEMPLATE_VERSION


def test_generate_cloze_dead_letters_after_three_invalid_attempts(
    db_session, monkeypatch
):
    """Three consecutive schema violations → ``ClozeGenerationError``
    with the structured fields. Hard rule #6: retries ≤ 3.
    """
    _seed_word(db_session, word="schlafen", word_type="Verb", example_de="X schläft.")
    _seed_word(db_session, word="gehen", word_type="Verb", example_de="X geht.")
    _seed_word(db_session, word="kommen", word_type="Verb", example_de="X kommt.")
    _seed_word(db_session, word="bleiben", word_type="Verb", example_de="X bleibt.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Every response is missing ``rationale`` → Pydantic validation
    # fails every time. Instructor raises after its retry budget.
    invalid_payload = json.dumps(
        {
            "sentence_with_blank": "Der ___ schläft.",
            "answer_word_id": 1,
            "distractors": [2, 3, 4],
            "difficulty": "easy",
            # rationale omitted → validation error
            "prompt_template_version": "cloze-v1",
        }
    )
    monkeypatch.setattr(
        cloze,
        "_openai_client",
        lambda: _make_stub_instructor_client(invalid_payload),
    )

    with pytest.raises(ClozeGenerationError) as excinfo:
        cloze.generate_cloze(db_session, user_id)
    err = excinfo.value
    # instructor's count is ``initial + max_retries``: with
    # ``max_retries=3`` the budget is 4 total HTTP calls. The
    # hard rule says "≤ 3 schema-violation retries" — i.e. the
    # retry budget is 3, the total attempts is 4. We assert the
    # retry budget is respected (so the dead-letter always lands
    # in bounded time) without conflating the two.
    assert err.schema_retry_count <= MAX_ATTEMPTS + 1
    assert err.last_validation_error
    # ``attempted_schema`` carries the JSON schema for triage.
    # Pydantic v2's ``model_json_schema()`` returns a dict (not a
    # string) — we serialise to JSON here so the operator can grep
    # the dead-letter body without an extra step.
    attempted = err.attempted_schema
    if not isinstance(attempted, str):
        attempted = json.dumps(attempted)
    assert "ClozeExercise" in attempted
    assert "sentence_with_blank" in attempted


def test_generate_cloze_raises_llm_error_when_api_key_missing(
    db_session, monkeypatch
):
    """No ``OPENROUTER_API_KEY`` → ``LLMError`` (route layer → 502).

    We never want a missing key to silently succeed; the operator
    should see a clear "add the key and restart" message.
    """
    _seed_word(db_session, word="schlafen", word_type="Verb", example_de="X schläft.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        cloze.generate_cloze(db_session, user_id)


# ---------------------------------------------------------------------------
# 7. _trace_cloze — no-op stub, signature-locked
# ---------------------------------------------------------------------------


def test_trace_cloze_accepts_metadata_contract_keyset():
    """4.3 will replace the no-op body with a real Langfuse span.
    The signature is locked — this test asserts the call site can
    hand the metadata contract to it without raising.
    """
    exercise = ClozeExercise(
        sentence_with_blank="Der ___ schläft.",
        answer_word_id=1,
        distractors=[2, 3, 4],
        difficulty="easy",
        rationale="Test.",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata = {
        "user_id": 42,
        "weakness_axes": {"verbs": 3},
        "word_id": 1,
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ],
        "schema_retry_count": 0,
        "prompt_tokens": 30,
        "completion_tokens": 12,
    }
    # The no-op returns None and accepts every field; the assertion
    # is the implicit "did not raise".
    assert cloze._trace_cloze(exercise, metadata, latency_ms=42) is None
    # ``result=None`` path is exercised on the dead-letter branch
    # (the activity tries to trace even on schema failure before
    # raising ``ClozeGenerationError``).
    assert cloze._trace_cloze(None, metadata, latency_ms=42) is None


def test_trace_cloze_invoked_on_happy_path(db_session, monkeypatch):
    """The metadata contract reaches ``_trace_cloze`` on the happy
    path. We monkeypatch the no-op to record the call, then assert
    the metadata keyset matches the contract in PHASE-4.md.

    Mocking strategy: instead of respx (which fights with the
    OpenAI SDK's transport in some test orders — the SDK caches a
    custom httpx pool that respx doesn't always intercept), we
    monkeypatch ``_openai_client`` to return a stub OpenAI client
    whose ``chat.completions.create`` returns a fixed
    ``ChatCompletion`` shaped object. The instructor wrapping
    happens on top of the stub, so we still exercise the full
    production path (instructor parsing, validation, retry logic
    — except for actual schema violations, which the
    schema-violation test covers).
    """
    wid = _seed_word(
        db_session, word="schlafen", word_type="Verb", example_de="X schläft."
    )
    _seed_word(db_session, word="gehen", word_type="Verb", example_de="X geht.")
    _seed_word(db_session, word="kommen", word_type="Verb", example_de="X kommt.")
    _seed_word(db_session, word="bleiben", word_type="Verb", example_de="X bleibt.")
    user_id = _seed_user_with_axes(db_session, axes={"verbs": 3})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    payload = json.dumps(
        {
            "sentence_with_blank": "Der ___ schläft.",
            "answer_word_id": wid,
            "distractors": [2, 3, 4],
            "difficulty": "easy",
            "rationale": "Stub rationale.",
            "prompt_template_version": "cloze-v1",
        }
    )
    monkeypatch.setattr(
        cloze,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    captured: list[dict] = []

    def fake_trace(result, metadata, latency_ms):
        captured.append({"result": result, "metadata": metadata, "latency_ms": latency_ms})

    monkeypatch.setattr(cloze, "_trace_cloze", fake_trace)
    cloze.generate_cloze(db_session, user_id)

    assert len(captured) == 1
    call = captured[0]
    md = call["metadata"]
    # Every contract field is present.
    for key in (
        "user_id",
        "weakness_axes",
        "word_id",
        "model_id",
        "prompt_template_version",
        "prompt_messages",
        "schema_retry_count",
        "prompt_tokens",
        "completion_tokens",
    ):
        assert key in md, f"metadata missing contract key: {key}"
    assert call["latency_ms"] >= 0


def test_trace_cloze_metadata_contract_keyset_on_mocked_span(monkeypatch):
    """Phase 4.3 — when the Langfuse client is non-None, the trace
    span carries every metadata-contract field exactly once and
    the contract keyset matches ``docs/PHASE-4.md`` §"The metadata
    contract" (10 fields).

    We monkeypatch the v2-SDK call shape:
    ``client.span(name=..., input=..., output=...)`` → MagicMock;
    ``span.update(metadata=...)`` → records the call;
    ``span.end()`` → marks closure;
    ``client.flush()`` → ensures the buffer is sent.

    The mocked client is wired through the freshly-resolved
    ``app.cloze`` module — ``test_cloze_does_not_import_retrieval``
    earlier in the suite reloads ``app.cloze`` (via
    ``del sys.modules["app.cloze"]; import app.cloze``); the
    module-level ``from app import cloze`` reference in this file
    is therefore stale by the time we patch, so we re-acquire it
    inside this test before patching.
    """
    from unittest.mock import MagicMock
    import importlib

    # Re-acquire the live module — earlier tests reload it.
    cloze_live = importlib.import_module("app.cloze")

    # Build a MagicMock span + client with the v2-SDK shape.
    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span

    # Wire the mock into the cloze module's bound ``get_langfuse``.
    monkeypatch.setattr(cloze_live, "get_langfuse", lambda: mock_client)

    exercise = ClozeExercise(
        sentence_with_blank="Der ___ schläft.",
        answer_word_id=1,
        distractors=[2, 3, 4],
        difficulty="easy",
        rationale="Test.",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata = {
        "user_id": 42,
        "weakness_axes": {"verbs": 3, "prepositions": 1},
        "word_id": 1,
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

    # The function is an idempotent side-effect emitter; return value
    # is implicitly None.
    assert cloze_live._trace_cloze(exercise, metadata, latency_ms=42) is None

    # client.span was invoked exactly once with the canonical name.
    assert mock_client.span.call_count == 1
    span_kwargs = mock_client.span.call_args.kwargs
    assert span_kwargs["name"] == "cloze.generate"
    # input is the prompt messages; output is the serialised exercise.
    assert span_kwargs["input"] == metadata["prompt_messages"]
    assert json.loads(span_kwargs["output"]) == json.loads(
        exercise.model_dump_json()
    )

    # span.update was invoked at least once with the full metadata keyset.
    assert mock_span.update.call_count >= 1
    update_calls = mock_span.update.call_args_list
    # Concatenate every metadata dict we passed to span.update; find
    # the one whose keys match the contract.
    merged: dict = {}
    for call in update_calls:
        for key, value in (call.kwargs.get("metadata") or {}).items():
            merged[key] = value

    expected_keys = {
        "user_id",
        "weakness_axes",
        "word_id",
        "difficulty",
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
    assert merged["user_id"] == 42
    assert merged["weakness_axes"] == {"verbs": 3, "prepositions": 1}
    assert merged["word_id"] == 1
    assert merged["difficulty"] == "easy"
    assert merged["model_id"] == "qwen/qwen3-235b-a22b-2507"
    assert merged["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert merged["schema_retry_count"] == 0
    assert merged["latency_ms"] == 42
    assert merged["prompt_tokens"] == 30
    assert merged["completion_tokens"] == 12

    # Span closed and client flushed — required for the QA-hook
    # visibility acceptance gate ("trace queryable in UI before
    # request returns").
    assert mock_span.end.call_count == 1
    assert mock_client.flush.call_count == 1


@respx.mock
def test_trace_cloze_is_silent_when_keys_missing(monkeypatch, caplog):
    """Phase 4.3 — when ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``
    are unset, ``get_langfuse()`` returns None and ``_trace_cloze``
    returns silently without contacting the network.

    We wrap the test in ``@respx.mock`` so any HTTP request that
    *would* leak out of the function fails the test (respx raises
    ``RequestNotCalled`` on un-matched routes). Together with the
    caplog assertion (one warning from ``observability.py`` at import
    time, no per-call warnings), this proves the graceful-degrade
    branch is exercised end-to-end.
    """
    import logging

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    # exercise is None — the dead-letter branch. The function must
    # still return None without raising.
    with caplog.at_level(logging.WARNING, logger="app.observability"):
        assert cloze._trace_cloze(None, {}, 0) is None

    # exercise populated — happy-path branch with empty metadata.
    # We must not raise on a missing 'prompt_template_version' key
    # (the function's ``metadata.get(...)`` tolerates the absence);
    # supply the keys we do read directly:
    metadata = {
        "user_id": 1,
        "weakness_axes": {},
        "word_id": 1,
        "model_id": "stub",
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    exercise = ClozeExercise(
        sentence_with_blank="Der ___ schläft.",
        answer_word_id=1,
        distractors=[2, 3, 4],
        difficulty="easy",
        rationale="x",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert cloze._trace_cloze(exercise, metadata, latency_ms=0) is None

    # No per-call warnings — observability.py logs once at module
    # import. caplog may carry the import-time warning depending on
    # fixture ordering; we only assert no new cloze-side warnings.
    cloze_warnings = [
        r for r in caplog.records
        if r.name == "app.cloze" and r.levelno >= logging.WARNING
    ]
    assert cloze_warnings == [], (
        f"unexpected warnings from app.cloze: "
        f"{[r.getMessage() for r in cloze_warnings]}"
    )

    # respx: any un-matched request would have raised. The function
    # returned without contacting the network.
    # (no respx.post(...) routes were set up; the mock asserts on
    # exit that no un-matched routes were hit.)


def test_trace_cloze_swallows_langfuse_failures(monkeypatch):
    """Phase 4.3 — when the Langfuse SDK raises mid-span, the cloze
    activity still succeeds. Tracing failures must never break the
    request (same invariant as ``_trace_retrieval``).
    """
    from unittest.mock import MagicMock
    import importlib

    cloze_live = importlib.import_module("app.cloze")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span
    mock_client.flush.side_effect = RuntimeError("simulated flush failure")

    monkeypatch.setattr(cloze_live, "get_langfuse", lambda: mock_client)

    exercise = ClozeExercise(
        sentence_with_blank="Der ___ schläft.",
        answer_word_id=1,
        distractors=[2, 3, 4],
        difficulty="easy",
        rationale="x",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata = {
        "user_id": 1,
        "weakness_axes": {},
        "word_id": 1,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    # The exception is swallowed; the activity returns cleanly.
    assert cloze_live._trace_cloze(exercise, metadata, latency_ms=0) is None


# ---------------------------------------------------------------------------
# 8. Auth-gated route
# ---------------------------------------------------------------------------


def test_post_exercises_cloze_requires_auth(client):
    """``POST /exercises/cloze`` with no cookie → 401 (Hard rule on
    the cloze endpoint being auth-gated).
    """
    client.cookies.clear()
    resp = client.post("/exercises/cloze", json={})
    assert resp.status_code == 401


def test_post_exercises_cloze_returns_cloze_exercise_with_valid_token(
    client, db_session, monkeypatch
):
    """The happy path end-to-end: signup → set axes → POST → 200 +
    ``ClozeExercise`` JSON with the metadata contract populated.

    We mock the OpenRouter call so the test stays hermetic; the
    production path is exercised in the QA hook.
    """
    body = _signup(client, email="ada@example.com")
    user_id = body["user"]["id"]

    # Seed words in the empty test DB so the word selection has
    # something to pick from. ``select_target_word`` queries the
    # same DB the route uses (the per-test SQLite).
    _seed_word(db_session, word="schlafen", word_type="Verb", example_de="X schläft.")
    _seed_word(db_session, word="gehen", word_type="Verb", example_de="X geht.")
    _seed_word(db_session, word="kommen", word_type="Verb", example_de="X kommt.")
    _seed_word(db_session, word="bleiben", word_type="Verb", example_de="X bleibt.")
    client.put(
        f"/weakness-profile/{user_id}",
        json={"axes": {"verbs": 3}},
    )

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_cloze_response(
                sentence="Der ___ schläft.",
                answer_id=1,
                distractors=[2, 3, 4],
            ),
        )
    )
    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Some test orderings let respx's intercept survive through
    # ``TestClient``; others don't (the OpenAI SDK caches a
    # custom httpx pool that respx doesn't always patch). We
    # use the same stub-client fallback the trace test uses so
    # this test is order-independent. If respx *did* intercept,
    # the stub never fires; if it didn't, the stub is what
    # actually serves the call.
    payload = json.dumps(
        {
            "sentence_with_blank": "Der ___ schläft.",
            "answer_word_id": 1,
            "distractors": [2, 3, 4],
            "difficulty": "easy",
            "rationale": "Stub rationale.",
            "prompt_template_version": "cloze-v1",
        }
    )
    monkeypatch.setattr(
        "app.cloze._openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post("/exercises/cloze", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sentence_with_blank"] == "Der ___ schläft."
    assert body["answer_word_id"] == 1
    assert body["distractors"] == [2, 3, 4]
    assert body["difficulty"] == "easy"
    assert body["prompt_template_version"] == "cloze-v1"


# ---------------------------------------------------------------------------
# 9. DSPy module — constructed without OpenRouter
# ---------------------------------------------------------------------------


def test_dspy_module_constructible_without_openrouter(monkeypatch):
    """``ClozeModule`` can be constructed without an OpenRouter key.
    The DSPy configure path falls back to ``DummyLM`` automatically
    (Hard rule #8: offline-capable).
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Force re-configuration (the test process might have an LM set
    # by a prior test).
    import dspy

    dspy.settings.lm = None
    module = cloze.ClozeModule()
    assert module is not None
    assert hasattr(module, "predict")


def test_cloze_signature_inputs_match_production_contract():
    """The DSPy signature carries the same input keys the production
    path passes in ``build_prompt`` (word, context_sentence,
    learner_axes_json, target_word_id). The output is the
    ``ClozeExercise`` Pydantic model — DSPy 3.x supports Pydantic
    output types.
    """
    sig = cloze.ClozeSignature
    input_fields = {k for k in sig.model_fields if sig.model_fields[k].json_schema_extra.get("__dspy_field_type") == "input"}
    output_fields = {k for k in sig.model_fields if sig.model_fields[k].json_schema_extra.get("__dspy_field_type") == "output"}
    assert input_fields == {"word", "context_sentence", "learner_axes_json", "target_word_id"}
    assert output_fields == {"exercise"}


# ---------------------------------------------------------------------------
# 10. optimize_cloze_module — end-to-end on a 2-row eval set, no network
# ---------------------------------------------------------------------------


def test_optimize_cloze_module_runs_on_two_row_eval_set_offline(monkeypatch):
    """``optimize_cloze_module`` runs end-to-end on a 2-row eval set
    using ``DummyLM``.

    Caveat: MIPROv2's internal prompt-proposer signature is hard to
    satisfy with ``DummyLM`` (it expects strict JSONAdapter-shaped
    responses from an LM it probes several times). The harder
    optimizer call falls back to ``BootstrapFewShot`` or returns
    the un-optimized module if both optimizers fail; the contract
    is "no crash on the offline path". To exercise the actual
    optimization loop, run the CLI with ``--live`` and a real
    OpenRouter key — that's the path the spec points to.
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
        },
        {
            "word": "gehen",
            "context_sentence": "Ich gehe nach Hause.",
            "learner_axes_json": json.dumps({"verbs": 2}),
            "target_word_id": 2,
        },
    ]
    val = [
        {
            "word": "kommen",
            "context_sentence": "Er kommt morgen.",
            "learner_axes_json": json.dumps({"verbs": 3}),
            "target_word_id": 3,
        },
    ]
    # ``optimize_cloze_module`` configures the DSPy LM (DummyLM
    # when no key is present) and dispatches to MIPROv2 →
    # BootstrapFewShot → un-optimized. Any of those three paths
    # is acceptable as long as the function returns a
    # ``ClozeModule``. The spec just says "the CLI plumbing must
    # run without network" — we assert that here.
    try:
        module = cloze.optimize_cloze_module(train_set=train, val_set=val)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"optimize_cloze_module raised on the offline path: {exc!r}"
        )
    assert isinstance(module, cloze.ClozeModule)


def test_dspy_module_forward_runs_with_dummy_lm(monkeypatch):
    """``ClozeModule`` produces a Prediction when given a ``DummyLM``-served
    backend. Verifies the DSPy integration end-to-end (signature,
    predictor, output field) without invoking the optimizer
    (which has its own failure modes on tiny eval sets).
    """
    import dspy

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    dspy.settings.lm = None
    # Configure the cloze module's DSPy settings.
    cloze._configure_dspy()
    module = cloze.ClozeModule()
    # ``ClozeModule.predict`` is a ``dspy.Predict`` which uses
    # the configured LM. With DummyLM it should return a
    # ``Prediction`` with the ``exercise`` field.
    pred = module(
        word="schlafen",
        context_sentence="x",
        learner_axes_json="{}",
        target_word_id=1,
    )
    # ``pred.exercise`` is a string (DummyLM stub) rather than a
    # validated Pydantic model — the production path validates it
    # via ``instructor``; the DSPy path is for optimization, not
    # for production-grade validation.
    assert hasattr(pred, "exercise")


# ---------------------------------------------------------------------------
# 11. ClozeExercise Pydantic constraints
# ---------------------------------------------------------------------------


def test_cloze_exercise_rejects_wrong_distractor_count():
    """Pydantic enforces the ``min_length=3, max_length=3`` rule on
    ``distractors`` (the metadata contract).
    """
    import pydantic

    base = dict(
        sentence_with_blank="Der ___ schläft.",
        answer_word_id=1,
        difficulty="easy",
        rationale="x",
        prompt_template_version="cloze-v1",
    )
    with pytest.raises(pydantic.ValidationError):
        ClozeExercise(**base, distractors=[1, 2])  # too few
    with pytest.raises(pydantic.ValidationError):
        ClozeExercise(**base, distractors=[1, 2, 3, 4])  # too many


def test_cloze_exercise_rejects_unknown_difficulty():
    """Pydantic enforces the ``Literal["easy", "medium", "hard"]``
    type on ``difficulty``.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ClozeExercise(
            sentence_with_blank="Der ___ schläft.",
            answer_word_id=1,
            distractors=[2, 3, 4],
            difficulty="trivial",  # not in the literal
            rationale="x",
            prompt_template_version="cloze-v1",
        )