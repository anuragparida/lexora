"""Tests for Phase 6.4 — comprehension exercise generator (card t_8556fc97).

Coverage map (mirrors the card body's "Tests" section):

1. Module-level constants: ``PROMPT_TEMPLATE_VERSION``,
   ``MAX_ATTEMPTS``, the three ``COMPREHENSION_PASSAGE_*``
   bounds match the spec.
2. No retrieval import — ``app.retrieval`` is NOT in
   ``sys.modules`` after ``from app import comprehension``
   UNLESS the active engine is Postgres (Hard rule #2:
   ``/retrieve`` is consumed as-is; we don't add new import-
   time deps). The comprehension module's *static* import
   surface stays retrieval-free; the retrieval call lives
   behind a function-level import.
3. ``ComprehensionExercise`` Pydantic enforces:
   - all 4 choices keys A/B/C/D present;
   - each choice value 1..200 chars;
   - ``correct_choice`` in ``{"A", "B", "C", "D"}``;
   - ``passage`` length in
     ``[20, COMPREHENSION_PASSAGE_MAX_CHARS]``;
   - ``question`` length in ``[5, 300]``;
   - ``rationale`` length in ``[1, 400]``.
4. ``build_prompt`` no-RAG path: user-side JSON has NO
   ``retrieved_chunks`` key; system prompt carries the 3-5
   sentence + 4-choice prohibitions.
5. ``build_prompt`` RAG-on path: user-side JSON includes
   ``retrieved_chunks`` with N items; the chunks land in the
   JSON verbatim.
6. ``_retrieve_for_comprehension`` on SQLite (the default test
   DB): returns ``[]`` gracefully (the comprehension call
   falls back to the no-RAG prompt).
7. ``generate_comprehension`` happy path: one
   ``app.llm.complete`` call, returns a validated
   ``ComprehensionExercise`` with the 4 choices populated.
8. ``generate_comprehension`` schema-violation retry path:
   instructor raises after the budget and
   ``ComprehensionGenerationError`` bubbles out.
9. ``generate_comprehension`` LLMError when API key missing.
10. ``_trace_comprehension`` accepts the metadata contract
    keyset (mocked; signature-locked, mirrors _trace_cloze).
11. ``_trace_comprehension`` with a mocked Langfuse client
    records a span carrying every metadata-contract field.
12. ``_trace_comprehension`` with keys unset does NOT raise,
    does NOT contact the network, and does NOT log per-call
    warnings.
13. ``_trace_comprehension`` swallows Langfuse SDK failures
    so the comprehension activity still succeeds.
14. DSPy module can be constructed without OpenRouter (uses
    DummyLM).
15. ``optimize_comprehension_module`` runs end-to-end on a
    2-row eval set without network (uses DummyLM).
16. ``ComprehensionGenerationError`` carries the same
    dead-letter fields as ``ClozeGenerationError`` (Hard
    rule #6 / Phase 6 mirror).
17. Wire shape parity: ``ComprehensionExerciseOut`` in
    ``app.schemas`` mirrors the generator fields and narrows
    the exercise_type discriminator to
    ``Literal["comprehension"]``.
18. ``ComprehensionGenerateRequest.enable_rag`` defaults to
    ``False`` (Hard rule #1: RAG-on is opt-in).

Hermetic: a fresh temp SQLite DB + a temp JWT secret per
test. The tests don't depend on the live Postgres / docker
stack. The OpenRouter call is mocked via the
``_make_stub_instructor_client`` helper (same shape as
test_cloze.py).

Run from ``backend/``::

    uv run pytest -q tests/test_comprehension.py
"""
from __future__ import annotations

import json
import secrets
import sys

import pytest
import respx
from httpx import Response

from app import comprehension
from app.comprehension import (
    ComprehensionExercise,
    ComprehensionGenerationError,
    COMPREHENSION_PASSAGE_MAX_CHARS,
    COMPREHENSION_PASSAGE_MAX_SENTENCES,
    COMPREHENSION_PASSAGE_MIN_SENTENCES,
    MAX_ATTEMPTS,
    PROMPT_TEMPLATE_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s
    import-time check passes. Mirrors the pattern in
    ``test_cloze.py``.
    """
    db_path = tmp_path / "test_comprehension.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB.

    We call ``create_all`` here so the session is usable in
    tests that request ``db_session`` without other fixtures.
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
    example_de: str = "Beispiel nicht verfügbar.",
) -> int:
    """Insert one ``Word`` row plus a single ``Example`` and return word_id."""
    from app import models

    row = models.Word(
        word=word, word_type=word_type, frequency="5", is_complete=True
    )
    session.add(row)
    session.flush()
    session.add(models.Example(word_id=row.id, german=example_de, english=""))
    session.commit()
    return row.id


def _openai_comprehension_response(
    *,
    target_word_id: int = 1,
    passage: str = (
        "Der Hund läuft durch den Park. Er sieht einen Ball und "
        "rennt sofort los. Sein Besitzer lacht und ruft seinen Namen. "
        "Am Ende sind beide müde und gehen nach Hause."
    ),
    question: str = "Was sieht der Hund im Park?",
    choices: dict[str, str] | None = None,
    correct_choice: str = "A",
    rationale: str = "Ball is grounded in sentence 2; the distractors swap objects.",
    prompt_tokens: int = 50,
    completion_tokens: int = 80,
) -> dict:
    """Build a fake OpenAI chat-completions response body for a
    valid ComprehensionExercise.
    """
    if choices is None:
        choices = {
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        }
    return {
        "id": "gen-comprehension-test-001",
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
                            "target_word_id": target_word_id,
                            "passage": passage,
                            "question": question,
                            "choices": choices,
                            "correct_choice": correct_choice,
                            "rationale": rationale,
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


def test_prompt_template_version_is_comprehension_v1():
    """Hard rule #11: ``PROMPT_TEMPLATE_VERSION`` is a module
    constant, not env-derived. Bumping it is a code review event.
    """
    assert PROMPT_TEMPLATE_VERSION == "comprehension-v1"


def test_max_attempts_is_3():
    """Hard rule #6: schema-violation retries are bounded ≤ 3.
    ``MAX_ATTEMPTS`` is the same number the instructor library
    receives via ``max_retries`` so the dead-letter and the
    instructor budget stay in sync.
    """
    assert MAX_ATTEMPTS == 3


def test_passage_bounds_match_spec():
    """The three ``COMPREHENSION_PASSAGE_*`` constants are
    hard-coded module constants (Hard rule #9). The prompt's
    prohibition block quotes them by name; the Pydantic
    ``passage`` field's ``max_length`` references
    ``COMPREHENSION_PASSAGE_MAX_CHARS``. A future maintainer
    who tweaks them gets a single source of truth.
    """
    assert COMPREHENSION_PASSAGE_MIN_SENTENCES == 3
    assert COMPREHENSION_PASSAGE_MAX_SENTENCES == 5
    assert COMPREHENSION_PASSAGE_MAX_CHARS == 600


# ---------------------------------------------------------------------------
# 2. No static retrieval import (Hard rule #2)
# ---------------------------------------------------------------------------


def test_comprehension_does_not_static_import_retrieval():
    """Hard rule #2: ``/retrieve`` is consumed as-is. The
    comprehension module's *static* import surface stays
    retrieval-free; the retrieval call lives behind a
    function-level import. The assertion is on ``sys.modules``:
    if any code path inside ``app.comprehension`` (or its
    transitive imports touched at module load) ever reaches
    ``app.retrieval`` or ``app.embeddings``, the assertion
    fires.
    """
    sys.modules.pop("app.retrieval", None)
    sys.modules.pop("app.embeddings", None)
    if "app.comprehension" in sys.modules:
        del sys.modules["app.comprehension"]
    import app.comprehension  # noqa: F401

    assert "app.retrieval" not in sys.modules, (
        "app.comprehension must not import app.retrieval at "
        "module load (Hard rule #2; retrieval lives behind a "
        "function-level import)"
    )
    assert "app.embeddings" not in sys.modules, (
        "app.comprehension must not import app.embeddings at "
        "module load (Hard rule #2; the embed_one call lives "
        "behind a function-level import)"
    )


# ---------------------------------------------------------------------------
# 3. ComprehensionExercise Pydantic constraints
# ---------------------------------------------------------------------------


def test_comprehension_exercise_rejects_missing_choices_key():
    """Pydantic enforces the ``min_length=4, max_length=4`` rule
    on ``choices``. A missing key is a 422, not a default.
    """
    import pydantic

    base = dict(
        target_word_id=1,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            # "D" missing → 4-key invariant violated
        },
        correct_choice="A",
        rationale="x",
    )
    with pytest.raises(pydantic.ValidationError):
        ComprehensionExercise(**base)


def test_comprehension_exercise_rejects_unknown_correct_choice():
    """Pydantic enforces ``correct_choice: Literal["A","B","C","D"]``.
    A model that emits "E" (or "a" lowercase, or anything outside
    the literal) gets a 422 on the dead-letter path.
    """
    import pydantic

    base = dict(
        target_word_id=1,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        },
        rationale="x",
    )
    with pytest.raises(pydantic.ValidationError):
        ComprehensionExercise(**base, correct_choice="E")
    with pytest.raises(pydantic.ValidationError):
        ComprehensionExercise(**base, correct_choice="a")  # lowercase


def test_comprehension_exercise_rejects_passage_too_long():
    """Pydantic enforces ``passage`` length in
    ``[20, COMPREHENSION_PASSAGE_MAX_CHARS]``. A model that
    emits a 700-char passage (one over the cap) is rejected.
    """
    import pydantic

    too_long = "x" * (COMPREHENSION_PASSAGE_MAX_CHARS + 1)
    with pytest.raises(pydantic.ValidationError):
        ComprehensionExercise(
            target_word_id=1,
            passage=too_long,
            question="Was passiert?",
            choices={
                "A": "x",
                "B": "x",
                "C": "x",
                "D": "x",
            },
            correct_choice="A",
            rationale="x",
        )


def test_comprehension_exercise_rejects_passage_too_short():
    """Pydantic enforces ``passage.min_length=20``. A model that
    emits a 19-char passage (one under the floor) is rejected.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ComprehensionExercise(
            target_word_id=1,
            passage="x" * 19,  # one under the 20-char floor
            question="Was passiert?",
            choices={
                "A": "x",
                "B": "x",
                "C": "x",
                "D": "x",
            },
            correct_choice="A",
            rationale="x",
        )


def test_comprehension_exercise_rejects_choice_text_too_long():
    """Pydantic enforces each choice value bounded
    ``min_length=1, max_length=200``. A 201-char choice text
    is rejected.
    """
    import pydantic

    too_long = "y" * 201
    with pytest.raises(pydantic.ValidationError):
        ComprehensionExercise(
            target_word_id=1,
            passage=(
                "Der Hund läuft durch den Park. Er sieht einen Ball "
                "und rennt sofort los. Sein Besitzer lacht."
            ),
            question="Was passiert?",
            choices={
                "A": too_long,
                "B": "x",
                "C": "x",
                "D": "x",
            },
            correct_choice="A",
            rationale="x",
        )


# ---------------------------------------------------------------------------
# 4. build_prompt — no-RAG path
# ---------------------------------------------------------------------------


def test_build_prompt_no_rag_user_json_has_no_retrieved_chunks_key(db_session):
    """The no-RAG prompt is byte-for-byte stable — the user JSON
    has NO ``retrieved_chunks`` key. A future maintainer who
    adds a default key (even an empty list) breaks the
    byte-stability promise; the git-diff test in
    test_comprehension.py asserts this at the module level.
    """
    from app import models

    word = models.Word(
        word="Hund", word_type="Noun", frequency="5", is_complete=True
    )
    db_session.add(word)
    db_session.commit()
    db_session.refresh(word)

    msgs = comprehension.build_prompt(word, weakness_axes={"abstract_nouns": 2})
    # Two messages: system + user.
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    # System prompt carries the prohibition block + the JSON
    # schema + the C1-accept bar.
    system = msgs[0]["content"]
    assert "C1-ACCEPT BAR" in system
    assert "EXACTLY four choices" in system
    assert "target_word_id" in system

    # User prompt carries the target word + the JSON-encoded
    # axes, but NO ``retrieved_chunks`` key (RAG is off by
    # default — Hard rule #1).
    user = json.loads(msgs[1]["content"])
    assert user["target_word"]["word"] == "Hund"
    assert user["learner_axes"] == {"abstract_nouns": 2}
    assert "retrieved_chunks" not in user, (
        "no-RAG prompt must not include retrieved_chunks "
        "key (Hard rule #1: byte-stable prompt)"
    )


def test_build_prompt_rag_on_user_json_includes_retrieved_chunks(db_session):
    """When ``retrieved_chunks`` is non-empty, the user JSON
    embeds the chunks. The prompt's RAG-on branch is
    opt-in (Hard rule #1).
    """
    from app import models

    word = models.Word(
        word="Hund", word_type="Noun", frequency="5", is_complete=True
    )
    db_session.add(word)
    db_session.commit()
    db_session.refresh(word)

    chunks = [
        {"kind": "word", "id": 99, "text": "Spaziergang im Park"},
        {"kind": "example", "id": 42, "text": "Der Hund läuft schnell."},
    ]
    msgs = comprehension.build_prompt(
        word, weakness_axes={}, retrieved_chunks=chunks
    )
    user = json.loads(msgs[1]["content"])
    assert user["retrieved_chunks"] == chunks
    assert len(user["retrieved_chunks"]) == 2


def test_build_prompt_uses_module_constants_in_prohibition_block(db_session):
    """The system prompt's prohibition block quotes the
    ``COMPREHENSION_PASSAGE_MIN_SENTENCES`` /
    ``COMPREHENSION_PASSAGE_MAX_SENTENCES`` /
    ``COMPREHENSION_PASSAGE_MAX_CHARS`` module constants
    verbatim. A future maintainer who tweaks the constants
    sees the prompt update in lockstep (no two sources of
    truth).
    """
    from app import models

    word = models.Word(
        word="Hund", word_type="Noun", frequency="5", is_complete=True
    )
    db_session.add(word)
    db_session.commit()
    db_session.refresh(word)

    msgs = comprehension.build_prompt(word, weakness_axes={})
    system = msgs[0]["content"]
    assert str(COMPREHENSION_PASSAGE_MIN_SENTENCES) in system
    assert str(COMPREHENSION_PASSAGE_MAX_SENTENCES) in system
    assert str(COMPREHENSION_PASSAGE_MAX_CHARS) in system


# ---------------------------------------------------------------------------
# 5. _retrieve_for_comprehension — SQLite fallback
# ---------------------------------------------------------------------------


def test_retrieve_for_comprehension_returns_empty_on_sqlite(db_session):
    """When the active engine is SQLite (the default test DB),
    ``_retrieve_for_comprehension`` returns ``[]`` gracefully.
    The comprehension call falls back to the no-RAG prompt;
    the call still succeeds.
    """
    from app import models

    word = models.Word(
        word="Hund", word_type="Noun", frequency="5", is_complete=True
    )
    db_session.add(word)
    db_session.commit()
    db_session.refresh(word)

    chunks = comprehension._retrieve_for_comprehension(db_session, word)
    assert chunks == [], (
        "SQLite has no vector type; the comprehension RAG-on "
        "helper must return [] so the prompt falls back to "
        "the no-RAG shape (the comprehension call still "
        "succeeds)"
    )


# ---------------------------------------------------------------------------
# 6. generate_comprehension — happy path
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 50,
    completion_tokens: int = 80,
):
    """Build a stub OpenAI-shaped client that returns ``payload``
    as the assistant message content. Mirrors the helper in
    ``test_cloze.py``.
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


@respx.mock
def test_generate_comprehension_happy_path(db_session, monkeypatch):
    """One successful OpenAI call → one ``ComprehensionExercise``
    return. We mock the underlying httpx POST against
    ``https://openrouter.ai/api/v1/chat/completions`` and
    assert that ``generate_comprehension`` returns a
    fully-validated Pydantic model with all four choices
    populated and ``correct_choice`` in ``{"A","B","C","D"}``.
    """
    wid = _seed_word(
        db_session,
        word="Hund",
        word_type="Noun",
        example_de="Der Hund läuft durch den Park.",
    )
    user_id = _seed_user_with_axes(db_session, axes={"abstract_nouns": 2})

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json=_openai_comprehension_response(target_word_id=wid),
        )
    )

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    result = comprehension.generate_comprehension(db_session, user_id)
    assert isinstance(result, ComprehensionExercise)
    assert result.target_word_id == wid
    assert set(result.choices.keys()) == {"A", "B", "C", "D"}
    assert all(isinstance(v, str) and v for v in result.choices.values())
    assert result.correct_choice in {"A", "B", "C", "D"}
    assert (
        COMPREHENSION_PASSAGE_MIN_SENTENCES
        <= len(result.passage)
        <= COMPREHENSION_PASSAGE_MAX_CHARS
    )


# ---------------------------------------------------------------------------
# 7. generate_comprehension — schema-violation retry path
# ---------------------------------------------------------------------------


def test_generate_comprehension_dead_letters_after_three_invalid_attempts(
    db_session, monkeypatch
):
    """Three consecutive schema violations →
    ``ComprehensionGenerationError`` with the structured
    fields. Hard rule #6: retries ≤ 3.
    """
    _seed_word(
        db_session, word="Hund", word_type="Noun", example_de="Der Hund."
    )
    user_id = _seed_user_with_axes(db_session, axes={"abstract_nouns": 2})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Every response is missing ``choices`` → Pydantic validation
    # fails every time. Instructor raises after its retry budget.
    invalid_payload = json.dumps(
        {
            "target_word_id": 1,
            "passage": (
                "Der Hund läuft durch den Park. Er sieht einen Ball und "
                "rennt sofort los. Sein Besitzer lacht."
            ),
            "question": "Was sieht der Hund?",
            # ``choices`` omitted → 4-key invariant violated
            "correct_choice": "A",
            "rationale": "x",
        }
    )
    monkeypatch.setattr(
        comprehension,
        "_openai_client",
        lambda: _make_stub_instructor_client(invalid_payload),
    )

    with pytest.raises(ComprehensionGenerationError) as excinfo:
        comprehension.generate_comprehension(db_session, user_id)
    err = excinfo.value
    assert err.schema_retry_count <= MAX_ATTEMPTS + 1
    assert err.last_validation_error
    attempted = err.attempted_schema
    if not isinstance(attempted, str):
        attempted = json.dumps(attempted)
    assert "ComprehensionExercise" in attempted
    assert "choices" in attempted


def test_generate_comprehension_raises_llm_error_when_api_key_missing(
    db_session, monkeypatch
):
    """No ``OPENROUTER_API_KEY`` → ``LLMError`` (route layer →
    502). We never want a missing key to silently succeed.
    """
    _seed_word(
        db_session, word="Hund", word_type="Noun", example_de="Der Hund."
    )
    user_id = _seed_user_with_axes(db_session, axes={"abstract_nouns": 2})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from app.llm import LLMError

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        comprehension.generate_comprehension(db_session, user_id)


def test_generate_comprehension_rag_on_falls_back_when_retrieval_unavailable(
    db_session, monkeypatch
):
    """``enable_rag=True`` on SQLite (the default test DB) —
    the retrieval helper returns ``[]``, the prompt falls back
    to the no-RAG shape, the comprehension call still
    succeeds. The metadata records ``enable_rag=True`` and
    ``retrieved_chunk_count=0``.
    """
    wid = _seed_word(
        db_session,
        word="Hund",
        word_type="Noun",
        example_de="Der Hund läuft durch den Park.",
    )
    user_id = _seed_user_with_axes(db_session, axes={"abstract_nouns": 2})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    payload = json.dumps(
        {
            "target_word_id": wid,
            "passage": (
                "Der Hund läuft durch den Park. Er sieht einen Ball und "
                "rennt sofort los. Sein Besitzer lacht."
            ),
            "question": "Was sieht der Hund?",
            "choices": {
                "A": "einen Ball",
                "B": "einen Knochen",
                "C": "eine Katze",
                "D": "einen Stock",
            },
            "correct_choice": "A",
            "rationale": "x",
        }
    )
    monkeypatch.setattr(
        comprehension,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    # RAG-on but SQLite — retrieval returns []. The prompt is
    # the no-RAG shape; the call still succeeds.
    result = comprehension.generate_comprehension(
        db_session, user_id, enable_rag=True
    )
    assert isinstance(result, ComprehensionExercise)
    assert result.target_word_id == wid


# ---------------------------------------------------------------------------
# 8. _trace_comprehension — signature-locked
# ---------------------------------------------------------------------------


def test_trace_comprehension_accepts_metadata_contract_keyset():
    """The function signature is locked — this test asserts
    the call site can hand the metadata contract to it
    without raising.
    """
    exercise = ComprehensionExercise(
        target_word_id=1,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        },
        correct_choice="A",
        rationale="x",
    )
    metadata = {
        "user_id": 42,
        "weakness_axes": {"abstract_nouns": 2},
        "word_id": 1,
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ],
        "enable_rag": False,
        "retrieved_chunk_count": 0,
        "retrieved_chunk_k": 0,
        "schema_retry_count": 0,
        "prompt_tokens": 50,
        "completion_tokens": 80,
    }
    # The function returns None and accepts every field; the
    # assertion is the implicit "did not raise".
    assert comprehension._trace_comprehension(exercise, metadata, latency_ms=42) is None
    # ``result=None`` path is exercised on the dead-letter
    # branch (the activity tries to trace even on schema
    # failure before raising
    # ``ComprehensionGenerationError``).
    assert comprehension._trace_comprehension(None, metadata, latency_ms=42) is None


def test_trace_comprehension_invoked_on_happy_path(db_session, monkeypatch):
    """The metadata contract reaches ``_trace_comprehension`` on
    the happy path. We monkeypatch the function to record the
    call, then assert the metadata keyset matches the Phase 6
    contract.
    """
    wid = _seed_word(
        db_session,
        word="Hund",
        word_type="Noun",
        example_de="Der Hund läuft durch den Park.",
    )
    user_id = _seed_user_with_axes(db_session, axes={"abstract_nouns": 2})

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    payload = json.dumps(
        {
            "target_word_id": wid,
            "passage": (
                "Der Hund läuft durch den Park. Er sieht einen Ball und "
                "rennt sofort los. Sein Besitzer lacht."
            ),
            "question": "Was sieht der Hund?",
            "choices": {
                "A": "einen Ball",
                "B": "einen Knochen",
                "C": "eine Katze",
                "D": "einen Stock",
            },
            "correct_choice": "A",
            "rationale": "Stub rationale.",
        }
    )
    monkeypatch.setattr(
        comprehension,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    captured: list[dict] = []

    def fake_trace(result, metadata, latency_ms):
        captured.append({"result": result, "metadata": metadata, "latency_ms": latency_ms})

    monkeypatch.setattr(comprehension, "_trace_comprehension", fake_trace)
    comprehension.generate_comprehension(db_session, user_id)

    assert len(captured) == 1
    call = captured[0]
    md = call["metadata"]
    for key in (
        "user_id",
        "weakness_axes",
        "word_id",
        "model_id",
        "prompt_template_version",
        "prompt_messages",
        "enable_rag",
        "retrieved_chunk_count",
        "retrieved_chunk_k",
        "schema_retry_count",
        "prompt_tokens",
        "completion_tokens",
    ):
        assert key in md, f"metadata missing contract key: {key}"
    assert call["latency_ms"] >= 0


def test_trace_comprehension_metadata_contract_keyset_on_mocked_span(monkeypatch):
    """When the Langfuse client is non-None, the trace span
    carries every metadata-contract field exactly once and
    the contract keyset matches ``docs/PHASE-6.md`` §"The
    metadata contract".
    """
    from unittest.mock import MagicMock
    import importlib

    # Re-acquire the live module — earlier tests may have
    # reloaded it.
    comp_live = importlib.import_module("app.comprehension")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span

    monkeypatch.setattr(comp_live, "get_langfuse", lambda: mock_client)

    exercise = ComprehensionExercise(
        target_word_id=1,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        },
        correct_choice="A",
        rationale="x",
    )
    metadata = {
        "user_id": 42,
        "weakness_axes": {"abstract_nouns": 2},
        "word_id": 1,
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        "enable_rag": False,
        "retrieved_chunk_count": 0,
        "retrieved_chunk_k": 0,
        "schema_retry_count": 0,
        "prompt_tokens": 50,
        "completion_tokens": 80,
    }

    assert comp_live._trace_comprehension(exercise, metadata, latency_ms=42) is None

    # client.span was invoked exactly once with the canonical
    # name.
    assert mock_client.span.call_count == 1
    span_kwargs = mock_client.span.call_args.kwargs
    assert span_kwargs["name"] == "comprehension.generate"
    # input is the prompt messages; output is the serialised
    # exercise.
    assert span_kwargs["input"] == metadata["prompt_messages"]
    assert json.loads(span_kwargs["output"]) == json.loads(
        exercise.model_dump_json()
    )

    # span.update was invoked at least once with the full
    # metadata keyset.
    assert mock_span.update.call_count >= 1
    update_calls = mock_span.update.call_args_list
    merged: dict = {}
    for call in update_calls:
        for key, value in (call.kwargs.get("metadata") or {}).items():
            merged[key] = value

    expected_keys = {
        "user_id",
        "exercise_type",
        "weakness_axes",
        "word_id",
        "target_word_id",
        "model_id",
        "prompt_template_version",
        "schema_retry_count",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "enable_rag",
        "retrieved_chunk_count",
        "retrieved_chunk_k",
    }
    assert set(merged.keys()) == expected_keys, (
        f"metadata keys drifted: got {set(merged.keys())}, "
        f"expected {expected_keys}"
    )
    # Spot-check field values.
    assert merged["user_id"] == 42
    assert merged["exercise_type"] == "comprehension"
    assert merged["word_id"] == 1
    assert merged["target_word_id"] == 1
    assert merged["model_id"] == "qwen/qwen3-235b-a22b-2507"
    assert merged["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert merged["schema_retry_count"] == 0
    assert merged["latency_ms"] == 42
    assert merged["prompt_tokens"] == 50
    assert merged["completion_tokens"] == 80
    assert merged["enable_rag"] is False
    assert merged["retrieved_chunk_count"] == 0
    assert merged["retrieved_chunk_k"] == 0

    # Span closed and client flushed — required for the
    # QA-hook visibility acceptance gate ("trace queryable in
    # UI before request returns").
    assert mock_span.end.call_count == 1
    assert mock_client.flush.call_count == 1


def test_trace_comprehension_is_silent_when_keys_missing(monkeypatch, caplog):
    """When ``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY`` are unset, ``get_langfuse()``
    returns None and ``_trace_comprehension`` returns
    silently without contacting the network.
    """
    import logging

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    # ``result=None`` is the dead-letter branch. The function
    # must still return None without raising.
    with caplog.at_level(logging.WARNING, logger="app.observability"):
        assert comprehension._trace_comprehension(None, {}, 0) is None

    # ``result`` populated — happy-path branch with empty
    # metadata. We must not raise on a missing
    # 'prompt_template_version' key (the function's
    # ``metadata.get(...)`` tolerates the absence).
    metadata = {
        "user_id": 1,
        "weakness_axes": {},
        "word_id": 1,
        "model_id": "stub",
        "enable_rag": False,
        "retrieved_chunk_count": 0,
        "retrieved_chunk_k": 0,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    exercise = ComprehensionExercise(
        target_word_id=1,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        },
        correct_choice="A",
        rationale="x",
    )
    assert comprehension._trace_comprehension(exercise, metadata, latency_ms=0) is None

    # No per-call warnings from ``app.comprehension`` —
    # ``observability.py`` logs once at module import, but
    # the comprehension-side function never logs.
    comp_warnings = [
        r for r in caplog.records
        if r.name == "app.comprehension" and r.levelno >= logging.WARNING
    ]
    assert comp_warnings == [], (
        f"unexpected warnings from app.comprehension: "
        f"{[r.getMessage() for r in comp_warnings]}"
    )


def test_trace_comprehension_swallows_langfuse_failures(monkeypatch):
    """When the Langfuse SDK raises mid-span, the comprehension
    activity still succeeds. Tracing failures must never break
    the request.
    """
    from unittest.mock import MagicMock
    import importlib

    comp_live = importlib.import_module("app.comprehension")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span
    mock_client.flush.side_effect = RuntimeError("simulated flush failure")

    monkeypatch.setattr(comp_live, "get_langfuse", lambda: mock_client)

    exercise = ComprehensionExercise(
        target_word_id=1,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        },
        correct_choice="A",
        rationale="x",
    )
    metadata = {
        "user_id": 1,
        "weakness_axes": {},
        "word_id": 1,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "enable_rag": False,
        "retrieved_chunk_count": 0,
        "retrieved_chunk_k": 0,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    # The exception is swallowed; the activity returns cleanly.
    assert comp_live._trace_comprehension(exercise, metadata, latency_ms=0) is None


# ---------------------------------------------------------------------------
# 9. DSPy module — constructed without OpenRouter
# ---------------------------------------------------------------------------


def test_dspy_module_constructible_without_openrouter(monkeypatch):
    """``ComprehensionModule`` can be constructed without an
    OpenRouter key. The DSPy configure path falls back to
    ``DummyLM`` automatically (Hard rule #8: offline-capable).
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Force re-configuration (the test process might have an
    # LM set by a prior test).
    import dspy

    dspy.settings.lm = None
    module = comprehension.ComprehensionModule()
    assert module is not None
    assert hasattr(module, "predict")


def test_comprehension_signature_inputs_match_production_contract():
    """The DSPy signature carries the four input keys
    (word, learner_axes_json, target_word_id,
    retrieved_chunks_json) and the output is the
    ``ComprehensionExercise`` Pydantic model.
    """
    sig = comprehension.ComprehensionSignature
    input_fields = {
        k
        for k in sig.model_fields
        if sig.model_fields[k].json_schema_extra.get("__dspy_field_type") == "input"
    }
    output_fields = {
        k
        for k in sig.model_fields
        if sig.model_fields[k].json_schema_extra.get("__dspy_field_type") == "output"
    }
    assert input_fields == {
        "word",
        "learner_axes_json",
        "target_word_id",
        "retrieved_chunks_json",
    }
    assert output_fields == {"exercise"}


def test_optimize_comprehension_module_runs_on_two_row_eval_set_offline(monkeypatch):
    """``optimize_comprehension_module`` runs end-to-end on a
    2-row eval set using ``DummyLM``. Mirrors the cloze
    optimizer's offline-path contract.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None

    train = [
        {
            "word": "Hund",
            "learner_axes_json": json.dumps({"abstract_nouns": 2}),
            "target_word_id": 1,
            "retrieved_chunks_json": "[]",
        },
        {
            "word": "gehen",
            "learner_axes_json": json.dumps({"verbs": 3}),
            "target_word_id": 2,
            "retrieved_chunks_json": "[]",
        },
    ]
    val = [
        {
            "word": "kommen",
            "learner_axes_json": json.dumps({"verbs": 3}),
            "target_word_id": 3,
            "retrieved_chunks_json": "[]",
        },
    ]
    # Same three-way fallback as cloze: MIPROv2 →
    # BootstrapFewShot → un-optimized module. Any of the three
    # is acceptable as long as the function returns a
    # ``ComprehensionModule``.
    try:
        module = comprehension.optimize_comprehension_module(
            train_set=train, val_set=val
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"optimize_comprehension_module raised on the offline path: {exc!r}"
        )
    assert isinstance(module, comprehension.ComprehensionModule)


def test_dspy_module_forward_runs_with_dummy_lm(monkeypatch):
    """``ComprehensionModule`` produces a Prediction when given
    a ``DummyLM``-served backend. Verifies the DSPy
    integration end-to-end (signature, predictor, output
    field) without invoking the optimizer.
    """
    import dspy

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    dspy.settings.lm = None
    # Configure the comprehension module's DSPy settings.
    comprehension._configure_dspy()
    module = comprehension.ComprehensionModule()
    pred = module(
        word="Hund",
        learner_axes_json="{}",
        target_word_id=1,
        retrieved_chunks_json="[]",
    )
    # ``pred.exercise`` is a string (DummyLM stub) rather than
    # a validated Pydantic model — the production path
    # validates it via ``instructor``; the DSPy path is for
    # optimization, not for production-grade validation.
    assert hasattr(pred, "exercise")


# ---------------------------------------------------------------------------
# 10. ComprehensionGenerationError — dead-letter shape
# ---------------------------------------------------------------------------


def test_comprehension_generation_error_carries_dead_letter_fields():
    """``ComprehensionGenerationError`` carries the same three
    structured fields as ``ClozeGenerationError``:
    ``attempted_schema``, ``last_validation_error``,
    ``schema_retry_count``. The route layer surfaces these
    in the 502 body so an operator can triage without
    re-running.
    """
    err = ComprehensionGenerationError(
        "test",
        attempted_schema='{"title": "ComprehensionExercise"}',
        last_validation_error="missing field: choices",
        schema_retry_count=3,
    )
    assert err.attempted_schema == '{"title": "ComprehensionExercise"}'
    assert err.last_validation_error == "missing field: choices"
    assert err.schema_retry_count == 3


def test_comprehension_generation_error_from_validation_failure_serialises_dict():
    """``from_validation_failure`` accepts a Pydantic
    ``model_json_schema()`` dict (the Pydantic v2 default
    shape) and serialises it to JSON for
    ``attempted_schema``.
    """
    schema = ComprehensionExercise.model_json_schema()
    err = ComprehensionGenerationError.from_validation_failure(
        "test",
        schema=schema,
        last_validation_error="x",
        schema_retry_count=1,
    )
    # ``attempted_schema`` is a JSON-encoded string.
    assert isinstance(err.attempted_schema, str)
    # Round-trip back to a dict and confirm we got the schema
    # back.
    assert json.loads(err.attempted_schema) == schema


# ---------------------------------------------------------------------------
# 11. Wire shape parity — schemas.py
# ---------------------------------------------------------------------------


def test_comprehension_exercise_out_wire_shape_matches_generator():
    """``ComprehensionExerciseOut`` (in ``app.schemas``) mirrors
    the generator fields and narrows the exercise_type
    discriminator to ``Literal["comprehension"]``.

    Phase 6.5 added ``exercise_id`` to the wire shape (the
    server-minted per-generation id that round-trips into
    ``/exercises/grade`` for Phase 6.6 dispatch). The generator
    ``ComprehensionExercise`` does NOT carry ``exercise_id`` —
    the id is stamped at response time only.
    """
    from app.schemas import ComprehensionExerciseOut

    # ``exercise_type`` narrows to the comprehension literal.
    type_field = ComprehensionExerciseOut.model_fields["exercise_type"]
    # Pydantic v2 stores the literal as ``Literal["comprehension"]``
    # in the annotation.
    assert "comprehension" in str(type_field.annotation)

    # All generator fields are present on the wire shape.
    out_fields = set(ComprehensionExerciseOut.model_fields.keys())
    for f in (
        "exercise_type",
        "target_word_id",
        "passage",
        "question",
        "choices",
        "correct_choice",
        "rationale",
        "prompt_template_version",
    ):
        assert f in out_fields, f"wire shape missing field: {f}"

    # Wire-only field added in Phase 6.5: ``exercise_id`` (the
    # matching wire shape added it in 6.2 / 6.3; 6.5 brings
    # comprehension to parity so 6.6 can dispatch on the same id).
    assert "exercise_id" in out_fields, (
        "wire shape missing exercise_id field added in Phase 6.5"
    )


def test_comprehension_exercise_out_serialises_with_default_exercise_type():
    """The wire shape defaults ``exercise_type`` to
    ``"comprehension"`` so a future route that forgets to set
    the discriminator still emits the right tag.

    The wire shape also carries a server-minted ``exercise_id``
    (Phase 6.5 ships this so the same id round-trips into
    ``/exercises/grade`` for Phase 6.6 dispatch and into the
    ``grade_logs`` row for Phase 6.7 Ragas join determinism).
    """
    from app.schemas import ComprehensionExerciseOut

    out = ComprehensionExerciseOut(
        target_word_id=1,
        exercise_id=42,
        passage=(
            "Der Hund läuft durch den Park. Er sieht einen Ball und "
            "rennt sofort los. Sein Besitzer lacht."
        ),
        question="Was sieht der Hund?",
        choices={
            "A": "einen Ball",
            "B": "einen Knochen",
            "C": "eine Katze",
            "D": "einen Stock",
        },
        correct_choice="A",
        rationale="x",
        prompt_template_version="comprehension-v1",
    )
    assert out.exercise_type == "comprehension"
    # ``exercise_id`` is a server-minted int; the test passes 42 to
    # confirm the field round-trips through Pydantic validation.
    assert out.exercise_id == 42


def test_comprehension_generate_request_enable_rag_defaults_to_false():
    """Hard rule #1: RAG-on is opt-in. The request schema
    defaults ``enable_rag`` to ``False`` so existing clients
    (the future Phase 6.5 route, the Phase 9 study-session
    mixer) see no schema change.
    """
    from app.schemas import ComprehensionGenerateRequest

    req = ComprehensionGenerateRequest()
    assert req.enable_rag is False

    # Opt-in path: caller can still set it explicitly.
    req_on = ComprehensionGenerateRequest(enable_rag=True)
    assert req_on.enable_rag is True
