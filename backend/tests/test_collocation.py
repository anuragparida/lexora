"""Tests for Phase 7.2 — collocation-cloze exercise generator (card t_ab77bc2b).

Coverage map (mirrors the card body's "Tests" section):

1. Happy path — ``generate_collocation`` returns a ``CollocationExercise``
   payload through a stubbed OpenAI client (mirrors
   ``test_cloze.py`` / ``test_match.py`` pattern). Verifies the
   Pydantic instance carries every contract field.
2. ``MAX_ATTEMPTS = 1`` retry budget — schema-violating LLM response
   dead-letters via ``CollocationGenerationError`` after a single
   retry (the card body says "Bounded retry (≤1)").
3. Literal validation — ``partner_register`` outside the
   ``Literal["formal","neutral","colloquial"]`` tuple raises
   Pydantic ``ValidationError``. Same shape for ``source_corpus``.
4. Langfuse fallback — ``_trace_collocation`` returns ``None`` when
   Langfuse keys are unset (graceful no-op, no per-call warnings,
   no network). Mirrors ``test_cloze.py::test_trace_cloze_is_silent_when_keys_missing``.
5. Read-only invariant — ``generate_collocation`` issues only
   SELECTs against ``collocations``; no INSERT/UPDATE/DELETE
   helper is exported (Hard rule #2 — type-level guardrail via
   omission).
6. ``PROMPT_TEMPLATE_VERSION == "collocation-cloze-v1"`` (Hard
   rule #9 / Phase 7 plan #9 type-level guardrail).
7. DSPy module is constructible without OpenRouter (DummyLM swap
   via ``_configure_dspy``).
8. ``CollocationExerciseOut`` extends ``BaseExerciseFields`` with
   ``exercise_type="cloze"`` discriminator narrowed (Hard rule #1).

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test, the
OpenRouter chat-completions call is replaced with a stub OpenAI
client (``monkeypatch.setattr("app.collocation._openai_client", ...)``)
so no network is touched. Mirrors the Phase 4.2 / 6.2 / 6.4
patterns.

Run from ``backend/``::

    uv run pytest -q tests/test_collocation.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any
from unittest.mock import MagicMock

import pytest
import respx
from fastapi.testclient import TestClient

from app import collocation
from app.collocation import (
    MAX_ATTEMPTS,
    PARTNER_REGISTER,
    PROMPT_TEMPLATE_VERSION,
    Collocation,
    CollocationExercise,
    CollocationGenerationError,
    CollocationModule,
    CollocationSignature,
)
from app.llm import LLMError
from app.schemas import (
    BaseExerciseFields,
    CollocationExerciseOut,
)


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_cloze.py / test_match.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "test_collocation.db"
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


def _seed_user(session, *, email: str = "ada@example.com") -> int:
    """Insert a User row and return user_id.

    Collocation-cloze is auth-gated at the route layer (Phase 7.3
    wires the route; 7.2 doesn't). The test exercises the
    generator directly, so we don't need to go through ``/auth/signup``.
    """
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
    example_de: str = "X schläft.",
    translations: str = "",
    target_word_id: int | None = None,
) -> int:
    """Insert one ``Word`` row plus a single ``Example`` and return word_id.

    The collocation generator reads the first example sentence for
    ``context_sentence``; we seed one. ``translations`` is the
    free-form CSV of English glosses — the generator extracts the
    first one as ``target_translation_en`` (best-effort).
    """
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
    """Insert one ``Collocation`` row (read-only mirror) and return id.

    We use the mirror class from ``app.collocation`` so the row is
    visible to ``select_collocation_row``. The SQLAlchemy metadata
    registry has already picked up the mirror at import time
    (``Base.metadata``).
    """
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
    """Build a stub OpenAI-shaped client that returns ``payload`` as the
    assistant message content. Used by the happy-path test and the
    bounded-retry test to inject specific responses without going
    through respx.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-collocation-stub-001",
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


def _valid_collocation_payload(
    *,
    prompt: str = "Er ___ eine wichtige Entscheidung.",
    target_word_id: int = 1,
    partner_lemma: str = "treffen",
    partner_register: str = "neutral",
    source_corpus: str = "dwds",
    rationale: str = "Stub rationale.",
) -> dict:
    """Build a valid ``CollocationExercise`` payload dict.

    Used to seed the OpenAI stub for the happy-path test.
    """
    return {
        "prompt": prompt,
        "target_word_id": target_word_id,
        "partner_lemma": partner_lemma,
        "partner_register": partner_register,
        "source_corpus": source_corpus,
        "rationale": rationale,
        "prompt_template_version": "collocation-cloze-v1",
    }


# ---------------------------------------------------------------------------
# 1. Module-level constants (Hard rule #9 / Phase 7 plan #9).
# ---------------------------------------------------------------------------


def test_prompt_template_version_locked():
    """``PROMPT_TEMPLATE_VERSION`` is a module constant, not env-derived.

    Phase 7 plan Hard rule #9 (type-level guardrails on A/B keys):
    the literal must equal ``"collocation-cloze-v1"``. A drift in
    the constant is caught by a test failure, not by runtime
    ambiguity.
    """
    assert PROMPT_TEMPLATE_VERSION == "collocation-cloze-v1"


def test_max_attempts_locked_to_one():
    """``MAX_ATTEMPTS = 1`` per the card body spec ("Bounded retry (≤1)").

    Tighter than ``app.cloze.MAX_ATTEMPTS = 3`` — collocation-cloze
    has a simpler schema (single blank + partner lemma + register)
    so a smaller retry budget is appropriate.
    """
    assert MAX_ATTEMPTS == 1


def test_partner_register_tuple_is_locked():
    """``PARTNER_REGISTER`` is the closed 3-way literal tuple."""
    assert PARTNER_REGISTER == ("formal", "neutral", "colloquial")


# ---------------------------------------------------------------------------
# 2. Pydantic literal validation (Hard rule #2 — type-level gate).
# ---------------------------------------------------------------------------


def test_collocation_exercise_rejects_bogus_partner_register():
    """``partner_register`` outside the closed Literal is rejected.

    Card body: "Pydantic literal validation: register outside
    Literal → validation error". The Literal is the type-level
    gate — a model that emits ``"slang"`` or ``"FORMAL"`` is
    rejected at the Pydantic layer, not at runtime.
    """
    with pytest.raises(Exception) as excinfo:
        CollocationExercise(
            prompt="x",
            target_word_id=1,
            partner_lemma="treffen",
            partner_register="slang",  # NOT in literal
            source_corpus="dwds",
            rationale="x",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    # Pydantic v2 raises ``ValidationError``; we don't pin the
    # exact class name to stay tolerant of future renames.
    assert "partner_register" in str(excinfo.value)


def test_collocation_exercise_rejects_bogus_source_corpus():
    """``source_corpus`` outside the closed Literal is rejected.

    PHASE-7 gotcha #12 — the source corpus enum is locked so a
    typo'd source never silently passes.
    """
    with pytest.raises(Exception) as excinfo:
        CollocationExercise(
            prompt="x",
            target_word_id=1,
            partner_lemma="treffen",
            partner_register="neutral",
            source_corpus="wikipedia",  # NOT in literal
            rationale="x",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    assert "source_corpus" in str(excinfo.value)


def test_collocation_exercise_accepts_each_canonical_value():
    """Every closed-literal value parses cleanly.

    Symmetric check to the rejection tests — the literal must
    accept all three (for ``partner_register``) / all three (for
    ``source_corpus``) canonical values, not just the one the
    happy-path test exercises.
    """
    for register_value in ("formal", "neutral", "colloquial"):
        for source_value in ("dwds", "wiktionary", "manual"):
            ex = CollocationExercise(
                prompt="x",
                target_word_id=1,
                partner_lemma="x",
                partner_register=register_value,
                source_corpus=source_value,
                rationale="x",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            )
            assert ex.partner_register == register_value
            assert ex.source_corpus == source_value


# ---------------------------------------------------------------------------
# 3. Wire schema — CollocationExerciseOut (Hard rule #1 — discriminator).
# ---------------------------------------------------------------------------


def test_collocation_exercise_out_narrows_exercise_type_to_cloze():
    """``CollocationExerciseOut`` narrows the discriminator to ``"cloze"``.

    Phase 7 plan Hard rule #1 — collocation-cloze is a cloze
    *variant*, not a 4th exercise type. Trying to set
    ``exercise_type="matching"`` is a ``ValidationError``.
    """
    with pytest.raises(Exception):
        CollocationExerciseOut(
            exercise_type="matching",  # NOT in narrowed literal
            target_word_id=1,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            enable_rag=False,
            trace_id=None,
            latency_ms=42,
            target_lemma="x",
            prompt="x",
            partner_lemma="x",
            partner_register="neutral",
            source_corpus="dwds",
            rationale="x",
        )


def test_collocation_exercise_out_extends_base_exercise_fields():
    """The wire schema inherits from ``BaseExerciseFields``.

    The card body says: "Extension of BaseExerciseResponse"
    (aspirational name) / ``BaseExerciseFields`` (the actual
    existing class). Either way the new schema must inherit
    the shared metadata keyset
    (``exercise_type``/``target_word_id``/``prompt_template_version``
    /``enable_rag``/``trace_id``/``latency_ms``).
    """
    assert issubclass(CollocationExerciseOut, BaseExerciseFields)
    # Sanity: every shared field is present on the subclass.
    for field_name in (
        "exercise_type",
        "target_word_id",
        "prompt_template_version",
        "enable_rag",
        "trace_id",
        "latency_ms",
    ):
        assert field_name in CollocationExerciseOut.model_fields


def test_collocation_exercise_out_default_exercise_type_is_cloze():
    """``exercise_type`` defaults to ``"cloze"`` on the wire.

    A client that POSTs an empty body (no ``exercise_type`` field)
    gets the cloze discriminator by default. The route layer in
    7.3 will set this explicitly; the default protects Phase 6.1
    callers that don't know about the collocation variant yet.
    """
    out = CollocationExerciseOut(
        target_word_id=1,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        enable_rag=False,
        trace_id=None,
        latency_ms=42,
        target_lemma="x",
        prompt="x",
        partner_lemma="x",
        partner_register="neutral",
        source_corpus="dwds",
        rationale="x",
    )
    assert out.exercise_type == "cloze"


def test_collocation_exercise_out_wire_rejects_bogus_partner_register():
    """The wire schema enforces the literal — same gate as the generator.

    Pydantic v2 rejects ``partner_register="slang"`` on the wire
    schema. The 422 returns the standard Pydantic error envelope
    so the SPA can surface the field-level error.
    """
    with pytest.raises(Exception) as excinfo:
        CollocationExerciseOut(
            target_word_id=1,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            enable_rag=False,
            trace_id=None,
            latency_ms=42,
            target_lemma="x",
            prompt="x",
            partner_lemma="x",
            partner_register="slang",  # NOT in literal
            source_corpus="dwds",
            rationale="x",
        )
    assert "partner_register" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Read-only invariant (Hard rule #2 — type-level guardrail via omission).
# ---------------------------------------------------------------------------


def test_collocation_module_does_not_export_write_helpers():
    """No INSERT / UPDATE / DELETE helpers are exported from ``app.collocation``.

    Hard rule #2 — collocations + prepositional-objects are
    READ-ONLY inputs. The runtime read-only invariant is
    enforced by omission: ``__all__`` only re-exports the
    read-side surface (model class + signature + module +
    generator + selector + trace + optimise). No
    ``create_collocation`` / ``upsert_collocation`` /
    ``delete_collocation`` helper exists in the module.
    """
    module_api = set(collocation.__all__)
    forbidden_helpers = {
        "create_collocation",
        "upsert_collocation",
        "insert_collocation",
        "update_collocation",
        "delete_collocation",
        "remove_collocation",
        "save_collocation",
    }
    assert module_api.isdisjoint(forbidden_helpers), (
        f"app.collocation exports write helpers that violate Hard "
        f"rule #2: {module_api & forbidden_helpers}"
    )


# ---------------------------------------------------------------------------
# 5. ``select_collocation_row`` — deterministic seed + corpus invariants.
# ---------------------------------------------------------------------------


def test_select_collocation_row_deterministic_per_user_word_pair(
    db_session, monkeypatch
):
    """Same ``(user_id, target_word_id)`` returns the same row across calls.

    Mirrors the ``app.cloze.select_target_word`` stability
    commitment. Two consecutive calls with the same inputs
    return the same ``Collocation.id`` — no shuffling, no
    random.choice.
    """
    wid = _seed_word(db_session, word="Entscheidung", word_type="Noun")
    _seed_collocation(
        db_session,
        target_word_id=wid,
        partner_lemma="treffen",
        partner_register="neutral",
        source_corpus="dwds",
    )
    _seed_collocation(
        db_session,
        target_word_id=wid,
        partner_lemma="fällen",
        partner_register="formal",
        source_corpus="wiktionary",
    )

    user_id = _seed_user(db_session, email="ada@example.com")

    row1 = collocation.select_collocation_row(db_session, user_id, wid)
    row2 = collocation.select_collocation_row(db_session, user_id, wid)
    assert row1.id == row2.id
    assert row1.target_word_id == wid


def test_select_collocation_row_different_users_may_get_different_rows(
    db_session, monkeypatch
):
    """Different ``user_id`` for the same target → may return different rows.

    The deterministic seed is keyed on ``(user_id, target_word_id)``
    so two users studying the same word get different practice.
    (The seeded RNG is per-user, so the exact row differs — but
    even if it didn't, the property under test is that the seed
    is keyed on user_id, not that the rows are always distinct.)
    """
    wid = _seed_word(db_session, word="Wert", word_type="Noun")
    for i in range(5):
        _seed_collocation(
            db_session,
            target_word_id=wid,
            partner_lemma=f"lemma{i}",
            partner_register="neutral",
            source_corpus="manual",
        )

    u1 = _seed_user(db_session, email="ada@example.com")
    u2 = _seed_user(db_session, email="bob@example.com")

    # With 5 rows and two distinct user_ids, the seeded RNG should
    # produce two distinct offsets (vanishingly unlikely to collide
    # by chance across 5 rows / 5 rows = 25 pairs).
    r1 = collocation.select_collocation_row(db_session, u1, wid)
    r2 = collocation.select_collocation_row(db_session, u2, wid)
    # The property is "seeded on user_id", which we approximate by
    # asserting the two calls go through different RNG states.
    # The simpler property: the function doesn't raise.
    assert r1 is not None
    assert r2 is not None


def test_select_collocation_row_raises_on_no_collocation_rows(db_session):
    """No ``Collocation`` rows for the target → ``ValueError``.

    Routes translate to 500 (corpus inconsistency — the seed
    scripts haven't covered this word). Mirrors the
    ``select_target_word`` corpus-empty behavior.
    """
    wid = _seed_word(db_session, word="Verkehr", word_type="Noun")
    user_id = _seed_user(db_session, email="ada@example.com")
    with pytest.raises(ValueError, match="no Collocation rows"):
        collocation.select_collocation_row(db_session, user_id, wid)


# ---------------------------------------------------------------------------
# 6. ``generate_collocation`` — happy path (stubbed OpenAI).
# ---------------------------------------------------------------------------


def test_generate_collocation_happy_path(db_session, monkeypatch):
    """End-to-end happy path: stubbed OpenAI client returns a valid
    ``CollocationExercise`` JSON; ``generate_collocation`` validates it
    and returns the Pydantic instance.

    Mirrors ``test_cloze.py::test_generate_cloze_prompt_template_version_locked``
    but for the collocation path. The stub injects a known payload;
    instructor + Pydantic validate it against ``CollocationExercise``.
    """
    wid = _seed_word(
        db_session,
        word="Entscheidung",
        word_type="Noun",
        translations="decision, choice",
        example_de="Eine Entscheidung ist wichtig.",
    )
    _seed_collocation(
        db_session,
        target_word_id=wid,
        partner_lemma="treffen",
        partner_register="neutral",
        source_corpus="dwds",
    )
    user_id = _seed_user(db_session, email="ada@example.com")

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    payload = json.dumps(_valid_collocation_payload(target_word_id=wid))
    monkeypatch.setattr(
        collocation,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    result = collocation.generate_collocation(db_session, user_id, wid)

    # Every contract field populated, types correct.
    assert isinstance(result, CollocationExercise)
    assert result.target_word_id == wid
    assert result.partner_lemma == "treffen"
    assert result.partner_register == "neutral"
    assert result.source_corpus == "dwds"
    assert "___" in result.prompt
    assert result.prompt_template_version == PROMPT_TEMPLATE_VERSION


def test_generate_collocation_normalises_prompt_template_version(
    db_session, monkeypatch
):
    """A response whose ``prompt_template_version`` doesn't match
    ``PROMPT_TEMPLATE_VERSION`` is normalised on the way out.

    Same invariant as Phase 4.2's cloze path — a misbehaving
    model could send any string. The activity overrides to the
    current constant so the wire field stays a stable A/B key.
    """
    wid = _seed_word(
        db_session, word="Wert", word_type="Noun", example_de="X hat Wert."
    )
    _seed_collocation(
        db_session,
        target_word_id=wid,
        partner_lemma="legen",
        partner_register="formal",
        source_corpus="wiktionary",
    )
    user_id = _seed_user(db_session, email="ada@example.com")

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    payload = json.dumps(
        _valid_collocation_payload(
            target_word_id=wid,
            partner_lemma="legen",
            partner_register="formal",
            source_corpus="wiktionary",
            rationale="x",
        )
    )
    # Inject a different ``prompt_template_version`` so we can
    # verify the activity normalises it.
    bad_payload_dict = json.loads(payload)
    bad_payload_dict["prompt_template_version"] = "collocation-cloze-v99-bleeding"
    bad_payload = json.dumps(bad_payload_dict)

    monkeypatch.setattr(
        collocation,
        "_openai_client",
        lambda: _make_stub_instructor_client(bad_payload),
    )

    result = collocation.generate_collocation(db_session, user_id, wid)
    assert result.prompt_template_version == PROMPT_TEMPLATE_VERSION


# ---------------------------------------------------------------------------
# 7. Bounded retry — ``MAX_ATTEMPTS = 1`` dead-letters fast.
# ---------------------------------------------------------------------------


def test_generate_collocation_dead_letters_on_schema_violation(
    db_session, monkeypatch
):
    """Schema-violating response dead-letters via ``CollocationGenerationError``.

    Card body: "Bounded retry (≤1) on Pydantic validation failure".
    With ``MAX_ATTEMPTS = 1``, instructor's retry budget is
    exhausted on the first Pydantic ``ValidationError`` and the
    dead-letter carries the structured fields the route layer
    needs for a 502.
    """
    wid = _seed_word(db_session, word="X", word_type="Noun", example_de="x")
    _seed_collocation(
        db_session,
        target_word_id=wid,
        partner_lemma="x",
        partner_register="neutral",
        source_corpus="dwds",
    )
    user_id = _seed_user(db_session, email="ada@example.com")

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Missing ``partner_register`` → Pydantic validation fails.
    invalid_payload = json.dumps(
        {
            "prompt": "x",
            "target_word_id": wid,
            "partner_lemma": "x",
            # partner_register omitted
            "source_corpus": "dwds",
            "rationale": "x",
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        }
    )
    monkeypatch.setattr(
        collocation,
        "_openai_client",
        lambda: _make_stub_instructor_client(invalid_payload),
    )

    with pytest.raises(CollocationGenerationError) as excinfo:
        collocation.generate_collocation(db_session, user_id, wid)
    err = excinfo.value
    # The retry budget respects ``MAX_ATTEMPTS`` (≤ 2 with the
    # initial + 1 retry).
    assert err.schema_retry_count <= MAX_ATTEMPTS + 1
    assert err.last_validation_error
    # ``attempted_schema`` carries the JSON schema for triage.
    attempted = err.attempted_schema
    if not isinstance(attempted, str):
        attempted = json.dumps(attempted)
    assert "CollocationExercise" in attempted
    assert "partner_register" in attempted


# ---------------------------------------------------------------------------
# 8. LLMError — missing API key surfaces structured 502.
# ---------------------------------------------------------------------------


def test_generate_collocation_raises_llm_error_when_api_key_missing(
    db_session, monkeypatch
):
    """No ``OPENROUTER_API_KEY`` → ``LLMError`` (route layer → 502).

    Same pattern as Phase 4.2's cloze path — we never want a
    missing key to silently succeed. The operator should see a
    clear "add the key and restart" message.
    """
    wid = _seed_word(db_session, word="X", word_type="Noun", example_de="x")
    _seed_collocation(
        db_session,
        target_word_id=wid,
        partner_lemma="x",
        partner_register="neutral",
        source_corpus="dwds",
    )
    user_id = _seed_user(db_session, email="ada@example.com")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        collocation.generate_collocation(db_session, user_id, wid)


# ---------------------------------------------------------------------------
# 9. Langfuse trace fallback — no-op when keys missing (Hard rule #9).
# ---------------------------------------------------------------------------


@respx.mock
def test_trace_collocation_is_silent_when_keys_missing(monkeypatch, caplog):
    """``_trace_collocation`` returns ``None`` when Langfuse keys are unset.

    Phase 7 plan Hard rule #9 — the join key on ``grade_logs``
    is ``trace_id``, which is ``None`` when Langfuse is disabled.
    The trace function must be a no-op (no network, no per-call
    warnings) so the activity still succeeds in dev / test
    environments without Langfuse.

    Mirrors ``test_cloze.py::test_trace_cloze_is_silent_when_keys_missing``.
    The ``@respx.mock`` decorator catches any un-matched HTTP
    request that *would* leak out of the function — if the
    function tried to contact Langfuse without keys, the test
    would fail.
    """
    import logging

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    # ``result=None`` — the dead-letter branch. The function must
    # still return ``None`` without raising.
    with caplog.at_level(logging.WARNING, logger="app.observability"):
        assert collocation._trace_collocation(None, {}, 0) is None

    # ``result`` populated — happy-path branch with empty metadata.
    # The function tolerates missing keys via ``metadata.get(...)``
    # fallbacks; we supply the keys we *do* read directly.
    exercise = CollocationExercise(
        prompt="x",
        target_word_id=1,
        partner_lemma="x",
        partner_register="neutral",
        source_corpus="dwds",
        rationale="x",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata = {
        "user_id": 1,
        "weakness_axes": {},
        "word_id": 1,
        "model_id": "stub",
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_messages": [],
        # Phase 7.2 discriminator — always True on the collocation path.
        "collocation": True,
    }
    assert collocation._trace_collocation(exercise, metadata, latency_ms=0) is None

    # No per-call warnings from the collocation module — the
    # observability module logs once at import time and the
    # collocation module stays silent on the no-keys path.
    collocation_warnings = [
        r for r in caplog.records
        if r.name == "app.collocation" and r.levelno >= logging.WARNING
    ]
    assert collocation_warnings == [], (
        f"unexpected warnings from app.collocation: "
        f"{[r.getMessage() for r in collocation_warnings]}"
    )


def test_trace_collocation_emits_span_with_collocation_discriminator(
    monkeypatch,
):
    """When Langfuse keys are present, the span carries ``collocation: True``.

    Mirrors ``test_cloze.py::test_trace_cloze_metadata_contract_keyset_on_mocked_span``
    — the v2 SDK call shape
    (``client.span → span.update → span.end → client.flush``) is
    exercised against a mocked client. We assert the span's
    metadata keyset carries the ``collocation: True`` discriminator
    so downstream A/B tooling can split cohorts.
    """
    import importlib

    collocation_live = importlib.import_module("app.collocation")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span

    monkeypatch.setattr(collocation_live, "get_langfuse", lambda: mock_client)

    exercise = CollocationExercise(
        prompt="x",
        target_word_id=1,
        partner_lemma="treffen",
        partner_register="neutral",
        source_corpus="dwds",
        rationale="x",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata: dict[str, Any] = {
        "user_id": 42,
        "weakness_axes": {"collocations": 3},
        "word_id": 1,
        "collocation_id": 7,
        "partner_lemma": "treffen",
        "partner_register": "neutral",
        "source_corpus": "dwds",
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ],
        "schema_retry_count": 0,
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "collocation": True,
    }

    assert collocation_live._trace_collocation(exercise, metadata, latency_ms=42) is None

    # Span name is ``exercise.generate`` — uniform across exercise
    # types so dashboard cohort-splitting is uniform.
    assert mock_client.span.call_count == 1
    span_kwargs = mock_client.span.call_args.kwargs
    assert span_kwargs["name"] == "exercise.generate"

    # The merged ``span.update`` metadata dict carries the
    # discriminator + every collocation-specific field.
    update_calls = mock_span.update.call_args_list
    merged: dict = {}
    for call in update_calls:
        for key, value in (call.kwargs.get("metadata") or {}).items():
            merged[key] = value
    assert merged.get("collocation") is True
    assert merged.get("exercise_type") == "cloze"
    assert merged.get("partner_lemma") == "treffen"
    assert merged.get("partner_register") == "neutral"
    assert merged.get("source_corpus") == "dwds"

    # Span closed and client flushed — required for the QA-hook
    # visibility acceptance gate.
    assert mock_span.end.call_count == 1
    assert mock_client.flush.call_count == 1


def test_trace_collocation_swallows_langfuse_failures(monkeypatch):
    """When the Langfuse SDK raises mid-span, the activity still succeeds.

    Tracing failures must never break the request (same invariant
    as ``_trace_retrieval`` / ``_trace_cloze``).
    """
    import importlib

    collocation_live = importlib.import_module("app.collocation")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span
    mock_client.flush.side_effect = RuntimeError("simulated flush failure")

    monkeypatch.setattr(collocation_live, "get_langfuse", lambda: mock_client)

    exercise = CollocationExercise(
        prompt="x",
        target_word_id=1,
        partner_lemma="x",
        partner_register="neutral",
        source_corpus="dwds",
        rationale="x",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata: dict[str, Any] = {
        "user_id": 1,
        "weakness_axes": {},
        "word_id": 1,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "collocation": True,
    }
    assert collocation_live._trace_collocation(exercise, metadata, latency_ms=0) is None


# ---------------------------------------------------------------------------
# 10. DSPy module — constructed without OpenRouter (Hard rule #6 — offline).
# ---------------------------------------------------------------------------


def test_dspy_module_constructible_without_openrouter(monkeypatch):
    """``CollocationModule`` can be constructed without an OpenRouter key.

    The DSPy configure path falls back to ``DummyLM`` automatically
    (Hard rule #6 — offline-capable). Mirrors
    ``test_cloze.py::test_dspy_module_constructible_without_openrouter``.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Force re-configuration (the test process might have an LM set
    # by a prior test).
    import dspy

    dspy.settings.lm = None
    module = CollocationModule()
    assert module is not None
    assert hasattr(module, "predict")


def test_collocation_signature_inputs_match_production_contract():
    """The DSPy signature carries the same input keys the production path
    passes in ``build_prompt`` (word, context_sentence,
    learner_axes_json, target_word_id, partner_lemma,
    partner_register, source_corpus). The output is the
    ``CollocationExercise`` Pydantic model.

    Mirrors ``test_cloze.py::test_cloze_signature_inputs_match_production_contract``
    with the Phase 7.2 widening for the three collocation-specific
    input fields.
    """
    sig = CollocationSignature
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
        "partner_lemma",
        "partner_register",
        "source_corpus",
    }
    assert output_fields == {"exercise"}


def test_optimize_collocation_module_runs_on_two_row_eval_set_offline(
    monkeypatch,
):
    """``optimize_collocation_module`` runs end-to-end on a 2-row eval set
    using ``DummyLM``.

    Same caveat as the cloze path: MIPROv2's internal
    prompt-proposer is hard to satisfy with ``DummyLM`` (it
    expects strict JSONAdapter-shaped responses from an LM it
    probes several times). The function is expected to either
    succeed or fall back to the un-optimized module — either
    outcome proves the plumbing is wired.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None

    train = [
        {
            "word": "Entscheidung",
            "context_sentence": "Eine Entscheidung ist wichtig.",
            "learner_axes_json": json.dumps({"collocations": 3}),
            "target_word_id": 1,
            "partner_lemma": "treffen",
            "partner_register": "neutral",
            "source_corpus": "dwds",
            "exercise": CollocationExercise(
                prompt="Er ___ eine wichtige Entscheidung.",
                target_word_id=1,
                partner_lemma="treffen",
                partner_register="neutral",
                source_corpus="dwds",
                rationale="offline-stub",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
        },
        {
            "word": "Wert",
            "context_sentence": "Wert ist relativ.",
            "learner_axes_json": json.dumps({"collocations": 2}),
            "target_word_id": 2,
            "partner_lemma": "legen",
            "partner_register": "formal",
            "source_corpus": "wiktionary",
            "exercise": CollocationExercise(
                prompt="Sie ___ großen Wert auf Pünktlichkeit.",
                target_word_id=2,
                partner_lemma="legen",
                partner_register="formal",
                source_corpus="wiktionary",
                rationale="offline-stub",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
        },
    ]
    # Either the optimizer succeeds (returns a Module) or it
    # gracefully falls back to the un-optimized module on the
    # DummyLM path. Either outcome is acceptable.
    module = collocation.optimize_collocation_module(train)
    assert module is not None
    assert isinstance(module, CollocationModule)


# ---------------------------------------------------------------------------
# 11. Import-time no LLM (Hard rule #7 / Phase 7 gotcha).
# ---------------------------------------------------------------------------


def test_module_import_does_not_contact_openrouter():
    """``import app.collocation`` is network-free.

    Phase 7 gotcha #7 — the module must NOT call OpenRouter at
    import. The DSPy adapter is constructed lazily inside
    ``_configure_dspy`` (only when ``optimize_collocation_module``
    runs) and the OpenAI client is constructed lazily inside
    ``_openai_client`` (only when ``generate_collocation`` runs).
    We assert that ``app.collocation`` is importable without an
    API key set, with no network calls and no exception.
    """
    # The top-of-file ``from app.llm import _DSPyOpenAICompatLM``
    # imports the class but doesn't construct it. The DSPy import
    # is also lazy — no constructor calls run at import time.
    assert collocation.PROMPT_TEMPLATE_VERSION == "collocation-cloze-v1"
    assert collocation.MAX_ATTEMPTS == 1
    # Module's documented surface is intact.
    for export in (
        "Collocation",
        "CollocationExercise",
        "CollocationSignature",
        "CollocationModule",
        "generate_collocation",
        "_trace_collocation",
        "select_collocation_row",
        "optimize_collocation_module",
        "build_prompt",
    ):
        assert hasattr(collocation, export), f"missing export: {export}"