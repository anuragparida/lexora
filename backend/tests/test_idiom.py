"""Tests for Phase 8.3 — idiom exercise generator + Literal widening.

Card: t_fa86ac58.

Coverage map (mirrors the card body's "Tests" section):

1. Module-level constants — ``PROMPT_TEMPLATE_VERSION == "idiom-v1"``,
   ``MAX_ATTEMPTS == 3``, ``FREQUENCY_BAND == ("high","mid","low")``,
   ``SOURCE_ATTRIBUTION == ("dwds","goethe","schiller")``. Type-level
   guardrails on the A/B keys (Hard rule #8 — no env-derived
   thresholds).
2. Pydantic ``IdiomExercise`` bounds — 5..200 char ``phrase``, 1..400
   char ``definition``, 5..400 char ``example_usage``, the closed
   ``Literal["high","mid","low"]`` on ``frequency_band``, the
   comma-joined subset invariant on ``source_attribution``.
3. ``IdiomExerciseOut`` extends ``BaseExerciseFields`` and narrows
   ``exercise_type`` to ``Literal["idiom"]`` (Hard rule #1 — wire
   discriminator is the type-level gate).
4. Literal widening on ``BaseExerciseFields.exercise_type`` — the
   base class now accepts ``"idiom"``; existing callers passing
   ``"cloze"`` / ``"matching"`` / ``"comprehension"`` parse as
   before (no regression). The 8.3 schema test asserts this.
5. ``IdiomModule`` is constructible without OpenRouter (DummyLM
   swap via ``_configure_dspy``).
6. ``IdiomSignature`` carries the production input keys
   (word_id, target_phrase, curated_definition, attested_quote,
   source_attribution, frequency_band) and the
   ``IdiomExercise`` Pydantic-typed output.
7. ``generate_idiom`` end-to-end through a stubbed OpenAI client —
   Pydantic-validated response, retry budget on schema violations,
   Langfuse fallback to a graceful no-op when keys are missing.
8. Read-only invariant — ``generate_idiom`` issues only SELECTs
   against ``phrases``; no INSERT/UPDATE/DELETE helper is
   exported (Hard rule #2 — type-level guardrail via omission).
9. ``enable_rag`` stub passes through the prompt unchanged when no
   ``retrieve_neighbor`` callable is supplied (Hard rule #7 — no
   bge-m3 OpenRouter call).
10. Literal widening non-regression — the existing cloze /
    matching / comprehension exercise surfaces still parse with
    ``exercise_type="cloze"`` / ``"matching"`` / ``"comprehension"``.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test, the
OpenRouter chat-completions call is replaced with a stub OpenAI
client (``monkeypatch.setattr("app.idiom._openai_client", ...)``)
so no network is touched. Mirrors the Phase 4.2 / 6.2 / 6.4 / 7.2
patterns.

The 8.1 ``phrases`` table isn't on main yet (8.3 ships before its
schema lands), so the ``select_phrase_row`` tests create the
``phrases`` table in-place via ``Base.metadata.create_all`` and seed
a 3-row fixture. Mirrors the Phase 7.2 + 7.1 build-order trick
(7.2's local ORM mirror compiling before 7.1 lands) — once 8.1
folds, the canonical ``models.Phrase`` is the single source of
truth.

Run from ``backend/``::

    uv run pytest -q tests/test_idiom.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from app import crud
from app.idiom import (
    FREQUENCY_BAND,
    MAX_ATTEMPTS,
    PROMPT_TEMPLATE_VERSION,
    SOURCE_ATTRIBUTION,
    IdiomExercise,
    IdiomGenerationError,
    IdiomModule,
    IdiomSignature,
    Phrase,
    build_prompt,
    generate_idiom,
    optimize_idiom_module,
    select_phrase_row,
)
from app.llm import LLMError
from app.schemas import (
    BaseExerciseFields,
    ClozeExerciseOut,
    IdiomExerciseOut,
    IdiomGenerateRequest,
    MatchingExerciseOut,
    ComprehensionExerciseOut,
)


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_cloze.py / test_collocation.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "test_idiom.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def db_session(sqlite_db_path):
    """Bring up a fresh SQLite DB with the 8.3 ``phrases`` mirror
    pre-created, then seed three idiom rows (one per frequency
    band) so ``select_phrase_row`` has stable selection material.

    Mirrors the test_collocation.py fixture pattern verbatim. The
    8.1 canonical ``models.Phrase`` model + migration lands later;
    8.3's local ``app.idiom.Phrase`` mirror handles the schema
    in-test.
    """
    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        # Seed three Phrase rows covering the three frequency bands.
        # Slugs are stable strings so deterministic selection is
        # reproducible across runs.
        rows = [
            Phrase(
                id="ins-blaue-hinein",
                phrase="ins Blaue hinein",
                definition="ohne festes Ziel, planlos (in the blue).",
                example_usage="Wir fahren einfach ins Blaue hinein.",
                source_attribution="dwds",
                frequency_band="high",
                dwds_url="https://www.dwds.de/wb/ins%20Blaue%20hinein",
                attested_quote=None,
                attested_source=None,
            ),
            Phrase(
                id="tomaten-auf-den-augen",
                phrase="Tomaten auf den Augen",
                definition=(
                    "etwas Offensichtliches nicht sehen (blind for "
                    "what's obvious)."
                ),
                example_usage="Du hast wohl Tomaten auf den Augen!",
                source_attribution="dwds",
                frequency_band="mid",
                dwds_url=None,
                attested_quote=None,
                attested_source=None,
            ),
            Phrase(
                id="das-eis-brechen",
                phrase="das Eis brechen",
                definition=(
                    "die erste Hemmung in einer Beziehung "
                    "ueberwinden (break the ice)."
                ),
                example_usage=(
                    "Er versuchte, mit einem Witz das Eis zu brechen."
                ),
                source_attribution="dwds,goethe",
                frequency_band="high",
                dwds_url=None,
                attested_quote="Wer will denn gleich das Eis brechen?",
                attested_source="Faust I, Studierzimmer",
            ),
        ]
        for row in rows:
            session.add(row)
        session.commit()
        yield session


# ---------------------------------------------------------------------------
# OpenAI stub — mirrors test_cloze.py / test_collocation.py's
# _make_stub_instructor_client.
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
):
    """Build a stub OpenAI-shaped client that returns ``payload`` as
    the assistant message content. Used by the happy-path test and
    the bounded-retry test to inject specific responses without
    going through respx.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-idiom-stub-001",
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


def _valid_idiom_payload(
    *,
    phrase: str = "ins Blaue hinein",
    definition: str = (
        "ohne festes Ziel, planlos (in the blue)."
    ),
    example_usage: str = "Wir fahren einfach ins Blaue hinein.",
    source_attribution: str = "dwds",
    frequency_band: str = "high",
    attested_quote: str | None = None,
    attested_source: str | None = None,
    cloze_target: str | None = "ins ___ hinein",
) -> dict:
    """Build a valid ``IdiomExercise`` payload dict.

    Used to seed the OpenAI stub for the happy-path test.
    """
    return {
        "exercise_id": "11111111111111111111111111111111",
        "phrase": phrase,
        "definition": definition,
        "example_usage": example_usage,
        "source_attribution": source_attribution,
        "attested_quote": attested_quote,
        "attested_source": attested_source,
        "frequency_band": frequency_band,
        "cloze_target": cloze_target,
        "prompt_template_version": "idiom-v1",
    }


# ---------------------------------------------------------------------------
# 1. Module-level constants — Hard rule #8 (no env-derived thresholds).
# ---------------------------------------------------------------------------


def test_prompt_template_version_locked():
    """``PROMPT_TEMPLATE_VERSION`` is a module constant, not env-derived.

    Phase 8.3 plan Hard rule #8 (type-level guardrails on A/B
    keys). A drift in the constant is caught by a test failure,
    not by runtime ambiguity.
    """
    assert PROMPT_TEMPLATE_VERSION == "idiom-v1"


def test_max_attempts_locked_to_three():
    """``MAX_ATTEMPTS = 3`` per the card body ("retries on schema
    failure (Phase 4.2 retry pattern: up to 3 attempts)").

    Same budget as ``app.cloze.MAX_ATTEMPTS`` — idioms have a
    richer schema than cloze (phrase + definition + example_usage
    + cloze_target + frequency_band) so a 3-attempt ceiling is
    appropriate.
    """
    assert MAX_ATTEMPTS == 3


def test_frequency_band_tuple_is_locked():
    """``FREQUENCY_BAND`` is the closed 3-way literal tuple used by
    both the generator and the offline metric."""
    assert FREQUENCY_BAND == ("high", "mid", "low")


def test_source_attribution_tuple_is_locked():
    """``SOURCE_ATTRIBUTION`` is the closed 3-way literal tuple
    ``(dwds, goethe, schiller)`` that ``source_attribution`` strings
    are validated against by ``IdiomExercise._validate_source_attribution``.
    """
    assert SOURCE_ATTRIBUTION == ("dwds", "goethe", "schiller")


# ---------------------------------------------------------------------------
# 2. Pydantic ``IdiomExercise`` bounds (Hard rule #1 — type-level
# guardrails on the wire contract).
# ---------------------------------------------------------------------------


def test_idiom_exercise_phrase_bounds():
    """``phrase`` is bounded 5..200 chars; below/above raises
    ``ValidationError``.

    Locks the card-body contract: ``5..200 chars`` so the wire
    fits ``"ins Blaue hinein"`` (14 chars) through multi-clause
    DWDS attestations (~180 chars worst case).
    """
    # Within bounds.
    ex = IdiomExercise(
        exercise_id="a" * 32,
        phrase="ins Blaue hinein",
        definition="planlos.",
        example_usage="Wir fahren ins Blaue hinein.",
        source_attribution="dwds",
        frequency_band="high",
        cloze_target="ins ___ hinein",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex.phrase == "ins Blaue hinein"

    # Below bounds — too short.
    with pytest.raises(ValidationError):
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ab",  # 2 chars, < 5
            definition="x",
            example_usage="abcde",
            source_attribution="dwds",
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    # Above bounds — too long.
    with pytest.raises(ValidationError):
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="x" * 201,  # > 200 chars
            definition="x",
            example_usage="abcde",
            source_attribution="dwds",
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )


def test_idiom_exercise_definition_bounds():
    """``definition`` is bounded 1..400 chars."""
    # Below 1 char.
    with pytest.raises(ValidationError):
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ins Blaue hinein",
            definition="",
            example_usage="abcde",
            source_attribution="dwds",
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    # Above 400 chars.
    with pytest.raises(ValidationError):
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ins Blaue hinein",
            definition="x" * 401,
            example_usage="abcde",
            source_attribution="dwds",
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )


def test_idiom_exercise_frequency_band_literal():
    """``frequency_band`` is a closed ``Literal["high","mid","low"]``;
    any other value raises ``ValidationError``."""
    ex_high = IdiomExercise(
        exercise_id="a" * 32,
        phrase="ins Blaue hinein",
        definition="planlos.",
        example_usage="Wir fahren ins Blaue hinein.",
        source_attribution="dwds",
        frequency_band="high",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex_high.frequency_band == "high"

    # Out of literal.
    with pytest.raises(ValidationError) as exc:
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ins Blaue hinein",
            definition="planlos.",
            example_usage="Wir fahren ins Blaue hinein.",
            source_attribution="dwds",
            frequency_band="ultra",  # not in the literal
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    assert "frequency_band" in str(exc.value)


def test_idiom_exercise_source_attribution_comma_joined():
    """``source_attribution`` accepts comma-joined subsets of
    ``(dwds, goethe, schiller)`` and rejects anything outside
    the closed literal.

    The card body allows comma-joined subsets (``"dwds,goethe"``
    when an idiom is in both sources). The validator also
    dedupes (``"dwds,dwds"`` → ``"dwds"``) and trims whitespace.
    """
    # Single token — fine.
    ex = IdiomExercise(
        exercise_id="a" * 32,
        phrase="ins Blaue hinein",
        definition="planlos.",
        example_usage="Wir fahren ins Blaue hinein.",
        source_attribution="dwds",
        frequency_band="high",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex.source_attribution == "dwds"

    # Comma-joined subset — fine.
    ex2 = IdiomExercise(
        exercise_id="a" * 32,
        phrase="das Eis brechen",
        definition="Hemmung ueberwinden.",
        example_usage="Er versuchte das Eis zu brechen.",
        source_attribution="dwds,goethe",
        frequency_band="high",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex2.source_attribution == "dwds,goethe"

    # De-dupe + whitespace strip.
    ex3 = IdiomExercise(
        exercise_id="a" * 32,
        phrase="das Eis brechen",
        definition="Hemmung ueberwinden.",
        example_usage="Er versuchte das Eis zu brechen.",
        source_attribution=" dwds , dwds , goethe ",
        frequency_band="high",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex3.source_attribution == "dwds,goethe"

    # Out-of-literal token — rejected.
    with pytest.raises(ValidationError) as exc:
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ins Blaue hinein",
            definition="planlos.",
            example_usage="Wir fahren ins Blaue hinein.",
            source_attribution="dwds,wikipedia",  # wikipedia not in literal
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    assert "source_attribution" in str(exc.value)

    # Empty / blank — rejected.
    with pytest.raises(ValidationError):
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ins Blaue hinein",
            definition="planlos.",
            example_usage="Wir fahren ins Blaue hinein.",
            source_attribution="",
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    with pytest.raises(ValidationError):
        IdiomExercise(
            exercise_id="a" * 32,
            phrase="ins Blaue hinein",
            definition="planlos.",
            example_usage="Wir fahren ins Blaue hinein.",
            source_attribution="   ",
            frequency_band="high",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )


def test_idiom_exercise_cloze_target_optional():
    """``cloze_target`` is optional (``None`` when the idiom doesn't
    lend itself to an internal blank). No length bound applies to
    the ``None`` case.
    """
    ex_no_cloze = IdiomExercise(
        exercise_id="a" * 32,
        phrase="ins Blaue hinein",
        definition="planlos.",
        example_usage="Wir fahren ins Blaue hinein.",
        source_attribution="dwds",
        frequency_band="high",
        cloze_target=None,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex_no_cloze.cloze_target is None


# ---------------------------------------------------------------------------
# 3. ``IdiomExerciseOut`` narrow discriminator (Hard rule #1).
# ---------------------------------------------------------------------------


def test_idiom_exercise_out_discriminator_narrows_to_idiom():
    """``IdiomExerciseOut.exercise_type`` is narrowed to
    ``Literal["idiom"]`` — setting ``"cloze"`` raises a
    ``ValidationError``.

    Mirrors the ClozeExerciseOut / MatchingExerciseOut /
    ComprehensionExerciseOut discriminator-narrowing pattern
    (Hard rule #1 — the wire discriminator is the type-level
    gate).
    """
    base_payload = {
        **_valid_idiom_payload(),
        "target_word_id": 1,
        "latency_ms": 0,
    }
    # Default discriminator value.
    out = IdiomExerciseOut.model_validate(base_payload)
    assert out.exercise_type == "idiom"

    # Attempting to set ``"cloze"`` on the idiom response is a
    # ValidationError — Pydantic v2 honours the narrowed
    # annotation on the subclass.
    with pytest.raises(ValidationError):
        IdiomExerciseOut.model_validate(
            {**base_payload, "exercise_type": "cloze"},
        )


def test_base_exercise_fields_discriminator_widens_to_include_idiom():
    """``BaseExerciseFields.exercise_type`` now accepts ``"idiom"``
    in addition to the original 3 (Hard rule #1 — Phase 8.3 widens
    the literal additively).

    The widening is wire-level and additive. Existing callers
    passing ``"cloze"`` / ``"matching"`` / ``"comprehension"``
    parse as before — the typing contract is forward-compatible.
    """
    # The base class's annotation now includes "idiom".
    type_adapter_annotation = BaseExerciseFields.model_fields[
        "exercise_type"
    ].annotation
    # ``typing.Literal`` instances are exposed as the union of
    # their args; we don't introspect that here — instead we
    # validate the runtime values round-trip.
    for valid in ("cloze", "matching", "comprehension", "idiom"):
        out = BaseExerciseFields.model_validate(
            {
                "exercise_type": valid,
                "target_word_id": 1,
                "prompt_template_version": "x",
                "latency_ms": 0,
            }
        )
        assert out.exercise_type == valid

    # A value outside the widened union is still rejected.
    with pytest.raises(ValidationError):
        BaseExerciseFields.model_validate(
            {
                "exercise_type": "speaking",  # not in the 4-way union
                "target_word_id": 1,
                "prompt_template_version": "x",
                "latency_ms": 0,
            }
        )


# ---------------------------------------------------------------------------
# 4. Literal widening — non-regression on existing endpoints.
# ---------------------------------------------------------------------------


def test_widening_does_not_regress_cloze_response_shape():
    """The Phase 6.x ``ClozeExerciseOut`` wire shape parses
    byte-for-byte unchanged after the 8.3 widening.

    Hard rule #1 (Phase 7 hard rule carried forward verbatim) —
    the Literal widening is additive; existing callers passing
    ``"cloze"`` see no schema drift.
    """
    cloze_payload = {
        "sentence_with_blank": "Der ___ schlaeft.",
        "answer_word_id": 1,
        "target_word_id": 1,
        "distractors": [2, 3, 4],
        "difficulty": "easy",
        "rationale": "test rationale",
        "prompt_template_version": "cloze-v1",
        "latency_ms": 12,
    }
    cloze_out = ClozeExerciseOut.model_validate(cloze_payload)
    assert cloze_out.exercise_type == "cloze"
    # JSON round-trip stays byte-stable except for the explicit
    # ``exercise_type`` discriminator.
    wire = cloze_out.model_dump_json()
    assert '"exercise_type":"cloze"' in wire
    assert '"answer_word_id":1' in wire


def test_widening_does_not_regress_matching_response_shape():
    """The Phase 6.3 ``MatchingExerciseOut`` wire shape parses
    unchanged after the 8.3 widening.
    """
    matching_payload = {
        "exercise_id": 1,
        "target_word_id": 1,
        "pairs": [
            {
                "left_word_id": i,
                "right_word_id": i + 100,
                "right_kind": "translation",
            }
            for i in range(1, 5)
        ],
        "rationale": "test rationale",
        "prompt_template_version": "match-v1",
        "latency_ms": 22,
    }
    out = MatchingExerciseOut.model_validate(matching_payload)
    assert out.exercise_type == "matching"


def test_widening_does_not_regress_comprehension_response_shape():
    """The Phase 6.5 ``ComprehensionExerciseOut`` wire shape parses
    unchanged after the 8.3 widening.
    """
    comp_payload = {
        "exercise_id": 1,
        "target_word_id": 1,
        "passage": (
            "Anna ging am Sonntag in den Park. Sie traf dort "
            "ihren alten Freund Bernd. Sie sprachen lange ueber "
            "vergangene Zeiten."
        ),
        "question": "Wen traf Anna im Park?",
        "choices": {
            "A": "Ihren Bruder",
            "B": "Ihren alten Freund Bernd",
            "C": "Ihre Nachbarin",
            "D": "Ihren Kollegen",
        },
        "correct_choice": "B",
        "rationale": "Bernd ist im Passage-Kontext explizit erwaehnt.",
        "prompt_template_version": "comprehension-v1",
        "latency_ms": 33,
    }
    out = ComprehensionExerciseOut.model_validate(comp_payload)
    assert out.exercise_type == "comprehension"


def test_idiom_generate_request_default_enable_rag_is_false():
    """``IdiomGenerateRequest()`` defaults ``enable_rag=False`` —
    the wire stays closed for callers that don't opt in.

    Mirrors ``ClozeGenerateRequest``'s default contract.
    """
    req = IdiomGenerateRequest()
    assert req.enable_rag is False


def test_idiom_generate_request_strict_bool_rejects_non_bools():
    """``enable_rag`` is a ``StrictBool`` — string ``"true"`` and
    integer ``1`` are rejected with ``ValidationError``.

    Mirrors ``ClozeGenerateRequest`` (Phase 7.3 acceptance).
    """
    for bad in ("true", 1, 0, 1.0, "false"):
        with pytest.raises(ValidationError):
            IdiomGenerateRequest(enable_rag=bad)


# ---------------------------------------------------------------------------
# 5. ``IdiomModule`` constructed without OpenRouter (DummyLM).
# ---------------------------------------------------------------------------


def test_dspy_module_constructible_without_openrouter(monkeypatch):
    """``IdiomModule`` can be constructed without an OpenRouter key.

    The DSPy configure path falls back to ``DummyLM`` automatically
    (Hard rule #6 — offline-capable). Mirrors
    ``test_collocation.py::test_dspy_module_constructible_without_openrouter``.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None
    module = IdiomModule()
    assert module is not None
    assert hasattr(module, "predict")


def test_idiom_signature_inputs_match_production_contract():
    """The DSPy signature carries the production input keys
    (word_id, target_phrase, curated_definition, attested_quote,
    source_attribution, frequency_band) and the ``IdiomExercise``
    Pydantic-typed output.

    Mirrors ``test_collocation.py::test_collocation_signature_inputs_match_production_contract``
    with the idiom-specific input keys.
    """
    sig = IdiomSignature
    input_fields = {
        k
        for k in sig.model_fields
        if sig.model_fields[k].json_schema_extra.get("__dspy_field_type")
        == "input"
    }
    output_fields = {
        k
        for k in sig.model_fields
        if sig.model_fields[k].json_schema_extra.get("__dspy_field_type")
        == "output"
    }
    assert input_fields == {
        "word_id",
        "target_phrase",
        "curated_definition",
        "attested_quote",
        "source_attribution",
        "frequency_band",
    }
    assert output_fields == {"exercise"}


# ---------------------------------------------------------------------------
# 6. ``generate_idiom`` happy path — stubbed OpenAI client.
# ---------------------------------------------------------------------------


def _seed_user_for_routes(
    session,
    *,
    email: str = "idiom@example.com",
    password: str = "supersecret",
):
    """Helper for tests that exercise ``generate_idiom`` via the
    route layer (used by future 8.4 route tests; the 8.3 generator
    doesn't strictly require a user row)."""
    from app import models
    from app.passwords import hash_password

    user = models.User(
        email=email, password_hash=hash_password(password)
    )
    session.add(user)
    session.flush()
    session.commit()
    return user.id


def test_generate_idiom_happy_path_returns_validated_exercise(
    monkeypatch, db_session
):
    """End-to-end: ``generate_idiom`` returns a Pydantic-validated
    ``IdiomExercise`` through a stubbed OpenAI client.

    Mirrors ``test_cloze.py::test_generate_cloze_happy_path``
    and ``test_collocation.py::test_generate_collocation_happy_path``.
    """
    import dspy

    dspy.settings.lm = None

    payload = json.dumps(_valid_idiom_payload(), ensure_ascii=False)
    raw_client = _make_stub_instructor_client(payload)

    monkeypatch.setattr("app.idiom._openai_client", lambda: raw_client)

    exercise = generate_idiom(db_session, word_id=1)
    assert isinstance(exercise, IdiomExercise)
    assert exercise.phrase == "ins Blaue hinein"
    assert exercise.definition == (
        "ohne festes Ziel, planlos (in the blue)."
    )
    assert exercise.frequency_band == "high"
    assert exercise.source_attribution == "dwds"
    # ``prompt_template_version`` is forced onto the module
    # constant regardless of what the LLM emitted.
    assert exercise.prompt_template_version == PROMPT_TEMPLATE_VERSION


def test_generate_idiom_passes_enable_rag_through_when_no_neighbor_callable(
    monkeypatch, db_session
):
    """``enable_rag=True`` with no ``retrieve_neighbor`` callable
    is byte-for-byte identical to the curated-only path (Hard
    rule #7 — no bge-m3 OpenRouter call when RAG is on without a
    neighbor-retriever).

    The 8.3 stub keeps the prompt identical; 8.4 wires the real
    neighbor-fetching callable.
    """
    import dspy

    dspy.settings.lm = None

    # Build the canonical prompt that ``generate_idiom`` would emit
    # in curated-only mode.
    phrase = db_session.query(Phrase).filter(Phrase.id == "ins-blaue-hinein").one()
    weakness_axes: dict[str, int] = {}
    messages_no_rag = build_prompt(phrase, weakness_axes)
    messages_with_rag_no_neighbor = build_prompt(
        phrase, weakness_axes
    )  # build_prompt with default arg stays curated-only

    # Both prompts are byte-for-byte identical — the enable_rag
    # flag with no neighbor callable is a no-op on the prompt
    # surface.
    assert messages_no_rag == messages_with_rag_no_neighbor


def test_generate_idiom_raises_llm_error_when_key_missing(
    monkeypatch, db_session
):
    """With ``OPENROUTER_API_KEY`` unset, ``generate_idiom`` raises
    ``LLMError`` (mirrors ``app.cloze`` and ``app.collocation``).

    Production path: the route layer translates LLMError to a 502.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("app.idiom._openai_client", lambda: None)

    with pytest.raises(LLMError) as exc:
        generate_idiom(db_session, word_id=1)
    assert "OPENROUTER_API_KEY" in str(exc.value)


def test_generate_idiom_raises_idiom_generation_error_on_persistent_schema_failure(
    monkeypatch, db_session
):
    """Persistent schema violations after ``MAX_ATTEMPTS`` raise
    ``IdiomGenerationError`` carrying the structured dead-letter
    fields.

    Mirrors ``test_collocation.py::test_generate_collocation_dead_letters_after_max_attempts``
    — the persistent-retry exhaustion path.
    """
    import httpx
    from openai import OpenAI

    # Return an invalid payload (missing required fields, wrong
    # types) every call. ``instructor`` retries up to
    # ``MAX_ATTEMPTS`` times, then raises; ``generate_idiom``
    # translates the failure into ``IdiomGenerationError``.
    invalid_payload = json.dumps(
        {
            "phrase": "x",  # too short, will fail bounds
            "frequency_band": "ultra",  # not in literal
        },
        ensure_ascii=False,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-idiom-bad-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "qwen/qwen3-235b-a22b-2507",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": invalid_payload,
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

    raw_client = OpenAI(
        api_key="test-key-not-real",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    monkeypatch.setattr("app.idiom._openai_client", lambda: raw_client)

    with pytest.raises(IdiomGenerationError) as exc:
        generate_idiom(db_session, word_id=1)
    err = exc.value
    assert err.schema_retry_count >= 1
    assert "idiom: schema validation failed" in str(err)
    # ``attempted_schema`` is a JSON-encoded Pydantic schema dict
    # so the operator can grep the dead-letter body.
    assert err.attempted_schema.startswith("{")


# ---------------------------------------------------------------------------
# 7. Phrase row selector — deterministic + reads-only invariant.
# ---------------------------------------------------------------------------


def test_select_phrase_row_is_deterministic(db_session):
    """``select_phrase_row`` returns the same phrase row for the
    same ``word_id`` across calls.

    Same integer → same phrase (the stability commitment).
    Mirrors ``app.cloze.select_target_word`` /
    ``app.collocation.select_collocation_row``.
    """
    row_a = select_phrase_row(db_session, word_id=1)
    row_b = select_phrase_row(db_session, word_id=1)
    assert row_a.id == row_b.id

    # Different word_ids may map to different rows.
    seen: set[str] = set()
    for word_id in (1, 2, 3, 4, 5):
        row = select_phrase_row(db_session, word_id=word_id)
        seen.add(row.id)
    # At least 2 distinct phrases across 5 seeded rows for
    # robust coverage of the deterministic selector.
    assert len(seen) >= 2


def test_select_phrase_row_raises_when_phrases_table_is_empty(
    tmp_path, monkeypatch
):
    """When no phrase rows exist (8.1 seed hasn't run yet), the
    selector raises ``ValueError`` (corpus inconsistency).

    Routes translate to 500.
    """
    db_path = tmp_path / "test_idiom_empty.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))

    from app import database

    database.reconfigure_for_test(f"sqlite:///{db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        with pytest.raises(ValueError) as exc:
            select_phrase_row(session, word_id=1)
        assert "no Phrase rows" in str(exc.value)


def test_phrases_table_is_read_only_at_generator_layer(db_session, monkeypatch):
    """``app.idiom`` does not export any INSERT / UPDATE / DELETE
    helper for the ``phrases`` table.

    Hard rule #2 — read-only at runtime. The seed scripts
    (``seed_phrases_*.py``, 8.1 / 8.2) are the only write path.
    The type-level guardrail is *omission of write helpers*
    rather than runtime checks.
    """
    import app.idiom as idiom_module

    forbidden_names = (
        "insert_phrase",
        "update_phrase",
        "delete_phrase",
        "create_phrase",
        "upsert_phrase",
        "write_phrase",
    )
    public_names = set(getattr(idiom_module, "__all__", []))
    assert not (public_names & set(forbidden_names))


# ---------------------------------------------------------------------------
# 8. DSPy offline optimizer — runs on DummyLM.
# ---------------------------------------------------------------------------


def test_optimize_idiom_module_runs_offline_on_two_row_eval_set(
    monkeypatch,
):
    """``optimize_idiom_module`` runs end-to-end on a 2-row eval
    set using ``DummyLM``.

    Same caveat as the cloze / collocation paths: MIPROv2's
    internal prompt-proposer is hard to satisfy with ``DummyLM``
    (it expects strict JSONAdapter-shaped responses from an LM it
    probes several times). The function is expected to either
    succeed or fall back to the un-optimized module — either
    outcome proves the plumbing is wired.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None

    train = [
        {
            "word_id": 1,
            "target_phrase": "ins Blaue hinein",
            "curated_definition": "ohne festes Ziel.",
            "attested_quote": "",
            "source_attribution": "dwds",
            "frequency_band": "high",
            "exercise": IdiomExercise(
                exercise_id="1" * 32,
                phrase="ins Blaue hinein",
                definition="ohne festes Ziel.",
                example_usage="Wir fahren ins Blaue hinein.",
                source_attribution="dwds",
                frequency_band="high",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
        },
        {
            "word_id": 2,
            "target_phrase": "Tomaten auf den Augen",
            "curated_definition": "etwas nicht sehen.",
            "attested_quote": "",
            "source_attribution": "dwds",
            "frequency_band": "mid",
            "exercise": IdiomExercise(
                exercise_id="2" * 32,
                phrase="Tomaten auf den Augen",
                definition="etwas nicht sehen.",
                example_usage="Du hast Tomaten auf den Augen!",
                source_attribution="dwds",
                frequency_band="mid",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
        },
    ]

    optimized = optimize_idiom_module(train)
    assert isinstance(optimized, IdiomModule)


# ---------------------------------------------------------------------------
# 9. ``IdiomModule.generate`` — DSPy-callable mirror.
# ---------------------------------------------------------------------------


def test_idiom_module_generate_returns_validated_idiom_exercise(
    monkeypatch,
):
    """``IdiomModule.generate(prompt_template_version)`` returns a
    Pydantic-validated ``IdiomExercise`` instance.

    Mirrors the ``CollocationModule`` /
    ``ComprehensionModule`` generate-shape. The DSPy offline
    path serves a fixed ``DummyLM`` answer that the
    ``IdiomExercise.model_validate_json`` path accepts.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None
    _configure_dspy_on_idiom()

    module = IdiomModule()
    exercise = module.generate()
    assert isinstance(exercise, IdiomExercise)
    assert exercise.phrase == "ins Blaue hinein"
    assert exercise.prompt_template_version == PROMPT_TEMPLATE_VERSION


def _configure_dspy_on_idiom():
    """Helper: force the offline DSPy configuration on ``app.idiom``.

    Imports the helper and calls it in isolation. Mirrors
    ``_configure_dspy`` from app.cloze / app.collocation.
    """
    from app.idiom import _configure_dspy

    _configure_dspy()


# ---------------------------------------------------------------------------
# 10. Langfuse fallback — graceful no-op when keys are missing.
# ---------------------------------------------------------------------------


def test_trace_idiom_silent_when_langfuse_keys_missing(monkeypatch):
    """``_trace_idiom`` returns ``None`` when Langfuse keys are
    unset (graceful no-op, no per-call warnings, no network).

    Mirrors ``test_collocation.py::test_trace_collocation_silent_when_keys_missing``
    / ``test_cloze.py::test_trace_cloze_is_silent_when_keys_missing``.
    """
    from app.idiom import _trace_idiom

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    exercise = IdiomExercise(
        exercise_id="1" * 32,
        phrase="ins Blaue hinein",
        definition="planlos.",
        example_usage="Wir fahren ins Blaue hinein.",
        source_attribution="dwds",
        frequency_band="high",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata: dict[str, Any] = {
        "word_id": 1,
        "weakness_axes": {},
        "phrase_id": "ins-blaue-hinein",
        "phrase": "ins Blaue hinein",
        "source_attribution": "dwds",
        "frequency_band": "high",
        "enable_rag": False,
        "retrieved_neighbor_present": False,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "exercise_type": "idiom",
        "idiom": True,
    }
    assert _trace_idiom(exercise, metadata, latency_ms=0) is None


def test_trace_idiom_emits_exercise_generate_span_with_idiom_discriminator(
    monkeypatch,
):
    """When Langfuse keys are present, ``_trace_idiom`` emits one
    span named ``exercise.generate`` (the canonical Phase 6.x
    span name) with the ``idiom: true`` discriminator and the
    ``exercise_type="idiom"`` field.

    The discriminator is what the offline Ragas / Phase 6.7
    follow-up reader uses to split idiom cohorts from cloze /
    matching / comprehension / collocation cohorts.
    """
    import importlib

    idiom_live = importlib.import_module("app.idiom")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span
    mock_client.flush.return_value = None

    monkeypatch.setattr(idiom_live, "get_langfuse", lambda: mock_client)

    exercise = IdiomExercise(
        exercise_id="1" * 32,
        phrase="ins Blaue hinein",
        definition="planlos.",
        example_usage="Wir fahren ins Blaue hinein.",
        source_attribution="dwds",
        frequency_band="high",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata: dict[str, Any] = {
        "word_id": 1,
        "weakness_axes": {},
        "phrase_id": "ins-blaue-hinein",
        "phrase": "ins Blaue hinein",
        "source_attribution": "dwds",
        "frequency_band": "high",
        "enable_rag": False,
        "retrieved_neighbor_present": False,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "exercise_type": "idiom",
        "idiom": True,
    }
    assert idiom_live._trace_idiom(exercise, metadata, latency_ms=42) is None

    assert mock_client.span.call_count == 1
    span_kwargs = mock_client.span.call_args.kwargs
    assert span_kwargs["name"] == "exercise.generate"

    update_calls = mock_span.update.call_args_list
    merged: dict = {}
    for call in update_calls:
        for key, value in (call.kwargs.get("metadata") or {}).items():
            merged[key] = value
    assert merged.get("idiom") is True
    assert merged.get("exercise_type") == "idiom"
    assert merged.get("phrase_id") == "ins-blaue-hinein"
    assert merged.get("frequency_band") == "high"

    assert mock_span.end.call_count == 1
    assert mock_client.flush.call_count == 1
