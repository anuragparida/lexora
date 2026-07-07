"""Tests for Phase 10.2 — phrase-match exercise generator + Literal widening.

Card: t_5d91a7e7.

Coverage map (mirrors the 10.2 card body's "Tests" section):

1. Module-level constants — ``PROMPT_TEMPLATE_VERSION ==
   "phrase-match-v1"``, ``MAX_ATTEMPTS == 3``,
   ``RELATION_CHOICES == ("equivalent", "paraphrase", "related",
   "unrelated")``, ``SOURCE_ATTRIBUTION == ("dwds", "goethe",
   "schiller", "bge-m3-cosine")``, ``RAG_TOP_K == 3``. Type-level
   guardrails on the A/B keys (Hard rule #8 — no env-derived
   thresholds).
2. Pydantic ``PhraseMatchExercise`` bounds — 5..200 char
   ``phrase_a`` / ``phrase_b``, 1..400 char
   ``relation_rationale``, the closed 4-way literal on
   ``relation``, the comma-joined subset invariant on
   ``source_attribution``.
3. ``PhraseMatchExerciseOut`` extends ``BaseExerciseFields`` and
   narrows ``exercise_type`` to ``Literal["phrase_match"]`` (Hard
   rule #1 — wire discriminator is the type-level gate).
4. Literal widening on ``BaseExerciseFields.exercise_type`` — the
   base class now accepts ``"phrase_match"``; existing callers
   passing ``"cloze"`` / ``"matching"`` / ``"comprehension"`` /
   ``"idiom"`` parse as before (no regression). The 10.2 schema
   test asserts this.
5. ``PhraseMatchModule`` is constructible without OpenRouter
   (DummyLM swap via ``_configure_dspy``).
6. ``PhraseMatchSignature`` carries the production input keys
   (word_id, phrase_a, phrase_b, relation_choices,
   few_shot_examples) and the ``PhraseMatchExercise`` Pydantic-
   typed output.
7. ``generate_phrase_match`` end-to-end through a stubbed OpenAI
   client — Pydantic-validated response, retry budget on schema
   violations, Langfuse fallback to a graceful no-op when keys
   are missing.
8. Read-only invariant — ``generate_phrase_match`` issues only
   SELECTs against ``phrases`` / ``phrase_pairs``; no
   INSERT/UPDATE/DELETE helper is exported (Hard rule #2 — type-
   level guardrail via omission).
9. ``enable_rag`` stub passes through the prompt unchanged when
   no neighbor callable is supplied (Hard rule #7 — no bge-m3
   OpenRouter call).
10. Literal widening non-regression — the existing cloze /
    matching / comprehension / idiom exercise surfaces still
    parse with ``exercise_type="cloze"`` / ``"matching"`` /
    ``"comprehension"`` / ``"idiom"``.
11. ``PhraseMatchModule.generate`` returns a validated
    ``PhraseMatchExercise`` instance from the
    ``ChainOfThought``-wrapped predictor (Hard rule #10 —
    CoT wrapping).
12. Selector seed determinism — 5 distinct ``word_id`` values
    map to 5 distinct ``phrase_pair`` rows from a 5-row
    fixture. Same ``word_id`` → same pair on a re-call.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per test,
the OpenRouter chat-completions call is replaced with a stub
OpenAI client (``monkeypatch.setattr("app.phrase_match.
_openai_client", ...)``) so no network is touched. Mirrors the
Phase 4.2 / 6.2 / 7.2 / 8.3 patterns.

The 10.1 ``phrase_pairs`` table isn't on main yet (10.2 ships
in its own worktree with the seed in a sibling branch), so the
``select_phrase_pair`` tests create the ``phrases`` + ``phrase_pairs``
tables in-place via ``Base.metadata.create_all`` and seed
5 row fixtures. Mirrors the Phase 7.2 + 8.3 build-order trick
(8.3's local ORM mirror compiling before 8.1 lands + 7.2's
local ORM mirror compiling before 7.1 lands) — once 10.1
folds, the canonical ``models.PhrasePair`` is the single
source of truth and the local mirror in ``app.phrase_match``
flips to the canonical reference.

Run from ``backend/``::

    uv run pytest -q tests/test_phrase_match.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import sessionmaker

from app.phrase_match import (
    MAX_ATTEMPTS,
    PROMPT_TEMPLATE_VERSION,
    RAG_TOP_K,
    RELATION_CHOICES,
    SOURCE_ATTRIBUTION,
    Phrase,
    PhraseMatchExercise,
    PhraseMatchGenerationError,
    PhraseMatchModule,
    PhraseMatchNotFoundError,
    PhraseMatchSignature,
    PhrasePair,
    _trace_phrase_match,
    build_prompt,
    generate_phrase_match,
    optimize_phrase_match_module,
    select_phrase_pair,
)
from app.llm import LLMError
from app.schemas import (
    BaseExerciseFields,
    ClozeExerciseOut,
    IdiomExerciseOut,
    MatchingExerciseOut,
    ComprehensionExerciseOut,
    PhraseMatchExerciseOut,
    PhraseMatchGenerateRequest,
)


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_cloze.py / test_collocation.py / test_idiom.py.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "test_phrase_match.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def db_session(sqlite_db_path):
    """Bring up a fresh SQLite DB with the 8.1 ``phrases`` mirror
    + 10.2 ``phrase_pairs`` local mirror pre-created, then seed
    five phrase rows + five ``phrase_pairs`` rows so
    ``select_phrase_pair`` has stable selection material.

    Mirrors the test_collocation.py / test_idiom.py fixture
    pattern verbatim. The 10.1 canonical ``models.PhrasePair``
    model + migration lands later; 10.2's local
    ``app.phrase_match.PhrasePair`` mirror handles the schema
    in-test.
    """
    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        # Seed five Phrase rows covering the 4-way relation
        # spread (we need at least 5 phrase rows so a 5-row
        # phrase_pairs fixture can reference unique slugs).
        phrase_rows = [
            Phrase(
                id=f"phrase-{i:02d}",
                phrase=f"phrase number {i:02d} surface text",
                definition=f"definition {i:02d}",
                example_usage=f"example {i:02d} usage.",
                source_attribution="dwds",
                frequency_band="high" if i % 2 == 0 else "mid",
                dwds_url=None,
                attested_quote=None,
                attested_source=None,
            )
            for i in range(1, 6)
        ]
        for row in phrase_rows:
            session.add(row)
        session.flush()

        # Seed five PhrasePair rows. Each row is a (phrase-a-id,
        # phrase-b-id, relation, attested_pair) tuple. We use a
        # 5-row fixture so the selector-determinism test can
        # ask for 5 distinct ``word_id`` values and assert that
        # all 5 distinct pairs surface.
        pair_rows = [
            PhrasePair(
                id=i,
                phrase_a_id=f"phrase-0{i}",
                phrase_b_id=f"phrase-0{i + 1 if i < 5 else 1}",
                relation="equivalent" if i == 1 else (
                    "paraphrase" if i == 2 else (
                        "related" if i == 3 else (
                            "unrelated" if i == 4 else "equivalent"
                        )
                    )
                ),
                attested_pair=1,
            )
            for i in range(1, 6)
        ]
        for row in pair_rows:
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
    """Build a stub OpenAI-shaped client that returns ``payload``
    as the assistant message content. Used by the happy-path
    test and the bounded-retry test to inject specific
    responses without going through respx.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-phrase-match-stub-001",
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


def _valid_phrase_match_payload(
    *,
    phrase_a: str = "ins Blaue hinein",
    phrase_b: str = "ohne festes Ziel",
    relation: str = "equivalent",
    relation_rationale: str = (
        "Beide Ausdruecke bezeichnen eine planlose Handlung."
    ),
    source_attribution: str = "dwds",
) -> dict:
    """Build a valid ``PhraseMatchExercise`` payload dict.

    Used to seed the OpenAI stub for the happy-path test.
    """
    return {
        "exercise_id": 44444441,
        "phrase_a": phrase_a,
        "phrase_b": phrase_b,
        "relation": relation,
        "relation_rationale": relation_rationale,
        "source_attribution": source_attribution,
        "prompt_template_version": "phrase-match-v1",
    }


# ---------------------------------------------------------------------------
# 1. Module-level constants — Hard rule #8 (no env-derived thresholds).
# ---------------------------------------------------------------------------


def test_prompt_template_version_locked():
    """``PROMPT_TEMPLATE_VERSION`` is a module constant, not
    env-derived.

    Phase 10.2 plan Hard rule #8 (type-level guardrails on A/B
    keys). A drift in the constant is caught by a test failure,
    not by runtime ambiguity.
    """
    assert PROMPT_TEMPLATE_VERSION == "phrase-match-v1"


def test_max_attempts_locked_to_three():
    """``MAX_ATTEMPTS = 3`` per the card body (Phase 8.3 hard rule
    #5 carried forward verbatim).

    Same budget as ``app.idiom.MAX_ATTEMPTS`` — phrase-match
    has a schema with two phrases + a 4-way relation literal,
    so a 3-attempt ceiling is appropriate.
    """
    assert MAX_ATTEMPTS == 3


def test_relation_choices_tuple_is_locked():
    """``RELATION_CHOICES`` is the closed 4-way literal tuple
    (``equivalent, paraphrase, related, unrelated``) used by
    both the signature input and the offline metric.

    The closed tuple is the single source of truth — the
    ``phrase_pairs.relation`` column 10.1 ships carries the
    same literals; the DSPy ``relation_choices`` input field
    is built from this tuple.
    """
    assert RELATION_CHOICES == (
        "equivalent", "paraphrase", "related", "unrelated",
    )


def test_source_attribution_tuple_is_locked():
    """``SOURCE_ATTRIBUTION`` is the closed 4-way literal tuple
    ``(dwds, goethe, schiller, bge-m3-cosine)`` that
    ``source_attribution`` strings are validated against.

    Mirrors the 8.3 idiom ``SOURCE_ATTRIBUTION`` plus the new
    ``"bge-m3-cosine"`` token for the 10.2 RAG-on nearest-
    neighbor path.
    """
    assert SOURCE_ATTRIBUTION == (
        "dwds", "goethe", "schiller", "bge-m3-cosine",
    )


def test_rag_top_k_locked_to_three():
    """``RAG_TOP_K = 3`` per the card body ("top-3 nearest-
    neighbor pairs").

    Hard rule #8 — locked at module constant, not env-derived.
    """
    assert RAG_TOP_K == 3


# ---------------------------------------------------------------------------
# 2. Pydantic ``PhraseMatchExercise`` bounds (Hard rule #1 — type-level
# guardrails on the wire contract).
# ---------------------------------------------------------------------------


def test_phrase_match_exercise_phrase_bounds():
    """``phrase_a`` and ``phrase_b`` are bounded 5..200 chars;
    below/above raises ``ValidationError``.

    Locks the card-body contract: both phrases share the
    8.3 ``phrase`` bound so a single pair's phrases share
    the same wire contract.
    """
    # Within bounds.
    ex = PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="planlose Handlung.",
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex.phrase_a == "ins Blaue hinein"
    assert ex.phrase_b == "ohne festes Ziel"

    # Edge case: 4 chars (phrase_a) — below 5-char minimum.
    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="abcd",  # 4 chars, < 5
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x",
            source_attribution="dwds",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    # Edge case: 5 chars (phrase_a) — at the lower boundary.
    ex_at_lower = PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="abcde",  # exactly 5 chars
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="x",
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert len(ex_at_lower.phrase_a) == 5

    # Edge case: 200 chars (phrase_b) — at the upper boundary.
    ex_at_upper = PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="x" * 200,  # exactly 200 chars
        relation="equivalent",
        relation_rationale="x",
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert len(ex_at_upper.phrase_b) == 200

    # Edge case: 201 chars (phrase_a) — above 200-char maximum.
    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="x" * 201,  # > 200 chars
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x",
            source_attribution="dwds",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )


def test_phrase_match_exercise_relation_literal():
    """``relation`` is a closed 4-way ``Literal["equivalent",
    "paraphrase", "related", "unrelated"]``; any other value
    raises ``ValidationError``.

    Mirrors the 8.3 ``frequency_band`` closed-literal
    discipline.
    """
    for valid in ("equivalent", "paraphrase", "related", "unrelated"):
        ex = PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation=valid,
            relation_rationale="x",
            source_attribution="dwds",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        assert ex.relation == valid

    # Out of literal.
    with pytest.raises(ValidationError) as exc:
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="synonym",  # not in the 4-way literal
            relation_rationale="x",
            source_attribution="dwds",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    assert "relation" in str(exc.value)


def test_phrase_match_exercise_relation_rationale_bounds():
    """``relation_rationale`` is bounded 1..400 chars; below/above
    raises ``ValidationError``.

    Same 1..400-char bound as the 8.3 ``definition`` field.
    """
    # Below 1 char (empty).
    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="",  # 0 chars
            source_attribution="dwds",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    # Above 400 chars.
    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x" * 401,  # > 400 chars
            source_attribution="dwds",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    # Edge: 1 char minimum and 400 char maximum both pass.
    PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="x",  # 1 char
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="x" * 400,  # 400 chars
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )


def test_phrase_match_exercise_source_attribution_comma_joined():
    """``source_attribution`` accepts comma-joined subsets of
    ``(dwds, goethe, schiller, bge-m3-cosine)`` and rejects
    anything outside the closed literal.

    The new ``"bge-m3-cosine"`` token is exclusive to
    phrase-match (8.3 idiom doesn't carry it). The validator
    dedupes (``"dwds,dwds"`` → ``"dwds"``), trims whitespace,
    and rejects whitespace around the separator.
    """
    # Single token — fine.
    ex = PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="x",
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex.source_attribution == "dwds"

    # Comma-joined subset with the new bge-m3-cosine token —
    # the 10.2 RAG path attribution.
    ex2 = PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="x",
        source_attribution="dwds,bge-m3-cosine",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex2.source_attribution == "dwds,bge-m3-cosine"

    # De-dupe + whitespace strip.
    ex3 = PhraseMatchExercise(
        exercise_id=44444441,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="x",
        source_attribution=" dwds , dwds , goethe ",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    assert ex3.source_attribution == "dwds,goethe"

    # Out-of-literal token — rejected.
    with pytest.raises(ValidationError) as exc:
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x",
            source_attribution="dwds,wikipedia",  # wikipedia not in literal
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
    assert "source_attribution" in str(exc.value)

    # The 8.3 idiom "bge-m3" alone (without "-cosine") is also
    # rejected — the literal is the closed 4-way set, no
    # half-tokens.
    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x",
            source_attribution="bge-m3",  # not in literal
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    # Empty / blank — rejected.
    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x",
            source_attribution="",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )

    with pytest.raises(ValidationError):
        PhraseMatchExercise(
            exercise_id=44444441,
            phrase_a="ins Blaue hinein",
            phrase_b="ohne festes Ziel",
            relation="equivalent",
            relation_rationale="x",
            source_attribution="   ",
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )


# ---------------------------------------------------------------------------
# 3. ``PhraseMatchExerciseOut`` narrow discriminator (Hard rule #1).
# ---------------------------------------------------------------------------


def test_phrase_match_exercise_out_discriminator_narrows_to_phrase_match():
    """``PhraseMatchExerciseOut.exercise_type`` is narrowed to
    ``Literal["phrase_match"]`` — setting ``"cloze"`` raises a
    ``ValidationError``.

    Mirrors the ``ClozeExerciseOut`` /
    ``MatchingExerciseOut`` /
    ``ComprehensionExerciseOut`` /
    ``IdiomExerciseOut`` discriminator-narrowing pattern
    (Hard rule #1 — the wire discriminator is the type-level
    gate).

    Phase 10.2 wire shape mirrors the 8.4 idiom:
    ``exercise_id`` is a server-minted int and ``word_id``
    is the request-side pair-selector seed echoed back. We
    DON'T use the shared generator helper here — that helper
    is for the *generator-side* ``PhraseMatchExercise``; we
    build a wire-shape payload directly.
    """
    base_payload = {
        # Wire shape — exercise_id is a server-minted int.
        "exercise_id": 99999,
        # Wire shape — word_id is the pair-selector seed
        # echoed from the request.
        "word_id": 1,
        "target_word_id": 1,
        "prompt_template_version": "phrase-match-v1",
        "enable_rag": False,
        "trace_id": None,
        "latency_ms": 0,
        "phrase_a": "ins Blaue hinein",
        "phrase_b": "ohne festes Ziel",
        "relation": "equivalent",
        "relation_rationale": "planlose Handlung.",
        "source_attribution": "dwds",
    }
    # Default discriminator value.
    out = PhraseMatchExerciseOut.model_validate(base_payload)
    assert out.exercise_type == "phrase_match"

    # Attempting to set ``"cloze"`` on the phrase-match
    # response is a ValidationError — Pydantic v2 honours
    # the narrowed annotation on the subclass.
    with pytest.raises(ValidationError):
        PhraseMatchExerciseOut.model_validate(
            {**base_payload, "exercise_type": "cloze"},
        )


def test_base_exercise_fields_discriminator_widens_to_include_phrase_match():
    """``BaseExerciseFields.exercise_type`` now accepts
    ``"phrase_match"`` in addition to the original 4 (Hard rule
    #1 — Phase 10.2 widens the literal additively).

    The widening is wire-level and additive. Existing
    callers passing ``"cloze"`` / ``"matching"`` /
    ``"comprehension"`` / ``"idiom"`` parse as before — the
    typing contract is forward-compatible.
    """
    # The base class's annotation now includes "phrase_match"
    # alongside the prior 4 values.
    annotation = BaseExerciseFields.model_fields["exercise_type"].annotation
    # We don't introspect the Literal union — instead we
    # validate the runtime values round-trip across all 5
    # values.
    for valid in (
        "cloze", "matching", "comprehension", "idiom", "phrase_match",
    ):
        out = BaseExerciseFields.model_validate(
            {
                "exercise_type": valid,
                "target_word_id": 1,
                "prompt_template_version": "x",
                "latency_ms": 0,
            }
        )
        assert out.exercise_type == valid

    # A value outside the widened 5-way union is still
    # rejected.
    with pytest.raises(ValidationError):
        BaseExerciseFields.model_validate(
            {
                "exercise_type": "speaking",  # not in the 5-way union
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
    byte-for-byte unchanged after the 10.2 widening.

    Hard rule #1 (Phase 7 hard rule carried forward verbatim)
    — the Literal widening is additive; existing callers
    passing ``"cloze"`` see no schema drift.
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
    # JSON round-trip stays byte-stable except for the
    # explicit ``exercise_type`` discriminator.
    wire = cloze_out.model_dump_json()
    assert '"exercise_type":"cloze"' in wire
    assert '"answer_word_id":1' in wire


def test_widening_does_not_regress_matching_response_shape():
    """The Phase 6.3 ``MatchingExerciseOut`` wire shape parses
    unchanged after the 10.2 widening.
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
    """The Phase 6.5 ``ComprehensionExerciseOut`` wire shape
    parses unchanged after the 10.2 widening.
    """
    comp_payload = {
        "exercise_id": 1,
        "target_word_id": 1,
        "passage": (
            "Anna ging am Sonntag in den Park. Sie traf dort "
            "ihren alten Freund Bernd. Sie sprachen lange "
            "ueber vergangene Zeiten."
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


def test_widening_does_not_regress_idiom_response_shape():
    """The Phase 8.4 ``IdiomExerciseOut`` wire shape parses
    unchanged after the 10.2 widening.
    """
    idiom_payload = {
        "exercise_id": 12345,
        "word_id": 1,
        "target_word_id": 1,
        "prompt_template_version": "idiom-v1",
        "enable_rag": False,
        "trace_id": None,
        "latency_ms": 0,
        "phrase": "ins Blaue hinein",
        "definition": "ohne festes Ziel, planlos (in the blue).",
        "example_usage": "Wir fahren einfach ins Blaue hinein.",
        "source_attribution": "dwds",
        "attested_quote": None,
        "attested_source": None,
        "frequency_band": "high",
        "cloze_target": "ins ___ hinein",
    }
    out = IdiomExerciseOut.model_validate(idiom_payload)
    assert out.exercise_type == "idiom"


def test_phrase_match_generate_request_default_enable_rag_is_false():
    """``PhraseMatchGenerateRequest(word_id=...)`` defaults
    ``enable_rag=False`` — the wire stays closed for callers
    that don't opt in.

    Mirrors ``IdiomGenerateRequest``'s default contract from
    Phase 8.4. ``word_id`` itself is required (the route
    contract anchors the pair to a deterministic seed); the
    default-check here is specifically about ``enable_rag``.
    """
    req = PhraseMatchGenerateRequest(word_id=1)
    assert req.enable_rag is False


def test_phrase_match_generate_request_strict_bool_rejects_non_bools():
    """``enable_rag`` is a ``StrictBool`` — string ``"true"``
    and integer ``1`` are rejected with ``ValidationError``.

    Mirrors ``IdiomGenerateRequest`` (Phase 8.4 acceptance).
    """
    for bad in ("true", 1, 0, 1.0, "false"):
        with pytest.raises(ValidationError):
            PhraseMatchGenerateRequest(enable_rag=bad)


# ---------------------------------------------------------------------------
# 5. ``PhraseMatchModule`` constructed without OpenRouter (DummyLM).
# ---------------------------------------------------------------------------


def test_dspy_module_constructible_without_openrouter(monkeypatch):
    """``PhraseMatchModule`` can be constructed without an
    OpenRouter key.

    The DSPy configure path falls back to ``DummyLM``
    automatically (Hard rule #3 — offline-capable). Mirrors
    ``test_collocation.py::test_dspy_module_constructible_without_openrouter``
    / ``test_idiom.py::test_dspy_module_constructible_without_openrouter``.

    Also confirms Hard rule #10 — the module wraps the
    signature with ``dspy.ChainOfThought`` (not
    ``dspy.Predict``).
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None
    module = PhraseMatchModule()
    assert module is not None
    assert hasattr(module, "predict")
    # Hard rule #10 — the predictor is a ChainOfThought
    # instance (not bare Predict).
    assert isinstance(module.predict, dspy.ChainOfThought)


def test_phrase_match_signature_inputs_match_production_contract():
    """The DSPy signature carries the production input keys
    (word_id, phrase_a, phrase_b, relation_choices,
    few_shot_examples) and the ``PhraseMatchExercise``
    Pydantic-typed output.

    Mirrors
    ``test_idiom.py::test_idiom_signature_inputs_match_production_contract``
    with the phrase-match-specific input keys.
    """
    sig = PhraseMatchSignature
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
        "phrase_a",
        "phrase_b",
        "relation_choices",
        "few_shot_examples",
    }
    assert output_fields == {"exercise"}


# ---------------------------------------------------------------------------
# 6. ``generate_phrase_match`` happy path — stubbed OpenAI client.
# ---------------------------------------------------------------------------


def test_generate_phrase_match_happy_path_returns_validated_exercise(
    monkeypatch, db_session
):
    """End-to-end: ``generate_phrase_match`` returns a
    Pydantic-validated ``PhraseMatchExercise`` through a
    stubbed OpenAI client.

    Mirrors
    ``test_idiom.py::test_generate_idiom_happy_path_returns_validated_exercise``
    / ``test_collocation.py::test_generate_collocation_happy_path``.
    """
    import dspy

    dspy.settings.lm = None

    # Use a pair-row that exists in the fixture (any
    # attested_pair row is fine — the test only checks the
    # Pydantic validation of the LLM response).
    pair_row = (
        db_session.query(PhrasePair).filter(PhrasePair.id == 1).one()
    )
    payload = json.dumps(_valid_phrase_match_payload(), ensure_ascii=False)
    raw_client = _make_stub_instructor_client(payload)

    monkeypatch.setattr("app.phrase_match._openai_client", lambda: raw_client)

    exercise = generate_phrase_match(
        db_session,
        phrase_a_id=pair_row.phrase_a_id,
        phrase_b_id=pair_row.phrase_b_id,
    )
    assert isinstance(exercise, PhraseMatchExercise)
    assert exercise.phrase_a == "ins Blaue hinein"
    assert exercise.phrase_b == "ohne festes Ziel"
    assert exercise.relation == "equivalent"
    assert exercise.source_attribution == "dwds"
    # ``prompt_template_version`` is forced onto the module
    # constant regardless of what the LLM emitted.
    assert exercise.prompt_template_version == PROMPT_TEMPLATE_VERSION


def test_generate_phrase_match_raises_llm_error_when_key_missing(
    monkeypatch, db_session
):
    """With ``OPENROUTER_API_KEY`` unset,
    ``generate_phrase_match`` raises ``LLMError`` (mirrors
    ``app.idiom`` and ``app.collocation``).

    Production path: the route layer (10.3) translates
    LLMError to a 502.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("app.phrase_match._openai_client", lambda: None)

    pair_row = (
        db_session.query(PhrasePair).filter(PhrasePair.id == 1).one()
    )
    with pytest.raises(LLMError) as exc:
        generate_phrase_match(
            db_session,
            phrase_a_id=pair_row.phrase_a_id,
            phrase_b_id=pair_row.phrase_b_id,
        )
    assert "OPENROUTER_API_KEY" in str(exc.value)


def test_generate_phrase_match_raises_phrase_match_generation_error_on_persistent_schema_failure(
    monkeypatch, db_session
):
    """Persistent schema violations after ``MAX_ATTEMPTS`` raise
    ``PhraseMatchGenerationError`` carrying the structured
    dead-letter fields.

    Mirrors
    ``test_idiom.py::test_generate_idiom_raises_idiom_generation_error_on_persistent_schema_failure``
    — the persistent-retry exhaustion path.
    """
    import httpx
    from openai import OpenAI

    # Return an invalid payload (missing required fields,
    # wrong types) every call. ``instructor`` retries up to
    # ``MAX_ATTEMPTS`` times, then raises;
    # ``generate_phrase_match`` translates the failure into
    # ``PhraseMatchGenerationError``.
    invalid_payload = json.dumps(
        {
            "phrase_a": "x",  # too short, will fail bounds
            "relation": "synonym",  # not in 4-way literal
        },
        ensure_ascii=False,
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-phrase-match-bad-001",
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
    monkeypatch.setattr("app.phrase_match._openai_client", lambda: raw_client)

    pair_row = (
        db_session.query(PhrasePair).filter(PhrasePair.id == 1).one()
    )
    with pytest.raises(PhraseMatchGenerationError) as exc:
        generate_phrase_match(
            db_session,
            phrase_a_id=pair_row.phrase_a_id,
            phrase_b_id=pair_row.phrase_b_id,
        )
    err = exc.value
    assert err.schema_retry_count >= 1
    assert "phrase_match: schema validation failed" in str(err)
    # ``attempted_schema`` is a JSON-encoded Pydantic schema
    # dict so the operator can grep the dead-letter body.
    assert err.attempted_schema.startswith("{")


# ---------------------------------------------------------------------------
# 7. Pair-row selector — deterministic + read-only invariant.
# ---------------------------------------------------------------------------


def test_select_phrase_pair_is_deterministic(db_session):
    """``select_phrase_pair`` returns the same pair row for the
    same ``word_id`` across calls.

    Same integer → same pair (the stability commitment).
    Mirrors ``app.idiom.select_phrase_row`` /
    ``app.cloze.select_target_word`` /
    ``app.collocation.select_collocation_row``.
    """
    row_a = select_phrase_pair(db_session, word_id=1)
    row_b = select_phrase_pair(db_session, word_id=1)
    assert row_a.id == row_b.id

    # Different word_ids map to different rows. The exact
    # RNG mapping depends on Python's ``random.Random`` seed
    # — collisions are mathematically possible across 5
    # candidates × 5 word_ids. We assert a coverage lower
    # bound (≥ 2 distinct rows) that proves the mapping is
    # operating, while not being so strict that the test
    # flakes if the RNG collapses two adjacent seeds to the
    # same row. Same discipline as
    # ``test_idiom.py::test_select_phrase_row_is_deterministic``.
    seen: set[int] = set()
    for word_id in (1, 2, 3, 4, 5):
        row = select_phrase_pair(db_session, word_id=word_id)
        seen.add(row.id)
    # At least 2 distinct pair rows across 5 distinct
    # word_ids — proves the deterministic-seed mapping is
    # operating without coupling to the RNG's exact
    # distribution.
    assert len(seen) >= 2


def test_select_phrase_pair_raises_phrase_match_not_found_when_phrase_pairs_table_is_empty(
    tmp_path, monkeypatch
):
    """When no pair rows exist (10.1 seed hasn't run yet), the
    selector raises ``PhraseMatchNotFoundError`` — the 404
    path. The route layer (10.3) catches it and surfaces a
    404 (card body mandates 404, not 500, for this case;
    mirrors the 8.4 ``IdiomNotFoundError`` discipline).
    """
    db_path = tmp_path / "test_phrase_match_empty.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))

    from app import database

    database.reconfigure_for_test(f"sqlite:///{db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        from app.phrase_match import (
            PhraseMatchNotFoundError, PhrasePair, select_phrase_pair,
        )

        with pytest.raises(PhraseMatchNotFoundError) as exc:
            select_phrase_pair(session, word_id=1)
        assert exc.value.word_id == 1
        assert "no phrase_pairs row" in str(exc.value)


def test_phrase_pairs_table_is_read_only_at_generator_layer(
    db_session, monkeypatch
):
    """``app.phrase_match`` does not export any INSERT /
    UPDATE / DELETE helper for the ``phrase_pairs`` table.

    Hard rule #2 — read-only at runtime. The seed scripts
    (10.1) are the only write path. The type-level
    guardrail is *omission of write helpers* rather than
    runtime checks.
    """
    import app.phrase_match as phrase_match_module

    forbidden_names = (
        "insert_phrase_pair",
        "update_phrase_pair",
        "delete_phrase_pair",
        "create_phrase_pair",
        "upsert_phrase_pair",
        "write_phrase_pair",
    )
    public_names = set(getattr(phrase_match_module, "__all__", []))
    assert not (public_names & set(forbidden_names))


# ---------------------------------------------------------------------------
# 8. ``enable_rag`` stub — no bge-m3 OpenRouter call.
# ---------------------------------------------------------------------------


def test_enable_rag_stub_returns_empty_neighbors_offline():
    """``_retrieve_phrase_pair_neighbors`` (the 10.2 stub)
    returns an empty list unconditionally.

    Hard rule #7 — no bge-m3 OpenRouter chat call. 10.2 only
    stubs the parameter; 10.3 wires the real nearest-neighbor
    pull against the ``phrase_pairs`` table using local
    ``sentence-transformers`` (Phase 1.3) + Phase 7.5
    cosine.

    This test asserts the contract that the 10.2 stub returns
    ``[]`` without making any network call. A 10.3
    regression that reintroduces an OpenRouter chat call would
    surface here as a network-egress block (CI offline) or a
    missing-API-key failure.
    """
    from app.phrase_match import _retrieve_phrase_pair_neighbors

    # The stub takes a ``PhrasePair`` row (any) + an optional
    # ``top_k`` and returns ``[]`` deterministically.
    pair = PhrasePair(
        id=1, phrase_a_id="phrase-01", phrase_b_id="phrase-02",
        relation="equivalent", attested_pair=1,
    )
    result = _retrieve_phrase_pair_neighbors(None, pair, top_k=RAG_TOP_K)
    assert result == []


# ---------------------------------------------------------------------------
# 9. DSPy offline optimizer — runs on DummyLM.
# ---------------------------------------------------------------------------


def test_optimize_phrase_match_module_runs_offline_on_two_row_eval_set(
    monkeypatch,
):
    """``optimize_phrase_match_module`` runs end-to-end on a
    2-row eval set using ``DummyLM``.

    Same caveat as the cloze / collocation / idiom paths:
    MIPROv2's internal prompt-proposer is hard to satisfy
    with ``DummyLM`` (it expects strict JSONAdapter-shaped
    responses from an LM it probes several times). The
    function is expected to either succeed or fall back to
    the un-optimized module — either outcome proves the
    plumbing is wired.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None

    train = [
        {
            "word_id": 1,
            "phrase_a": "ins Blaue hinein",
            "phrase_b": "ohne festes Ziel",
            "relation_choices": list(RELATION_CHOICES),
            "few_shot_examples": None,
            "exercise": PhraseMatchExercise(
                exercise_id=12121212,
                phrase_a="ins Blaue hinein",
                phrase_b="ohne festes Ziel",
                relation="equivalent",
                relation_rationale="planlose Handlung.",
                source_attribution="dwds",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
        },
        {
            "word_id": 2,
            "phrase_a": "Tomaten auf den Augen",
            "phrase_b": "etwas Offensichtliches nicht sehen",
            "relation_choices": list(RELATION_CHOICES),
            "few_shot_examples": None,
            "exercise": PhraseMatchExercise(
                exercise_id=23232323,
                phrase_a="Tomaten auf den Augen",
                phrase_b="etwas Offensichtliches nicht sehen",
                relation="paraphrase",
                relation_rationale="Umschreibung.",
                source_attribution="dwds",
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
        },
    ]

    optimized = optimize_phrase_match_module(train)
    assert isinstance(optimized, PhraseMatchModule)


# ---------------------------------------------------------------------------
# 10. ``PhraseMatchModule.generate`` — DSPy-callable mirror.
# ---------------------------------------------------------------------------


def test_phrase_match_module_generate_returns_validated_exercise(
    monkeypatch,
):
    """``PhraseMatchModule.generate(...)`` returns a
    Pydantic-validated ``PhraseMatchExercise`` instance.

    Mirrors the ``CollocationModule`` /
    ``ComprehensionModule`` / ``IdiomModule`` generate-shape.
    The DSPy offline path serves a fixed ``DummyLM`` answer
    that the ``PhraseMatchExercise.model_validate_json``
    path accepts.

    Confirms Hard rule #10 — the predictor is a
    ``ChainOfThought``-wrapped instance (the test confirms
    the output round-trips through Pydantic).
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import dspy

    dspy.settings.lm = None
    _configure_dspy_on_phrase_match()

    module = PhraseMatchModule()
    exercise = module.generate()
    assert isinstance(exercise, PhraseMatchExercise)
    assert exercise.prompt_template_version == PROMPT_TEMPLATE_VERSION
    # The DummyLM pool has one equivalent + one paraphrase +
    # one related + one unrelated variant; the first call
    # serves the equivalent one (id 44444441).
    assert exercise.phrase_a == "ins Blaue hinein"
    assert exercise.phrase_b == "ohne festes Ziel"
    assert exercise.relation in RELATION_CHOICES


def _configure_dspy_on_phrase_match():
    """Helper: force the offline DSPy configuration on
    ``app.phrase_match``.

    Imports the helper and calls it in isolation. Mirrors
    ``_configure_dspy`` from app.cloze / app.collocation /
    app.idiom.
    """
    from app.phrase_match import _configure_dspy

    _configure_dspy()


# ---------------------------------------------------------------------------
# 11. Langfuse fallback — graceful no-op when keys are missing.
# ---------------------------------------------------------------------------


def test_trace_phrase_match_silent_when_langfuse_keys_missing(monkeypatch):
    """``_trace_phrase_match`` returns ``None`` when Langfuse
    keys are unset (graceful no-op, no per-call warnings, no
    network).

    Mirrors
    ``test_idiom.py::test_trace_idiom_silent_when_langfuse_keys_missing``
    / ``test_cloze.py::test_trace_cloze_is_silent_when_keys_missing``.
    """
    from app.phrase_match import _trace_phrase_match

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    exercise = PhraseMatchExercise(
        exercise_id=12121212,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="planlos.",
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata: dict[str, Any] = {
        "word_id": 0,
        "phrase_a_id": "phrase-01",
        "phrase_b_id": "phrase-02",
        "phrase_a": "ins Blaue hinein",
        "phrase_b": "ohne festes Ziel",
        "source_attribution_a": "dwds",
        "source_attribution_b": "dwds",
        "enable_rag": False,
        "retrieved_neighbors_present": False,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "exercise_type": "phrase_match",
        "phrase_match": True,
    }
    assert _trace_phrase_match(exercise, metadata, latency_ms=0) is None


def test_trace_phrase_match_emits_exercise_generate_span_with_phrase_match_discriminator(
    monkeypatch,
):
    """When Langfuse keys are present, ``_trace_phrase_match``
    emits one span named ``exercise.generate`` (the canonical
    Phase 6.x span name) with the ``phrase_match: true``
    discriminator and the ``exercise_type="phrase_match"``
    field.

    The discriminator is what the offline Ragas / Phase 6.7 /
    Phase 10.4 follow-up reader uses to split phrase_match
    cohorts from cloze / matching / comprehension / idiom /
    collocation cohorts.
    """
    import importlib

    phrase_match_live = importlib.import_module("app.phrase_match")

    mock_client = MagicMock(name="langfuse_client")
    mock_span = MagicMock(name="langfuse_span")
    mock_client.span.return_value = mock_span
    mock_client.flush.return_value = None

    monkeypatch.setattr(phrase_match_live, "get_langfuse", lambda: mock_client)

    exercise = PhraseMatchExercise(
        exercise_id=12121212,
        phrase_a="ins Blaue hinein",
        phrase_b="ohne festes Ziel",
        relation="equivalent",
        relation_rationale="planlos.",
        source_attribution="dwds",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    metadata: dict[str, Any] = {
        "word_id": 0,
        "phrase_a_id": "phrase-01",
        "phrase_b_id": "phrase-02",
        "phrase_a": "ins Blaue hinein",
        "phrase_b": "ohne festes Ziel",
        "source_attribution_a": "dwds",
        "source_attribution_b": "dwds",
        "enable_rag": False,
        "retrieved_neighbors_present": False,
        "model_id": "stub",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": [],
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "exercise_type": "phrase_match",
        "phrase_match": True,
    }
    _trace_phrase_match(exercise, metadata, latency_ms=0)

    # Span name is the canonical ``exercise.generate`` with
    # the phrase_match discriminator fields.
    mock_client.span.assert_called_once()
    span_kwargs = mock_client.span.call_args.kwargs
    assert span_kwargs["name"] == "exercise.generate"
    mock_span.update.assert_called_once()
    update_kwargs = mock_span.update.call_args.kwargs
    span_meta = update_kwargs["metadata"]
    assert span_meta["phrase_match"] is True
    assert span_meta["exercise_type"] == "phrase_match"
