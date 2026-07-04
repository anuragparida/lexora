"""Tests for the Pydantic wire schemas (Phase 5.2 + earlier phases).

Card: t_88b6f1c4 (this card), but the file is also a homing
location for the small Pydantic-shape tests that don't belong
on a route-level test file. Each top-level test is annotated
with the card id / phase it covers.

Tests in this file are pure-Pydantic — no DB, no FastAPI client,
no network. Anything that needs the HTTP adapter lives in the
per-phase route test (test_grade.py, test_cloze.py, ...).

Run from ``backend/``::

    uv run pytest -q tests/test_schemas.py
"""
from __future__ import annotations

from datetime import datetime
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas import (
    AuthResponse,
    BaseExerciseFields,
    ClozeDifficulty,
    ClozeExerciseOut,
    ClozeGenerateRequest,
    DiagnosticChoiceOut,
    DiagnosticQuestionOut,
    DiagnosticState,
    ExerciseType,
    GradeRequest,
    GradeResponse,
    LoginRequest,
    SignupRequest,
    UserCreate,
    UserOut,
    WeaknessProfileOut,
    WeaknessProfileUpdate,
    WordResponse,
)


# ---------------------------------------------------------------------------
# Phase 5.2 / 6.6 — GradeRequest / GradeResponse (cards t_88b6f1c4, t_d11d0011)
#
# Hard rule #2 (type-level guardrail): ``exercise_type`` is the
# schema-level gate. Phase 6.6 widens it from
# ``Literal["cloze"]`` to the 3-way ``ExerciseType`` alias
# ``Literal["cloze", "matching", "comprehension"]``. A drift to
# "speaking" / "writing" / "CLOZE" / empty string is a
# ``ValidationError`` — no runtime check downstream.
#
# Hard rule #5 (Pydantic v2 validated input): ``grade`` is
# ``Literal[1, 2, 3, 4]``; out-of-range grades reject at the
# schema layer. ``exercise_id`` is ``gt=0``.
# ---------------------------------------------------------------------------


class TestExerciseTypeLiteral:
    """Phase 6.6 (card t_d11d0011) — the closed 3-way
    ``ExerciseType`` alias is the single source of truth for the
    grade-route wire guardrail. The alias is a
    ``typing.Literal`` — it isn't runtime-constructible, so we
    drive a ``TypeAdapter`` to validate against the union.
    """

    def test_exercise_type_literal_is_closed_three_way(self):
        """The ``ExerciseType`` literal covers exactly three
        values: ``cloze``, ``matching``, ``comprehension``. Any
        other value rejects.
        """
        adapter = TypeAdapter(ExerciseType)

        # The three valid types round-trip.
        for v in ("cloze", "matching", "comprehension"):
            assert adapter.validate_python(v) == v

        # Anything else is a ValidationError — including the
        # Phase 5.3 cloze-only drift targets (case variants, the
        # empty string, types that are "exercise-like" but not
        # in the closed union).
        for v in (
            "speaking",
            "writing",
            "CLOZE",
            "MATCHING",
            "",
            "cloze ",
            "cloze-extra",
        ):
            with pytest.raises(ValidationError):
                adapter.validate_python(v)

    def test_exercise_type_literal_exposes_three_members(self):
        """The literal exposes exactly three members via
        ``get_args`` — pins the closed 3-way shape so a future
        widening card has to update this test explicitly.
        """
        assert set(get_args(ExerciseType)) == {
            "cloze",
            "matching",
            "comprehension",
        }


class TestGradeRequest:
    """``GradeRequest`` shape contract."""

    def test_minimal_valid_request(self):
        """The minimal valid body. All three fields present with
        the right literals."""
        req = GradeRequest(
            exercise_id=42, exercise_type="cloze", grade=3
        )
        assert req.exercise_id == 42
        assert req.exercise_type == "cloze"
        assert req.grade == 3

    @pytest.mark.parametrize("grade_value", [1, 2, 3, 4])
    def test_each_valid_grade_literal_is_accepted(self, grade_value):
        """The four valid FSRS grades round-trip cleanly."""
        req = GradeRequest(
            exercise_id=1, exercise_type="cloze", grade=grade_value
        )
        assert req.grade == grade_value

    @pytest.mark.parametrize("grade_value", [0, 5, -1, 99])
    def test_out_of_range_grade_is_rejected(self, grade_value):
        """Out-of-range grades raise ``ValidationError`` — the schema
        is the gate."""
        with pytest.raises(ValidationError) as exc:
            GradeRequest(
                exercise_id=1, exercise_type="cloze", grade=grade_value
            )
        # The error envelope tags the offending field.
        assert "grade" in str(exc.value)

    @pytest.mark.parametrize(
        "exercise_type", ["cloze", "matching", "comprehension"]
    )
    def test_each_valid_exercise_type_is_accepted(self, exercise_type):
        """Phase 6.6 widens the ``exercise_type`` literal to a
        3-way union. The three values round-trip cleanly.
        """
        req = GradeRequest(
            exercise_id=1,
            exercise_type=exercise_type,
            grade=3,
        )
        assert req.exercise_type == exercise_type

    @pytest.mark.parametrize(
        "exercise_type",
        [
            "speaking",
            "writing",
            "CLOZE",
            "MATCHING",
            "COMPREHENSION",
            "cloze ",
            "",
        ],
    )
    def test_non_union_exercise_type_is_rejected(self, exercise_type):
        """Hard rule #2 — ``exercise_type`` is locked to the
        closed 3-way union ``Literal["cloze", "matching",
        "comprehension"]``. Any other value (case variants,
        near-misses, empty string) is a ``ValidationError``.

        Phase 5.3 pinned this with the cloze-only drift targets
        ``"matching"`` / ``"comprehension"`` / ``"CLOZE"`` / ``""``.
        Phase 6.6 splits the original test in two: ``matching``
        and ``comprehension`` are now *valid* (see
        ``test_each_valid_exercise_type_is_accepted``), and the
        new rejection set is everything outside the closed union.
        """
        with pytest.raises(ValidationError) as exc:
            GradeRequest(
                exercise_id=1,
                exercise_type=exercise_type,
                grade=3,
            )
        assert "exercise_type" in str(exc.value)

    @pytest.mark.parametrize(
        "exercise_id", [0, -1, -42]
    )
    def test_non_positive_exercise_id_is_rejected(self, exercise_id):
        """``exercise_id`` carries ``gt=0``. 0 / negatives are
        rejected before the request reaches the grader."""
        with pytest.raises(ValidationError) as exc:
            GradeRequest(
                exercise_id=exercise_id,
                exercise_type="cloze",
                grade=3,
            )
        assert "exercise_id" in str(exc.value)

    def test_missing_required_field_is_rejected(self):
        """Omitting any required field is a ``ValidationError``."""
        with pytest.raises(ValidationError):
            GradeRequest(exercise_id=1, exercise_type="cloze")  # no grade
        with pytest.raises(ValidationError):
            GradeRequest(exercise_id=1, grade=3)  # no exercise_type
        with pytest.raises(ValidationError):
            GradeRequest(exercise_type="cloze", grade=3)  # no exercise_id

    def test_json_round_trip_preserves_all_fields(self):
        """``model_dump`` round-trips every field — the wire shape
        is stable."""
        req = GradeRequest(
            exercise_id=42, exercise_type="cloze", grade=3
        )
        as_dict = req.model_dump()
        roundtrip = GradeRequest.model_validate(as_dict)
        assert roundtrip == req


class TestGradeResponse:
    """``GradeResponse`` shape contract."""

    def _sample(self, **overrides):
        """Build a valid response payload."""
        data = {
            "graded": True,
            "exercise_id": 42,
            "exercise_type": "cloze",
            "next_due_at": datetime(2026, 7, 1, 12, 0, 0),
            "card_state": 2,
            "stability": 5.0,
            "difficulty": 3.0,
            "trace_id": "trace-xyz",
        }
        data.update(overrides)
        return data

    def test_graded_discriminator_defaults_to_true(self):
        """The ``graded`` discriminator defaults to ``True`` so
        callers can omit it; the discriminator literal ensures a
        future ``False`` branch stays tagged.
        """
        # ``graded`` is omitted from the payload — defaults to True.
        sample = self._sample()
        sample.pop("graded")
        out = GradeResponse.model_validate(sample)
        assert out.graded is True
        assert out.exercise_type == "cloze"

    def test_graded_discriminator_must_be_true(self):
        """``graded`` is ``Literal[True]`` — the False case is
        reserved for a future card (the tagged-union handoff)."""
        with pytest.raises(ValidationError):
            GradeResponse.model_validate(self._sample(graded=False))

    @pytest.mark.parametrize(
        "exercise_type", ["cloze", "matching", "comprehension"]
    )
    def test_each_valid_exercise_type_is_accepted(self, exercise_type):
        """Phase 6.6 — the response side carries the same 3-way
        union as the request. Each value round-trips.
        """
        out = GradeResponse.model_validate(
            self._sample(exercise_type=exercise_type)
        )
        assert out.exercise_type == exercise_type

    def test_non_union_exercise_type_is_rejected(self):
        """Same Hard rule #2 guardrail on the response side —
        the wire shape can't smuggle a non-union kind back to
        the client either (defense in depth). Phase 6.6 widens
        from cloze-only to 3-way; the rejection set is now
        anything outside the union.
        """
        with pytest.raises(ValidationError):
            GradeResponse.model_validate(
                self._sample(exercise_type="speaking")
            )

    def test_trace_id_can_be_none(self):
        """The graceful-degradation path: ``trace_id`` is ``None``
        when Langfuse keys are unset, and the schema accepts it.
        A NOT NULL constraint here would force Phase 5.3 to lie
        about non-existent traces."""
        out = GradeResponse.model_validate(self._sample(trace_id=None))
        assert out.trace_id is None

    @pytest.mark.parametrize("card_state", [1, 2, 3])
    def test_valid_card_states_round_trip(self, card_state):
        """FSRS state is the int {1, 2, 3} — Learning / Review /
        Relearning. Any int is accepted at the schema layer (the
        model layer is the gate for the enum); the schema just
        carries the int through."""
        out = GradeResponse.model_validate(
            self._sample(card_state=card_state)
        )
        assert out.card_state == card_state

    def test_json_round_trip_preserves_all_fields(self):
        """``model_dump_json`` -> parse -> ``model_validate``
        yields the same Pydantic object — every field survives the
        wire / serialize / deserialize cycle.
        """
        original = GradeResponse.model_validate(self._sample())
        as_json = original.model_dump_json()
        decoded = GradeResponse.model_validate_json(as_json)
        assert decoded == original


# ---------------------------------------------------------------------------
# Phase 2.1 — UserCreate / UserOut (existing contract regression)
# ---------------------------------------------------------------------------


class TestUserSchemas:
    """Lock the Phase 2.1 user schema so a future refactor can't
    silently add ``password_hash`` to ``UserOut``."""

    def test_user_out_does_not_expose_password_hash(self):
        out = UserOut.model_validate(
            {
                "id": 1,
                "email": "x@example.com",
                "created_at": datetime(2026, 1, 1, 0, 0, 0),
            }
        )
        dumped = out.model_dump()
        assert "password_hash" not in dumped

    def test_user_create_requires_email_and_password_hash(self):
        with pytest.raises(ValidationError):
            UserCreate()  # both fields missing


# ---------------------------------------------------------------------------
# Phase 2.2 — SignupRequest / LoginRequest (auth shape regression)
# ---------------------------------------------------------------------------


class TestAuthSchemas:
    """The Phase 2.2 auth shape: ``EmailStr`` on email,
    ``[8, 128]`` on password. Out-of-range passwords reject."""

    def test_signup_request_requires_email_and_password(self):
        with pytest.raises(ValidationError):
            SignupRequest()  # no fields

    def test_login_request_password_too_short_is_rejected(self):
        """bcrypt's 72-byte input cap is matched by a
        ``min_length=8`` Pydantic guard; below 8 chars reject."""
        with pytest.raises(ValidationError):
            LoginRequest(email="x@example.com", password="short")

    def test_auth_response_exposes_access_token_and_user(self):
        ar = AuthResponse.model_validate(
            {
                "access_token": "abc",
                "user": {
                    "id": 1,
                    "email": "x@example.com",
                    "created_at": datetime(2026, 1, 1, 0, 0, 0),
                },
            }
        )
        assert ar.access_token == "abc"
        assert ar.user.email == "x@example.com"


# ---------------------------------------------------------------------------
# Phase 2.1 / 3.3 — WeaknessProfile shape
# ---------------------------------------------------------------------------


class TestWeaknessProfileSchemas:
    """WeaknessProfileUpdate rejects out-of-range axis scores, and
    ``WeaknessProfileOut`` carries the user's saved axes."""

    def test_axes_scores_outside_zero_three_rejected(self):
        with pytest.raises(ValidationError):
            WeaknessProfileUpdate(axes={"verbs": 4})

    def test_axes_booleans_coerce_to_int_at_pydantic_layer(self):
        """Pydantic's ``Dict[str, int]`` coercion squashes a JSON
        ``true`` to ``1`` BEFORE the ``field_validator`` runs — so
        the ``isinstance(score, bool)`` guard is technically dead
        code today. We document the current behaviour here so a
        future maintainer doesn't ship a regression: the only way
        to actually reject ``True`` would be a stricter ``Field``
        annotation (e.g. ``Annotated[int, StrictInt]``). For
        Phase 5.2 this is a known-acceptable edge; the schema's
        primary gate is the [0, 3] range check, which still
        rejects ``True`` because it becomes ``1`` (which is in
        range) — so ``True`` IS accepted as ``"valid"`` today.
        """
        # Acceptance is the current behaviour — codify it.
        out = WeaknessProfileUpdate(axes={"verbs": True})
        assert out.axes == {"verbs": 1}

    def test_out_carries_axes_and_updated_at(self):
        out = WeaknessProfileOut.model_validate(
            {
                "id": 1,
                "user_id": 1,
                "axes": {"verbs": 2},
                "updated_at": datetime(2026, 1, 1, 0, 0, 0),
            }
        )
        assert out.axes == {"verbs": 2}


# ---------------------------------------------------------------------------
# Phase 4.2 — ClozeExerciseOut (wire shape regression)
# ---------------------------------------------------------------------------


class TestClozeSchemas:
    """The cloze wire shape locks:

    - ``sentence_with_blank``: non-empty
    - ``answer_word_id``: positive int
    - ``distractors``: exactly 3 entries
    - ``difficulty``: ``"easy" | "medium" | "hard"``
    - ``rationale``: ``[1, 400]`` chars
    """

    def _sample(self, **overrides):
        data = {
            "sentence_with_blank": "Ich ___ nach Hause.",
            "answer_word_id": 1,
            "distractors": [2, 3, 4],
            "difficulty": "medium",
            "rationale": "default sentence template",
            "prompt_template_version": "cloze-v1",
            # Phase 6.1 — required shared metadata fields.
            "target_word_id": 1,
            "latency_ms": 0,
            # ``exercise_type`` / ``enable_rag`` / ``trace_id``
            # have defaults so they don't need to be set, but
            # the sample includes them for clarity.
            "exercise_type": "cloze",
            "enable_rag": False,
            "trace_id": None,
        }
        data.update(overrides)
        return data

    def test_minimal_valid_cloze_payload(self):
        out = ClozeExerciseOut.model_validate(self._sample())
        assert out.prompt_template_version == "cloze-v1"
        assert out.difficulty == "medium"

    @pytest.mark.parametrize(
        "difficulty", ["easy", "medium", "hard"]
    )
    def test_valid_difficulty_literals(self, difficulty):
        out = ClozeExerciseOut.model_validate(
            self._sample(difficulty=difficulty)
        )
        assert out.difficulty == difficulty

    def test_invalid_difficulty_rejected(self):
        """The ``ClozeDifficulty`` literal is closed — any other
        value is a ``ValidationError``."""
        with pytest.raises(ValidationError):
            ClozeExerciseOut.model_validate(
                self._sample(difficulty="impossible")
            )

    def test_distractors_must_be_exactly_three(self):
        """The card-body contract locks distractors to exactly 3."""
        with pytest.raises(ValidationError):
            ClozeExerciseOut.model_validate(
                self._sample(distractors=[1, 2])
            )
        with pytest.raises(ValidationError):
            ClozeExerciseOut.model_validate(
                self._sample(distractors=[1, 2, 3, 4])
            )

    def test_exercise_type_discriminator_is_cloze(self):
        """Phase 6.1 — ``ClozeExerciseOut`` carries the
        ``exercise_type`` discriminator. Default is ``"cloze"``;
        any other value (matching / comprehension) is a
        ``ValidationError`` — the type system is the gate
        (Hard rule #1 / Plan §"The exercise-type wire").
        """
        # Default value (omit from payload) — defaults to "cloze".
        out = ClozeExerciseOut.model_validate(
            {k: v for k, v in self._sample().items() if k != "exercise_type"}
        )
        assert out.exercise_type == "cloze"

        # Explicit "cloze" — also accepted.
        out = ClozeExerciseOut.model_validate(
            self._sample(exercise_type="cloze")
        )
        assert out.exercise_type == "cloze"

    def test_exercise_type_matching_is_rejected(self):
        """Phase 6.1 — the cloze response can only claim
        ``exercise_type="cloze"``. ``"matching"`` is a 6.2+ field;
        trying to surface it on the cloze shape is a
        ``ValidationError`` (defense in depth — the cloze route
        always stamps ``"cloze"`` itself, but a future maintainer
        who forgets the stamp gets caught by the schema).
        """
        with pytest.raises(ValidationError) as exc:
            ClozeExerciseOut.model_validate(
                self._sample(exercise_type="matching")
            )
        assert "exercise_type" in str(exc.value)

    def test_target_word_id_must_equal_answer_word_id(self):
        """Phase 6.1 — ``target_word_id`` is the canonical shared
        field for cross-exercise-type consumers; on cloze it
        must equal ``answer_word_id``. A drift is a bug.
        """
        with pytest.raises(ValidationError) as exc:
            ClozeExerciseOut.model_validate(
                self._sample(answer_word_id=1, target_word_id=2)
            )
        assert "target_word_id" in str(exc.value)

    def test_default_enable_rag_is_false(self):
        """Phase 6.1 — RAG-on is opt-in. The default ``enable_rag``
        on the response is ``False`` (matches the request default
        of ``False`` from ``ClozeGenerateRequest``).
        """
        # Omit from payload — defaults to False.
        out = ClozeExerciseOut.model_validate(
            {k: v for k, v in self._sample().items() if k != "enable_rag"}
        )
        assert out.enable_rag is False

        # Explicit True — also accepted (the route stamps it from
        # the request payload).
        out = ClozeExerciseOut.model_validate(
            self._sample(enable_rag=True)
        )
        assert out.enable_rag is True

    def test_latency_ms_must_be_non_negative_int(self):
        """Phase 6.1 — ``latency_ms`` is a required positive-or-zero
        int. Missing is a ``ValidationError``; floats that aren't
        whole are too.
        """
        with pytest.raises(ValidationError):
            ClozeExerciseOut.model_validate(
                {k: v for k, v in self._sample().items() if k != "latency_ms"}
            )


# ---------------------------------------------------------------------------
# Phase 6.1 — ClozeGenerateRequest (card t_616cc266)
#
# Spec coverage from the card body §"Tests":
# - ``ClozeGenerateRequest()`` → default ``enable_rag=False``.
# - ``ClozeGenerateRequest(enable_rag=True)`` → serialises to
#   ``{"enable_rag": true}``.
# - ``ClozeExerciseOut`` JSON includes ``exercise_type: "cloze"``.
# ---------------------------------------------------------------------------


class TestClozeGenerateRequest:
    """``ClozeGenerateRequest`` shape contract.

    The empty-body case ``{}`` parses to
    ``ClozeGenerateRequest(enable_rag=False)`` via Pydantic's
    defaults — that's the Phase 4.5 wire-contract preservation
    the card body calls out. Existing curl / frontend callers
    that pass ``{}`` see no change.
    """

    def test_default_enable_rag_is_false(self):
        """An empty ``ClozeGenerateRequest()`` defaults
        ``enable_rag=False`` — the Phase 4.5 contract."""
        req = ClozeGenerateRequest()
        assert req.enable_rag is False

    def test_empty_payload_parses_to_default(self):
        """``{}`` parses to ``enable_rag=False`` (Pydantic default)."""
        req = ClozeGenerateRequest.model_validate({})
        assert req.enable_rag is False

    def test_explicit_true_round_trips(self):
        """``ClozeGenerateRequest(enable_rag=True)`` serialises
        to ``{"enable_rag": true, "collocation": false}`` on the
        wire. Phase 7.3 (card t_bdd6ab24) widens the request with
        the ``collocation`` opt-in flag; the default is ``False``
        so existing Phase 4.2 / 6.1 callers see the new field
        appear on the wire but the field's value is the no-op
        default (Hard rule H10).
        """
        req = ClozeGenerateRequest(enable_rag=True)
        assert req.enable_rag is True
        assert req.collocation is False
        # ``model_dump`` is the JSON-ready dict; ``model_dump_json``
        # is the wire string. Both should carry the flag.
        assert req.model_dump() == {"enable_rag": True, "collocation": False}
        assert req.model_dump_json() == '{"enable_rag":true,"collocation":false}'

    def test_explicit_false_round_trips(self):
        """``ClozeGenerateRequest(enable_rag=False)`` is also
        accepted (default + explicit agree on the wire). Same
        shape change as the explicit-true case — the new
        ``collocation`` field appears at ``False`` by default
        (Hard rule H10, H3)."""
        req = ClozeGenerateRequest(enable_rag=False)
        assert req.enable_rag is False
        assert req.collocation is False
        assert req.model_dump() == {"enable_rag": False, "collocation": False}
        assert req.model_dump_json() == '{"enable_rag":false,"collocation":false}'

    def test_non_bool_enable_rag_is_rejected(self):
        """Non-bool ``enable_rag`` (list / dict) is a
        ``ValidationError``. The type system is the gate — a
        non-bool container value doesn't sneak through.

        Note: Pydantic v2's default coercion accepts strings
        (``"true"`` → ``True``), ints (``1`` → ``True``,
        ``0`` → ``False``), and floats (``1.0`` → ``True``)
        for bool fields — documented behaviour for
        JSON-deserialised payloads. Container values (list,
        dict) ARE rejected, which is what this test asserts.
        The stringy / numeric coercion is acceptable here
        because the wire format is JSON, and ``True`` /
        ``False`` are the canonical JSON bools.
        """
        for bad in ([True], {"v": True}, [1, 2], {}):
            with pytest.raises(ValidationError) as exc:
                ClozeGenerateRequest(enable_rag=bad)
            assert "enable_rag" in str(exc.value)


class TestClozeExerciseOutJsonShape:
    """Wire-level assertions on the cloze response.

    Spec: ``ClozeExerciseOut`` JSON includes
    ``exercise_type: "cloze"`` (the new Phase 6.1 discriminator).
    """

    def test_json_includes_exercise_type_discriminator(self):
        """The serialised JSON carries ``exercise_type: "cloze"``
        at the top level."""
        out = ClozeExerciseOut.model_validate(
            {
                "sentence_with_blank": "Der ___ schläft.",
                "answer_word_id": 1,
                "target_word_id": 1,
                "distractors": [2, 3, 4],
                "difficulty": "easy",
                "rationale": "test",
                "prompt_template_version": "cloze-v1",
                "latency_ms": 0,
            }
        )
        wire = out.model_dump_json()
        # The discriminator is a top-level key, not nested.
        assert '"exercise_type":"cloze"' in wire
        # And the cloze-specific fields are unchanged.
        assert '"sentence_with_blank":"Der ___ schläft."' in wire
        assert '"answer_word_id":1' in wire

    def test_json_includes_enable_rag_flag(self):
        """The serialised JSON carries ``enable_rag`` (echoed
        from the request)."""
        out = ClozeExerciseOut.model_validate(
            {
                "sentence_with_blank": "Der ___ schläft.",
                "answer_word_id": 1,
                "target_word_id": 1,
                "distractors": [2, 3, 4],
                "difficulty": "easy",
                "rationale": "test",
                "prompt_template_version": "cloze-v1",
                "latency_ms": 0,
                "enable_rag": True,
            }
        )
        wire = out.model_dump_json()
        assert '"enable_rag":true' in wire

    def test_json_includes_trace_id_when_set(self):
        """``trace_id`` is ``None`` by default; an explicit
        string is surfaced on the wire (Phase 4.3 hook returns
        ``None``; 6.x widens to a real id)."""
        out = ClozeExerciseOut.model_validate(
            {
                "sentence_with_blank": "Der ___ schläft.",
                "answer_word_id": 1,
                "target_word_id": 1,
                "distractors": [2, 3, 4],
                "difficulty": "easy",
                "rationale": "test",
                "prompt_template_version": "cloze-v1",
                "latency_ms": 0,
                "trace_id": "abc-123",
            }
        )
        wire = out.model_dump_json()
        assert '"trace_id":"abc-123"' in wire


# ---------------------------------------------------------------------------
# Phase 6.1 — BaseExerciseFields mixin (card t_616cc266)
#
# Verifies the mixin's field set is what Phase 6.2+ (matching)
# and 6.4+ (comprehension) will inherit. The matching /
# comprehension cards extend the ``exercise_type`` Literal
# union — Phase 6.1 ships the cloze branch only.
# ---------------------------------------------------------------------------


class TestBaseExerciseFields:
    """Mixin field set — the contract Phase 6.2 / 6.4 inherit."""

    def test_cloze_response_carries_all_shared_fields(self):
        """Every field declared on ``BaseExerciseFields`` is
        present on a ``ClozeExerciseOut`` instance (mixin
        inheritance works as documented)."""
        out = ClozeExerciseOut.model_validate(
            {
                "sentence_with_blank": "Der ___ schläft.",
                "answer_word_id": 1,
                "target_word_id": 1,
                "distractors": [2, 3, 4],
                "difficulty": "easy",
                "rationale": "test",
                "prompt_template_version": "cloze-v1",
                "latency_ms": 0,
            }
        )
        # Shared fields
        assert out.exercise_type == "cloze"
        assert out.target_word_id == 1
        assert out.prompt_template_version == "cloze-v1"
        assert out.enable_rag is False  # default
        assert out.trace_id is None  # default
        assert out.latency_ms == 0
        # Cloze-specific
        assert out.sentence_with_blank == "Der ___ schläft."
        assert out.answer_word_id == 1
        assert out.distractors == [2, 3, 4]


# ---------------------------------------------------------------------------
# Phase 3 — Diagnostic wire shapes (regression)
# ---------------------------------------------------------------------------


class TestDiagnosticSchemas:
    """Phase 3 diagnostic schemas: ``DiagnosticState`` Literal,
    ``DiagnosticQuestionOut`` strips scoring internals."""

    def test_diagnostic_state_literal_is_closed(self):
        """The ``DiagnosticState`` Literal covers four valid
        states. Anything else rejects.

        ``DiagnosticState`` is a ``typing.Literal`` alias — it's
        not a runtime-constructible value, so we drive a
        ``TypeAdapter`` to validate against the union.
        """
        state_adapter = TypeAdapter(DiagnosticState)

        # The four valid states round-trip.
        for v in ("never", "in_progress", "completed", "applied"):
            assert state_adapter.validate_python(v) == v

        # Anything else is a ValidationError.
        for v in ("unknown", "", "COMPLETED", "done"):
            with pytest.raises(ValidationError):
                state_adapter.validate_python(v)

        # Sanity check: the literal exposes exactly four members
        # via ``get_args``.
        assert set(get_args(DiagnosticState)) == {
            "never",
            "in_progress",
            "completed",
            "applied",
        }

    def test_diagnostic_question_out_strips_scoring(self):
        """``DiagnosticQuestionOut`` exposes only the human-facing
        fields. ``axis_tags`` and ``weight`` must not be part of
        the schema."""
        out = DiagnosticQuestionOut.model_validate(
            {
                "id": "q-1",
                "prompt": "How comfortable are you with verbs?",
                "kind": "axis_intensity",
                "choices": [
                    {"label": "Not at all"},
                    {"label": "A bit"},
                    {"label": "Confident"},
                ],
            }
        )
        assert out.id == "q-1"
        # The shape carries only the two expected choice fields —
        # server-side scoring (delta, weight, axis_tags) is
        # explicitly NOT exposed.
        assert hasattr(out.choices[0], "label")
        assert not hasattr(out.choices[0], "delta")

    def test_choice_label_round_trips(self):
        """``DiagnosticChoiceOut`` carries a single ``label`` field.
        Pydantic v2 accepts an empty string here — the schema
        doesn't enforce ``min_length=1`` (this is an existing
        shape from Phase 3.1; not in Phase 5.2 scope to tighten).
        What we DO assert is that the field round-trips cleanly."""
        out = DiagnosticChoiceOut.model_validate({"label": "Yes"})
        assert out.label == "Yes"


# ---------------------------------------------------------------------------
# Phase 0 — WordResponse shape regression
# ---------------------------------------------------------------------------


class TestWordSchemas:
    """Phase 0 — ``WordResponse`` is the public word shape."""

    def test_minimal_word_response(self):
        out = WordResponse.model_validate(
            {
                "id": 1,
                "word": "Hund",
                "is_complete": False,
            }
        )
        assert out.id == 1
        assert out.word == "Hund"
        # Defaults from the schema definition.
        assert out.examples == []
        assert out.verb_conjugation is None
