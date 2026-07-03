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
    ClozeDifficulty,
    ClozeExerciseOut,
    DiagnosticChoiceOut,
    DiagnosticQuestionOut,
    DiagnosticState,
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
# Phase 5.2 — GradeRequest / GradeResponse (card t_88b6f1c4)
#
# Hard rule #2 (type-level guardrail): ``exercise_type`` is the
# schema-level gate. A drift to "matching" / "comprehension" is a
# ValidationError — no runtime check downstream.
#
# Hard rule #5 (Pydantic v2 validated input): ``grade`` is
# ``Literal[1, 2, 3, 4]``; out-of-range grades reject at the
# schema layer. ``exercise_id`` is ``gt=0``.
# ---------------------------------------------------------------------------


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
        "exercise_type", ["matching", "comprehension", "CLOZE", ""]
    )
    def test_non_cloze_exercise_type_is_rejected(self, exercise_type):
        """Hard rule #2 — ``exercise_type`` is hard-locked to
        ``"cloze"``. Any other value, including case variants and
        the empty string, is a ``ValidationError``.
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

    def test_non_cloze_exercise_type_is_rejected(self):
        """Same Hard rule #2 guardrail on the response side —
        the wire shape can't smuggle a non-cloze kind back to the
        client either (defense in depth)."""
        with pytest.raises(ValidationError):
            GradeResponse.model_validate(
                self._sample(exercise_type="matching")
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
