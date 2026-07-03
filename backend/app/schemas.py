from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Literal


class VerbConjugationBase(BaseModel):
    infinitive: str
    present_3rd_person: Optional[str] = None
    simple_past: Optional[str] = None
    participle: Optional[str] = None


class VerbConjugationResponse(VerbConjugationBase):
    id: int

    class Config:
        from_attributes = True


class ExampleBase(BaseModel):
    german: str
    english: str


class ExampleResponse(ExampleBase):
    id: int

    class Config:
        from_attributes = True


class WordBase(BaseModel):
    word: str
    word_type: Optional[str] = None
    frequency: Optional[str] = None
    level: Optional[str] = None
    translations: Optional[str] = None
    conjugation: Optional[str] = None
    additional_info: Optional[str] = None
    is_complete: bool = False


class WordResponse(WordBase):
    id: int
    examples: List[ExampleResponse] = []
    verb_conjugation: Optional[VerbConjugationResponse] = None

    class Config:
        from_attributes = True


class WordListResponse(BaseModel):
    items: List[WordResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Phase 2.1 — User + WeaknessProfile schemas
#
# This card ships the data layer only. NO auth, NO bcrypt, NO JWT. The
# ``POST /users`` route is intentionally open and accepts a raw
# ``password_hash`` string; Phase 2.2 will add ``/auth/signup`` /
# ``/auth/login`` that hash internally and gate the new routes.
#
# ``UserOut`` deliberately omits ``password_hash``. The route layer
# queries ``models.User`` then serialises via ``UserOut.model_validate``;
# Pydantic's ``from_attributes=True`` config means missing fields raise
# a clean validation error if a future maintainer accidentally
# adds ``password_hash`` to the schema.
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """Request body for ``POST /users`` (Phase 2.1, auth-free).

    ``password_hash`` is sent raw for this card. Phase 2.2 swaps this
    for a plain ``password: str`` field once ``/auth/signup`` hashes
    internally. Email format validation is NOT enforced here — that's
    Phase 2.2 too (the auth flow owns email format guarantees).
    """

    email: str
    password_hash: str = Field(
        ...,
        min_length=1,
        description=(
            "Raw password hash. Phase 2.1 accepts a pre-hashed value; "
            "Phase 2.2's /auth/signup will hash internally."
        ),
    )


class UserOut(BaseModel):
    """Response shape for ``User`` rows.

    NEVER exposes ``password_hash``. If you find yourself adding that
    field here, stop — ``schemas.py`` is the public API surface, and
    returning the hash would leak credential material.
    """

    id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class WeaknessProfileUpdate(BaseModel):
    """Request body for ``PUT /weakness-profile/{user_id}``.

    Each axis value must be an int in [0, 3]. The validation runs in
    a ``field_validator`` so 422 responses surface on the standard
    Pydantic error envelope. Empty ``axes={}`` is a valid reset (the
    user can clear their declaration without dropping the profile row).
    """

    axes: Dict[str, int] = Field(default_factory=dict)

    @field_validator("axes")
    @classmethod
    def _validate_axes(cls, v: Dict[str, int]) -> Dict[str, int]:
        if not isinstance(v, dict):
            raise ValueError("axes must be an object mapping axis name to score")
        for name, score in v.items():
            if not isinstance(score, int) or isinstance(score, bool):
                # ``bool`` is a subclass of ``int`` in Python — exclude
                # it explicitly so a JSON ``true`` doesn't sneak through.
                raise ValueError(f"axis '{name}' must be an integer in [0, 3]")
            if not 0 <= score <= 3:
                raise ValueError(
                    f"axis '{name}' score must be in [0, 3]; got {score}"
                )
        return v


class WeaknessProfileOut(BaseModel):
    """Response shape for ``WeaknessProfile`` rows.

    ``axes`` is always serialised as a dict on the wire regardless of
    the storage dialect (Postgres JSON vs SQLite Text). The CRUD layer
    is responsible for deserialising on read and serialising on write.
    """

    id: int
    user_id: int
    axes: Dict[str, int]
    updated_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Phase 2.2 — Auth request / response schemas
#
# Card t_74c3aa1e. Replaces the Phase 2.1 ``UserCreate`` /
# ``UserOut`` flow with a proper bcrypt + JWT surface:
#
# - ``SignupRequest`` and ``LoginRequest`` both accept a plaintext
#   ``password`` (NOT a pre-hashed value). ``EmailStr`` enforces
#   RFC-5321-ish format at the Pydantic layer; ``password`` is
#   bounded ``[8, 128]`` — the lower bound is the spec minimum, the
#   upper bound stops a pathological request from making bcrypt do
#   72-byte truncation work for a multi-MB body.
# - ``AuthResponse`` returns ``{access_token, user}``. The
#   ``user`` field is a ``UserOut`` (no ``password_hash``) — the
#   field name ``access_token`` is the convention used by the
#   frontend auth card 2.3, which stores it for now (it'll move
#   to a pure-cookie path once the SPA proxies all requests
#   through the same origin).
# ---------------------------------------------------------------------------


# bcrypt's 72-byte input limit caps the meaningful password length.
# We accept up to 128 chars in the schema and let ``app.passwords``
# truncate to 72 bytes — the upper bound is just a sanity cap.
_PASSWORD_MIN = 8
_PASSWORD_MAX = 128


class _PasswordBody(BaseModel):
    """Shared base for signup/login request bodies.

    Kept private (underscore prefix) because the public surface is
    the two request models below. The ``EmailStr`` field requires
    ``email-validator`` (Phase 2.2 added it to ``pyproject.toml``).
    """

    email: EmailStr
    password: str = Field(
        ...,
        min_length=_PASSWORD_MIN,
        max_length=_PASSWORD_MAX,
        description=(
            "Plaintext password. Min 8 chars, max 128 (the bcrypt "
            "library truncates to 72 bytes internally)."
        ),
    )


class SignupRequest(_PasswordBody):
    """Request body for ``POST /auth/signup``."""


class LoginRequest(_PasswordBody):
    """Request body for ``POST /auth/login``."""


class AuthResponse(BaseModel):
    """Response body for ``/auth/signup`` and ``/auth/login``.

    ``user`` is a ``UserOut`` — never carries ``password_hash`` even
    though the SQLAlchemy row has the column. The cookie is set as a
    side effect of the response (the route calls
    ``app.auth.set_auth_cookie``), independent of this body shape,
    so curl / manual tests that ignore the body still work.
    """

    access_token: str
    user: "UserOut"


# ---------------------------------------------------------------------------
# Phase 3.1 — Diagnostic probe request / response schemas
#
# Card t_41d85c32. The probe is a deterministic, auth-gated,
# LLM-free questionnaire. Four endpoints:
#
#   POST /diagnostic/start   -> DiagnosticStartOut
#   POST /diagnostic/answer  -> {"answered": N, "total": 10}
#   GET  /diagnostic/result  -> DiagnosticResultOut
#   POST /diagnostic/apply   -> WeaknessProfileOut (reuses 2.x shape)
#
# The question bank's scoring fields (delta / weight / axis_tags)
# NEVER cross the wire — ``DiagnosticQuestionOut`` exposes only the
# client-facing fields (id / prompt / kind / choices[].label).
# ---------------------------------------------------------------------------


class DiagnosticChoiceOut(BaseModel):
    """A single client-facing choice. Only the human-readable label
    is exposed — the server-side ``delta`` map is stripped so the
    client can't reverse-engineer the scoring."""

    label: str


class DiagnosticQuestionOut(BaseModel):
    """A client-facing question. Excludes ``axis_tags`` / ``weight``
    (scoring internals); exposes only what the UI needs to render the
    question and its options."""

    id: str
    prompt: str
    kind: str
    choices: List[DiagnosticChoiceOut]


class DiagnosticStartOut(BaseModel):
    """Response for ``POST /diagnostic/start``: the session handle
    plus the full (stripped) question bank to render."""

    session_id: str
    questions: List[DiagnosticQuestionOut]


class DiagnosticAnswerIn(BaseModel):
    """Request body for ``POST /diagnostic/answer``.

    ``session_id`` scopes the answer to a probe run; ``question_id``
    and ``choice_label`` are validated against the bank in the route
    layer (400 on an unknown id / label, 404 on a session that isn't
    the caller's). All three fields are required and non-empty.
    """

    session_id: str = Field(..., min_length=1)
    question_id: str = Field(..., min_length=1)
    choice_label: str = Field(..., min_length=1)


class DiagnosticAnswerOut(BaseModel):
    """Response for ``POST /diagnostic/answer``: progress counters.

    ``answered`` is the count of distinct questions answered in this
    session; ``total`` is the fixed bank size (10).
    """

    answered: int
    total: int


class DiagnosticResultOut(BaseModel):
    """Response for ``GET /diagnostic/result``.

    ``axes`` maps every axis -> 0..3 (axes no answer touched are 0).
    ``reasons`` maps only axes with score > 0 -> a one-line string
    naming the top contributing questions. Matches the
    ``WeaknessProfile`` axes shape so Apply is a direct UPSERT.
    """

    axes: Dict[str, int]
    reasons: Dict[str, str]


class DiagnosticApplyIn(BaseModel):
    """Request body for ``POST /diagnostic/apply``: which session's
    computed result to UPSERT into the caller's weakness profile."""

    session_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Phase 3.3 — ``MeOut`` (response for ``GET /auth/me``)
#
# Card: t_ff6fa637. Extends the previous ``UserOut`` payload with the
# two fields the post-signup first-login gate (frontend) needs:
#
# - ``weakness_profile`` — the user's saved axes (or ``None`` if the
#   row hasn't been created yet; the gate treats ``None`` as empty).
#   Reuses ``WeaknessProfileOut`` so the shape is byte-identical to
#   ``GET /weakness-profile/{user_id}``.
# - ``diagnostic_state`` — a coarse summary of the user's diagnostic
#   probe history, computed from the most recent
#   ``diagnostic_sessions`` row:
#
#   - ``"never"``       — no session has ever been started
#   - ``"in_progress"`` — a session exists with status ``in_progress``
#   - ``"completed"``   — most recent session is ``completed`` (the
#     user has answered all 10 questions but hasn't applied yet)
#   - ``"applied"``     — most recent session is ``applied`` (Apply
#     has been clicked; the score is in the profile, even if the
#     user then zeroed the sliders manually)
#
# The gate logic on the client is:
#
#   axes non-empty                              -> /weakness-profile
#   axes empty AND state in {never,in_progress} -> /diagnostic
#   axes empty AND state in {completed,applied} -> /weakness-profile
#     (the user has been through the probe; respect their choice)
#
# The ``Literal`` keeps the response self-documenting — OpenAPI
# surfaces the four valid values and the frontend gets a string
# union on the wire.
# ---------------------------------------------------------------------------


DiagnosticState = Literal["never", "in_progress", "completed", "applied"]


class MeOut(BaseModel):
    """Response shape for ``GET /auth/me``.

    Returned to the SPA on every auth probe (the protected-route
    gate fires ``getMe()`` on mount, the post-login ``AuthForm``
    uses it to decide where to land, the header re-probes on
    ``lexora:auth-change``). The new fields are non-breaking — any
    client that only reads ``id``/``email`` keeps working.
    """

    id: int
    email: str
    created_at: datetime
    # ``None`` means the user has never had a profile row created
    # (pre-Phase-2.1 schema, or simply hasn't loaded the profile
    # page yet). The frontend treats ``None`` as ``{axes: {}}``.
    weakness_profile: Optional[WeaknessProfileOut] = None
    diagnostic_state: DiagnosticState = "never"

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Phase 4.2 — Cloze exercise response (card t_bdd9ffbe)
#
# Mirrors ``app.cloze.ClozeExercise`` 1:1. The split is intentional:
# ``app.cloze`` owns the *generator* contract (used by the
# instructor-wrapped chat call); ``app.schemas`` owns the *wire*
# contract (used by the FastAPI response_model and the SPA).
# Field-for-field equivalence today; if they ever diverge, the
# schemas shape is the one Phase 5's grading loop will read off the
# persisted row.
# ---------------------------------------------------------------------------


ClozeDifficulty = Literal["easy", "medium", "hard"]


class ClozeExerciseOut(BaseModel):
    """Response shape for ``POST /exercises/cloze``."""

    sentence_with_blank: str = Field(
        ...,
        description=(
            "German sentence with '___' marking the cloze position."
        ),
    )
    answer_word_id: int = Field(..., description="FK to words.id of the correct answer.")
    distractors: list[int] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Exactly 3 FKs to words.id of plausible wrong answers. "
            "Same word_type as answer_word_id."
        ),
    )
    difficulty: ClozeDifficulty
    rationale: str = Field(..., min_length=1, max_length=400)
    prompt_template_version: str = Field(
        ..., description="Should always equal 'cloze-v1' for production generations."
    )


# ---------------------------------------------------------------------------
# Phase 5.2 / 6.6 — Grade request / response (cards t_88b6f1c4, t_d11d0011)
#
# Wire contract for ``POST /exercises/grade``. Phase 5.3 imports this
# shape — keeping schemas + models here means 5.3 and 5.4 read the
# same Pydantic types and the same SQLAlchemy rows.
#
# Phase 6.6 widens the ``exercise_type`` literal from a 1-way
# ``Literal["cloze"]`` to a 3-way union
# ``Literal["cloze", "matching", "comprehension"]``. The widening
# is the wire-level guardrail — Pydantic rejects any other value
# (``"speaking"``, ``"CLOZE"``, empty string) with a 422. The
# existing 5.2/5.3 cloze callers are unaffected: Pydantic accepts
# ``"cloze"`` as a subset of the 3-way union. The 5.3 grader logic
# stays exercise-type-agnostic (it goes through ``apply_grade`` +
# a ``grade_logs`` row, both of which already key on
# ``exercise_type``), so the route layer is the only thing that
# fans out per type.
#
# Hard rule #2 (type-level guardrail): ``exercise_type`` is a
# closed 3-way literal on BOTH request and response. Any value
# outside the union is a Pydantic ``ValidationError``, not a
# runtime check downstream — the type system is the gate.
#
# Hard rule #5 (Pydantic v2 validated input): ``grade`` is
# ``Literal[1, 2, 3, 4]`` — out-of-range grades (0, 5, -1) reject
# at the schema layer. ``exercise_id`` carries ``gt=0`` so a 0
# (or negative) id is rejected before it reaches the grader.
# ---------------------------------------------------------------------------

# The single source of truth for the closed 3-way exercise-type
# union. The route layer in ``app.main`` dispatches on this same
# value via a ``match`` statement (Python 3.10+). If a future card
# widens the union (Phase 7+), the only edits are here + the
# ``match`` arms in ``app.main`` + the matching handler functions.
ExerciseType = Literal["cloze", "matching", "comprehension"]


class GradeRequest(BaseModel):
    """Request body for ``POST /exercises/grade``.

    The grader (5.3) uses ``exercise_id`` to look up the FSRS card
    that backs this exercise — for the cloze kind, the cloze
    generator (4.2) embeds ``answer_word_id`` in the
    ``ClozeExercise`` payload, so 5.3 derives the ``word_id``
    from the card row, not the request. For matching /
    comprehension (Phase 6.6), the same derivation holds: the
    card row keys on ``word_id``, and the exercise type only
    changes the trace span name and the ``grade_logs.exercise_type``
    label — the FSRS scheduling path is exercise-type-agnostic.

    The request body carries the ``grade`` score
    (1=Again, 2=Hard, 3=Good, 4=Easy) only; the rest of the
    snapshot is reconstructed from the card row + the Langfuse span.
    """

    exercise_id: int = Field(..., gt=0)
    exercise_type: ExerciseType
    grade: Literal[1, 2, 3, 4]


class GradeResponse(BaseModel):
    """Response body for ``POST /exercises/grade``.

    Returns the post-grade snapshot: when the next review is due,
    the FSRS card state (1=Learning / 2=Review / 3=Relearning),
    and the two scalar params the Langfuse span will surface
    (``stability``, ``difficulty``). The ``trace_id`` is the
    Langfuse span id when keys are set, ``None`` otherwise
    (graceful-degradation path).

    ``exercise_type`` mirrors the request's literal — it's
    repeated on the wire so the client can confirm which handler
    served the call without re-deriving from the trace.

    The leading ``graded: Literal[True] = True`` discriminator is
    forward-leaning: when 5.3 / 5.4 evolve to return a richer
    payload (e.g. a 202 for "queued"), a new response model can
    introduce ``graded: Literal[False]`` and a tagged-union on
    the wire. Phase 5 / Phase 6 only ship the True branch.
    """

    graded: Literal[True] = True
    exercise_id: int
    exercise_type: ExerciseType
    next_due_at: datetime
    card_state: int  # 1/2/3
    stability: float
    difficulty: float
    trace_id: str | None


# ---------------------------------------------------------------------------
# Phase 5.4 — ``/exercises/due`` response (card t_e8548d6d)
#
# Same cloze payload as ``POST /exercises/cloze`` (Phase 4.2), plus one
# boolean so the frontend can distinguish the two return modes:
#
# - ``due_from_fsrs=True``  — the picked word had a row in ``fsrs_cards``
#   with ``due_date <= now()``. The user has graded this word before and
#   the FSRS scheduler says it's time for another review. The frontend
#   should treat this as "you've seen this word before" and surface the
#   usual grade buttons (the Phase 5.5 grading surface — not 5.4's
#   concern).
# - ``due_from_fsrs=False`` — no card was due; the route picked a fresh
#   word from the corpus (one with no ``fsrs_cards`` row yet) and
#   created a fresh Learning row inline. The user is seeing this word
#   for the first time. The frontend can still grade it (the next
#   ``POST /exercises/grade`` will see the row); the flag is just a
#   hint that "this is new" so the UI can choose a different empty-state
#   message or starter animation.
#
# The boolean is locked here on the wire so a Phase 6 / Phase 7
# frontend can branch on it without re-deriving the heuristic from the
# underlying row count.
# ---------------------------------------------------------------------------


class ClozeDueExerciseOut(ClozeExerciseOut):
    """Response shape for ``GET /exercises/due``.

    Inherits every field from ``ClozeExerciseOut`` and adds the
    ``due_from_fsrs`` discriminator. Pydantic v2 subclassing with
    extra fields is the canonical extension pattern; FastAPI's
    ``response_model=...`` accepts the subclass and serialises the
    merged field set.
    """

    due_from_fsrs: bool = Field(
        ...,
        description=(
            "True if the picked word had an existing fsrs_cards row "
            "that was due (FSRS-driven re-grade). False if the route "
            "picked a fresh corpus word with no fsrs_cards row yet "
            "and created a new Learning row inline (first encounter)."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 6.4 — Shared exercise fields + comprehension wire shape
# (card t_8556fc97)
#
# Phase 6.4 lands the comprehension exercise type. The shared wire
# fields documented in ``docs/PHASE-6.md`` §"The exercise-type wire"
# live on ``BaseExerciseFields`` so Phase 6.2 (matching), 6.4
# (comprehension), and the future Phase 6.1 (cloze-with-RAG) can all
# subclass a single base. This keeps ``exercise_type`` and the
# activity-boundary metadata in one place — the alternative (each
# type re-declaring the same fields) would mean a future schema
# bump needs three separate edits and a per-type test.
#
# Note: the shared fields here are the **server-side** wire contract
# only. The actual generator Pydantic models
# (``app.cloze.ClozeExercise``, ``app.comprehension.ComprehensionExercise``)
# are separate — the generator side is the *instructor* contract (what
# the LLM is asked to produce); the schema side is the *wire* contract
# (what the route returns). The split mirrors the Phase 4.2 /
# Phase 5.2 pattern: ``app.cloze`` owns the generator, ``app.schemas``
# owns the wire.
# ---------------------------------------------------------------------------


class BaseExerciseFields(BaseModel):
    """Fields shared by every exercise-type wire response.

    The comprehension response (6.4) subclasses this directly. The
    matching response (6.2) will subclass this too — when 6.2 lands
    the change is ``class MatchingExerciseOut(BaseExerciseFields)``
    plus the matching-specific fields, and the ``exercise_type``
    discriminator on the subclass narrows to ``Literal["matching"]``.

    Pydantic v2 makes subclassing a clean extension path: each
    concrete type re-declares ``exercise_type`` with a narrower
    Literal so the wire surface carries the precise discriminator
    value, not the union.
    """

    exercise_type: Literal["cloze", "matching", "comprehension"]
    target_word_id: int = Field(
        ...,
        description=(
            "FK to words.id of the target word the exercise was "
            "built around. Same id re-appears on the grade_logs row "
            "when the user grades this exercise."
        ),
    )
    prompt_template_version: str = Field(
        ...,
        description=(
            "Module constant ('comprehension-v1' for 6.4, 'match-v1' "
            "for 6.2, 'cloze-v1' for 4.2). Bumped on prompt change; "
            "used as the A/B key by the Phase 6.7 Ragas eval runner."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 6.4 — Comprehension exercise request / response
# (card t_8556fc97)
#
# Mirrors ``app.comprehension.ComprehensionExercise`` field-for-field,
# the same split as Phase 4.2's ``ClozeExerciseOut`` /
# ``app.cloze.ClozeExercise``: the *generator* contract on the
# instructor side, the *wire* contract on the response_model side.
#
# Hard rule #3 (Phase 6 hard rules): three exercise types only.
# ``exercise_type: Literal["comprehension"] = "comprehension"`` on
# the response narrows the discriminator — the union on
# ``BaseExerciseFields`` is wider so the base model is reusable
# across the three concrete types.
#
# Hard rule #1: ``ComprehensionGenerateRequest.enable_rag: bool = False``
# by default. RAG-on stays opt-in. The route layer in 6.5 will pass
# this through to ``generate_comprehension``. There is no ``count``
# knob on this type — comprehension generates one passage + one
# question per call (mirroring cloze, not matching).
# ---------------------------------------------------------------------------


ComprehensionChoice = Literal["A", "B", "C", "D"]


class ComprehensionExerciseOut(BaseExerciseFields):
    """Response shape for ``POST /exercises/comprehension`` (Phase 6.5).

    The comprehension wire surface: a 3-5 sentence German passage
    on the target word's topic, a multiple-choice question with
    four options A-D, the correct answer key, and a one-sentence
    rationale explaining the design.

    Field bounds (locked by the card body):

    - ``passage``: 3-5 sentences, 20..600 chars. The LLM prompt's
      prohibitions are load-bearing here — without them the model
      drifts toward longer, multi-paragraph passages that the
      frontend can't render in the study-session card.
    - ``question``: 5..300 chars. The comprehension prompt asks
      for ONE question, not a battery — the frontend renders the
      passage above, the question below, and the four choices in
      a grid.
    - ``choices``: all four keys A/B/C/D required, each 1..200 chars.
      The Pydantic ``min_length=4, max_length=4`` on the dict
      enforces the four-options invariant; ``min_length=1,
      max_length=200`` on each value bounds the choice text. A
      missing key is a validation error, not a default — a future
      maintainer who forgets one of the keys gets a 422 on the
      dead-letter path.
    - ``correct_choice``: ``Literal["A", "B", "C", "D"]`` — the
      answer key, not the index. The frontend uses it to mark the
      correct answer after the user submits.
    - ``rationale``: 1..400 chars. One sentence explaining the
      distractor design — what semantic axis separates the correct
      answer from the wrong ones, so a hand-reviewer can verify the
      model isn't a coin flip.

    ``exercise_id`` is the server-minted per-generation id (the
    same shape 6.2 / 6.3 ship on the matching wire). Phase 6.6
    expects to round-trip the same id on the ``grade_logs`` row so
    the Ragas join in 6.7 is deterministic; adding the field here
    is what 6.5 ships so 6.6 can dispatch on it.
    """

    exercise_type: Literal["comprehension"] = "comprehension"
    exercise_id: int = Field(
        ...,
        description=(
            "Server-minted per generation: "
            "int.from_bytes(os.urandom(8), 'big', signed=True). "
            "The same id re-appears on the grade_logs row for the "
            "same exercise so the Ragas join is deterministic "
            "(Phase 6.7 follow-up). Mirrors the matching wire shape "
            "(6.2 / 6.3) and the GradeRequest.exercise_id: int "
            "discriminator on the write side."
        ),
    )
    passage: str = Field(
        ...,
        min_length=20,
        max_length=600,
        description=(
            "3-5 sentence German passage on the target word's topic. "
            "Bounded 20..600 chars so the frontend can render the "
            "passage in a single study-session card without scrolling."
        ),
    )
    question: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description=(
            "ONE multiple-choice question whose answer is grounded in "
            "the passage. The frontend renders the question below the "
            "passage and the four choices in a grid."
        ),
    )
    choices: Dict[ComprehensionChoice, str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "All four keys A/B/C/D required. Each value 1..200 chars. "
            "Pydantic enforces the key set; a missing key is a 422."
        ),
    )
    correct_choice: ComprehensionChoice = Field(
        ...,
        description=(
            "The correct answer key. NOT an index — the frontend "
            "uses this directly to mark the right answer."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "One sentence (1..400 chars) explaining the distractor "
            "design — what semantic axis separates the correct "
            "answer from the three wrong ones."
        ),
    )

    @field_validator("choices")
    @classmethod
    def _validate_choices(cls, v: Dict) -> Dict:
        """Per-value length validation on ``choices``.

        Pydantic v2's ``min_length`` / ``max_length`` on a
        ``dict`` field validate the *number of keys* (4 here),
        not the per-value length. This validator closes that
        gap: each value must be a non-empty string bounded
        ``[1, 200]`` chars. The generator side
        (``app.comprehension.ComprehensionExercise``) has the
        same validator; the wire side mirrors it so the
        two surfaces stay byte-equivalent on validation
        behaviour.
        """
        if not isinstance(v, dict):
            raise ValueError("choices must be a dict mapping A/B/C/D to a string")
        for key, value in v.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"choices[{key!r}] must be a string; got {type(value).__name__}"
                )
            if len(value) < 1:
                raise ValueError(
                    f"choices[{key!r}] must be at least 1 char; got {len(value)}"
                )
            if len(value) > 200:
                raise ValueError(
                    f"choices[{key!r}] must be at most 200 chars; got {len(value)}"
                )
        return v


class ComprehensionGenerateRequest(BaseModel):
    """Request body for ``POST /exercises/comprehension`` (Phase 6.5).

    Comprehension doesn't have a ``count`` knob (one passage + one
    question per call, mirroring cloze). The only field is the
    RAG-on opt-in flag, same shape as the cloze request (6.1) and
    the matching request (6.2).
    """

    enable_rag: bool = Field(
        default=False,
        description=(
            "Opt-in flag for retrieval-augmented prompting. When "
            "True, the comprehension generator calls /retrieve for "
            "the target word and embeds the chunks in the user-side "
            "JSON. When False (default), the prompt template is "
            "byte-for-byte identical to the no-RAG path — git-diff "
            "test asserts this in test_comprehension.py."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 6.2 / 6.3 — Matching exercise request / response
# (cards t_ddaf9cf9, t_39d85400)
#
# Wire contract for ``POST /exercises/match`` (the route ships in 6.3).
# Mirrors the comprehension wire shape (6.4 / 6.5): the response
# subclasses ``BaseExerciseFields`` and adds matching-specific fields.
# ``exercise_id`` lives on BOTH the matching and comprehension
# subclasses — it was in 6.2's narrower ``BaseExerciseFields`` but
# 6.4 widened the base to the 3-way Literal and dropped per-type
# bookkeeping from the shared mixin. Phase 6.5 added it back on
# ``ComprehensionExerciseOut`` so both wire shapes round-trip the
# same id into ``/exercises/grade`` (Phase 6.6's ``ExerciseType``
# dispatch) and into the ``grade_logs`` row for Ragas join
# determinism (Phase 6.7 follow-up).
# ---------------------------------------------------------------------------

# Re-export ``MatchingPair`` from ``app.match`` so the wire schema and
# the generator contract reference the same Pydantic model. The split
# between ``app.schemas`` and ``app.match`` is a module-organisation
# convention; the Pydantic model itself must be the same so a
# response built from the generator's ``MatchingExercise`` validates
# against the wire's ``MatchingExerciseOut`` without an adapter step.
#
# The wire-side class adds Pydantic ``Field(...)`` descriptions that
# don't appear on the generator-side class (so a frontend consumer
# reading the OpenAPI doc gets the description text). The shape —
# ``left_word_id`` / ``right_word_id`` / ``right_kind`` — is identical,
# so the two classes are interchangeable for instance-vs-instance
# assignment.
from app.match import MatchingPair  # type: ignore  # noqa: E402,F401  (re-export)


class MatchingExerciseOut(BaseExerciseFields):
    """Response shape for ``POST /exercises/match`` (Phase 6.3 route).

    ``pairs`` is always in ``[MATCH_MIN_COUNT, MATCH_MAX_COUNT]`` — the
    Pydantic model ``MatchingExercise`` in ``app.match`` enforces the
    same bounds on the generator side, so the wire is the mirror of
    the generator contract.

    ``exercise_id`` is server-minted per generation
    (``int.from_bytes(os.urandom(8), "big", signed=True)``). The same
    id re-appears on the ``grade_logs`` row for the same exercise so
    the Ragas join is deterministic (Phase 6.7 follow-up). Mirrors
    the ``GradeRequest.exercise_id: int`` discriminator on the write
    side.
    """

    exercise_type: Literal["matching"] = "matching"
    exercise_id: int = Field(
        ...,
        description=(
            "Server-minted per generation: "
            "int.from_bytes(os.urandom(8), 'big', signed=True). "
            "The same id re-appears on the grade_logs row for the "
            "same exercise so the Ragas join is deterministic."
        ),
    )
    pairs: list[MatchingPair] = Field(
        ...,
        min_length=2,
        max_length=8,
        description=(
            "Match pairs the user connects (left -> right). Length "
            "is bounded in [MATCH_MIN_COUNT=2, MATCH_MAX_COUNT=8] by "
            "the generator; the wire constraint mirrors it so a "
            "validation drift surfaces as 422, not as a runtime "
            "shape mismatch."
        ),
    )


# These bounds mirror the generator's hard-coded module constants in
# ``app.match`` (Hard rule #9: type-level guardrails, not env). The
# Pydantic field uses them directly so a drift between the two
# modules is a test failure, not a silent footgun.
class MatchGenerateRequest(BaseModel):
    """Request body for ``POST /exercises/match`` (Phase 6.3 route).

    Mirrors the ``GradeRequest`` shape (Phase 5.2): minimal request
    body, the server picks the target word via ``select_target_word``
    and builds the rest. The two knobs the caller can tweak are
    ``count`` (how many pairs to generate) and ``enable_rag``
    (RAG-on is opt-in — Hard rule #1).
    """

    count: int = Field(
        default=4,
        ge=2,
        le=8,
        description=(
            "How many match pairs to generate. Must be in [2, 8]. "
            "Default 4. Out-of-range values are rejected at the "
            "Pydantic layer (422)."
        ),
    )
    enable_rag: bool = Field(
        default=False,
        description=(
            "Opt-in flag: True augments the prompt with retrieval "
            "chunks from the /retrieve endpoint. False (default) "
            "keeps the prompt byte-for-byte identical to the "
            "no-RAG fixture so the offline eval stays reproducible."
        ),
    )
