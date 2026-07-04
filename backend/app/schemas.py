from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
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


# Phase 6.1 (card t_616cc266) — shared exercise metadata fields
# surfaced on every exercise response. Defined here as a Pydantic
# mixin so the cloze / matching / comprehension response models can
# all inherit it without a deep class hierarchy. The Phase 6.2+
# cards extend this union (``exercise_type: Literal["cloze",
# "matching", "comprehension"]``) and the per-exercise models add
# only their type-specific fields.
#
# Field roster (locked by ``docs/PHASE-6.md`` §"The exercise-type
# wire"):
#
# - ``exercise_type``: wire discriminator (Phase 6 widens from
#   ``Literal["cloze"]`` to the 3-way union; Phase 6.1 ships the
#   cloze-only branch because matching + comprehension aren't
#   built yet).
# - ``target_word_id``: FK to ``words.id`` of the answer.
# - ``prompt_template_version``: A/B eval key. Always equals the
#   module constant for production generations.
# - ``enable_rag``: echoed from the request. ``False`` for cloze
#   when the caller didn't pass ``enable_rag=True``; ``False`` for
#   matching / comprehension (Phase 6.1 only ships the cloze
#   opt-in; the other types default to ``False`` until their own
#   cards extend the contract).
# - ``trace_id``: Langfuse span id, ``None`` when keys are unset
#   (graceful degradation).
# - ``latency_ms``: end-to-end wall-clock from the activity
#   boundary.
class BaseExerciseFields(BaseModel):
    """Shared metadata on every exercise response.

    Pydantic v2 mixin pattern: a ``BaseModel`` with no required
    fields acts as a field-bag that other models can subclass.
    Subclasses call ``class ClozeExerciseOut(BaseExerciseFields):
    ...`` and inherit every field listed here.
    """

    exercise_type: Literal["cloze", "matching", "comprehension", "idiom"] = Field(
        default="cloze",
        description=(
            "Wire discriminator. Phase 6.1 ships the cloze-only "
            "branch; matching + comprehension widen this literal in "
            "6.2 / 6.4; **Phase 8.3 (card t_fa86ac58) adds "
            "``\"idiom\"``** as the 4th branch. Hard rule: widening "
            "is wire-level + additive only; existing callers "
            "parsing ``\"cloze\"`` / ``\"matching\"`` / "
            "``\"comprehension\"`` see no change. The opposite "
            "direction — narrowing — would silently break callers; "
            "Phase 8 forbids it."
        ),
    )
    target_word_id: int = Field(
        ...,
        description=(
            "FK to words.id of the answer / central token for this "
            "exercise."
        ),
    )
    prompt_template_version: str = Field(
        ...,
        description=(
            "Should always equal the module constant for production "
            "generations. Used as an A/B key by Ragas (Phase 6.7)."
        ),
    )
    enable_rag: bool = Field(
        default=False,
        description=(
            "Echoed from the request. Phase 6.1 — only the cloze "
            "endpoint honours ``enable_rag=True``; matching + "
            "comprehension are non-RAG in this card."
        ),
    )
    trace_id: Optional[str] = Field(
        default=None,
        description=(
            "Langfuse span id for the generation. ``None`` when "
            "LANGFUSE_*_KEY env vars are unset (graceful "
            "degradation — the activity still succeeds)."
        ),
    )
    latency_ms: int = Field(
        ...,
        description=(
            "End-to-end wall-clock from the activity boundary to "
            "the response. Includes the LLM round-trip; recorded "
            "for offline A/B comparison."
        ),
    )


class ClozeExerciseOut(BaseExerciseFields):
    """Response shape for ``POST /exercises/cloze``.

    Inherits the shared ``BaseExerciseFields`` (Phase 6.1) and adds
    the cloze-specific fields from Phase 4.2. ``target_word_id`` is
    a duplicate of ``answer_word_id`` on purpose: the shared field
    is the canonical wire name for cross-exercise-type consumers
    (Phase 9's study-session mixer), while ``answer_word_id`` is
    kept for Phase 5.x backward compatibility with the cloze-only
    clients built against the Phase 4.2 wire shape.

    **Discriminator lock.** ``exercise_type`` is narrowed here
    from the base class's 3-way ``Literal["cloze", "matching",
    "comprehension"]`` down to the cloze-only branch
    (``Literal["cloze"]``). Trying to set
    ``exercise_type="matching"`` on a ``ClozeExerciseOut`` is a
    ``ValidationError`` — the type system is the gate (Phase 6
    plan §"Hard rules" #1). The matching / comprehension cards
    (6.2 / 6.4) introduce their own subclasses that narrow the
    discriminator to their own branch.
    """

    # Re-declare ``exercise_type`` here so it narrows the
    # ``Literal`` to ``"cloze"`` only. Pydantic v2 honours the
    # narrowed annotation on the subclass; the base class's
    # broader union doesn't bleed through.
    exercise_type: Literal["cloze"] = Field(
        default="cloze",
        description=(
            "Wire discriminator. Always ``\"cloze\"`` on this "
            "response — matching / comprehension are 6.2 / 6.4."
        ),
    )

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
    partner_translation: str | None = Field(
        default=None,
        description=(
            "Phase 7.4 — bilingual read-through. Populated from "
            "``collocations.partner_lemma`` for the target word "
            "when the request's ``partner_lang=\"en\"`` AND a "
            "collocation row exists for the target. ``None`` when "
            "``partner_lang=\"de\"`` (default) or no row exists. "
            "Mirrors the matching wire field on the same flag."
        ),
    )

    @model_validator(mode="after")
    def _target_matches_answer(self) -> "ClozeExerciseOut":
        """Cross-field check: ``target_word_id`` must equal
        ``answer_word_id`` for cloze. The shared field exists for
        cross-exercise-type consumers; on cloze the two are
        semantically the same — a drift would be a bug.

        Uses ``model_validator(mode="after")`` (Pydantic v2)
        because the check needs both fields populated; the
        ``field_validator`` machinery runs per-field and doesn't
        have a reliable view of sibling fields without a
        ``model_validator`` wrapper.
        """
        if self.target_word_id != self.answer_word_id:
            raise ValueError(
                f"target_word_id={self.target_word_id} must equal "
                f"answer_word_id={self.answer_word_id} for cloze exercises"
            )
        return self


# The single source of truth for the 2-way partner-language union
# (Phase 7.4 / card t_d621bb4f). The Pydantic ``Literal`` is the
# wire-level guardrail — anything outside ``{"de", "en"}`` is a
# 422 at the request body layer, NOT a runtime check downstream
# (Hard rule H4). The matching and cloze request schemas both
# reference this alias; a future widening (``"fr"``) is a single
# edit here plus the matching accept-arm in the generator.
#
# This union is deliberately narrow. PHASE-7.md §"What is NOT in
# Phase 7" defers ``"fr"`` (and any 3rd partner language) out of
# Phase 7; the comprehension endpoint is left for Phase 8 — it
# doesn't import this alias.
PartnerLang = Literal["de", "en"]


# Phase 6.1 (card t_616cc266) — request body for
# ``POST /exercises/cloze``. The Phase 4.5 endpoint accepted an
# empty body; this card adds the optional ``enable_rag`` field.
# Default ``False`` means existing callers (curl, the Phase 4.5/5.5
# frontend) keep working without any change.
class ClozeGenerateRequest(BaseModel):
    """Request body for ``POST /exercises/cloze``.

    Phase 6.1 — the only field is ``enable_rag`` (default
    ``False``). An empty body ``{}`` parses to
    ``ClozeGenerateRequest(enable_rag=False)`` via Pydantic's
    defaults, so the Phase 4.5 wire contract is preserved.

    Phase 7.3 (card t_bdd6ab24) — the request widens with
    ``collocation: bool = False`` (mirroring Phase 5.6's
    ``enable_rag`` and Phase 6.1's ``enable_rag`` widening
    pattern). When ``True``, the cloze prompt routes through
    Phase 7.2's ``generate_collocation`` and consumes the
    ``collocations`` table for the target word. Default
    ``False`` keeps the Phase 4.2 + 6.1 callers byte-for-byte
    stable (Hard rule #10 / H3) — ``{}`` parses to a request
    identical to the previous contract.

    Phase 7.4 (card t_d621bb4f) — widens with
    ``partner_lang: PartnerLang = "de"``. Default ``"de"``
    keeps the Phase 4.5 / 6.1 / 7.3 wire contract byte-for-byte
    unchanged. When ``partner_lang="en"``, the cloze generator
    stamps ``partner_translation`` onto the response from
    ``collocations.partner_lemma`` (fail-soft). Values outside
    the literal (``"fr"``) are rejected with a 422 (Hard rule
    H4).
    """

    # ``StrictBool`` (Phase 7.3 acceptance) — Pydantic v2's
    # default bool field coerces ``"true"`` (str) and ``1`` (int)
    # to ``True`` automatically. PHASE-7 card body Hard rule #5
    # says the field must be a real Pydantic type AND the spec
    # test list says ``collocation="true"`` → 422. ``StrictBool``
    # opts out of coercion: any non-bool raises
    # ``ValidationError``, FastAPI surfaces as 422.
    enable_rag: bool = Field(
        default=False,
        strict=True,
        description=(
            "Phase 6.1 — opt-in flag for the retrieval-augmented "
            "cloze prompt path. When ``True``, the cloze generator "
            "calls ``/retrieve`` (Postgres + pgvector) and embeds "
            "the top chunks in the user prompt. When ``False`` "
            "(default), the prompt is byte-for-byte identical to "
            "Phase 4.2's — keeps the offline eval reproducible "
            "for A/B comparison."
        ),
    )
    collocation: bool = Field(
        default=False,
        strict=True,
        description=(
            "Phase 7.3 (card t_bdd6ab24) — opt-in flag for the "
            "collocation-cloze prompt path. When ``True``, the "
            "endpoint routes through Phase 7.2's "
            "``app.collocation.generate_collocation`` and the "
            "response payload carries the collocation-specific "
            "fields (``partner_lemma``, ``partner_register``, "
            "``source_corpus``, ``prompt``). When ``False`` "
            "(default), the endpoint produces the standard Phase "
            "6.1 cloze response — ``partner_lemma`` echoes ``None`` "
            "and every standard cloze field is identical to the "
            "Phase 6.1 wire shape (Hard rule #10). Strict bool: "
            "string ``\"true\"`` and integer ``1`` are rejected "
            "with HTTP 422 (Phase 7.3 Hard rule #5)."
        ),
    )
    partner_lang: PartnerLang = Field(
        default="de",
        description=(
            "Phase 7.4 — opt-in bilingual flag. When ``\"en\"``, "
            "the response's ``partner_translation`` field is "
            "populated from ``collocations.partner_lemma`` for "
            "the target word (when such a row exists). When "
            "``\"de\"`` (default), ``partner_translation`` is "
            "always ``None`` — bilingual is opt-in (Hard rule "
            "H3). Values outside the literal (``\"fr\"``) are "
            "rejected at the Pydantic layer with a 422 (Hard "
            "rule H4)."
        ),
    )


# Phase 7.3 (card t_bdd6ab24) — ``collocation`` echo and
# ``partner_lemma`` optional field on ``ClozeExerciseOut``.
#
# The endpoint's response is a discriminated shape:
# ``collocation=False`` returns a ``ClozeExerciseOut`` with
# ``partner_lemma=None`` (Phase 6.1 default cloze shape, plus
# the new echo fields); ``collocation=True`` returns a
# ``CollocationExerciseOut`` (which already has ``partner_lemma``
# as a required field, mirror of ``collocations.partner_lemma``).
# The two subclasses share ``BaseExerciseFields`` (Phase 6.1) and
# each narrows ``exercise_type`` to ``Literal["cloze"]`` —
# PHASE-7 gotcha #5 keeps the wire discriminator ``"cloze"`` for
# both branches (collocation-cloze is a *variant*, not a new
# exercise type literal).
#
# We add two new fields to ``ClozeExerciseOut`` rather than
# building a wrapper response_model so the existing SPA keeps
# working unchanged:
#
# - ``collocation``: ``bool = False`` — discriminator echoed
#   from the request. Always serialised by Pydantic v2's default
#   inclusion policy. ALWAYS ``False`` on ``ClozeExerciseOut``
#   (it IS the standard branch) and ALWAYS ``True`` on
#   ``CollocationExerciseOut`` (it IS the collocation branch)
#   once Phase 7.3 lands. The Phase 4.2 / 6.1 SPA sees a new
#   field on the response that it can ignore; reading the field
#   is opt-in.
# - ``partner_lemma``: ``Optional[str] = None`` — populated only
#   when ``collocation=True``. Always ``None`` on the standard
#   branch.
#
# These two fields make the cloze endpoint a discriminated
# response on the wire:
#
#     { ..., "collocation": false, "partner_lemma": null }
#     { ..., "collocation": true,  "partner_lemma": "treffen" }
#
# Hard rule #10 (H10) — the no-flag branch must produce a
# prompt-bytes-identical result to Phase 6.1. We measure this
# via the generator's rendered user-prompt hash, not the JSON
# wire shape (the wire SHAPE has the two new echo fields added
# by Phase 7.3; the JSON ordering is alphabetical via Pydantic
# v2's default, so a dict hash on the prompt bytes is the
# stable invariant). Tests assert the prompt-bytes hash.
class ClozeGenerateResponse(BaseModel):
    """Single-class response body for ``POST /exercises/cloze``.

    Phase 7.3 (card t_bdd6ab24) — the route layer returns either
    ``ClozeExerciseOut``-shaped data (``collocation=False``) or
    ``CollocationExerciseOut``-shaped data (``collocation=True``)
    via a free-form ``payload: Dict[str, Any]`` bag. The two
    always-present top-level fields are the discriminator + the
    collocation partner word:

    - ``collocation`` (bool, required) — echoed from the request.
      ``True`` on the collocation-cloze branch, ``False`` on the
      standard cloze branch.
    - ``partner_lemma`` (``str | None``, default ``None``) —
      populated only when ``collocation=True``; ``None`` on the
      standard cloze branch (Hard rule #10).

    Why a single-class wrapper rather than a Pydantic
    ``TaggedUnion``? FastAPI's ``response_model`` validates the
    returned value against the schema, and a discriminated union
    ties the schema to one branch. The two branches are
    disjoint field rosters (``sentence_with_blank`` vs ``prompt``,
    ``distractors`` vs ``partner_register``); a single union class
    with ``extra='allow'`` lets the route stamp either branch
    while validating the shared discriminator fields.

    Wire shape (after Pydantic serialisation, Phase 7.3):

        { "collocation": false, "partner_lemma": null, <ClozeExerciseOut fields>... }
        { "collocation": true,  "partner_lemma": "treffen", <CollocationExerciseOut fields>... }

    Tests asserting H10 (Prompt-bytes identical to Phase 6.1)
    normalise the response by stripping ``collocation`` and
    ``partner_lemma`` before hashing — see
    ``tests/test_cloze_collocation_flag.py``.
    """

    collocation: bool = Field(
        ...,
        description=(
            "Phase 7.3 discriminator — ``True`` on the "
            "collocation-cloze branch, ``False`` on the standard "
            "cloze branch. Always present (echoed from the "
            "request, never ``Optional``)."
        ),
    )
    partner_lemma: Optional[str] = Field(
        default=None,
        description=(
            "Collocation partner word the user has to fill in. "
            "``None`` on the standard cloze branch; populated "
            "only on the collocation-cloze branch (Hard rule "
            "#10). Carries ``collocations.partner_lemma`` "
            "verbatim when ``collocation=True``."
        ),
    )
    # The bulk of the response is forwarded as a free-form
    # payload — Pydantic v2 preserves it via ``model_config``.
    # We can't put it on a single class because the two branches
    # have disjoint field rosters (sentence_with_blank vs prompt,
    # distractors vs partner_register); the route layer stamps
    # the appropriate branch dict, and the wire surface accepts
    # either.
    model_config = ConfigDict(extra="allow")  # type: ignore[assignment]


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
# live on ``BaseExerciseFields`` (defined earlier in this file at the
# Phase 6.1 cloze section, with the superset of fields including
# ``enable_rag`` / ``trace_id`` / ``latency_ms``) so Phase 6.1 (cloze),
# Phase 6.2 (matching), Phase 6.4 (comprehension), and any future
# exercise type can subclass a single base. This keeps
# ``exercise_type`` and the activity-boundary metadata in one place —
# the alternative (each type re-declaring the same fields) would mean
# a future schema bump needs three separate edits and a per-type
# test.
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

    Phase 7.4 (card t_d621bb4f) — ``partner_translation`` is the
    bilingual read-through field. It's ``None`` when ``partner_lang``
    on the request is ``"de"`` (default) OR when no collocations row
    exists for the target word. It's the ``partner_lemma`` string
    from the ``collocations`` table when ``partner_lang="en"`` AND a
    collocation row exists for the target word.
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
    partner_translation: str | None = Field(
        default=None,
        description=(
            "Phase 7.4 — bilingual read-through. Populated from "
            "``collocations.partner_lemma`` for the target word when "
            "the request's ``partner_lang=\"en\"`` AND a collocation "
            "row exists for the target. ``None`` when "
            "``partner_lang=\"de\"`` (default) or no row exists."
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
    and builds the rest. The knobs the caller can tweak are
    ``count`` (how many pairs to generate), ``enable_rag``
    (RAG-on is opt-in — Hard rule #1), and Phase 7.4's
    ``partner_lang`` (bilingual exercise is opt-in — Hard rule H3).
    The default ``partner_lang="de"`` keeps the Phase 6.2 / 6.3
    wire contract byte-for-byte unchanged (Hard rule H10).
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
    partner_lang: PartnerLang = Field(
        default="de",
        description=(
            "Phase 7.4 — opt-in bilingual flag. When ``\"en\"``, "
            "the response's ``partner_translation`` field is "
            "populated from ``collocations.partner_lemma`` for "
            "the target word (when such a row exists). When "
            "``\"de\"`` (default), ``partner_translation`` is "
            "always ``None``. Values outside the literal "
            "(``\"fr\"``) are rejected at the Pydantic layer "
            "with a 422 (Hard rule H4)."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 7.1 (card t_96ab949e) — Collocations + PrepositionalObjects schemas
# ---------------------------------------------------------------------------
#
# Wire-level shapes for the two new read-only corpus tables. The
# underlying SQLAlchemy models live in ``app.models.Collocation`` +
# ``app.models.PrepositionalObject``; the schemas here are the
# Pydantic v2 outbound views (Hard rule #4 — Pydantic v2 + Alembic).
#
# The literal enums (``register``, ``source_corpus``, ``case``) are
# the **wire-level guardrails** that prevent typos at the seed
# boundary (PHASE-7.md gotcha #12). The DB column is a loose String
# (dialect-agnostic), so a raw-SQL INSERT could in principle smuggle
# a misspelled value; this Pydantic layer closes that gap. Any future
# schema change that adds a 4th literal must widen both the Pydantic
# type AND the seed-row validator AND the test matrix.

# Source-corpus enum: shared by both tables (Hard rule #4).
SourceCorpus = Literal["dwds", "wiktionary", "manual"]


class CollocationOut(BaseModel):
    """Phase 7.1 — outbound view of a single ``collocations`` row.

    Wire-level fields match the SQLAlchemy model column-for-column.
    The ``created_at`` field is exposed because some Phase 9
    audit-style endpoints (not in this card) may want to filter by
    insertion batch.
    """

    # Pydantic v2 idiom (replaces the legacy ``class Config`` pattern).
    # ``from_attributes=True`` lets ``CollocationOut.model_validate(row)``
    # round-trip from a SQLAlchemy ORM instance without a manual dict
    # conversion. Mirrors the legacy ``Config`` blocks on the older
    # Phase 0/1 schemas in this file.
    model_config = ConfigDict(from_attributes=True)

    collocation_id: int
    headword_id: Optional[int] = None
    partner_lemma: str
    frequency_score: float
    # Field name ``register_label`` instead of ``register``: the
    # latter shadows ``BaseModel.register`` (a Pydantic v2 method
    # used internally for hook registration). ``register_label``
    # matches the SQLAlchemy model 1:1 visually while staying clear
    # of the v2 API surface.
    register_label: Literal["formal", "neutral", "colloquial"] = Field(
        ..., alias="register", validation_alias="register",
        serialization_alias="register",
        description=(
            "Register of the collocation: formal / neutral / "
            "colloquial. Pydantic field name is "
            "``register_label`` to avoid shadowing "
            "``BaseModel.register``; the wire-level JSON key is "
            "``register`` for client compatibility."
        ),
    )
    source_corpus: SourceCorpus
    created_at: datetime


class CollocationListOut(BaseModel):
    """Phase 7.1 — list wrapper for ``CollocationOut`` rows.

    The Phase 7.2 collocation-cloze generator will query this list
    to enrich the cloze prompt (Hard rule #11 — opt-in flag only).
    For now the shape is the canonical envelope; later phases may
    add a ``partner_translation`` field when ``partner_lang="en"``
    lands.
    """

    model_config = ConfigDict(from_attributes=True)

    items: List[CollocationOut]
    total: int


class CollocationSeedRow(BaseModel):
    """Phase 7.1 — inbound shape for a single seed-file row.

    Used by ``backend/scripts/seed_collocations.py`` to validate
    each JSON-Lines row before INSERT. Deliberately omits
    ``collocation_id`` (autoincrement PK) and ``created_at``
    (server-side default) — the seed input is the user-authored
    payload, not the round-trip view.

    The literal enums are the wire-level guardrails (PHASE-7.md
    gotcha #12): a typo'd ``register`` or ``source_corpus`` value
    is caught here, at the seed boundary, not later when a row
    silently propagates into the cloze generator (PHASE-7.2).
    """

    model_config = ConfigDict(from_attributes=True)

    headword_id: Optional[int] = None
    partner_lemma: str
    frequency_score: float = Field(..., ge=0.0, le=1.0)
    # Same rename pattern as ``CollocationOut``: ``register_label``
    # avoids shadowing Pydantic v2's ``BaseModel.register`` method;
    # the wire-level JSON key is ``register``.
    register_label: Literal["formal", "neutral", "colloquial"] = Field(
        ..., alias="register", validation_alias="register",
        serialization_alias="register",
    )
    source_corpus: SourceCorpus


class PrepositionalObjectOut(BaseModel):
    """Phase 7.1 — outbound view of a single ``prepositional_objects`` row.

    Wire-level fields match the SQLAlchemy model column-for-column.
    ``example_sentence`` is exposed because it's the primary teaching
    surface — the Phase 7.2 collocation module may render the
    sentence with the preposition blanked out as a cloze variant.
    """

    model_config = ConfigDict(from_attributes=True)

    prepositional_object_id: int
    verb_lemma: str
    preposition: str
    case: Literal["Akk", "Dat", "Gen"]
    example_sentence: str
    frequency_score: float
    source_corpus: SourceCorpus
    created_at: datetime


class PrepositionalObjectListOut(BaseModel):
    """Phase 7.1 — list wrapper for ``PrepositionalObjectOut`` rows.

    The Phase 7.2 module's prepositional-object cloze variant
    consumes this list. Same envelope shape as
    ``CollocationListOut`` so future endpoints can swap one for
    the other without changing client code.
    """

    model_config = ConfigDict(from_attributes=True)

    items: List[PrepositionalObjectOut]
    total: int


class PrepositionalObjectSeedRow(BaseModel):
    """Phase 7.1 — inbound shape for a single seed-file row.

    Used by ``backend/scripts/seed_prepositional_objects.py`` to
    validate each JSON-Lines row before INSERT. Deliberately omits
    ``prepositional_object_id`` (autoincrement PK) and ``created_at``
    (server-side default) — the seed input is the user-authored
    payload, not the round-trip view.

    The literal enums are the wire-level guardrails (PHASE-7.md
    gotcha #12): a typo'd ``case`` or ``source_corpus`` value is
    caught here, at the seed boundary, not later when a row
    silently propagates into the cloze generator (PHASE-7.2).
    """

    model_config = ConfigDict(from_attributes=True)

    verb_lemma: str
    preposition: str
    case: Literal["Akk", "Dat", "Gen"]
    example_sentence: str
    frequency_score: float = Field(..., ge=0.0, le=1.0)
    source_corpus: SourceCorpus


# ---------------------------------------------------------------------------
# Phase 7.2 — Collocation-cloze exercise wire shape (card t_ab77bc2b)
#
# Mirrors ``app.collocation.CollocationExercise`` field-for-field,
# the same generator / wire split as Phase 4.2 / 6.2 / 6.4:
#
# - ``app.collocation`` owns the *generator* contract (used by the
#   instructor-wrapped chat call).
# - ``app.schemas`` owns the *wire* contract (used by FastAPI's
#   ``response_model=...`` and the SPA).
#
# Hard rule #1 (Phase 7 plan §"Hard rules"): collocation-cloze is a
# cloze *variant*, NOT a 4th exercise type. The discriminator
# ``exercise_type`` stays ``Literal["cloze"]`` — the same literal
# the regular ``ClozeExerciseOut`` narrows to. The route layer in
# Phase 7.3 will accept a ``collocation: bool = False`` flag on
# ``ClozeGenerateRequest``; when ``True``, the route dispatches to
# ``generate_collocation`` and returns this ``CollocationExerciseOut``
# shape. The base-class ``trace_id`` is the same join key Phase 5.3
# grades use on the ``grade_logs`` row — collocation-cloze is one
# type of cloze row, not a separate grading-log kind.
#
# Hard rule #2 (Pydantic v2 validated output): ``partner_register``
# is a closed 3-way literal. Any value outside the union is a
# Pydantic ``ValidationError`` at generation time, not a runtime
# check downstream — the type system is the gate. ``source_corpus``
# is similarly a closed 3-way literal (PHASE-7 gotcha #12).
#
# Hard rule #6 (Langfuse ``trace_id``): the base-class ``trace_id``
# field carries the same ``grade_logs`` join key Phase 5.3 ships.
# ``collocation: True`` is NOT a wire field — it's a discriminator
# baked into the Langfuse span metadata (the collocation generator
# stamps it on the trace, not on the response, so the SPA's
# response shape stays identical to the regular cloze surface).
# ---------------------------------------------------------------------------

CollocationRegister = Literal["formal", "neutral", "colloquial"]
CollocationSourceCorpus = Literal["dwds", "wiktionary", "manual"]


class CollocationExerciseOut(BaseExerciseFields):
    """Response shape for the collocation-cloze generator (Phase 7.2).

    The collocation-cloze wire surface: a German sentence with a
    single ``___`` blank (the partner_lemma goes in), the target
    word's lemma + English translation for the SPA's header, the
    closed 3-way register label, the closed 3-way source corpus
    label, and an empty-by-default ``retrieval_chunks`` list for
    forward-compatibility with a future Phase 9 RAG-on flag (not
    wired in 7.2 — ``retrieval_chunks=[]`` is the contract today,
    so the SPA can rely on the field's presence even when the
    list is empty).

    Field bounds (locked by ``docs/PHASE-7.md`` §"Concrete cards"
    item 2 and the card body):

    - ``target_lemma``: the German lemma of the target word.
    - ``target_translation_en``: best-effort English gloss from
      ``words.translations`` (may be empty when the corpus row
      doesn't carry one — the SPA treats empty as "no gloss
      available").
    - ``prompt``: 1..400 chars, contains exactly one ``___``.
    - ``partner_lemma``: 1..80 chars (a single German word /
      short phrase).
    - ``register``: ``Literal["formal","neutral","colloquial"]`` —
      Pydantic rejects any other value at validation time. (The
      wire field name is ``partner_register`` — matches the
      storage column name; ``register`` would shadow
      ``BaseModel.model_fields``.)
    - ``source_corpus``: ``Literal["dwds","wiktionary","manual"]`` —
      PHASE-7 gotcha #12, locked so a typo'd source never silently
      passes.
    - ``rationale``: 1..400 chars.
    - ``retrieval_chunks``: empty list ``[]`` in 7.2 (RAG-on is a
      Phase 9 follow-on; the field is reserved now so a future
      wire bump doesn't need to add it).

    **Discriminator lock.** ``exercise_type`` is narrowed here
    from the base class's 3-way ``Literal["cloze","matching",
    "comprehension"]`` down to the cloze-only branch
    (``Literal["cloze"]``). Trying to set
    ``exercise_type="matching"`` on a ``CollocationExerciseOut``
    is a ``ValidationError`` — the type system is the gate
    (Phase 7 plan §"Hard rules" #1).
    """

    # Re-declare ``exercise_type`` here so it narrows the Literal
    # to ``"cloze"`` only. Pydantic v2 honours the narrowed
    # annotation on the subclass.
    exercise_type: Literal["cloze"] = Field(
        default="cloze",
        description=(
            "Wire discriminator. Always ``\"cloze\"`` — "
            "collocation-cloze is a cloze *variant*, not a new "
            "exercise type literal (Phase 7 plan Hard rule #1)."
        ),
    )

    target_lemma: str = Field(
        ...,
        description=(
            "The German lemma of the target word (echoed from "
            "words.word so the SPA doesn't need a second round-trip "
            "to fetch the word row)."
        ),
    )
    target_translation_en: str = Field(
        default="",
        description=(
            "Best-effort English gloss of the target word, "
            "extracted from words.translations. Empty when the "
            "corpus row doesn't carry one — the SPA treats empty "
            "as \"no gloss available\" rather than an error."
        ),
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "German sentence with '___' marking the cloze position. "
            "The LLM must embed partner_lemma verbatim."
        ),
    )
    partner_lemma: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description=(
            "The collocation partner word the user has to fill in. "
            "Mirrors collocations.partner_lemma verbatim."
        ),
    )
    # ``partner_register`` on the wire (matches ``collocations.partner_register``
    # on the storage side and ``app.collocation.CollocationExercise`` on
    # the generator side). Avoids Pydantic's ``register``-shadows-
    # ``BaseModel`` warning.
    partner_register: CollocationRegister = Field(
        ...,
        description=(
            "Register label of partner_lemma. Pydantic rejects "
            "any value outside the closed Literal at validation "
            "time — the type system is the gate (Hard rule #2)."
        ),
    )
    source_corpus: CollocationSourceCorpus = Field(
        ...,
        description=(
            "Provenance of the collocation row. PHASE-7 gotcha #12: "
            "locked to the 3-value enum so a typo'd source never "
            "silently passes."
        ),
    )
    rationale: str = Field(..., min_length=1, max_length=400)
    # Forward-compatibility shim for a Phase 9 RAG-on flag. The
    # field is always ``[]`` in 7.2 — the route layer doesn't
    # accept a ``enable_rag`` flag on the collocation path yet.
    # Adding the field now means a future wire bump doesn't need
    # to add a new key.
    retrieval_chunks: list[dict] = Field(
        default_factory=list,
        description=(
            "Phase 9 placeholder. Always ``[]`` in Phase 7.2 — "
            "the route layer doesn't accept a RAG-on flag on the "
            "collocation path yet. The field is reserved now so a "
            "future wire bump doesn't need to add a new key."
        ),
    )
    partner_lang: PartnerLang = Field(
        default="de",
        description=(
            "Phase 7.4 — opt-in bilingual flag. When ``\"en\"``, "
            "the response's ``partner_translation`` field is "
            "populated from ``collocations.partner_lemma`` for "
            "the target word (when such a row exists). When "
            "``\"de\"`` (default), ``partner_translation`` is "
            "always ``None``. Values outside the literal "
            "(``\"fr\"``) are rejected at the Pydantic layer with "
            "a 422 (Hard rule H4)."
        ),
    )
    partner_translation: Optional[str] = Field(
        default=None,
        description=(
            "Phase 7.4 — English gloss of the partner word, sourced "
            "from ``collocations.partner_lemma`` when ``partner_lang='en'``. "
            "Always ``None`` when ``partner_lang='de'`` (default) or when no "
            "matching collocations row exists (fail-soft)."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 8.4 — Idiom exercise request / response (card t_7c21c3f0)
#
# Wire surface for ``POST /exercises/idiom``. Mirrors the Phase 6.5
# comprehension shape (request + response split) with three Phase 8
# differences:
#
# 1. ``IdiomGenerateRequest`` carries a **required** ``word_id`` field
#    (the caller picks the target word; the comprehension endpoint
#    lets the server pick). The card body rationale: the
#    ``phrases`` table is a curated corpus where each row is
#    attached to a specific anchor word, so the caller has to
#    name it. There's no ``count`` knob — idiom generates ONE
#    exercise per call (mirrors comprehension, not matching).
#
# 2. ``source_attribution`` is a **closed literal** of three values
#    (``"dwds"`` / ``"goethe"`` / ``"schiller"``) or a comma-joined
#    subset (e.g. ``"dwds,goethe"`` when an idiom is attested in
#    both). Each token must be a valid literal value — the
#    ``field_validator`` enforces this so a typoed ``"goeth"`` is
#    a 422 (the type system is the gate, per Phase 6 hard rule #2).
#
# 3. ``BaseExerciseFields.exercise_type`` is widened to include
#    ``"idiom"`` (Phase 8.3 Hard rule #1 — additive only; existing
#    callers parsing the 3-value union see no change). The
#    discriminator is narrowed back to ``Literal["idiom"]`` on
#    ``IdiomExerciseOut.exercise_type`` so an attempt to set
#    ``exercise_type="cloze"`` on an ``IdiomExerciseOut`` is a 422.
#
# Hard rule surface (from card body):
#   - #1 RAG-on is opt-in via ``enable_rag: bool = False`` on the
#     request body (mirrors comprehension + cloze).
#   - #2 ``/retrieve`` consumed as-is by the RAG-on branch of
#     ``generate_idiom`` — the route layer never imports the
#     retrieval helper directly.
#   - #3 Literal widening is wire-level. The 3 existing endpoints
#     (cloze / matching / comprehension) parse as before.
#   - #6 422 on ``source_attribution`` outside the literal is
#     tested explicitly in ``tests/test_idiom_endpoint.py``.
# ---------------------------------------------------------------------------


IdiomSource = Literal["dwds", "goethe", "schiller"]
IdiomFrequencyBand = Literal["high", "mid", "low"]


# Canonical separator for comma-joined source attribution values.
# Locked to ``,`` (no spaces) so the round-trip
# parse / serialise preserves the original tokens — a future
# `set in ["dwds", "goethe"]`-style check on the wire side stays
# trivially implementable.
_IDIOM_SOURCE_SEP: str = ","

# Allowed per-token values (used by the field_validator below).
# Re-exported so tests can assert against the canonical set
# without re-declaring it.
_IDIOM_SOURCE_TOKENS: frozenset[str] = frozenset({"dwds", "goethe", "schiller"})


class IdiomExerciseOut(BaseExerciseFields):
    """Response shape for ``POST /exercises/idiom`` (Phase 8.4 route).

    Wire contract (locked by the card body's "Same as Phase 6.5"
    clause + the Phase 8.4 idiomatic fields):

    - ``exercise_type`` is narrowed to ``Literal["idiom"]`` from
      the Phase 8.3-widened base literal. Setting
      ``exercise_type="cloze"`` on this model is a
      ``ValidationError``.
    - ``word_id`` is required on the request *and* round-trips on
      the response — the caller has chosen the target word, so
      the response echoes it for the frontend's study-session
      log row.
    - ``phrase`` is the German idiom (5..200 chars).
    - ``definition`` is the learner-friendly meaning (1..400 chars;
      the LLM is asked to compress long DWDS definitions, mirroring
      the Phase 6.5 comprehension rationale rule — PHASE-8.md
      gotcha #5).
    - ``example_usage`` is a worked German example sentence
      (5..400 chars).
    - ``cloze_target`` is the idiom phrase with ONE word blanked
      for the cloze-within-idiom variant (Phase 8.4 card body
      item 1). ``None`` when the LLM emits a non-cloze variant;
      always populated when the prompt asks for the cloze variant.
    - ``source_attribution`` is a comma-joined subset of the
      closed ``IdiomSource`` literal (e.g. ``"dwds"`` or
      ``"dwds,goethe"``). The ``field_validator`` enforces the
      per-token constraint — a typoed token is a 422.
    - ``attested_quote`` / ``attested_source`` carry the Goethe /
      Schiller attestation from the Phase 8.2 seed (the readings
      that prove the idiom is in the literary corpus). ``None``
      when the row only has a DWDS source.

    ``exercise_id`` is server-minted per generation
    (``int.from_bytes(os.urandom(8), "big", signed=True)``); the
    same id re-appears on the ``grade_logs`` row for Ragas join
    determinism (Phase 6.6 pattern carries forward).
    """

    exercise_type: Literal["idiom"] = "idiom"
    exercise_id: int = Field(
        ...,
        description=(
            "Server-minted per generation: "
            "int.from_bytes(os.urandom(8), 'big', signed=True). "
            "Mirrors the matching wire shape (6.2 / 6.3) and the "
            "comprehension wire shape (6.4 / 6.5) so a future "
            "/exercises/grade call (Phase 6.6) can dispatch on the "
            "id without caring about the exercise type literal."
        ),
    )
    word_id: int = Field(
        ...,
        description=(
            "FK to words.id of the target word the idiom is "
            "anchored to. Echoed from the request — the caller "
            "picked it, the response confirms it. Phase 8.4 / "
            "IdiomGenerateRequest requires the field; the "
            "response mirrors it for the frontend's study-session "
            "log row."
        ),
    )
    phrase: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description=(
            "The German idiom. 5..200 chars. Bounds locked by the "
            "card body — most idioms are 2-5 words, so 5 chars "
            "rules out accidental single-word submissions and "
            "200 caps any forced long-form explosion."
        ),
    )
    definition: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Learner-friendly definition. 1..400 chars. The LLM "
            "prompt asks for a short learner-friendly "
            "paraphrase; long DWDS dictionary forms are "
            "compressed at generation time. PHASE-8.md gotcha "
            "#5 documents the variance in raw DWDS length."
        ),
    )
    example_usage: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description=(
            "Worked German example sentence. 5..400 chars. "
            "Mirrors cloze's ``sentence_with_blank`` shape "
            "(4.2), but without the ``___`` marker — the idiom "
            "exercise uses ``phrase`` as the cloze surface in a "
            "future Phase 9 UX variant."
        ),
    )
    cloze_target: Optional[str] = Field(
        default=None,
        description=(
            "The idiom phrase with ONE word blanked (e.g. "
            "``'Tomaten auf ___ Augen'``) for the cloze-within-"
            "idiom variant. ``None`` when the prompt asks for "
            "the non-cloze variant — the field is always "
            "populated when cloze is requested. The Phase 9 UX "
            "may render this as the cloze; Phase 8.4's wire only "
            "carries it."
        ),
    )
    source_attribution: str = Field(
        ...,
        description=(
            "Provenance of the idiom row, comma-joined from the "
            "closed ``IdiomSource`` literal. Examples: "
            "``\"dwds\"``, ``\"goethe\"``, ``\"schiller\"``, "
            "``\"dwds,goethe\"``. Per-token validation runs via "
            "_validate_source_attribution; a typoed token like "
            "``\"goeth\"`` is rejected with a 422 — the type "
            "system is the gate (Hard rule #2 of Phase 6)."
        ),
    )
    attested_quote: Optional[str] = Field(
        default=None,
        description=(
            "Attested quote from the Goethe/Schiller corpus "
            "(Phase 8.2). ``None`` when the row is DWDS-only. "
            "The frontend renders the quote in the study-"
            "session card to ground the idiom in literature."
        ),
    )
    attested_source: Optional[str] = Field(
        default=None,
        description=(
            "Work + chapter / page citation for ``attested_quote``. "
            "``None`` when no literary attestation exists for "
            "the row. Phase 9 may add a citation-formatter; "
            "Phase 8.4 carries the raw string from the seed."
        ),
    )
    frequency_band: IdiomFrequencyBand = Field(
        ...,
        description=(
            "Manual frequency bucketing from the Phase 8.1 seed. "
            "``\"high\"`` = top-100 most common, ``\"mid\"`` = "
            "next 100, ``\"low\"`` = the rest. Drives the Phase "
            "9 high-band-first cloze variant."
        ),
    )
    enable_rag: bool = Field(
        default=False,
        description=(
            "Echoed from the request. ``True`` when the caller "
            "passed ``enable_rag=True`` AND the RAG-on branch of "
            "``generate_idiom`` actually injected a neighbour "
            "into the prompt. ``False`` on the no-RAG path "
            "(default) or on the SQLite fallback (the retrieval "
            "helper returns ``[]`` and the prompt stays no-RAG)."
        ),
    )

    @field_validator("source_attribution")
    @classmethod
    def _validate_source_attribution(cls, v: str) -> str:
        """Per-token validation on the comma-joined source.

        Pydantic v2's ``Field(...)`` doesn't support
        ``min_length`` / ``max_length`` on a comma-joined string
        with per-token semantics, so we run a ``field_validator``
        instead.

        The set of allowed tokens is closed (``IdiomSource``
        literal). Empty strings and stray tokens are rejected at
        validation time so a typoed ``"goeth"`` is a 422 on the
        dead-letter path, not a runtime surprise in Phase 9.

        Mirrors the matching ``_validate_choices`` pattern
        (Phase 6.4) — type-level guardrail at the wire boundary.
        """
        if not isinstance(v, str):
            raise ValueError(
                f"source_attribution must be a string; got "
                f"{type(v).__name__}"
            )
        # Empty string is rejected — every row is attributed to
        # at least one source.
        if v == "":
            raise ValueError(
                "source_attribution must be a non-empty comma-"
                "joined subset of {dwds, goethe, schiller}"
            )
        tokens = v.split(_IDIOM_SOURCE_SEP)
        for tok in tokens:
            tok_stripped = tok.strip()
            if not tok_stripped:
                # Consecutive separators or trailing separator
                # (``"dwds,"`` or ``"dwds,,goethe"``) — reject
                # explicitly. A future caller who strips at the
                # client side is encouraged, but the wire form
                # is strict.
                raise ValueError(
                    f"source_attribution tokens must not be empty; "
                    f"got {v!r} (split on {repr(_IDIOM_SOURCE_SEP)})"
                )
            if tok_stripped not in _IDIOM_SOURCE_TOKENS:
                raise ValueError(
                    f"source_attribution token {tok_stripped!r} is "
                    f"not in {sorted(_IDIOM_SOURCE_TOKENS)} "
                    f"(full value: {v!r})"
                )
        # Round-trip check: serialising the parsed tokens with
        # the canonical separator must equal the input value
        # (forwards-compat: rejects ``"dwds; goethe"`` style
        # values up-front instead of silently normalising).
        rebuilt = _IDIOM_SOURCE_SEP.join(t.strip() for t in tokens)
        if rebuilt != v:
            raise ValueError(
                f"source_attribution must be a comma-joined string "
                f"with no whitespace around the separator; got "
                f"{v!r}, expected {rebuilt!r}"
            )
        return v


class IdiomGenerateRequest(BaseModel):
    """Request body for ``POST /exercises/idiom`` (Phase 8.4).

    Shape:

    - ``word_id`` (required) — FK to ``words.id`` of the target
      word the idiom row is anchored to. Mirrors the comprehension
      request's missing-field shape (``word_id`` is idiomatic to
      the idiom corpus; cloze / comprehension let the server pick
      the word, but the idiom table has per-word curation so the
      caller has to pick).
    - ``enable_rag`` (default ``False``) — opt-in flag for the
      RAG-on branch of ``generate_idiom``. When ``True``, the
      generator fetches the top-1 nearest-neighbour from the
      ``phrases`` table by embedding cosine and embeds its
      ``definition`` + ``attested_quote`` in the prompt.
      When ``False`` (default), the prompt is byte-for-byte
      identical to the no-RAG variant (Hard rule #2 of Phase 6).

    There is **no** ``count`` knob — idiom generates one exercise
    per call (mirrors comprehension, not matching).
    """

    word_id: int = Field(
        ...,
        ge=1,
        description=(
            "FK to words.id of the target word. Required: the "
            "``phrases`` table is per-word curated (Phase 8.1) "
            "so the caller must pick the anchor word. A "
            "``word_id`` with no ``phrases`` row surfaces as 404 "
            "in the route layer."
        ),
    )
    enable_rag: bool = Field(
        default=False,
        description=(
            "Opt-in flag for retrieval-augmented prompting. "
            "When ``True``, the generator fetches the top-1 "
            "nearest-neighbour from the ``phrases`` table by "
            "embedding cosine and embeds its ``definition`` + "
            "``attested_quote`` in the prompt. Falls back to the "
            "no-RAG variant on any retrieval / embedding failure. "
            "When ``False`` (default), the prompt is the no-RAG "
            "variant — byte-for-byte stable."
        ),
    )

