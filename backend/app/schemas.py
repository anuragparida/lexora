from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict


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
