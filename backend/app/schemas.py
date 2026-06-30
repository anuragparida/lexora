from datetime import datetime
from pydantic import BaseModel, Field, field_validator
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
