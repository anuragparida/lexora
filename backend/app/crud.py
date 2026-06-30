import json
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app import models
from app.models import _is_pg
from app.passwords import verify_password


def get_words(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    word_types: list = None,
    frequencies: list = None,
):
    query = db.query(models.Word)

    if word_types:
        query = query.filter(models.Word.word_type.in_(word_types))
    if frequencies:
        query = query.filter(models.Word.frequency.in_(frequencies))

    total = query.count()
    words = query.offset(skip).limit(limit).all()
    return {
        "items": words,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "page_size": limit,
    }


def get_word(db: Session, word_id: int):
    return db.query(models.Word).filter(models.Word.id == word_id).first()


def search_words(
    db: Session,
    query: str,
    skip: int = 0,
    limit: int = 100,
    word_types: list = None,
    frequencies: list = None,
):
    db_query = db.query(models.Word).filter(models.Word.word.contains(query))

    if word_types:
        db_query = db_query.filter(models.Word.word_type.in_(word_types))
    if frequencies:
        db_query = db_query.filter(models.Word.frequency.in_(frequencies))

    total = db_query.count()
    words = db_query.offset(skip).limit(limit).all()
    return {
        "items": words,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "page_size": limit,
    }


def get_word_types(db: Session):
    types = db.query(models.Word.word_type).distinct().all()
    return [t[0] for t in types if t[0]]


def get_frequencies(db: Session):
    freqs = db.query(models.Word.frequency).distinct().all()
    return sorted(
        [f[0] for f in freqs if f[0]], key=lambda x: int(x) if x.isdigit() else x
    )


# ---------------------------------------------------------------------------
# Phase 2.1 — User + WeaknessProfile CRUD
#
# No auth here. The functions are intentionally simple and rely on the
# route layer to enforce business rules (duplicate-email detection,
# auth gating once 2.2 lands, etc.). The ``axes`` column is stored as
# a JSON object on Postgres and as a JSON-encoded string on SQLite;
# these helpers hide the dialect difference so route code always sees
# a dict.
# ---------------------------------------------------------------------------


def _serialize_axes(axes: Dict[str, int]) -> object:
    """Return the storage shape for the current dialect.

    Postgres: return the dict directly (the JSON column accepts it).
    SQLite: json.dumps the dict into a Text-compatible string.
    """
    if _is_pg():
        return axes
    return json.dumps(axes)


def _deserialize_axes(raw: object) -> Dict[str, int]:
    """Return a dict regardless of the storage shape.

    Postgres: raw is already a dict (or None for an unset row).
    SQLite: raw is a JSON string; json.loads back to a dict.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Defensive: a row written by an older schema version or a
            # manual edit could be malformed. Surface an empty dict
            # rather than crashing the read path.
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    """Lookup helper used by both signup (duplicate-email check) and
    Phase 2.2's login route. Returns ``None`` if no row matches.
    """
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[models.User]:
    """Lookup helper used by the weakness-profile routes. Returns
    ``None`` if no row matches — the route layer translates this
    into a 404.
    """
    return db.query(models.User).filter(models.User.id == user_id).first()


def authenticate_user(
    db: Session, email: str, password: str
) -> Optional[models.User]:
    """Phase 2.2 — verify a (email, password) pair and return the
    ``User`` row on success, ``None`` on any failure.

    Constant-time on the bcrypt side (``bcrypt.checkpw``); the
    pre-lookup by email is a regular index hit. A 401 on this
    function's ``None`` return must not leak which of email vs
    password was wrong — the spec calls that out explicitly. The
    function itself just returns ``None`` for "not found" and
    ``None`` for "wrong password"; the route layer does not
    distinguish.

    A malformed hash row (the verify call raises) is also mapped
    to ``None`` so a probe cannot tell "row exists with broken
    hash" apart from "no row at all".
    """
    user = get_user_by_email(db, email)
    if user is None:
        # Run a verify against a dummy hash so the time cost of a
        # missing email roughly matches the time cost of a wrong
        # password. This is a low-impact timing equaliser — not
        # cryptographic — and avoids the trivial "look the email
        # up first, no row → fast 401" username-enumeration path.
        verify_password(password, "$2b$12$" + "x" * 53)
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def create_user(db: Session, email: str, password_hash: str) -> models.User:
    """Insert a new ``User`` row and commit.

    Caller is expected to have already validated ``email`` (length,
    format) and ``password_hash`` (non-empty). On a duplicate-email
    conflict the IntegrityError is re-raised — the route layer
    catches it and returns 409.
    """
    user = models.User(email=email, password_hash=password_hash)
    db.add(user)
    db.flush()  # Surface IntegrityError here, not at commit()
    db.commit()
    db.refresh(user)
    return user


def get_weakness_profile(
    db: Session, user_id: int
) -> Optional[models.WeaknessProfile]:
    """Return the ``WeaknessProfile`` for a user, or ``None`` if no row
    exists yet.

    The route layer (``GET /weakness-profile/{user_id}``) auto-creates
    an empty profile on first request; this function only reads. The
    dialect-aware ``axes`` deserialization happens in
    ``serialize_weakness_profile_for_response`` below.
    """
    return (
        db.query(models.WeaknessProfile)
        .filter(models.WeaknessProfile.user_id == user_id)
        .first()
    )


def create_empty_weakness_profile(
    db: Session, user_id: int
) -> models.WeaknessProfile:
    """Insert a default ``WeaknessProfile`` for a user. Called by the
    route when ``GET /weakness-profile/{user_id}`` finds no row.

    Idempotent: if the profile already exists, returns the existing
    row without inserting a duplicate (the ``user_id`` UNIQUE
    constraint would otherwise raise ``IntegrityError``).
    """
    existing = get_weakness_profile(db, user_id)
    if existing is not None:
        return existing
    profile = models.WeaknessProfile(
        user_id=user_id,
        axes=_serialize_axes({}),
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def upsert_weakness_profile(
    db: Session, user_id: int, axes: Dict[str, int]
) -> models.WeaknessProfile:
    """Insert-or-update the ``WeaknessProfile`` for a user.

    ``axes`` must already be a dict with validated values (the route
    layer runs ``WeaknessProfileUpdate`` first). The storage shape is
    chosen per dialect via ``_serialize_axes``.
    """
    profile = get_weakness_profile(db, user_id)
    if profile is None:
        profile = models.WeaknessProfile(
            user_id=user_id,
            axes=_serialize_axes(axes),
        )
        db.add(profile)
    else:
        profile.axes = _serialize_axes(axes)
        # ``onupdate=datetime.utcnow`` covers the SQLAlchemy-side path,
        # but explicitly setting it here keeps the value fresh even
        # when SQLAlchemy's ``onupdate`` doesn't fire (e.g. SQLite
        # has different UPDATE semantics under some versions).
        profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile


def serialize_weakness_profile_axes(
    profile: models.WeaknessProfile,
) -> Dict[str, int]:
    """Public hook for the route layer to read the axes dict out of a
    profile row, regardless of the underlying storage dialect.

    Returns ``{}`` for a profile with a NULL or malformed axes value.
    """
    return _deserialize_axes(profile.axes)
