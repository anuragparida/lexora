"""Phase 2.2 — auth helpers + ``get_current_user`` FastAPI dependency.

Card: t_74c3aa1e.

Public surface:

- ``hash_password(plain: str) -> str`` — bcrypt with rounds=12 (explicit).
- ``verify_password(plain: str, hashed: str) -> bool`` — constant-time.
- ``create_access_token(user_id: int) -> str`` — HS256 JWT, 7-day exp.
- ``decode_token(token: str) -> dict | None`` — None on any failure.
- ``get_current_user(...)`` — FastAPI dependency used by the
  ``/auth/*`` and ``/weakness-profile/*`` routes.

Configuration is read from the ``Settings`` singleton at module import
time (mirrors the convention in ``app.database``):

- ``JWT_SECRET`` — required. 32+ random bytes (the spec suggests
  ``openssl rand -hex 32``). The module raises ``RuntimeError`` at
  import if the env var is missing or still the placeholder used in
  ``docker-compose.yml`` (``change-me-in-production``). The first
  such boot on a fresh host is loud, not silent.
- ``JWT_ALGORITHM`` — default ``HS256``. Symmetric: same secret signs
  and verifies.
- ``JWT_EXPIRY_DAYS`` — default ``7``.

Cookie name: ``lexora_token``. ``httpOnly=True``, ``secure=False``
(dev), ``samesite="lax"``, max-age 7 days. The ``/auth/me``
dependency reads the cookie first, then falls back to
``Authorization: Bearer <token>`` so curl / manual testing works.

The dependency raises ``HTTPException(401)`` on any of: missing
token, malformed token, expired token, JWT signature mismatch, JWT
payload missing ``sub``, the ``sub`` not being an int, or the user
row no longer existing. Each path returns the same opaque
``{"detail": "..."}`` so a curl probe cannot distinguish "no cookie"
from "bad token" from "user deleted" — that's deliberate
(username-enumeration defense).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Cookie, Depends, Header, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app import crud, models
from app.database import get_db
from app.passwords import BCRYPT_ROUNDS, hash_password, verify_password  # noqa: F401

# re-export so the route layer can keep importing from ``app.auth`` —
# the bcrypt logic lives in ``app.passwords`` to avoid the
# auth<->crud circular import, but the public surface stays here.
__all__ = [
    "BCRYPT_ROUNDS",
    "COOKIE_NAME",
    "JWT_ALGORITHM",
    "JWT_EXPIRY_DAYS",
    "JWT_SECRET",
    "clear_auth_cookie",
    "cookie_kwargs",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "hash_password",
    "set_auth_cookie",
    "verify_password",
]


# ---------------------------------------------------------------------------
# Settings (env-driven, read at import time)
# ---------------------------------------------------------------------------


COOKIE_NAME = "lexora_token"
JWT_ALGORITHM_DEFAULT = "HS256"
JWT_EXPIRY_DAYS_DEFAULT = 7


def _load_settings() -> dict:
    """Resolve the JWT settings once at import time.

    Raises ``RuntimeError`` if ``JWT_SECRET`` is missing or still the
    docker-compose placeholder. The error surfaces on app boot — a
    misconfigured auth is loud, not silent.
    """
    secret = os.getenv("JWT_SECRET", "").strip()
    placeholder = "change-me-in-production"
    if not secret or secret == placeholder:
        raise RuntimeError(
            "JWT_SECRET is not configured. Set it in the env "
            "(e.g. `export JWT_SECRET=$(openssl rand -hex 32)`) or "
            "via docker compose. The Phase 2.2 card refuses to boot "
            "with the placeholder value used in docker-compose.yml."
        )
    if len(secret) < 32:
        # Spec says 32+ random bytes. We accept the value as a string
        # (a 64-char hex string is 32 bytes raw, a 32-char string is
        # only 32 bytes if ASCII), so check character length is also
        # at least 32 to give the operator headroom in either form.
        raise RuntimeError(
            "JWT_SECRET must be at least 32 characters. Generate a "
            "fresh value with `openssl rand -hex 32` (64 hex chars)."
        )
    return {
        "jwt_secret": secret,
        "jwt_algorithm": os.getenv("JWT_ALGORITHM", JWT_ALGORITHM_DEFAULT),
        "jwt_expiry_days": int(
            os.getenv("JWT_EXPIRY_DAYS", str(JWT_EXPIRY_DAYS_DEFAULT))
        ),
    }


_SETTINGS = _load_settings()
JWT_SECRET: str = _SETTINGS["jwt_secret"]
JWT_ALGORITHM: str = _SETTINGS["jwt_algorithm"]
JWT_EXPIRY_DAYS: int = _SETTINGS["jwt_expiry_days"]


# ---------------------------------------------------------------------------
# Password hashing (re-exported from app.passwords; see that module
# for the bcrypt implementation and the 72-byte truncation note).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_access_token(user_id: int) -> str:
    """Mint a short JWT for a user.

    Payload: ``{"sub": <user_id_as_str>, "exp": <now + 7d>}``. The
    ``sub`` is stringified because JSON integers are 53-bit-safe but
    ``python-jose``'s ``decode`` returns whatever was encoded — and
    the convention in the spec sketch is ``int(payload["sub"])``,
    which is robust to either form.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=JWT_EXPIRY_DAYS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT. Returns ``None`` on any failure.

    Failures handled silently: missing sub, expired, bad signature,
    malformed. Callers translate ``None`` into a 401 — they never
    branch on the failure mode.
    """
    if not token:
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def _extract_token(
    lexora_token: Optional[str], authorization: Optional[str]
) -> Optional[str]:
    """Pull the bearer token from the cookie first, then the
    ``Authorization: Bearer <token>`` header. Returns ``None`` if
    neither path yields a token.
    """
    if lexora_token:
        return lexora_token
    if authorization and authorization.lower().startswith("bearer "):
        # Split once on the first space — the token itself can be
        # any printable ASCII, but it must not contain a space.
        return authorization.split(" ", 1)[1].strip() or None
    return None


def get_current_user(
    lexora_token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    """FastAPI dependency: resolve the current user from the cookie
    or Authorization header. 401 on any failure mode.

    Usage::

        @app.get("/me")
        def me(user: models.User = Depends(get_current_user)):
            return user

    The route handler receives the SQLAlchemy ``User`` row. Route
    code must serialise via ``schemas.UserOut`` to keep
    ``password_hash`` out of the response (the Pydantic model omits
    the column by construction).
    """
    token = _extract_token(lexora_token, authorization)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    sub = payload.get("sub")
    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ---------------------------------------------------------------------------
# Cookie set / clear helpers (used by /auth/signup + /auth/login + /auth/logout)
# ---------------------------------------------------------------------------


def cookie_kwargs(max_age_seconds: Optional[int] = None) -> dict:
    """The kwargs for ``Response.set_cookie`` / ``delete_cookie``.

    Centralised so the path, samesite, httpOnly, and secure settings
    are defined in exactly one place. ``max_age`` is omitted on
    delete-cookie calls (the caller passes ``max_age_seconds=0`` to
    clear, which we then drop — a zero max-age on a delete is
    redundant with the ``Set-Cookie`` empty-value semantics).
    """
    out = {
        "key": COOKIE_NAME,
        "httponly": True,
        "secure": False,  # dev only — flip True behind HTTPS
        "samesite": "lax",
        "path": "/",
    }
    if max_age_seconds:
        out["max_age"] = int(max_age_seconds)
    return out


def set_auth_cookie(response, token: str) -> None:
    """Attach the JWT cookie to a FastAPI response. 7-day max-age."""
    response.set_cookie(
        value=token,
        **cookie_kwargs(max_age_seconds=JWT_EXPIRY_DAYS * 24 * 3600),
    )


def clear_auth_cookie(response) -> None:
    """Clear the JWT cookie (logout). max-age=0 tells the browser
    to drop it immediately."""
    response.set_cookie(value="", max_age=0, **cookie_kwargs())
