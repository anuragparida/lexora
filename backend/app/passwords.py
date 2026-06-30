"""Password hashing helpers — shared by ``app.auth`` and ``app.crud``.

Card: t_74c3aa1e.

Extracted to its own module so the bcrypt dependency is defined in
exactly one place. ``app.auth`` needs ``hash_password`` /
``verify_password`` to mint and verify JWTs; ``app.crud`` needs
``authenticate_user`` to do the same lookup but cannot import from
``app.auth`` (circular — ``app.auth`` imports ``get_user_by_id`` from
``app.crud``).

Bcrypt is invoked at rounds=12 (explicit, not relying on the
library default). The 72-byte input limit is enforced by truncation
rather than by raising — long passphrases are still accepted, just
hashed on the first 72 bytes.
"""
from __future__ import annotations

import bcrypt


BCRYPT_ROUNDS = 12


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt at rounds=12.

    Returns the standard ``$2b$12$...`` hash string (utf-8 decoded).
    """
    if not isinstance(plain, str):
        raise TypeError("password must be a str")
    if not plain:
        raise ValueError("password must be non-empty")
    pw_bytes = plain.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time bcrypt verification.

    Returns ``False`` on any malformed-hash failure rather than
    raising — the login route translates "no match" into a 401 and
    must not leak the difference between "wrong password" and
    "corrupted hash row".
    """
    if not isinstance(plain, str) or not isinstance(hashed, str):
        return False
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(
            plain.encode("utf-8")[:72], hashed.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False
