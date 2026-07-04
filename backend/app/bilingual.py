"""Phase 7.4 — bilingual lookup helper (card t_d621bb4f).

Read-through for the ``partner_lang`` opt-in flag on
``/exercises/match`` and ``/exercises/cloze``. Given a target
word id, ``lookup_partner_translation`` returns the English
counterpart (``collocations.partner_lemma``) when one exists,
or ``None`` when:

- the request didn't opt in to bilingual (``partner_lang="de"``);
- no row in ``collocations`` exists for the target word;
- the ``collocations`` table doesn't exist yet (the 7.1 schema
  hasn't been folded onto main when this card lands);
- or any other DB error prevents the lookup.

The helper is intentionally **fail-soft**: it never raises.
A missing translation is a valid runtime outcome (most words
have no curated EN partner). The wire-level ``partner_translation``
field on the response is ``str | None``; ``None`` is the
default — bilingual is opt-in (Hard rule H3 of PHASE-7.md).

The 7.1 schema migration creates ``collocations`` with at least
the columns ``headword_id`` (FK to ``words.id``) and
``partner_lemma`` (the curated EN string). We query via raw SQL
because no SQLAlchemy model for ``collocations`` exists on
``main`` yet — this card lands before 7.1 ships, so adding a
model would create a merge conflict with 7.1. Once 7.1 lands,
this helper continues to work unchanged because the column
names + ``headword_id`` FK are stable on both sides.

Reconciliation note (post-7.1 fold): the original helper queried
``WHERE word_id = :word_id`` assuming 7.1's FK column would be
``word_id``. The canonical ``app.models.Collocation`` (7.1)
ships with ``headword_id`` instead. The fix below updates the
query to ``headword_id``; the helper still queries by raw SQL
and is still fail-soft, so a future schema rename will land as
a one-line edit here.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Type-level guardrail for the partner_lang flag. Mirrors
# ``app.schemas.PartnerLang`` but lives here as a runtime alias
# so callers that haven't imported the wire schema can still
# reference the union. Hard rule H4 (PHASE-7.md): values outside
# the literal are rejected at the wire layer; this module
# accepts whatever the caller passes and only acts when the
# value is ``"en"``.
PartnerLang = Literal["de", "en"]


def lookup_partner_translation(
    db: Session,
    word_id: int,
    partner_lang: PartnerLang,
) -> str | None:
    """Return the curated English counterpart for ``word_id`` or
    ``None`` when bilingual is off, no row exists, or the table
    isn't present.

    Parameters
    ----------
    db
        Active SQLAlchemy session against the ``words`` /
        ``collocations`` schema.
    word_id
        FK to ``words.id`` of the exercise target.
    partner_lang
        ``"de"`` (default — return ``None`` immediately) or
        ``"en"`` (run the SELECT).

    Returns
    -------
    str | None
        The ``partner_lemma`` string when present, ``None`` when
        the flag is off, no row exists, or the table is missing.

    Notes
    -----
    The SELECT is wrapped in ``try / except`` so a missing
    ``collocations`` table (the pre-7.1 schema state) is a
    silent no-op rather than a 500. The exception path is
    logged at DEBUG so the unit-test suite doesn't fill the
    log buffer with warnings from the no-collocation-table
    fixtures; production INFO-level observation happens via
    the structured fields in the trace span emitted by the
    caller (``generate_match`` / ``generate_cloze``).
    """
    if partner_lang != "en":
        # Default path — bilingual is opt-in (Hard rule H3).
        # Returning ``None`` here is the same behaviour as
        # "no row found", which is the right wire outcome:
        # ``partner_translation`` is ``None``.
        return None

    try:
        # Raw SQL because no SQLAlchemy ``Collocations`` model
        # exists on ``main`` yet. The columns are stable
        # (``headword_id`` + ``partner_lemma``) so 7.1's migration
        # doesn't change this query.
        row = db.execute(
            text(
                "SELECT partner_lemma FROM collocations "
                "WHERE headword_id = :word_id LIMIT 1"
            ),
            {"word_id": int(word_id)},
        ).first()
    except (OperationalError, ProgrammingError) as exc:
        # Most common cause: ``collocations`` table doesn't
        # exist (Phase 7.1 hasn't shipped). The fail-soft
        # contract says return ``None`` and let the
        # ``partner_translation`` field stay ``None``.
        logger.debug(
            "bilingual.lookup_partner_translation: collocations "
            "table unavailable for word_id=%d: %s",
            word_id,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — fail-soft guardrail
        # Any other DB-side failure (connection drop, dialect
        # mismatch, etc.) is also fail-soft. The bilingual
        # feature is best-effort — a missing translation is
        # never an excuse to 500 the route.
        logger.warning(
            "bilingual.lookup_partner_translation: unexpected "
            "failure for word_id=%d partner_lang=%s: %s",
            word_id,
            partner_lang,
            exc,
        )
        return None

    if row is None:
        return None

    # ``row`` is a ``Row`` tuple from SQLAlchemy's Core
    # ``.first()``; the only column we selected is
    # ``partner_lemma``. We return ``str(row[0])`` defensively
    # so a non-string ``partner_lemma`` (e.g. a numeric column
    # drift in a future migration) doesn't crash the route —
    # we coerce or return ``None`` on coercion failure.
    raw: Any = row[0]
    if raw is None:
        return None
    if not isinstance(raw, str):
        try:
            raw = str(raw)
        except Exception:  # noqa: BLE001
            return None
    return raw


__all__ = ["PartnerLang", "lookup_partner_translation"]