"""Seed the ``collocations`` table from the hand-curated JSON-Lines file.

Phase 7.1 (card t_96ab949e). Idempotent: re-running on a table that
already has rows is a no-op. Reads
``backend/app/seeds/collocations_seed.json`` (one JSON object per
line) and INSERTs each row via the SQLAlchemy ORM.

The JSON-Lines schema (per the card body contract):

    {
      "headword_id": int,         # FK to words.id (nullable)
      "partner_lemma": str,       # co-occurring word
      "register": str,            # "formal" | "neutral" | "colloquial"
      "source_corpus": str,       # "dwds" | "wiktionary" | "manual"
      "frequency_score": float    # 0..1 (DWDS-normalized)
    }

Usage::

    cd backend
    DATABASE_URL=postgresql+psycopg://... \
        uv run python -m scripts.seed_collocations

Expected output: ``inserted=N skipped=M`` where ``N >= 200``.

The seed script is the single write path to ``collocations``
outside Alembic (Hard rule #2 of PHASE-7.md). It does NOT do
``update`` / ``delete`` — rows are immutable once seeded.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Collocation

logger = logging.getLogger("lexora.seed_collocations")

SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "seeds"
    / "collocations_seed.json"
)


def _already_seeded(session) -> bool:
    """Return True if the table already has rows.

    Idempotent contract: re-running the seed against an already
    populated DB is a clean no-op. This matches the bootstrap.py
    pattern (Phase 0) where the loader skips when ``words`` already
    has rows.

    Commits the implicit read transaction so the subsequent
    ``session.begin()`` block doesn't trip the "transaction is
    already begun" guard. The check itself is a no-op SELECT; we
    don't need the row data, just the count.
    """
    from sqlalchemy import func

    n = session.scalar(select(func.count()).select_from(Collocation))
    # Commit the implicit SELECT transaction so the subsequent
    # ``session.begin()`` block doesn't trip "A transaction is
    # already begun on this Session".
    session.commit()
    return (n or 0) > 0


def _load_seed_rows(path: Path) -> list[dict]:
    """Parse JSON-Lines (one object per line) into a list of dicts.

    Empty lines and trailing whitespace are tolerated. Each row is
    passed through ``CollocationSeedRow`` to surface Literal-enum
    violations before they hit the DB (the DB column is loose
    String — we want the validation at the seed boundary).
    """
    from app.schemas import CollocationSeedRow  # local import — Pydantic runtime

    out: list[Collocation] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"seed file {path}:{lineno} not valid JSON: {e}"
                )
            # Validate via Pydantic so a typo'd register /
            # source_corpus is caught here, not later at the
            # wire layer. The Pydantic ``alias='register'`` maps
            # the JSON key ``register`` back to the SQLAlchemy
            # column name.
            try:
                validated = CollocationSeedRow.model_validate(payload)
            except Exception as e:
                raise SystemExit(
                    f"seed file {path}:{lineno} validation failed: {e}"
                )
            # We pull the literal value via the Pydantic attribute
            # (``register_label``) so the ORM gets the canonical
            # column name back (``register``).
            out.append(
                Collocation(
                    headword_id=validated.headword_id,
                    partner_lemma=validated.partner_lemma,
                    frequency_score=validated.frequency_score,
                    register=validated.register_label,
                    source_corpus=validated.source_corpus,
                )
            )
    return out


def main() -> int:
    if not SEED_PATH.exists():
        print(f"ERROR: seed file not found at {SEED_PATH}", file=sys.stderr)
        return 1

    session = SessionLocal()
    try:
        if _already_seeded(session):
            from sqlalchemy import func

            n = session.scalar(select(func.count()).select_from(Collocation))
            print(
                f"collocations table already populated ({n} rows) — "
                f"skipping seed."
            )
            return 0

        rows = _load_seed_rows(SEED_PATH)
        if len(rows) < 200:
            print(
                f"WARNING: only {len(rows)} rows in seed file (target >= 200)",
                file=sys.stderr,
            )
        with session.begin():
            session.add_all(rows)
        print(f"inserted {len(rows)} collocation rows.")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())