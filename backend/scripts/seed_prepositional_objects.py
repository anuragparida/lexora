"""Seed the ``prepositional_objects`` table from the hand-curated JSON-Lines file.

Phase 7.1 (card t_96ab949e). Idempotent: re-running on a table that
already has rows is a no-op. Reads
``backend/app/seeds/prepositional_objects_seed.json`` (one JSON
object per line) and INSERTs each row via the SQLAlchemy ORM.

The JSON-Lines schema (per the card body contract):

    {
      "verb_lemma": str,            # head verb (free-form)
      "preposition": str,           # German preposition
      "case": str,                  # "Akk" | "Dat" | "Gen"
      "example_sentence": str,      # worked German example
      "frequency_score": float,     # 0..1 (DWDS-normalized)
      "source_corpus": str          # "dwds" | "wiktionary" | "manual"
    }

Usage::

    cd backend
    DATABASE_URL=postgresql+psycopg://... \
        uv run python -m scripts.seed_prepositional_objects

Expected output: ``inserted=N skipped=M`` where ``N >= 200``.

The seed script is the single write path to
``prepositional_objects`` outside Alembic (Hard rule #2 of
PHASE-7.md). It does NOT do ``update`` / ``delete`` — rows are
immutable once seeded.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.models import PrepositionalObject

logger = logging.getLogger("lexora.seed_prepositional_objects")

SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "seeds"
    / "prepositional_objects_seed.json"
)


def _already_seeded(session) -> bool:
    """Return True if the table already has rows.

    Same idempotent contract as ``seed_collocations.py``: a re-run
    against an already-populated DB is a clean no-op.

    Commits the implicit read transaction so the subsequent
    ``session.begin()`` block doesn't trip the "transaction is
    already begun" guard. The check itself is a no-op SELECT.
    """
    from sqlalchemy import func

    n = session.scalar(select(func.count()).select_from(PrepositionalObject))
    session.commit()
    return (n or 0) > 0


def _load_seed_rows(path: Path) -> list[PrepositionalObject]:
    """Parse JSON-Lines into ORM instances, validating each via Pydantic.

    The Pydantic validation surfaces Literal-enum violations
    (``case``, ``source_corpus``) at the seed boundary rather than
    letting a typo'd value sneak into the loose String column.
    """
    from app.schemas import PrepositionalObjectSeedRow  # local — Pydantic runtime

    out: list[PrepositionalObject] = []
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
            try:
                validated = PrepositionalObjectSeedRow.model_validate(payload)
            except Exception as e:
                raise SystemExit(
                    f"seed file {path}:{lineno} validation failed: {e}"
                )
            out.append(
                PrepositionalObject(
                    verb_lemma=validated.verb_lemma,
                    preposition=validated.preposition,
                    case=validated.case,
                    example_sentence=validated.example_sentence,
                    frequency_score=validated.frequency_score,
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

            n = session.scalar(
                select(func.count()).select_from(PrepositionalObject)
            )
            print(
                f"prepositional_objects table already populated "
                f"({n} rows) — skipping seed."
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
        print(f"inserted {len(rows)} prepositional_object rows.")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())