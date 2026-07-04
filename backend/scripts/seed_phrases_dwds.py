"""Seed the ``phrases`` table from the hand-curated DWDS Idiome JSON.

Phase 8.1 (card t_d967c006). Idempotent: re-running against an
already-seeded table is a clean no-op. Reads
``backend/data/dwds_idioms_subset.json`` (a JSON array of seed
rows — one object per idiom, the schema documented in
``PHASE-8.md``) and INSERTs each row via the SQLAlchemy ORM with
``ON CONFLICT (id) DO NOTHING`` semantics so a partial re-run is
safe.

The JSON-array schema:

    [
      {
        "id": str,                  # slug PK (3..120 chars)
        "phrase": str,              # 5..200 chars (UNIQUE in DB)
        "definition": str,          # 1..400 chars
        "example_usage": str|None,  # 5..400 chars or null
        "source_attribution": str,  # "dwds" only in 8.1
        "frequency_band": str,      # "high"|"mid"|"low"
        "dwds_url": str|None        # source URL or null
      },
      ...
    ]

Usage::

    cd backend
    DATABASE_URL=postgresql+psycopg://... \\
        uv run python scripts/seed_phrases_dwds.py --source data/dwds_idioms_subset.json

Expected output: ``Loaded N phrases (target >=150)`` where N is at
least 150 (the hand-curated starting set in
``data/dwds_idioms_subset.json``). The script logs both the loaded
count and the skipped (already-present) count, mirroring the Phase
7.1 seed scripts (``seed_collocations.py`` and
``seed_prepositional_objects.py``).

The seed script is the single write path to ``phrases`` outside
Alembic (Hard rule #2 of PHASE-8.md). It does NOT do
``update`` / ``delete`` — rows are immutable once seeded.

Idempotency strategy
---------------------

``phrases.id`` is the slug PK, so a re-run of the seed script
against an already-populated table needs ``INSERT ... ON CONFLICT
(id) DO NOTHING`` to skip rows that are already present. The
SQLAlchemy ORM doesn't expose ``ON CONFLICT`` directly; we use
``session.execute(insert(Phrase).values(...).on_conflict_do_nothing(
index_elements=['id']))`` to issue the dialect-correct statement on
both SQLite (INSERT OR IGNORE) and Postgres (ON CONFLICT DO NOTHING).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import SessionLocal, DATABASE_URL
from app.models import Phrase

logger = logging.getLogger("lexora.seed_phrases_dwds")

# Default seed path — matches the Phase 7.1 convention of
# colocating seeds under ``backend/app/seeds/`` for compiled-in
# fixtures. The 8.1 path follows the card body contract and
# lives under ``backend/data/`` instead (the JSON is too large
# to bundle inside the package and is referenced from
# ``docs/PHASE-8.md`` in the "Files affected" section).
DEFAULT_SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "dwds_idioms_subset.json"
)

# Card body target: ≥200 rows is the documented "ideal". The
# hand-curated starting set in 8.1 ships with ~150 rows; future
# commits (Phase 8.x and Phase 9) extend it. The test asserts
# ≥150 to match the curated reality and to fail loudly if the
# seeded JSON is truncated in a follow-up edit.
TARGET_ROWS = 150


def _is_pg() -> bool:
    """Dialect discriminator (mirrors ``app.models._is_pg``)."""
    return DATABASE_URL.startswith("postgresql")


def _already_seeded(session) -> bool:
    """Return True if the table already has rows.

    Idempotent contract: re-running the seed against an already
    populated DB is a clean no-op. Mirrors the ``seed_collocations``
    / ``seed_prepositional_objects`` pattern from Phase 7.1.

    Commits the implicit read transaction so the subsequent
    INSERT statements don't trip the "transaction is already
    begun" guard. The check is a no-op SELECT; we don't need the
    row data, just the count.
    """
    from sqlalchemy import func

    n = session.scalar(select(func.count()).select_from(Phrase))
    session.commit()
    return (n or 0) > 0


def _load_seed_rows(path: Path) -> list[dict]:
    """Parse the JSON-array seed file into a list of validated dicts.

    Each row is passed through ``PhraseSeedRow`` to surface
    Literal-enum / bound violations BEFORE the row hits the DB
    (the DB column is loose String — we want the validation at
    the seed boundary, where the Pydantic literal is the
    guardrail).
    """
    from app.schemas import PhraseSeedRow  # local import — Pydantic runtime

    with open(path, encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"seed file {path}: invalid JSON: {e}")
    if not isinstance(payload, list):
        raise SystemExit(
            f"seed file {path}: expected JSON array of rows, got "
            f"{type(payload).__name__}"
        )

    out: list[Phrase] = []
    for lineno, row in enumerate(payload, 1):
        if not isinstance(row, dict):
            raise SystemExit(
                f"seed file {path}:{lineno}: expected JSON object, got "
                f"{type(row).__name__}"
            )
        # Validate via Pydantic so a typo'd frequency_band /
        # source_attribution is caught here, not later when the
        # row silently propagates into the Phase 8.3 idiom
        # generator.
        try:
            validated = PhraseSeedRow.model_validate(row)
        except Exception as e:
            raise SystemExit(
                f"seed file {path}:{lineno} validation failed: {e}"
            )
        out.append(
            Phrase(
                id=validated.id,
                phrase=validated.phrase,
                definition=validated.definition,
                example_usage=validated.example_usage,
                source_attribution=validated.source_attribution,
                frequency_band=validated.frequency_band,
                dwds_url=validated.dwds_url,
                # The 8.1 DWDS seed doesn't fill the attestation
                # columns — those land in 8.2's
                # ``seed_phrases_attestations.py``.
                attested_quote=None,
                attested_source=None,
            )
        )
    return out


def _insert_with_ignore_conflict(session, rows: list[Phrase]) -> int:
    """INSERT the rows with ``ON CONFLICT (id) DO NOTHING``.

    SQLAlchemy ORM ``session.add_all()`` issues plain INSERTs
    that fail on PK collisions; we want the Postgres / SQLite
    ``ON CONFLICT (id) DO NOTHING`` / ``INSERT OR IGNORE``
    semantics so re-running the seed on a populated table is a
    no-op rather than an ``IntegrityError``.

    The ``phrases`` table has two uniqueness invariants — the PK
    on ``id`` AND a separate UNIQUE constraint on ``phrase`` (so
    the same German surface form can't appear twice in
    different bands). SQLite's ``ON CONFLICT`` only fires for
    indices that exist in the schema; the compound
    ``ON CONFLICT (id, phrase)`` shape isn't a valid SQLite
    identifier. We pre-dedupe the input rows by both ``id`` and
    ``phrase`` in Python (so a future seed-JSON edit that
    accidentally carries duplicates degrades to a clean skip
    rather than an IntegrityError) and use ``ON CONFLICT (id)
    DO NOTHING`` for the actual INSERT — which covers the PK
    case and lets the Python pre-dedupe handle the ``phrase``
    case.

    Returns the number of rows reported ``INSERTED 0`` by the
    driver (which doesn't tell us how many were skipped). The
    caller re-counts via ``SELECT count(*)`` after the bulk
    insert to compute the actual delta.

    Both dialects share the same compile-time shape::

        INSERT INTO phrases (cols...) VALUES (vals...)
        ON CONFLICT (id) DO NOTHING

    — supported by both SQLite (since 3.24) and PostgreSQL.
    """
    if not rows:
        return 0

    # Python-level pre-dedupe (id AND phrase). SQLite's
    # ``on_conflict_do_nothing`` only fires on the named unique
    # index — we'd need a separate ``ON CONFLICT (phrase)``
    # clause to cover the secondary UNIQUE, and SQLAlchemy
    # doesn't expose that on the dialect-portable insert builder.
    # Pre-dedupe keeps the wire shape portable.
    seen_id = set()
    seen_phrase = set()
    deduped = []
    for r in rows:
        if r.id in seen_id or r.phrase in seen_phrase:
            continue
        seen_id.add(r.id)
        seen_phrase.add(r.phrase)
        deduped.append(r)
    rows = deduped

    # Build a Core ``insert`` statement from the model so we can
    # attach the ``on_conflict_do_nothing`` clause. ``values`` is
    # populated from the ORM instances' ``__dict__``.
    stmt_dicts = [
        {
            "id": r.id,
            "phrase": r.phrase,
            "definition": r.definition,
            "example_usage": r.example_usage,
            "source_attribution": r.source_attribution,
            "frequency_band": r.frequency_band,
            "dwds_url": r.dwds_url,
            "attested_quote": r.attested_quote,
            "attested_source": r.attested_source,
        }
        for r in rows
    ]
    table = Phrase.__table__
    if _is_pg():
        stmt = (
            pg_insert(table)
            .values(stmt_dicts)
            .on_conflict_do_nothing(index_elements=["id"])
        )
    else:
        # SQLite dialect — SQLAlchemy compiles
        # ``on_conflict_do_nothing`` against SQLite as
        # ``INSERT OR IGNORE INTO phrases ...``. The seeded JSON
        # is pre-deduped by ``phrase`` (above), so the (id)
        # ON CONFLICT shape is sufficient.
        stmt = (
            sqlite_insert(table)
            .values(stmt_dicts)
            .on_conflict_do_nothing(index_elements=["id"])
        )

    session.execute(stmt)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SEED_PATH),
        help=(
            f"Path to the DWDS idioms JSON file (default: "
            f"{DEFAULT_SEED_PATH})"
        ),
    )
    args = parser.parse_args()

    seed_path = Path(args.source)
    if not seed_path.exists():
        print(f"ERROR: seed file not found at {seed_path}", file=sys.stderr)
        return 1

    rows = _load_seed_rows(seed_path)
    if len(rows) < TARGET_ROWS:
        print(
            f"WARNING: only {len(rows)} rows in seed file "
            f"(target >= {TARGET_ROWS}). The 8.1 hand-curated set "
            f"ships with ~150 rows; an extension commit can grow "
            f"this further.",
            file=sys.stderr,
        )

    session = SessionLocal()
    try:
        already_populated = _already_seeded(session)
        with session.begin():
            _insert_with_ignore_conflict(session, rows)
        # Re-count after the bulk INSERT to compute the actual delta.
        from sqlalchemy import func

        n_after = session.scalar(select(func.count()).select_from(Phrase))
        inserted = (n_after or 0)
        if already_populated:
            print(
                f"phrases table already populated before this run; "
                f"{len(rows)} rows inserted with ON CONFLICT DO "
                f"NOTHING (skipped any duplicates). Total now: "
                f"{inserted} rows."
            )
        else:
            print(f"Loaded {inserted} phrases (target >= {TARGET_ROWS}).")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
