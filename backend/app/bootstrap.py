"""One-time corpus loader: seeds Postgres from the shipped SQLite file.

Phase 0 ships the vocabulary corpus as a SQLite database at
``backend/data/vocabeo_words.db`` so a fresh clone runs out of the
box. When the backend container starts against a fresh Postgres DB,
the alembic baseline migration creates the (empty) schema but no rows
— this module fills them in.

Idempotent by design: if ``words`` already has rows (e.g. compose was
restarted, or the operator ran the seed manually), the function
returns early. Safe to call from the compose entrypoint on every
boot.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from sqlalchemy import inspect, text

from app.database import DATABASE_URL, engine

logger = logging.getLogger(__name__)


# Path to the shipped SQLite corpus inside the backend container.
SQLITE_CORPUS_PATH = os.getenv(
    "LEXORA_SQLITE_CORPUS_PATH",
    "/app/data/vocabeo_words.db",
)


def _is_postgres_target() -> bool:
    return DATABASE_URL.startswith("postgresql")


def _already_seeded(conn) -> bool:
    """Return True if the target DB already has the corpus loaded."""
    inspector = inspect(engine)
    if not inspector.has_table("words"):
        return False
    n = conn.execute(text("SELECT count(*) FROM words")).scalar_one()
    return n > 0


def _read_sqlite_corpus(path: Path) -> dict[str, list[tuple]]:
    """Stream every row from each table in the SQLite corpus."""
    if not path.exists():
        raise FileNotFoundError(f"SQLite corpus not found at {path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        out: dict[str, list[tuple]] = {}
        for table in ("verb_conjugations", "words", "examples", "fsrs_cards"):
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
            out[table] = [tuple(r) for r in rows]
        return out
    finally:
        conn.close()


def _insert_batch(conn, table: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    # All four tables have a single-column INTEGER primary key named
    # ``id``; columns are inserted in the order they were read.
    cols = (
        "id, word, word_type, frequency, level, translations, conjugation, "
        "additional_info, is_complete, conjugation_id"
        if table == "words"
        else "id, word_id, german, english"
        if table == "examples"
        else "id, infinitive, present_3rd_person, simple_past, participle"
        if table == "verb_conjugations"
        else "id, word_id, difficulty, stability, retrievability, due_date, "
        "last_review, reps, lapses, state, elapsed_days, scheduled_days"
    )
    placeholders = ", ".join([f":p{i}" for i in range(len(rows[0]))])
    stmt = text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})")

    # Postgres rejects integer values for BOOLEAN columns. The shipped
    # SQLite corpus stores ``is_complete`` as 0/1 ints; coerce to bool
    # on the Postgres path. (SQLite's BOOLEAN affinity accepts int
    # either way, so no transformation needed there.)
    is_postgres = _is_postgres_target()

    def _coerce(idx: int, v):
        if is_postgres and table == "words" and idx == 8:
            return bool(v)
        return v

    for row in rows:
        conn.execute(stmt, {f"p{i}": _coerce(i, v) for i, v in enumerate(row)})
    return len(rows)


def seed_corpus() -> bool:
    """Seed the active DB from the shipped SQLite corpus.

    Returns True if seed ran, False if it was skipped (already seeded
    or non-Postgres target). Logs are emitted at INFO level so the
    compose startup log shows what happened.
    """
    if not _is_postgres_target():
        logger.info(
            "seed_corpus: non-Postgres target (%s); assuming SQLite corpus is authoritative.",
            DATABASE_URL,
        )
        return False

    with engine.begin() as conn:
        if _already_seeded(conn):
            logger.info(
                "seed_corpus: words table is non-empty; skipping (idempotent)."
            )
            return False

    logger.info("seed_corpus: loading corpus from %s ...", SQLITE_CORPUS_PATH)
    data = _read_sqlite_corpus(Path(SQLITE_CORPUS_PATH))

    with engine.begin() as conn:
        # Insert order respects FKs: verb_conjugations -> words ->
        # examples -> fsrs_cards.
        n_vc = _insert_batch(conn, "verb_conjugations", data["verb_conjugations"])
        n_w = _insert_batch(conn, "words", data["words"])
        n_e = _insert_batch(conn, "examples", data["examples"])
        n_f = _insert_batch(conn, "fsrs_cards", data["fsrs_cards"])

    logger.info(
        "seed_corpus: inserted verb_conjugations=%d words=%d examples=%d fsrs_cards=%d",
        n_vc,
        n_w,
        n_e,
        n_f,
    )
    return True