"""Backfill embeddings for Word and Example rows.

Phase 1 offline batch job. Reads every row in ``words`` and
``examples`` that doesn't yet have an embedding, computes a vector
via the OpenRouter embedding endpoint, and writes it back in
batches. Idempotent — re-running skips rows that already have a
non-null embedding.

Usage::

    cd backend
    DATABASE_URL=postgresql+psycopg://... uv run python -m scripts.backfill_embeddings

Expected runtime: ~5-10 minutes against the shipped corpus
(12,430 words + ~12,430 examples) with ``EMBEDDING_BATCH_SIZE=32``.

## Text composition

The embedding input for a Word row is a short composite string
that gives the embedding model enough signal to discriminate
synonyms and grammatical context::

    "{word} ({word_type}, frequency {frequency}): {translations}"

For Examples it's the German sentence alone — the English
translation lives in the linked Word row and would otherwise
double-count. The retrieval endpoint scores on the same
representation, so the query text and the corpus text need to be
in the same semantic space.

## Output

Progress is logged at INFO level with the running count so the
operator can ``docker compose logs -f backend`` and watch it
move. Final line: per-table counts of embedded vs. skipped rows.

Exit code: 0 on success, 1 on any unrecoverable embedding error
(network, 4xx from the provider). Partial progress is committed
between batches, so a retry resumes from the last successful
checkpoint.
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Iterable

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from app.database import DATABASE_URL, SessionLocal, engine
from app.embeddings import embed
from app.models import Example, Word


logger = logging.getLogger("lexora.backfill")

# ``embedding`` writes need a SQL string the pgvector extension can
# parse — ``'[1.0,2.0,...]'::vector``. On SQLite the column is a
# BLOB and we pack the raw float32 bytes instead.
EMBED_DIM = 1024
BATCH_SIZE = 32  # matches app/embeddings.py default


def _word_text(word: Word) -> str:
    """Composite string used as the embedding input for a Word row."""
    pieces = [word.word or ""]
    if word.word_type:
        pieces.append(f"({word.word_type}")
        if word.frequency:
            pieces.append(f", frequency {word.frequency}")
        pieces.append(")")
    if word.translations:
        pieces.append(f": {word.translations}")
    return "".join(pieces).strip()


def _example_text(example: Example) -> str:
    """Composite string used as the embedding input for an Example row."""
    return (example.german or "").strip()


def _iter_chunks(items: list, size: int) -> Iterable[list]:
    """Yield successive ``size``-element chunks from ``items``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _format_vec(vec: list[float]) -> str:
    """Render a Python list[float] as a pgvector string literal."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _is_postgres() -> bool:
    return engine.dialect.name == "postgresql"


def _write_embedding(db: Session, table: str, row_id: int, vec: list[float]) -> None:
    """Persist a single embedding back to the row.

    Postgres: cast the literal so pgvector accepts it. SQLite: pack
    raw float32 bytes (the column is a BLOB; nothing reads it back
    on the SQLite path).
    """
    if _is_postgres():
        vec_lit = _format_vec(vec)
        db.execute(
            text(f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE id = :id"),
            {"v": vec_lit, "id": row_id},
        )
    else:
        # SQLite fallback — pack raw float32. Not used in production
        # but kept so a developer can ``DATABASE_URL=sqlite:///...``
        # backfill locally.
        import struct
        packed = struct.pack(f"{len(vec)}f", *vec)
        db.execute(
            text(f"UPDATE {table} SET embedding = :v WHERE id = :id"),
            {"v": packed, "id": row_id},
        )


def _fetch_pending_words(db: Session) -> list[Word]:
    """Return all Word rows whose embedding column is NULL."""
    stmt = select(Word).where(Word.embedding.is_(None))
    return list(db.execute(stmt).scalars())


def _fetch_pending_examples(db: Session) -> list[Example]:
    """Return all Example rows whose embedding column is NULL."""
    stmt = select(Example).where(Example.embedding.is_(None))
    return list(db.execute(stmt).scalars())


def _backfill_words(db: Session) -> tuple[int, int]:
    """Embed and persist pending Word embeddings. Returns (embedded, skipped).

    Skipped is always 0 in this version — pending rows are exactly
    the ones without embeddings. Returning the pair keeps the
    signature symmetric with a future "skip stale" path.
    """
    pending = _fetch_pending_words(db)
    logger.info("words: %d rows pending embedding", len(pending))
    if not pending:
        return 0, 0

    embedded = 0
    started = time.time()
    for chunk in _iter_chunks(pending, BATCH_SIZE):
        texts = [_word_text(w) for w in chunk]
        vectors = embed(texts)
        for word, vec in zip(chunk, vectors):
            _write_embedding(db, "words", word.id, vec)
            embedded += 1
        db.commit()
        logger.info(
            "words: %d/%d embedded (%.1fs elapsed)",
            embedded,
            len(pending),
            time.time() - started,
        )
    return embedded, 0


def _backfill_examples(db: Session) -> tuple[int, int]:
    """Embed and persist pending Example embeddings. See _backfill_words."""
    pending = _fetch_pending_examples(db)
    logger.info("examples: %d rows pending embedding", len(pending))
    if not pending:
        return 0, 0

    embedded = 0
    started = time.time()
    for chunk in _iter_chunks(pending, BATCH_SIZE):
        texts = [_example_text(e) for e in chunk]
        vectors = embed(texts)
        for example, vec in zip(chunk, vectors):
            _write_embedding(db, "examples", example.id, vec)
            embedded += 1
        db.commit()
        logger.info(
            "examples: %d/%d embedded (%.1fs elapsed)",
            embedded,
            len(pending),
            time.time() - started,
        )
    return embedded, 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not DATABASE_URL.startswith("postgresql"):
        logger.warning(
            "backfill: target is non-Postgres (%s); embeddings will be "
            "stored as packed float32 BLOBs. The /retrieve endpoint "
            "will return 503 on this DB — Phase 1 plumbing expects "
            "Postgres + pgvector.",
            DATABASE_URL,
        )

    db = SessionLocal()
    try:
        w_done, w_skip = _backfill_words(db)
        e_done, e_skip = _backfill_examples(db)
    finally:
        db.close()

    logger.info(
        "backfill: done. words: %d embedded/%d skipped, examples: %d embedded/%d skipped",
        w_done,
        w_skip,
        e_done,
        e_skip,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())