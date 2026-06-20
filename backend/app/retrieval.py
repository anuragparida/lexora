"""Retrieval queries over pgvector-backed embedding columns.

Phase 1 plumbing — provides the ``/retrieve`` endpoint with
nearest-neighbour lookups by cosine distance. The exercise
generator (Phase 4) and RAG prompt (Phase 6) will both import from
this module.

The dialect-aware helpers below branch on the active bind dialect:

- Postgres (production): use the ``<=>`` cosine-distance operator
  directly in raw SQL. The HNSW index from the Phase 1 migration
  makes this O(log n) per query.
- SQLite (dev fallback): ``/retrieve`` returns 503. SQLite has no
  vector type; the ``embedding`` column stays NULL. Trying to
  compute distance on the client would lie about the result, so we
  refuse the query instead.

Score convention: the endpoint returns ``score = 1 - distance`` so
larger = more similar. Cosine distance is in [0, 2] for any pair of
unit vectors; for normalized embeddings (which OpenRouter's
embedding endpoints all return) it's in [0, 1], so the score lives
in [0, 1] with 1.0 being identical.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import engine

logger = logging.getLogger(__name__)


Source = Literal["words", "examples", "both"]


def _is_postgres_target() -> bool:
    """Return True if the active engine is bound to Postgres."""
    return engine.dialect.name == "postgresql"


def _format_vector_literal(vec: list[float]) -> str:
    """Render a Python list[float] as a pgvector string literal.

    pgvector accepts ``'[1.0,2.0,...]'::vector``. ``str()`` on a
    Python list gives the right shape, but we want to be explicit
    so a future reader sees the contract.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _query_words(
    db: Session,
    query_vec: list[float],
    k: int,
) -> list[dict]:
    """Return top-k nearest words by cosine distance (lower = closer).

    Each row has ``id``, ``word``, ``word_type``, ``frequency``,
    ``score`` (= 1 - distance, so higher = more similar).
    """
    vec_lit = _format_vector_literal(query_vec)
    sql = text(
        """
        SELECT
            id,
            word,
            word_type,
            frequency,
            translations,
            (embedding <=> CAST(:qvec AS vector)) AS distance
        FROM words
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:qvec AS vector)
        LIMIT :k
        """
    )
    rows = db.execute(sql, {"qvec": vec_lit, "k": k}).fetchall()
    return [
        {
            "id": r.id,
            "word": r.word,
            "word_type": r.word_type,
            "frequency": r.frequency,
            "translations": r.translations,
            "score": 1.0 - float(r.distance),
        }
        for r in rows
    ]


def _query_examples(
    db: Session,
    query_vec: list[float],
    k: int,
) -> list[dict]:
    """Return top-k nearest examples by cosine distance."""
    vec_lit = _format_vector_literal(query_vec)
    sql = text(
        """
        SELECT
            e.id,
            e.word_id,
            e.german,
            e.english,
            w.word AS word,
            (e.embedding <=> CAST(:qvec AS vector)) AS distance
        FROM examples e
        LEFT JOIN words w ON w.id = e.word_id
        WHERE e.embedding IS NOT NULL
        ORDER BY e.embedding <=> CAST(:qvec AS vector)
        LIMIT :k
        """
    )
    rows = db.execute(sql, {"qvec": vec_lit, "k": k}).fetchall()
    return [
        {
            "id": r.id,
            "word_id": r.word_id,
            "word": r.word,
            "german": r.german,
            "english": r.english,
            "score": 1.0 - float(r.distance),
        }
        for r in rows
    ]


def retrieve(
    db: Session,
    query_vec: list[float],
    k: int,
    source: Source,
) -> list[dict]:
    """Top-k nearest rows by cosine distance.

    For ``source="both"`` we pull ``k`` rows from each table
    independently and merge them, sorted by score. The merged list
    is capped at ``2k`` rows so the endpoint stays bounded under
    combined-source calls.
    """
    if source == "words":
        return _query_words(db, query_vec, k)
    if source == "examples":
        return _query_examples(db, query_vec, k)
    if source == "both":
        words = _query_words(db, query_vec, k)
        examples = _query_examples(db, query_vec, k)
        # Tag each item with its source so the client can tell them
        # apart (rows from the two tables share no id namespace).
        for w in words:
            w["source"] = "words"
        for e in examples:
            e["source"] = "examples"
        merged = words + examples
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[: 2 * k]
    raise ValueError(f"unknown source: {source!r}")