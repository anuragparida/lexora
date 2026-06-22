"""add embedding columns + HNSW index on words and examples

Revision ID: 496091d14711
Revises: ca03ee90afb6
Create Date: 2026-06-20 21:45:00.000000

Phase 1 of the personalized-learner roadmap. Plumbs embedding storage
and retrieval without wiring any consumer (RAG prompt lands in
Phase 6; exercise generation lands in Phase 4).

Schema shape:

- Postgres (production target): ``embedding vector(1024)`` on both
  ``words`` and ``examples``, plus a HNSW index using
  ``vector_cosine_ops`` for fast nearest-neighbour queries. Index
  creation uses ``IF NOT EXISTS`` so re-running the migration is a
  no-op.
- SQLite (dev fallback): ``embedding BLOB`` storing the raw float32
  bytes. No HNSW, no vector math — the SQLite path is for tests and
  local dev only. The CRUD layer detects the dialect and skips
  vector writes on SQLite (the column exists but stays NULL), so
  ``/retrieve`` against a SQLite DB returns 503 rather than lying
  about cosine scores.

Idempotent: each ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` is
safe to re-run on a DB that already has the column. The HNSW index
creation is gated on the dialect name. Downgrade drops the indexes
first (child -> parent) then the columns, mirroring the baseline's
ordering convention.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "496091d14711"
down_revision: Union[str, Sequence[str], None] = "ca03ee90afb6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Phase 1 fix (t_2e386ba9 / Helena review §5): SQLite's
    # ``ALTER TABLE ADD COLUMN`` does NOT support inline
    # ``IF NOT EXISTS`` — the keyword set is reserved in that
    # grammar position and SQLite raises ``near "EXISTS": syntax
    # error``. The portable fix is dialect branching: keep
    # ``IF NOT EXISTS`` on Postgres (where it's a no-op when the
    # column already exists) and emit plain ``ADD COLUMN`` on
    # SQLite (a fresh dev DB always lacks the column, so the
    # qualifier buys nothing on that path).
    #
    # Idempotency on Postgres is preserved by ``IF NOT EXISTS``;
    # on SQLite, re-running against an already-migrated DB will
    # fail with ``duplicate column name: embedding`` — that's
    # acceptable because the SQLite path is dev-fallback only and
    # callers always start from a fresh file (see QA check 2).
    if dialect == "postgresql":
        # pgvector + HNSW
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute(
            "ALTER TABLE words ADD COLUMN IF NOT EXISTS embedding vector(1024)"
        )
        op.execute(
            "ALTER TABLE examples ADD COLUMN IF NOT EXISTS embedding vector(1024)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_words_embedding_hnsw "
            "ON words USING hnsw (embedding vector_cosine_ops)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_examples_embedding_hnsw "
            "ON examples USING hnsw (embedding vector_cosine_ops)"
        )
    elif dialect == "sqlite":
        # BLOB stores raw float32 bytes (1024 * 4 = 4096 bytes). The
        # value is opaque on the SQLite path; nothing reads it back.
        # HNSW is Postgres-only by design — no vector math runs here.
        op.execute("ALTER TABLE words ADD COLUMN embedding BLOB")
        op.execute("ALTER TABLE examples ADD COLUMN embedding BLOB")
    else:
        raise NotImplementedError(
            f"Embedding migration not implemented for dialect {dialect}"
        )


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # Indexes first (child -> parent ordering matches baseline).
    if is_postgres:
        op.execute("DROP INDEX IF EXISTS ix_examples_embedding_hnsw")
        op.execute("DROP INDEX IF EXISTS ix_words_embedding_hnsw")

    # Drop columns. ``DROP COLUMN`` errors out if the column doesn't
    # exist, which is the right behaviour for a clean downgrade. On
    # partial re-runs, swallow the error so the migration is
    # idempotent in reverse.
    for table in ("examples", "words"):
        try:
            op.execute(f"ALTER TABLE {table} DROP COLUMN embedding")
        except Exception:
            # Column doesn't exist on this DB — downgrade already
            # applied. Safe to skip.
            pass