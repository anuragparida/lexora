"""collocations table (Phase 7.1, card t_96ab949e)

Revision ID: 7a1_collocations_table
Revises: p5a2_unique_fsrs_grade_logs
Create Date: 2026-07-04 03:30:00.000000

Phase 7.1 — first of two schema additions for the read-only curated
German collocation + prepositional-object corpus. This migration
creates the ``collocations`` table.

The table is **read-only at runtime** (Hard rule #2 of PHASE-7.md).
The exercise generator in Phase 7.2 reads from it; it never writes
back. The only write paths outside Alembic are the seed scripts
(``backend/scripts/seed_collocations.py``) reading the hand-curated
``backend/app/seeds/collocations_seed.json`` JSON-Lines file.

Schema shape mirrors ``app.models.Collocation`` column-for-column:

- ``collocation_id`` — autoincrement PK.
- ``headword_id`` — FK to ``words.id`` (``ondelete=SET NULL`` so a
  future word deletion doesn't cascade-wipe the curated row).
  Nullable because partner lemmas are not always anchored on a
  row in the ``words`` table.
- ``partner_lemma`` — free-form string (the co-occurring word).
- ``frequency_score`` — Float (DWDS-normalized 0..1).
- ``register`` — ``formal`` / ``neutral`` / ``colloquial`` as a
  loose ``String`` (the wire-layer Pydantic literal is the
  guardrail; the DB column is dialect-agnostic).
- ``source_corpus`` — ``dwds`` / ``wiktionary`` / ``manual``
  (same dialect-agnostic pattern).
- ``created_at`` — DB default to ``sa.func.now()`` so raw-SQL
  inserts (``psql -c "INSERT ..."``) get a sane value.

There is no ``updated_at``: rows are immutable once seeded (Hard
rule #2). The seed scripts are the single write path outside
Alembic.

Both ops are guarded by ``inspect()`` so the migration is
idempotent: re-running ``upgrade head`` against an
already-migrated DB is a clean no-op. This mirrors the Phase 5.2
pattern (card t_88b6f1c4) and the Phase 4.5 two-file split
(card t_da712d54). Downgrade drops the table unconditionally —
the FK target (``words``) outlives the curated rows.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op


# Revision identifiers, used by Alembic.
revision: str = "7a1_collocations_table"
down_revision: Union[str, Sequence[str], None] = "p5a2_unique_fsrs_grade_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "collocations" not in existing_tables:
        op.create_table(
            "collocations",
            sa.Column(
                "collocation_id", sa.Integer, primary_key=True, index=True
            ),
            sa.Column(
                "headword_id",
                sa.Integer,
                sa.ForeignKey("words.id", ondelete="SET NULL"),
                nullable=True,
                index=True,
            ),
            sa.Column("partner_lemma", sa.String, nullable=False),
            sa.Column(
                "frequency_score", sa.Float, nullable=False, default=0.0
            ),
            # Loose String columns on both ``register`` and
            # ``source_corpus``: the wire-layer Pydantic literal is
            # the type-level guardrail (PHASE-7.md gotcha #12). The
            # DB column stays dialect-agnostic so SQLite + Postgres
            # agree on the shape.
            sa.Column("register", sa.String, nullable=False),
            sa.Column("source_corpus", sa.String, nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # Mirror the project's child-then-parent convention. The
    # ``collocations`` table has no inbound FKs from other tables
    # (it is a leaf in the FK graph; ``headword_id`` only points
    # *out* to ``words``), so the drop is unconditional.
    if "collocations" in existing_tables:
        op.drop_table("collocations")