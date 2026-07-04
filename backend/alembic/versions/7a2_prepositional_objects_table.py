"""prepositional_objects table (Phase 7.1, card t_96ab949e)

Revision ID: 7a2_prepositional_objects_table
Revises: 7a1_collocations_table
Create Date: 2026-07-04 03:30:00.000000

Phase 7.1 — second of two schema additions for the read-only curated
German collocation + prepositional-object corpus. This migration
creates the ``prepositional_objects`` table.

The table is **read-only at runtime** (Hard rule #2 of PHASE-7.md).
The exercise generator in Phase 7.2 reads from it; it never writes
back. The only write paths outside Alembic are the seed scripts
(``backend/scripts/seed_prepositional_objects.py``) reading the
hand-curated ``backend/app/seeds/prepositional_objects_seed.json``
JSON-Lines file.

Schema shape mirrors ``app.models.PrepositionalObject``
column-for-column:

- ``prepositional_object_id`` — autoincrement PK.
- ``verb_lemma`` — free-form string (the head verb). Not a FK
  because partner lemmas are not always anchored on a row in the
  ``words`` table (the corpus here is curated, not auto-linked).
- ``preposition`` — German preposition (``auf``, ``mit``, ``über``).
- ``case`` — ``Akk`` / ``Dat`` / ``Gen`` (governed case).
- ``example_sentence`` — worked German example showing the verb
  + preposition + case in context.
- ``frequency_score`` — Float (DWDS-normalized 0..1).
- ``source_corpus`` — ``dwds`` / ``wiktionary`` / ``manual``.
- ``created_at`` — DB default to ``sa.func.now()``.

There is no ``updated_at``: rows are immutable once seeded (Hard
rule #2). The seed scripts are the single write path outside
Alembic.

Same idempotency pattern as ``7a1_collocations_table.py``:
``inspect()`` guards short-circuit re-runs, the Phase 5.2 + 4.5
two-file split (collocation → prepositional_object) is preserved.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op


# Revision identifiers, used by Alembic.
revision: str = "7a2_prepositional_objects_table"
down_revision: Union[str, Sequence[str], None] = "7a1_collocations_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "prepositional_objects" not in existing_tables:
        op.create_table(
            "prepositional_objects",
            sa.Column(
                "prepositional_object_id",
                sa.Integer,
                primary_key=True,
                index=True,
            ),
            sa.Column("verb_lemma", sa.String, nullable=False),
            sa.Column("preposition", sa.String, nullable=False),
            # ``case`` is loose String at the DB layer; Pydantic
            # literal at the wire layer enforces ``Akk`` / ``Dat``
            # / ``Gen``. Same dialect-agnostic pattern as
            # ``collocations.register``.
            sa.Column("case", sa.String, nullable=False),
            sa.Column("example_sentence", sa.Text, nullable=False),
            sa.Column(
                "frequency_score", sa.Float, nullable=False, default=0.0
            ),
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
    # ``prepositional_objects`` table is a leaf in the FK graph
    # (no inbound FKs), so the drop is unconditional.
    if "prepositional_objects" in existing_tables:
        op.drop_table("prepositional_objects")