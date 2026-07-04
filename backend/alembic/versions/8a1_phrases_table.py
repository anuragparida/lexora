"""phrases table (Phase 8.1, card t_d967c006)

Revision ID: 8a1_phrases_table
Revises: 7a2_prepositional_objects_table
Create Date: 2026-07-04 12:15:00.000000

Phase 8.1 — third of the read-only curated corpus tables. This
migration creates the ``phrases`` table for curated German idioms
(multi-word fixed expressions that are not compositional).

The table is **read-only at runtime** (Hard rule #2 of PHASE-8.md).
The exercise generator in Phase 8.3 reads from it; it never writes
back. The only write paths outside Alembic are the seed scripts:

- ``backend/scripts/seed_phrases_dwds.py`` — DWDS Idiome subset
  (≥200 rows, hand-curated from public DWDS Idiome export).
- ``backend/scripts/seed_phrases_attestations.py`` — Phase 8.2
  Goethe/Schiller attestation extension (200–300 rows, lands
  as a separate card; this migration pre-reserves the
  ``attested_quote`` / ``attested_source`` columns so 8.2
  doesn't need an Alembic rewrite).

Schema shape mirrors ``app.models.Phrase`` column-for-column:

- ``id`` — slug VARCHAR(120) PK. NOT autoincrement. The seed
  script slugifies DWDS's ``<Lemma>`` (e.g. ``ins Blaue hinein``
  → ``ins-blaue-hinein``) and uses the slug as the PK, so the
  idempotent ``INSERT OR IGNORE`` / ``ON CONFLICT (id) DO NOTHING``
  re-runs cleanly on a populated DB.
- ``phrase`` — TEXT, NOT NULL, UNIQUE. The German surface form
  (5–200 chars per the ``PhraseSeedRow`` Pydantic model; the DB
  column is loose Text — bound is at the wire/seed layer).
- ``definition`` — TEXT, NOT NULL. Learner-friendly English gloss
  (1–400 chars).
- ``example_usage`` — TEXT, NULL allowed. Some DWDS
  ``<Lemma>`` entries lack a ``<Example>`` child; the seed
  script tolerates the omission.
- ``source_attribution`` — VARCHAR, NOT NULL. Comma-joined
  literal of ``dwds`` / ``goethe`` / ``schiller`` (and a
  future ``manual``); loose String on the DB; the wire-layer
  ``PhraseSeedRow`` Pydantic validator enforces the per-element
  literal. **Indexed** (``ix_phrases_source_attribution``) for
  Phase 9 attribution queries.
- ``frequency_band`` — VARCHAR, NOT NULL. ``high`` / ``mid`` /
  ``low``; loose String on the DB; Pydantic literal at the
  seed / wire layer. **Indexed** (``ix_phrases_frequency_band``)
  for the Phase 8.4 high-band-first cloze variant.
- ``dwds_url`` — TEXT, NULL allowed. The source DWDS Idiome URL.
- ``attested_quote`` — TEXT, NULL allowed. Phase 8.2 Goethe /
  Schiller quotation (reserved here, additive-only).
- ``attested_source`` — TEXT, NULL allowed. Phase 8.2 citation
  (reserved here, additive-only).
- ``created_at`` — DATETIME, NOT NULL, server default to
  ``sa.func.now()`` so raw-SQL inserts (``psql -c "INSERT ..."``)
  get a sane value.

Both ops are guarded by ``inspect()`` so the migration is
idempotent: re-running ``upgrade head`` against an
already-migrated DB is a clean no-op. This mirrors the Phase 7.1
pattern (card t_96ab949e) and the Phase 5.2 pattern (card
t_88b6f1c4). Downgrade drops the table unconditionally — the
``phrases`` table has no inbound FKs from other tables (it's a
leaf in the FK graph), so the drop is safe.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op


# Revision identifiers, used by Alembic.
revision: str = "8a1_phrases_table"
down_revision: Union[str, Sequence[str], None] = (
    "7a2_prepositional_objects_table"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "phrases" not in existing_tables:
        op.create_table(
            "phrases",
            # Slug PK — VARCHAR(120) bounds at the DB layer; the
            # seed-row Pydantic model enforces 3..120 chars, but
            # the DB cap is the safety belt (a broken seed can't
            # write an unbounded string into the PK).
            sa.Column("id", sa.String(120), primary_key=True),
            sa.Column("phrase", sa.Text, nullable=False, unique=True),
            sa.Column("definition", sa.Text, nullable=False),
            # Nullable — DWDS sometimes ships lemmas without an
            # <Example> child. The seed script tolerates the
            # omission; the column has to permit NULL.
            sa.Column("example_usage", sa.Text, nullable=True),
            # Comma-joined literal of ``dwds`` / ``goethe`` /
            # ``schiller`` / ``manual``. Loose VARCHAR on the DB;
            # the wire-layer ``PhraseSeedRow`` enforces the
            # per-element literal. ``index=True`` for Phase 9
            # attribution queries.
            sa.Column(
                "source_attribution",
                sa.String,
                nullable=False,
            ),
            sa.Column(
                "frequency_band",
                sa.String,
                nullable=False,
            ),
            sa.Column("dwds_url", sa.Text, nullable=True),
            # Phase 8.2 attestation columns — reserved here
            # (additive, nullable) so the 8.2 Goethe/Schiller
            # seed script doesn't need an Alembic rewrite.
            sa.Column("attested_quote", sa.Text, nullable=True),
            sa.Column("attested_source", sa.Text, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        # Index on ``source_attribution`` for Phase 9 attribution
        # queries (membership filters: ``WHERE source_attribution
        # LIKE 'dwds,%'``). Created explicitly (vs. relying on
        # ``index=True`` on the column) so we control the index
        # name and reuse the same idempotency-guard pattern as
        # the table creation.
        op.create_index(
            "ix_phrases_source_attribution",
            "phrases",
            ["source_attribution"],
        )
        # Index on ``frequency_band`` for the Phase 8.4
        # high-band-first cloze variant. Same explicit-create
        # rationale as ``source_attribution``.
        op.create_index(
            "ix_phrases_frequency_band",
            "phrases",
            ["frequency_band"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # The ``phrases`` table has no inbound FKs from other tables
    # (it's a leaf in the FK graph — Phase 9 may add an FK from a
    # future audit log table, but not yet). Drop the indexes first
    # (child-then-parent order), then the table. The migration is
    # the single drop path for ``phrases`` (the seed scripts never
    # delete).
    if "phrases" in existing_tables:
        # Drop indexes if they exist (defensive — the table might
        # have been created manually without the indexes in dev).
        for ix_name in (
            "ix_phrases_source_attribution",
            "ix_phrases_frequency_band",
        ):
            ix_names = {
                ix["name"]
                for ix in inspector.get_indexes("phrases")
            }
            if ix_name in ix_names:
                op.drop_index(ix_name, table_name="phrases")
        op.drop_table("phrases")
