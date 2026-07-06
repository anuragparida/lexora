"""phrase_pairs table (Phase 10.1, card t_18c90a68)

Revision ID: 10a1_phrase_pairs_table
Revises: 9a1_fsrs_cards_exercise_type
Create Date: 2026-07-06 17:30:00.000000

Phase 10.1 — paired-pair table for the cross-phrase match exercise
type. ``phrase_pairs`` stores a relation between two ``phrases``
rows (Phase 8.1 / 8.2) with a 4-way ``relation`` literal::

    equivalent, paraphrase, related, unrelated

The table is **read-only at runtime** (Hard rule #2 of PHASE-8.md,
restated verbatim for Phase 10 by the plan body). The Phase 10.3
``/exercises/phrase_match`` endpoint *reads* from it (e.g. for the
``enable_rag=True`` nearest-neighbor pull); nothing writes to it at
request time. The single write path outside Alembic is
``backend/scripts/seed_phrase_pairs.py`` (this card).

## Schema shape

Mirrors ``app.models.PhrasePair`` column-for-column:

- ``id`` — INTEGER PK, autoincrement (NOT slug — paired-pair
  identities aren't human-readable handles, unlike ``phrases.id``).
  Index is implicit on the PK.
- ``phrase_a_id`` — VARCHAR(120), FK to ``phrases.id``,
  ``ondelete="RESTRICT"`` (never cascade-delete a phrase because
  its pair row goes away). Indexed
  (``ix_phrase_pairs_phrase_a_id``).
- ``phrase_b_id`` — same shape, FK to ``phrases.id``,
  ``ondelete="RESTRICT"``. Indexed
  (``ix_phrase_pairs_phrase_b_id``).
- ``relation`` — VARCHAR, NOT NULL. Loose-String-on-the-DB /
  Literal-at-the-wire-layer (``PhrasePairRelation`` literal in
  ``app.schemas``). Indexed
  (``ix_phrase_pairs_relation``) for the Phase 10.3
  pinned-relation nearest-neighbor pull.
- ``attested_pair`` — BOOLEAN, NOT NULL. ``True`` for Goethe /
  Schiller attested rows from ``backend/data/attested_pairs.json``
  (Phase 10.4); ``False`` for the seed-script's bge-m3-bucketed
  rows. Indexed (``ix_phrase_pairs_attested_pair``) so the planner
  can quickly fetch the attested subset.
- ``created_at`` — DATETIME, NOT NULL, ``server_default`` to
  ``sa.func.now()`` so raw-SQL inserts get a sane value.

## Hard rules at the DB level

- ``phrase_a_id != phrase_b_id`` — CHECK constraint named
  ``check_phrase_pairs_distinct``. The Pydantic
  ``PhrasePairSeedRow`` mirror lives in ``app.schemas``; the seed
  script also pre-filters self-pairs before INSERT as belt-and-braces.
- ``UNIQUE(phrase_a_id, phrase_b_id)`` — same pair can't appear
  twice with swapped order. Named
  ``uq_phrase_pairs_a_b``. The seed script sorts ``(a, b)``
  lexicographically before INSERT so the ``(a, b)`` pair never
  collides with its ``(b, a)`` mirror.

## Idempotency

``inspect()``-guarded so re-running ``alembic upgrade head`` is a
clean no-op. The downgrade drops the indexes and the table
unconditionally. ``phrase_pairs`` has no inbound FKs from other
tables (the Phase 10.3 endpoint *reads* it but doesn't FK to it),
so the drop is safe.

## Hard rule — Alembic owns this table

Phase 7.1 (card t_96ab949e) removed ``Base.metadata.create_all``
from the ``lifespan``. Phase 10.1's model declaration adds
``phrase_pairs`` to ``Base.metadata`` (SA can't round-trip the new
table on INSERT/SELECT otherwise), but the ``lifespan`` does NOT
call ``create_all``. Only Alembic creates the table on a fresh DB.

This mirrors the Phase 8.1 ``phrases`` migration's discipline
(card t_d967c006) — the ``Base.metadata`` declaration is required
for SA's session to find the table, but the table's column shape
and its indexes belong to Alembic.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op


# Revision identifiers, used by Alembic.
revision: str = "10a1_phrase_pairs_table"
down_revision: Union[str, Sequence[str], None] = (
    "9a1_fsrs_cards_exercise_type"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Stable, predictable names — the downgrade needs to find the
# indexes and the constraints by name (SQLAlchemy's
# ``drop_index`` / ``drop_constraint`` take a name, not an
# introspection handle).
_TABLE_NAME = "phrase_pairs"
_INDEX_A = "ix_phrase_pairs_phrase_a_id"
_INDEX_B = "ix_phrase_pairs_phrase_b_id"
_INDEX_RELATION = "ix_phrase_pairs_relation"
_INDEX_ATTESTED = "ix_phrase_pairs_attested_pair"
_UNIQUE_NAME = "uq_phrase_pairs_a_b"
_CHECK_NAME = "check_phrase_pairs_distinct"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if _TABLE_NAME not in existing_tables:
        # UNIQUE(phrase_a_id, phrase_b_id) is declared **inline**
        # at table creation — SQLite's ``ALTER TABLE ADD
        # CONSTRAINT`` is not supported; the constraint has to be
        # baked into the ``CREATE TABLE`` statement. The PG path
        # accepts the same declaration, so the dialect-portable
        # approach is a single ``create_table`` with the
        # ``UniqueConstraint`` baked in.
        #
        # The CHECK (phrase_a_id != phrase_b_id) constraint
        # ships the same way: declared inline as part of the
        # table DDL so a single ``CREATE TABLE`` statement covers
        # both columns and constraints. SQLite + Postgres both
        # honor CHECK in the DDL.
        #
        # Same pair can't appear twice with swapped order — the
        # seed script sorts ``(a, b)`` lexicographically before
        # INSERT so the ``(a, b)`` pair never collides with its
        # ``(b, a)`` mirror.
        op.create_table(
            _TABLE_NAME,
            # Autoincrement integer PK — paired-pair identities
            # aren't human-readable handles (unlike ``phrases.id``).
            # Index is implicit on the PK.
            sa.Column("id", sa.Integer, primary_key=True),
            # FK columns — VARCHAR(120) matches ``phrases.id`` at
            # the DB level. The seed script's idempotent INSERT
            # uses ``ON CONFLICT (phrase_a_id, phrase_b_id) DO
            # NOTHING`` against the composite UNIQUE constraint
            # below, so re-runs against an already-seeded table
            # are a clean no-op.
            sa.Column(
                "phrase_a_id",
                sa.String(120),
                sa.ForeignKey("phrases.id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column(
                "phrase_b_id",
                sa.String(120),
                sa.ForeignKey("phrases.id", ondelete="RESTRICT"),
                nullable=False,
            ),
            # Loose VARCHAR on the DB; the Pydantic
            # ``PhrasePairRelation`` literal in ``app.schemas``
            # is the wire-level guardrail.
            sa.Column("relation", sa.String, nullable=False),
            # The Python-side default mirrors ``Base``'s
            # ``default=False``; ``server_default`` is set so
            # raw-SQL inserts (``psql -c "INSERT ... "``) get
            # ``False`` without a Python-side default.
            sa.Column(
                "attested_pair",
                sa.Boolean,
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column(
                "created_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
            # Composite UNIQUE — same pair can't appear twice.
            sa.UniqueConstraint(
                "phrase_a_id", "phrase_b_id", name=_UNIQUE_NAME
            ),
            # CHECK(phrase_a_id <> phrase_b_id) — the
            # "no self-pair" hard rule at the DB level.
            sa.CheckConstraint(
                "phrase_a_id <> phrase_b_id", name=_CHECK_NAME
            ),
        )

        # Four explicit indexes — same explicit-create rationale
        # as the Phase 8.1 ``ix_phrases_source_attribution``:
        # we control the index names, and ``inspect()``-guarded
        # creation is idempotent on re-run.
        op.create_index(
            _INDEX_A, _TABLE_NAME, ["phrase_a_id"]
        )
        op.create_index(
            _INDEX_B, _TABLE_NAME, ["phrase_b_id"]
        )
        op.create_index(
            _INDEX_RELATION, _TABLE_NAME, ["relation"]
        )
        op.create_index(
            _INDEX_ATTESTED, _TABLE_NAME, ["attested_pair"]
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if _TABLE_NAME not in existing_tables:
        return

    # Drop indexes first (child-then-parent order, matching the
    # Phase 7.1 / 8.1 / 9.1 pattern). The names are stable so
    # the ``inspect()`` lookup is belt-and-braces in case the
    # table was hand-created in dev without the indexes.
    existing_indexes = {
        ix["name"]
        for ix in inspector.get_indexes(_TABLE_NAME)
    }
    for ix_name in (
        _INDEX_A,
        _INDEX_B,
        _INDEX_RELATION,
        _INDEX_ATTESTED,
    ):
        if ix_name in existing_indexes:
            op.drop_index(ix_name, table_name=_TABLE_NAME)

    # The UNIQUE + CHECK constraints are declared inline in the
    # ``create_table`` above. SQLite treats them as part of the
    # table definition; the explicit ``drop_constraint`` is the
    # portable path. Postgres requires an explicit
    # ``DROP CONSTRAINT IF EXISTS`` before the table drop (the
    # constraint FK may otherwise prevent the drop on PG).
    if bind.dialect.name == "postgresql":
        op.execute(
            sa.text(
                f"ALTER TABLE {_TABLE_NAME} "
                f"DROP CONSTRAINT IF EXISTS {_CHECK_NAME}"
            )
        )
        op.execute(
            sa.text(
                f"ALTER TABLE {_TABLE_NAME} "
                f"DROP CONSTRAINT IF EXISTS {_UNIQUE_NAME}"
            )
        )
    op.drop_table(_TABLE_NAME)
