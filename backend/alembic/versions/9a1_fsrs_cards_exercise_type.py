"""fsrs_cards.exercise_type column (Phase 9.1, card t_0bfdb7ed)

Revision ID: 9a1_fsrs_cards_exercise_type
Revises: 8a1_phrases_table
Create Date: 2026-07-05 17:05:00.000000

Phase 9.1 (card t_0bfdb7ed) — schema widening for the Phase 9
cross-type study session. Adds ``fsrs_cards.exercise_type`` so
each FSRS card row carries the exercise kind it backs (cloze,
matching, comprehension, idiom). Until now the column lived only
on ``grade_logs`` (per-grade audit row) — the card itself was
implicitly cloze because Phase 5 was the only exercise kind that
read it.

## Why now

Phase 9 widens the union of exercise kinds a single study session
can contain (card t_6e784cf1). 9.2 widens ``/exercises/due`` to
return a tagged union; 9.3/9.4 widen the optimizer scripts; 9.5
widens the frontend. All of them assume each ``fsrs_cards`` row
already knows its kind — so the scheduler can decide "due cloze"
vs "due matching" without a join through ``grade_logs``. Adding
the column here (9.1) is the schema prerequisite.

## Column shape

- ``exercise_type`` — VARCHAR (TEXT in SQLite), ``NOT NULL``,
  ``server_default='cloze'`` so the column backfills cleanly
  for every existing row created by Phase 0/5/6/8 (all of which
  were cloze-only by construction). ``server_default`` is set so
  raw-SQL inserts (``psql -c "INSERT ... "``) get a sane value
  without a Python-side default.

- Same loose-String-on-the-DB / Pydantic-Literal-at-the-wire
  pattern used by ``grade_logs.exercise_type`` (Phase 5.2 card
  t_88b6f1c4). The wire layer's ``ExerciseType`` literal in
  ``app/schemas.py`` already covers all four values; this
  migration just mirrors the contract onto the card row.

- Indexed (``ix_fsrs_cards_exercise_type``) because Phase 9.2's
  ``/exercises/due`` union query filters on
  ``WHERE exercise_type = ?`` for each kind before the cross-kind
  merge. Same explicit-create rationale as the Phase 8.1
  ``ix_phrases_source_attribution`` index.

## Idempotency

``inspect()``-guarded so re-running ``upgrade head`` against an
already-migrated DB is a clean no-op (the Phase 5.2 / 7.1 /
8.1 pattern). The downgrade drops the column AND the index
unconditionally — ``fsrs_cards.exercise_type`` has no inbound
FK from any other table (only the SA model and the
``/exercises/grade`` writer read it), so the drop is safe.

## Hard rule — Alembic owns this column

Phase 7.1 (card t_96ab949e) removed ``Base.metadata.create_all``
from the ``lifespan``. The Phase 9.1 model declaration adds the
column to ``Base.metadata`` (it has to — SA can't round-trip
the column on INSERT/SELECT otherwise), but the ``lifespan``
itself does NOT call ``create_all``. Only Alembic adds the
column on a fresh DB. The matching test
(``test_fsrs_card_exercise_type.py::test_lifespan_does_not_create_exercise_type_column``)
asserts this end-to-end.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op


# Revision identifiers, used by Alembic.
revision: str = "9a1_fsrs_cards_exercise_type"
down_revision: Union[str, Sequence[str], None] = "8a1_phrases_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Stable, predictable names — the downgrade needs to find the
# column and the index by name (SQLAlchemy's ``drop_column`` /
# ``drop_index`` take a name, not an introspection handle).
_COLUMN_NAME = "exercise_type"
_INDEX_NAME = "ix_fsrs_cards_exercise_type"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    existing_fsrs_columns: set[str] = set()
    existing_fsrs_indexes: set[str] = set()
    if "fsrs_cards" in existing_tables:
        existing_fsrs_columns = {
            col["name"] for col in inspector.get_columns("fsrs_cards")
        }
        existing_fsrs_indexes = {
            ix["name"] for ix in inspector.get_indexes("fsrs_cards")
        }

    # --- Op 1: add the column ---
    if _COLUMN_NAME not in existing_fsrs_columns:
        # ``server_default='cloze'`` backfills every existing row
        # to the Phase 0/5/6/8 default. ``nullable=False`` mirrors
        # the model declaration. SA's ``add_column`` emits
        # ``ALTER TABLE ADD COLUMN`` on both Postgres and SQLite
        # (SQLite ≥ 3.35 supports ``ALTER TABLE ... ADD COLUMN``
        # with a constant ``DEFAULT``; the lexora test matrix is
        # 3.45+, well above the floor).
        op.add_column(
            "fsrs_cards",
            sa.Column(
                _COLUMN_NAME,
                sa.String,
                nullable=False,
                server_default="cloze",
            ),
        )

    # --- Op 2: add the index ---
    if _INDEX_NAME not in existing_fsrs_indexes:
        # Phase 9.2's ``/exercises/due`` union filters on
        # ``exercise_type`` per-kind before the cross-kind merge.
        # Explicit ``create_index`` (vs. relying on
        # ``index=True`` on the column) so we control the index
        # name and reuse the same idempotency-guard pattern as
        # the Phase 8.1 ``ix_phrases_source_attribution``.
        op.create_index(
            _INDEX_NAME,
            "fsrs_cards",
            [_COLUMN_NAME],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "fsrs_cards" not in existing_tables:
        return

    # Drop the index first (child-then-parent order, matching the
    # Phase 7.1 / 8.1 pattern). ``DROP INDEX IF EXISTS`` is
    # portable across Postgres + SQLite; the ``inspect()`` guard
    # in the index lookup is belt-and-braces in case the table
    # was hand-created in dev without the index.
    existing_indexes = {
        ix["name"] for ix in inspector.get_indexes("fsrs_cards")
    }
    if _INDEX_NAME in existing_indexes:
        op.drop_index(_INDEX_NAME, table_name="fsrs_cards")

    # Drop the column. No inbound FK from any other table, so the
    # drop is unconditional. ``inspect()``-guarded so a re-run
    # against an already-downgraded DB is a clean no-op.
    existing_columns = {
        col["name"] for col in inspector.get_columns("fsrs_cards")
    }
    if _COLUMN_NAME in existing_columns:
        op.drop_column("fsrs_cards", _COLUMN_NAME)