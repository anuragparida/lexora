"""phase 5.2: fsrs_cards.word_id unique constraint + grade_logs table

Revision ID: p5a2_unique_fsrs_grade_logs
Revises: 003_phase3_diagnostic
Create Date: 2026-07-03 09:30:00.000000

Phase 5.2 (card t_88b6f1c4). Two schema ops on the FSRS grading
pipeline, both required before 5.3 can wire ``/exercises/grade``:

- ``fsrs_cards.word_id`` UNIQUE — Phase 0's baseline created the
  table without a uniqueness constraint, so the same word could end
  up with multiple cards. Phase 5 needs one card per word so the
  grader in 5.3 can do a clean ``WHERE word_id = ?`` lookup. The
  constraint is added via the dialect-agnostic
  ``create_unique_constraint`` op, which renders as
  ``ALTER TABLE ... ADD CONSTRAINT ... UNIQUE (word_id)`` on
  Postgres and an equivalent ``CREATE UNIQUE INDEX`` on SQLite.

- ``grade_logs`` table — the per-grade audit trail. Every grade
  request in 5.3 writes one row here, the Langfuse ``trace_id``
  threads through the row so a Phase 6 Ragas evaluator can join
  the offline eval set back to the Langfuse traces. The card
  body locks the column set; this migration matches it 1:1.

Both ops are idempotent:

- On Postgres, ``CREATE UNIQUE INDEX IF NOT EXISTS`` is the right
  primitive; ``add_constraint`` would error if the constraint
  already exists, so the migration uses the index path and lets
  PG treat ``word_id`` unique as a logical uniqueness constraint
  (enforced via the index, which is portable with SQLite).
- On SQLite, ``CREATE UNIQUE INDEX IF NOT EXISTS`` is the only
  portable form — the language has no ``ALTER TABLE ADD
  CONSTRAINT UNIQUE`` syntax. The same index shape works on both
  dialects.

Downgrade reverses both ops: drop the unique index, drop the
``grade_logs`` table. ``grade_logs`` has no FK constraints on
``fsrs_cards`` (the audit row outlives the card deletion), so the
drop is unconditional.

Out of scope: any data migration (no historical rows to backfill),
the actual FSRS grading logic (5.1), the HTTP routes (5.3 / 5.4),
and Langfuse tracing on the write (4.3 / 5.3).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "p5a2_unique_fsrs_grade_logs"
down_revision: Union[str, Sequence[str], None] = "003_phase3_diagnostic"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Stable, predictable constraint name. 12-char Alembic revs are hex;
# the ``ops`` below name this constraint the same way so a future
# downgrade can find and drop it deterministically.
_FSRS_WORD_ID_UNIQUE_INDEX = "ix_fsrs_cards_word_id_unique"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    existing_indexes: set[str] = set()
    if "fsrs_cards" in existing_tables:
        existing_indexes = {
            ix["name"]
            for ix in inspector.get_indexes("fsrs_cards")
        }

    # --- Op 1: UNIQUE (word_id) on fsrs_cards ---
    # CREATE UNIQUE INDEX IF NOT EXISTS is portable across Postgres +
    # SQLite and is also what ``add_constraint`` lowers to on a
    # unique constraint. Skipping when the index already exists keeps
    # a re-run against an already-migrated DB a clean no-op.
    if _FSRS_WORD_ID_UNIQUE_INDEX not in existing_indexes:
        # Index on the existing column. ``word_id`` is the only
        # FSRS-side lookup key the grader (5.3) uses to find a card
        # for a graded word; the unique index serves both the
        # uniqueness guarantee and the read path.
        op.execute(
            f"CREATE UNIQUE INDEX {_FSRS_WORD_ID_UNIQUE_INDEX} "
            f"ON fsrs_cards (word_id)"
        )

    # --- Op 2: grade_logs audit table ---
    if "grade_logs" not in existing_tables:
        # Column set mirrors ``models.GradeLog`` 1:1 — see that
        # class docstring for the per-field rationale.
        #
        # - ``exercise_type`` is TEXT (not Enum) because Phase 5
        #   hard-locks the value to ``"cloze"`` at the Pydantic
        #   wire layer (``GradeRequest.exercise_type: Literal["cloze"]``).
        #   The DB column stays loose so a future exercise kind can
        #   be added without a schema rewrite; the wire contract
        #   gate is the single source of truth for Phase 5.
        # - ``trace_id`` is NULLABLE because Langfuse keys are not
        #   set in tests / dev; the graceful-degradation path
        #   writes ``NULL`` and the request still succeeds.
        # - ``latency_ms`` is INTEGER (millisecond granularity is
        #   sufficient — sub-ms resolution would burn the
        #   Langfuse-tracing budget on noise).
        # - ``graded_at`` defaults to ``datetime.utcnow`` at the
        #   Python level; ``server_default`` mirrors it at the DB
        #   level so a raw-SQL insert (Phase 6 backfill, manual
        #   psql) also gets a sane timestamp.
        op.create_table(
            "grade_logs",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column(
                "user_id",
                sa.Integer,
                sa.ForeignKey("users.id"),
                nullable=False,
                index=True,
            ),
            sa.Column("exercise_id", sa.Integer, nullable=False),
            sa.Column("exercise_type", sa.String, nullable=False),
            sa.Column("word_id", sa.Integer, nullable=False),
            sa.Column("grade", sa.Integer, nullable=False),
            sa.Column(
                "scheduled_next_due_at", sa.DateTime, nullable=False
            ),
            sa.Column("prev_due_at", sa.DateTime, nullable=False),
            sa.Column("state", sa.Integer, nullable=False),
            sa.Column("stability", sa.Float, nullable=False),
            sa.Column("difficulty", sa.Float, nullable=False),
            sa.Column("reps", sa.Integer, nullable=False),
            sa.Column("lapses", sa.Integer, nullable=False),
            sa.Column("trace_id", sa.String, nullable=True),
            sa.Column("latency_ms", sa.Integer, nullable=False),
            sa.Column(
                "graded_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # Reverse order: child first. ``grade_logs`` has no FK back to
    # ``fsrs_cards`` so the order is symbolic, but it matches the
    # project's child-then-parent convention.
    if "grade_logs" in existing_tables:
        op.drop_table("grade_logs")

    if "fsrs_cards" in existing_tables:
        # Drop the unique index if it exists. ``DROP INDEX IF EXISTS``
        # is portable across Postgres + SQLite; an idempotent
        # downgrade matters for the ``downgrade -1 && upgrade head``
        # CI smoke path.
        op.execute(
            f"DROP INDEX IF EXISTS {_FSRS_WORD_ID_UNIQUE_INDEX}"
        )
