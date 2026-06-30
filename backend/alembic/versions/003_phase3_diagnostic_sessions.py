"""phase 3: diagnostic_sessions

Revision ID: 003_phase3_diagnostic
Revises: a15ec4b9f736
Create Date: 2026-07-01 10:00:00.000000

Phase 3.1 of the personalized-learner roadmap. Adds the
``diagnostic_sessions`` table that the auth-gated probe endpoints
read and write:

- ``id`` — UUID stored as ``String(36)`` (portable across
  Postgres + SQLite; no ``pg uuid`` extension needed).
- ``user_id`` — ``Integer``, FK to ``users.id``, indexed.
- ``started_at`` — ``DateTime``, default ``datetime.utcnow``, NOT
  NULL.
- ``completed_at`` — ``DateTime``, nullable (set when the result
  is first computed, or when Apply runs).
- ``status`` — ``String(16)``, NOT NULL, default ``'in_progress'``.
  Allowed values: ``in_progress | completed | applied | skipped``.
  A CHECK constraint enforces the enum on Postgres; SQLite
  ignores it (SQLite has no native CHECK for enums in the same
  shape, and the API layer validates the values anyway).
- ``answers_json`` — dialect-aware (``JSON`` on Postgres, ``Text``
  on SQLite). Stores ``Dict[str, int]`` mapping
  ``question_id -> choice_index``. The CRUD helper hides the
  serialization so route code always sees a dict.

Idempotent on both dialects. The ``upgrade`` function uses
``Inspector.from_engine`` to check ``get_table_names()`` before
calling ``op.create_table`` — re-running the migration against an
already-migrated DB is a clean no-op (the alembic_version row gets
re-asserted, but the tables don't get re-created).

Out of scope: the question bank lives in
``backend/app/diagnostic/questions.py`` (code, not DB) per the
Phase 3.1 spec — versioning + review are a code-review concern,
not a schema concern. The deterministic scoring module
(``backend/app/diagnostic/scoring.py``) is also code-only.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "003_phase3_diagnostic"
down_revision: Union[str, Sequence[str], None] = "a15ec4b9f736"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "diagnostic_sessions" not in existing_tables:
        # Dialect-aware ``answers_json`` column: JSON on Postgres,
        # JSON-as-Text on SQLite. Both default to ``'{}'`` so a
        # row inserted by raw SQL (backfill, manual psql) still
        # round-trips cleanly through the deserialization helper.
        if bind.dialect.name == "postgresql":
            answers_type = sa.JSON()
            answers_default = "{}"
        else:
            answers_type = sa.Text()
            answers_default = "{}"

        op.create_table(
            "diagnostic_sessions",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.Integer,
                sa.ForeignKey("users.id"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "started_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "completed_at",
                sa.DateTime,
                nullable=True,
            ),
            sa.Column(
                "status",
                sa.String(16),
                nullable=False,
                server_default="in_progress",
            ),
            sa.Column(
                "answers_json",
                answers_type,
                nullable=False,
                server_default=answers_default,
            ),
        )

        # Postgres-only CHECK constraint for the status enum. SQLite
        # has no native enum and the API layer validates the values
        # on write anyway, so the CHECK is belt-and-braces on
        # Postgres only.
        if bind.dialect.name == "postgresql":
            op.create_check_constraint(
                "ck_diagnostic_sessions_status",
                "diagnostic_sessions",
                sa.text(
                    "status IN ('in_progress', 'completed', 'applied', 'skipped')"
                ),
            )


def downgrade() -> None:
    # Drop the CHECK constraint (Postgres) and then the table.
    # ``drop_constraint`` is a no-op on SQLite because the
    # constraint wasn't created there — wrap in a try/except so the
    # downgrade is clean on both dialects.
    try:
        op.drop_constraint(
            "ck_diagnostic_sessions_status", "diagnostic_sessions"
        )
    except Exception:
        # Constraint doesn't exist on this dialect (or already
        # dropped). Safe to continue.
        pass
    op.drop_table("diagnostic_sessions")