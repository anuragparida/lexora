"""phase 2: users + weakness_profiles

Revision ID: a15ec4b9f736
Revises: 496091d14711
Create Date: 2026-06-30 17:30:00.000000

Phase 2.1 of the personalized-learner roadmap. Adds the two new
tables that the auth-free data layer (card t_6318d0e1) needs:

- ``users`` — one row per learner. Stores ``email`` (unique,
  indexed) + ``password_hash`` (Phase 2.2 hashes here; this
  card ships the data layer only) + ``created_at``.
- ``weakness_profiles`` — one row per user (FK with UNIQUE on
  ``user_id``). The ``axes`` column is dialect-aware:

  * Postgres: ``JSON`` column (with ``server_default='{}'``).
  * SQLite: ``TEXT`` column storing JSON-encoded dicts.

Idempotent on both dialects. The ``upgrade`` function uses
``Inspector.from_engine`` to check ``get_table_names()`` before
calling ``op.create_table`` — re-running the migration against an
already-migrated DB is a clean no-op (the alembic_version row gets
re-asserted, but the tables don't get re-created).

The downgrade drops ``weakness_profiles`` first (child) and then
``users`` (parent) — same child-then-parent ordering the Phase 1
baseline uses. The ``password_hash`` column is dropped with the
table, so a downgrade + re-upgrade cycle starts from an empty
users table (no historical data loss because no real auth flow
has run yet at this point).

Out of scope: bcrypt hashing, JWT tokens, signup/login/logout
routes, ``auth.py``, ``Depends(get_current_user)`` — those are
card t_<2.2-id>.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "a15ec4b9f736"
down_revision: Union[str, Sequence[str], None] = "496091d14711"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "users" not in existing_tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer, primary_key=True, index=True),
            sa.Column("email", sa.String, unique=True, nullable=False, index=True),
            sa.Column("password_hash", sa.String, nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
        )

    if "weakness_profiles" not in existing_tables:
        # Dialect-aware ``axes`` column: JSON on Postgres, JSON-as-Text
        # on SQLite. Both columns default to ``{}`` via ``server_default``
        # so a backfill row inserted by raw SQL still round-trips
        # cleanly through ``crud._deserialize_axes``.
        if bind.dialect.name == "postgresql":
            axes_type = sa.JSON()
            axes_default = "{}"
        else:
            axes_type = sa.Text()
            axes_default = "{}"

        op.create_table(
            "weakness_profiles",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "user_id",
                sa.Integer,
                sa.ForeignKey("users.id"),
                unique=True,
                nullable=False,
                index=True,
            ),
            sa.Column(
                "axes",
                axes_type,
                nullable=False,
                server_default=axes_default,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime,
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    # Child first (weakness_profiles -> users) to avoid FK violations
    # on a Postgres target. SQLite doesn't enforce FKs by default but
    # the order is harmless there.
    op.drop_table("weakness_profiles")
    op.drop_table("users")