"""baseline: words, examples, verb_conjugations, fsrs_cards

Revision ID: ca03ee90afb6
Revises:
Create Date: 2026-06-20 19:37:01.074571

Phase 0 baseline. Creates the four tables that exist in the shipped
SQLite corpus (words, examples, verb_conjugations, fsrs_cards) using
``CREATE TABLE IF NOT EXISTS`` so the migration is idempotent:

- On a fresh Postgres DB (the production target): creates the empty
  schema. The corpus data is seeded by ``app.bootstrap.seed_corpus()``
  if and only if ``words`` is empty after this migration runs.
- On the shipped SQLite corpus (dev fallback): every ``IF NOT EXISTS``
  clause matches an existing table, so this is a no-op apart from
  recording the alembic_version row.

Both dialects support ``CREATE TABLE IF NOT EXISTS`` natively. The
types here match the SQLAlchemy models in ``app/models.py`` so
``alembic upgrade`` and ``alembic downgrade`` are clean reversals on
either dialect.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "ca03ee90afb6"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Postgres rejects ``BOOLEAN DEFAULT 0`` (type mismatch — 0 is int,
    # not bool). SQLite accepts both 0 and '0'. Render the right
    # literal per dialect by inspecting the bind URL the migration
    # runs against.
    bind = op.get_bind()
    is_complete_default = "false" if bind.dialect.name == "postgresql" else "0"

    # Postgres validates foreign keys at CREATE TABLE time (SQLite
    # defers). Tables must be created in dependency order:
    # verb_conjugations -> words -> examples -> fsrs_cards.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS verb_conjugations (
            id INTEGER NOT NULL PRIMARY KEY,
            infinitive VARCHAR NOT NULL UNIQUE,
            present_3rd_person VARCHAR,
            simple_past VARCHAR,
            participle VARCHAR
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER NOT NULL PRIMARY KEY,
            word VARCHAR NOT NULL,
            word_type VARCHAR,
            frequency VARCHAR,
            level VARCHAR,
            translations TEXT,
            conjugation TEXT,
            additional_info TEXT,
            is_complete BOOLEAN DEFAULT {is_complete_default},
            conjugation_id INTEGER,
            FOREIGN KEY (conjugation_id) REFERENCES verb_conjugations(id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER NOT NULL PRIMARY KEY,
            word_id INTEGER,
            german TEXT,
            english TEXT,
            FOREIGN KEY (word_id) REFERENCES words(id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS fsrs_cards (
            id INTEGER NOT NULL PRIMARY KEY,
            word_id INTEGER,
            difficulty FLOAT,
            stability FLOAT,
            retrievability FLOAT,
            due_date TIMESTAMP,
            last_review TIMESTAMP,
            reps INTEGER,
            lapses INTEGER,
            state INTEGER,
            elapsed_days INTEGER,
            scheduled_days INTEGER
        )
        """
    )
    # Indexes that the SQLAlchemy models declare as ``index=True`` on
    # the primary key (and the FK on words.conjugation_id for join
    # performance). IF NOT EXISTS keeps these idempotent.
    op.execute("CREATE INDEX IF NOT EXISTS ix_words_conjugation_id ON words (conjugation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_examples_word_id ON examples (word_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_fsrs_cards_word_id ON fsrs_cards (word_id)")


def downgrade() -> None:
    # Order matters: child tables first to avoid FK violations.
    op.execute("DROP INDEX IF EXISTS ix_fsrs_cards_word_id")
    op.execute("DROP INDEX IF EXISTS ix_examples_word_id")
    op.execute("DROP INDEX IF EXISTS ix_words_conjugation_id")
    op.execute("DROP TABLE IF EXISTS fsrs_cards")
    op.execute("DROP TABLE IF EXISTS examples")
    op.execute("DROP TABLE IF EXISTS words")
    op.execute("DROP TABLE IF EXISTS verb_conjugations")