"""Alembic environment for the Lexora backend.

Reads ``DATABASE_URL`` from the environment so the same alembic config
works against both the bundled Postgres container and the SQLite
fallback (used in tests and offline dev). Falls back to the
SQLite corpus path if no env var is set.

Targets ``app.models.Base.metadata`` for autogenerate support.
"""
from logging.config import fileConfig
import os
import sys

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make ``app`` importable when alembic runs from the backend/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DATABASE_URL as DEFAULT_DATABASE_URL  # noqa: E402
from app.models import Base  # noqa: E402

config = context.config

# Override sqlalchemy.url from env so both DB dialects are supported.
db_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live engine."""
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()