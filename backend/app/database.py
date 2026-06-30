import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./data/vocabeo_words.db"
)


def _engine_kwargs(url: str) -> dict:
    """Per-dialect engine kwargs."""
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    if url.startswith("postgresql"):
        # pool_pre_ping guards against stale connections when the
        # postgres container restarts between requests.
        return {"pool_pre_ping": True}
    return {}


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reconfigure_for_test(database_url: str) -> None:
    """Phase 2.2 — point the module's engine and session factory at
    a new DATABASE_URL.

    The default ``engine`` is created at import time, so test
    fixtures that need a fresh per-test database (a different
    ``tmp_path`` SQLite file) must call this to rebind. The test
    fixture in ``tests/test_auth.py`` uses it to isolate the
    SQLite file per test — without it, the first test that signs
    up a user would have its row visible to the next test, and
    the second test's signup would 409.

    Production code never calls this — the live docker stack binds
    to its real ``DATABASE_URL`` at import time and the engine
    stays bound for the lifetime of the process.
    """
    global engine, SessionLocal, DATABASE_URL
    DATABASE_URL = database_url
    engine = create_engine(database_url, **_engine_kwargs(database_url))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)