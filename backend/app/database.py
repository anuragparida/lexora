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