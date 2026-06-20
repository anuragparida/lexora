from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/vocabeo_words.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations():
    """Run simple migrations to add missing columns"""
    with engine.connect() as conn:
        # Check if is_complete column exists
        result = conn.execute(text("PRAGMA table_info(words)"))
        columns = [row[1] for row in result]

        if "is_complete" not in columns:
            conn.execute(
                text("ALTER TABLE words ADD COLUMN is_complete BOOLEAN DEFAULT 0")
            )
            conn.commit()
            print("Added is_complete column to words table")
