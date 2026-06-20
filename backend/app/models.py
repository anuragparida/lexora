from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    Boolean,
    Float,
    DateTime,
)
from sqlalchemy.orm import relationship
from app.database import Base


class Word(Base):
    __tablename__ = "words"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, nullable=False)
    word_type = Column(String)
    frequency = Column(String)
    level = Column(String)
    translations = Column(Text)
    conjugation = Column(Text)
    additional_info = Column(Text)
    is_complete = Column(Boolean, default=False)
    conjugation_id = Column(Integer, ForeignKey("verb_conjugations.id"), nullable=True)

    examples = relationship(
        "Example", back_populates="word", cascade="all, delete-orphan"
    )
    verb_conjugation = relationship("VerbConjugation", back_populates="words")


class Example(Base):
    __tablename__ = "examples"

    id = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer, ForeignKey("words.id"))
    german = Column(Text)
    english = Column(Text)

    word = relationship("Word", back_populates="examples")


class VerbConjugation(Base):
    __tablename__ = "verb_conjugations"

    id = Column(Integer, primary_key=True, index=True)
    infinitive = Column(String, nullable=False, unique=True)
    present_3rd_person = Column(String)
    simple_past = Column(String)
    participle = Column(String)

    words = relationship("Word", back_populates="verb_conjugation")


class FsrsCard(Base):
    """FSRS-algorithm spaced-repetition state for a Word.

    Added in Phase 0 baseline because the shipped SQLite corpus already
    includes this table (created empty by the loader for forward
    compatibility with Phase 6's py-fsrs wiring). Phase 6 will evolve
    it; for now the schema is fixed.
    """

    __tablename__ = "fsrs_cards"

    id = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer)
    difficulty = Column(Float)
    stability = Column(Float)
    retrievability = Column(Float)
    due_date = Column(DateTime)
    last_review = Column(DateTime)
    reps = Column(Integer)
    lapses = Column(Integer)
    state = Column(Integer)
    elapsed_days = Column(Integer)
    scheduled_days = Column(Integer)
