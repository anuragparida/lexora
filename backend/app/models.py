from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    Boolean,
    Float,
    DateTime,
    LargeBinary,
    JSON,
)
from sqlalchemy.orm import relationship
from app.database import Base, DATABASE_URL

# Phase 1: embeddings live in a pgvector ``vector(1024)`` column on
# Postgres and a BLOB on SQLite (dev fallback). The pgvector type
# is only importable when the lib is installed — wrap the import so
# the SQLite fallback path doesn't break in environments where the
# pgvector python lib isn't available (the runtime branch below
# falls back to ``LargeBinary``).
if DATABASE_URL.startswith("postgresql"):
    try:
        from pgvector.sqlalchemy import Vector
        _EMBEDDING_TYPE = Vector(1024)
    except ImportError:
        # pgvector lib missing — degrade to LargeBinary so the model
        # still loads. The retrieval endpoint will detect this and
        # return 503.
        _EMBEDDING_TYPE = LargeBinary
else:
    _EMBEDDING_TYPE = LargeBinary


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
    # Phase 1: nullable so the backfill script can populate incrementally
    # without a schema rewrite. The ``vector(1024)`` (Postgres) /
    # ``BLOB`` (SQLite) type is selected at import time above.
    embedding = Column(_EMBEDDING_TYPE, nullable=True)


class Example(Base):
    __tablename__ = "examples"

    id = Column(Integer, primary_key=True, index=True)
    word_id = Column(Integer, ForeignKey("words.id"))
    german = Column(Text)
    english = Column(Text)

    word = relationship("Word", back_populates="examples")
    # Phase 1: nullable embedding column. Same dialect-aware type
    # selection as ``Word.embedding``.
    embedding = Column(_EMBEDDING_TYPE, nullable=True)


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
    compatibility with Phase 6's py-fsrs wiring). Phase 5.2 (card
    t_88b6f1c4) adds the ``word_id`` UNIQUE constraint — one card
    per word, so the Phase 5.3 grader can do a clean
    ``WHERE word_id = ?`` lookup. The constraint is enforced at the
    DB level via the ``ix_fsrs_cards_word_id_unique`` index created
    by the matching Alembic migration (portable across SQLite +
    Postgres). The column here carries the matching ``unique=True``
    flag so SQLAlchemy metadata and the DB agree; the canonical
    constraint is the index, not the SA-level flag (CREATE UNIQUE
    INDEX is portable; ALTER TABLE ADD CONSTRAINT UNIQUE is not).
    """

    __tablename__ = "fsrs_cards"

    id = Column(Integer, primary_key=True, index=True)
    # Phase 5.2: unique=True mirrors the DB-level unique index. The
    # column itself is nullable on the old schema because Phase 0
    # shipped it that way (a card row pre-existed the word in some
    # backfill scenarios); the constraint still fires on non-null
    # duplicates.
    word_id = Column(Integer, unique=True)
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


class GradeLog(Base):
    """Phase 5.2 (card t_88b6f1c4) — per-grade audit trail.

    Every call to ``POST /exercises/grade`` (5.3) writes one row
    here, capturing the full grading snapshot at the moment the
    grade landed. The row's ``trace_id`` (NULL when Langfuse keys
    are unset) is the join key Phase 6's Ragas evaluator will use
    to replay an offline eval back against the Langfuse trace.

    Append-only by design (no UPDATEs, no DELETEs). Phase 6 may add
    a Postgres trigger to enforce the append-only invariant at the
    DB level; Phase 5 relies on application-layer discipline.

    Column set mirrors the Phase 5 metadata contract (``PHASE-5.md``
    §"The metadata contract"). ``exercise_type`` is a plain
    ``String`` (not an Enum) because Phase 5 hard-locks the value
    at the Pydantic wire layer (``Literal["cloze"]``); the DB
    column stays loose so a future exercise kind can be added
    without a schema rewrite.
    """

    __tablename__ = "grade_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    exercise_id = Column(Integer, nullable=False)
    exercise_type = Column(String, nullable=False)  # "cloze" only in Phase 5
    word_id = Column(Integer, nullable=False)
    grade = Column(Integer, nullable=False)  # 1-4
    scheduled_next_due_at = Column(DateTime, nullable=False)
    prev_due_at = Column(DateTime, nullable=False)
    state = Column(Integer, nullable=False)
    stability = Column(Float, nullable=False)
    difficulty = Column(Float, nullable=False)
    reps = Column(Integer, nullable=False)
    lapses = Column(Integer, nullable=False)
    # NULL when Langfuse keys are unset (tests, dev). The Phase 6
    # Ragas join tolerates NULL rows and skips them.
    trace_id = Column(String, nullable=True)
    latency_ms = Column(Integer, nullable=False)
    # ``default=datetime.utcnow`` fires on the Python-side INSERT
    # path; the matching Alembic migration sets ``server_default``
    # for raw-SQL inserts so the column is never NULL.
    graded_at = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )


def _is_pg() -> bool:
    """Dialect discriminator used by ``WeaknessProfile.axes``.

    Mirrors the convention in ``app.database`` (DATABASE_URL is the single
    source of truth). Returns True for any URL starting with
    ``postgresql``; False for SQLite (the dev fallback) and any other
    dialect. Phase 2.1 only ships the Postgres + SQLite pair — the
    Postgres-vs-other branch in the migration handles future dialects.
    """
    return DATABASE_URL.startswith("postgresql")


class User(Base):
    """Phase 2 schema: a single learner account.

    Phase 2.1 ships the data layer only — no auth. ``password_hash``
    is intentionally ``nullable=False`` from the start (the
    auth-free ``POST /users`` route accepts a pre-hashed value as
    input for now). Phase 2.2 wires bcrypt + JWT on top via a
    stricter write path (``/auth/signup`` hashes internally) — no
    schema change needed there.

    The relationship to ``WeaknessProfile`` is one-to-one with
    cascade: deleting a user drops their profile. The profile is
    not strictly required at signup time (the route auto-creates
    an empty profile on first ``GET /weakness-profile/{user_id}``),
    so a ``nullable=True`` relationship is unnecessary here — but
    the SQLAlchemy relationship doesn't enforce presence.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    # Raw hash lives here. Never returned in any response shape —
    # ``schemas.UserOut`` does not expose this column. Phase 2.2
    # wires hashing; this card only stores the value.
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    weakness_profile = relationship(
        "WeaknessProfile",
        uselist=False,
        back_populates="user",
        cascade="all, delete-orphan",
    )
    # Phase 3.1: one-to-many. A user can run the diagnostic probe
    # multiple times (re-running after an ``applied`` session creates
    # a fresh row). ``backref="user"`` gives ``DiagnosticSession.user``
    # without a separate relationship declaration on that model.
    # Cascade drops a user's sessions when the user is deleted.
    diagnostic_sessions = relationship(
        "DiagnosticSession",
        backref="user",
        cascade="all, delete-orphan",
    )


class WeaknessProfile(Base):
    """Phase 2 schema: per-user axes declaration (0-3 per axis).

    The ``axes`` column stores a JSON object shaped like
    ``{verbs: 2, collocations: 1, ...}``. The score scale is
    0=unknown / 1=shaky / 2=developing / 3=critical (declared by
    the user; the spec does not lock the meaning — the frontend
    shows tick labels).

    Storage is dialect-aware:
    - Postgres: ``JSON`` (the SA default maps to JSON on modern
      PG; for production we prefer JSONB but the spec card uses
      ``JSON`` which is sufficient for read-write round-trips).
    - SQLite: ``Text`` storing JSON-encoded ``str(dict)``. CRUD
      helpers ``get_weakness_profile`` / ``upsert_weakness_profile``
      hide the serialization so callers see a dict either way.
    """

    __tablename__ = "weakness_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(
        Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True
    )
    # Note: ``default=dict`` only fires on Python-side INSERTs via
    # SQLAlchemy. The Alembic migration sets ``server_default='{}'``
    # for raw SQL inserts (e.g. backfill, manual psql). Both paths
    # converge to the same shape.
    axes = (
        Column(JSON, nullable=False, default=dict)
        if _is_pg()
        else Column(Text, nullable=False, default="{}")
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="weakness_profile")


class DiagnosticSession(Base):
    """Phase 3 schema: one probe attempt per row.

    A diagnostic session collects 10 multiple-choice answers from a
    user about how comfortable they are with specific German
    grammar axes (verbs, prepositions, collocations, ...). The
    deterministic scorer in ``app.diagnostic.scoring`` consumes
    the recorded ``answers_json`` and produces a 0..3 score per
    axis; ``/diagnostic/apply`` UPSERTs the result into the user's
    ``WeaknessProfile``.

    Storage is dialect-aware (mirrors ``WeaknessProfile.axes``):

    - Postgres: ``JSON`` column (the SA default maps to JSON on
      modern PG). The route layer hands a dict in / out, the
      CRUD layer is a no-op for the JSON shape.
    - SQLite: ``Text`` storing JSON-encoded ``str(dict)``. The CRUD
      helpers ``serialize_diagnostic_answers`` /
      ``deserialize_diagnostic_answers`` hide the dialect.

    Lifecycle:

    - On ``POST /diagnostic/start`` a row is inserted with
      ``status='in_progress'`` and ``answers_json={}``.
    - Each ``POST /diagnostic/answer`` patches one
      ``question_id -> choice_index`` entry.
    - ``GET /diagnostic/result`` flips ``status`` to
      ``'completed'`` on first read (idempotent — re-reads stay
      ``completed``).
    - ``POST /diagnostic/apply`` flips ``status`` to ``'applied'``
      and UPSERTs the result into the user's
      ``WeaknessProfile``.

    The row is keyed by a UUID string (``id``) rather than an
    autoincrement int so the client-side handle is opaque — no
    enumeration of other users' sessions through sequential ids.
    """

    __tablename__ = "diagnostic_sessions"

    # UUID stored as String(36) — portable across Postgres + SQLite
    # without the ``pg uuid`` extension. The route layer mints via
    # ``uuid.uuid4()``.
    id = Column(String(36), primary_key=True)
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    started_at = Column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    # Nullable until the first ``/diagnostic/result`` call lands.
    completed_at = Column(DateTime, nullable=True)
    # Allowed values: ``in_progress | completed | applied | skipped``.
    # A CHECK constraint on Postgres enforces the enum; the API
    # layer validates values on write so a SQLite target is equally
    # safe.
    status = Column(
        String(16),
        nullable=False,
        default="in_progress",
        server_default="in_progress",
    )
    # Dialect-aware: JSON on Postgres, JSON-as-Text on SQLite.
    # Both columns default to ``'{}'`` so a backfill row inserted
    # by raw SQL still round-trips through the deserialization helper.
    answers_json = (
        Column(JSON, nullable=False, default=dict)
        if _is_pg()
        else Column(Text, nullable=False, default="{}")
    )

    # ``user`` is provided by the ``backref="user"`` on
    # ``User.diagnostic_sessions`` (declared above) — no explicit
    # relationship is declared here to avoid a name collision.
