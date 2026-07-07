from datetime import datetime
import sqlalchemy as sa
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

    Phase 9.1 (card t_0bfdb7ed) adds the ``exercise_type`` column
    so each card row carries the exercise kind it backs (cloze,
    matching, comprehension, idiom). Until Phase 9 the column lived
    only on ``grade_logs``; the card itself was implicitly cloze
    because Phase 5 was the only exercise kind that read it.
    Phase 9 widens ``/exercises/due`` (9.2) to a tagged union of
    exercise kinds, and the scheduler decides "due cloze" vs "due
    matching" per row — the ``fsrs_cards.exercise_type`` lookup is
    the join key. Column shape mirrors the Phase 5.2
    ``grade_logs.exercise_type`` pattern: loose ``String`` on the
    DB (the wire layer's ``ExerciseType`` literal in
    ``app/schemas.py`` is the source of truth), ``nullable=False``
    with a Python-side ``default='cloze'`` for ORM-level inserts
    AND ``server_default='cloze'`` on the matching Alembic
    migration so raw-SQL inserts (``psql -c "INSERT ..."``) get a
    sane value. ``index=True`` on the SA side mirrors the
    ``ix_fsrs_cards_exercise_type`` index owned by Alembic.

    Hard rule — Alembic owns this column on a fresh DB. The SA
    declaration below is necessary so the ORM can INSERT/SELECT the
    column, but on a fresh DB the column is added by the
    ``9a1_fsrs_cards_exercise_type`` migration, NOT by
    ``Base.metadata.create_all``. Phase 7.1 (card t_96ab949e)
    removed ``create_all`` from the ``lifespan``; that invariant
    stays — see ``test_fsrs_card_exercise_type.py`` for the
    end-to-end lifespan assertion.
    """

    __tablename__ = "fsrs_cards"

    id = Column(Integer, primary_key=True, index=True)
    # Phase 5.2: unique=True mirrors the DB-level unique index. The
    # column itself is nullable on the old schema because Phase 0
    # shipped it that way (a card row pre-existed the word in some
    # backfill scenarios); the constraint still fires on non-null
    # duplicates.
    word_id = Column(Integer, unique=True)
    # Phase 9.1: see the class docstring. ``default='cloze'`` is the
    # Python-side mirror of the migration's ``server_default``; it
    # fires on ORM INSERTs that omit the column so a card created
    # by Phase 5/6/8 code paths that haven't been updated to pass
    # ``exercise_type`` explicitly still round-trips. ``index=True``
    # mirrors the ``ix_fsrs_cards_exercise_type`` Alembic index.
    exercise_type = Column(
        String,
        nullable=False,
        default="cloze",
        index=True,
    )
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


class Collocation(Base):
    """Phase 7.1 (card t_96ab949e) — curated collocation rows.

    A *collocation* is a frequent word-pair or short phrase: a
    *headword* (looked up via FK to ``words.id``) plus a
    *partner_lemma* that co-occurs with it. Example: the German
    headword ``Entscheidung`` (decision) collocates strongly with
    the verb ``treffen`` (to make) — yielding the phrase
    *eine Entscheidung treffen*.

    The table is **read-only at runtime** (Hard rule #2 of
    PHASE-7.md). The exercise generator (Phase 7.2) consumes it;
    it never writes back. The only write paths outside Alembic
    are the seed scripts (``backend/scripts/seed_collocations.py``)
    and a hand-curated DWDS / Wiktionary subset shipped as
    ``backend/app/seeds/collocations_seed.json``.

    Columns mirror the card body exactly:

    - ``collocation_id`` — autoincrement PK.
    - ``headword_id`` — FK to ``words.id`` (the *anchor* word of
      the collocation). ``ondelete=SET NULL`` so a future word
      deletion doesn't cascade-wipe the curated row.
    - ``partner_lemma`` — the co-occurring word (free-form string,
      not FK — partner lemmas are not necessarily in the corpus).
    - ``frequency_score`` — Float in [0.0, 1.0] (DWDS-normalized).
      Not used for ranking yet; reserved for the Phase 9 optimizer.
    - ``register`` — ``formal`` / ``neutral`` / ``colloquial``.
      The DB column is loose String; the wire-layer (Pydantic)
      and the seed-row validator enforce the literal at the
      application boundary.
    - ``source_corpus`` — ``dwds`` / ``wiktionary`` / ``manual``.
      Same dialect-agnostic pattern as ``register``.
    - ``created_at`` — DB default to ``datetime.utcnow`` so raw-SQL
      inserts (e.g. ``psql -c "INSERT ..."``) get a sane value
      without going through SQLAlchemy.

    There is **no** ``updated_at`` column: rows are immutable once
    seeded (Hard rule #2 — the table is a curated corpus, not a
    learned one). The seed scripts are the single write path
    outside Alembic.
    """

    __tablename__ = "collocations"

    collocation_id = Column(Integer, primary_key=True, index=True)
    headword_id = Column(
        Integer,
        ForeignKey("words.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    partner_lemma = Column(String, nullable=False)
    frequency_score = Column(Float, nullable=False, default=0.0)
    register = Column(String, nullable=False)
    source_corpus = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=sa.func.now(),
    )


class PrepositionalObject(Base):
    """Phase 7.1 (card t_96ab949e) — curated verb + preposition + case rows.

    A *prepositional object* (German: *Präpositionalobjekt*) is the
    preposition + case pair a verb governs. Example: ``warten auf
    + Akk`` — the verb ``warten`` (to wait) takes the preposition
    ``auf`` in the accusative case, yielding *auf den Zug warten*
    (to wait for the train).

    Same read-only contract as ``Collocations`` (Hard rule #2):
    the exercise generator consumes these rows; it never writes
    back. The seed script
    (``backend/scripts/seed_prepositional_objects.py``) and the
    JSON-Lines seed
    (``backend/app/seeds/prepositional_objects_seed.json``) are the
    only write paths outside Alembic.

    Columns mirror the card body:

    - ``prepositional_object_id`` — autoincrement PK.
    - ``verb_lemma`` — the head verb, free-form (not FK — verbs
      here are lemmas, not always backed by a ``words`` row).
    - ``preposition`` — the German preposition (e.g. ``auf``,
      ``mit``, ``über``).
    - ``case`` — ``Akk`` / ``Dat`` / ``Gen`` — the governed case.
      Tight String column; Pydantic literal at the wire layer.
    - ``example_sentence`` — a worked German example showing the
      verb + preposition + case in context.
    - ``frequency_score`` — Float in [0.0, 1.0], same shape as
      ``Collocations.frequency_score``.
    - ``source_corpus`` — ``dwds`` / ``wiktionary`` / ``manual``.
    - ``created_at`` — DB default for raw-SQL safety.

    No ``updated_at`` — same immutability invariant as
    ``Collocations``.
    """

    __tablename__ = "prepositional_objects"

    prepositional_object_id = Column(Integer, primary_key=True, index=True)
    verb_lemma = Column(String, nullable=False)
    preposition = Column(String, nullable=False)
    case = Column(String, nullable=False)
    example_sentence = Column(Text, nullable=False)
    frequency_score = Column(Float, nullable=False, default=0.0)
    source_corpus = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=sa.func.now(),
    )


class Phrase(Base):
    """Phase 8.1 (card t_d967c006) — curated German idioms.

    A *phrase* here is a fixed multi-word expression that is NOT
    compositional: *ins Blaue hinein* (literally "into the blue in",
    meaning "without a clear plan"), *Tomaten auf den Augen*
    ("tomatoes on the eyes", meaning "blind to what's obvious").
    These are the building blocks of the ``app.idiom`` exercise
    generator (Phase 8.3) and the ``/exercises/idiom`` endpoint
    (Phase 8.4); Phase 9 may also feed the study-session mixer.

    The table is **read-only at runtime** (Hard rule #2 of
    PHASE-8.md). The generator (Phase 8.3) consumes it; it never
    writes back. The only write paths outside Alembic are the seed
    scripts (``backend/scripts/seed_phrases_dwds.py`` for the DWDS
    Idiome subset and ``backend/scripts/seed_phrases_attestations.py``
    for Goethe/Schiller, both in 8.2).

    Columns mirror the card body exactly (the Pydantic shape lives
    in ``app.schemas.Phrase``):

    - ``id`` — slug PK (NOT autoincrement). The seed script
      slugifies the DWDS ``Lemma`` (e.g. ``"ins Blaue hinein"`` →
      ``"ins-blaue-hinein"``). Slug-PK gives the FK reference for
      Phase 9 attribution a stable handle and lets the seed script
      INSERT OR IGNORE cleanly (idempotent on re-run).
    - ``phrase`` — the German surface form (5–200 chars). Pydantic
      bounds enforced at the wire layer; DB column is loose Text
      (the generator reads, never validates, so the column is
      pragmatic). UNIQUE — the same idiom can't appear twice in
      different bands.
    - ``definition`` — a learner-friendly English gloss (1–400
      chars). The Pydantic cap forces the seed script to compress
      DWDS's verbose multi-sentence definitions into a single
      tight sentence.
    - ``example_usage`` — optional German usage example (5–400
      chars). Some DWDS ``<Lemma>`` entries have no ``<Example>``
      child; the seed script tolerates the omission (NULL).
    - ``source_attribution`` — comma-joined literal of
      ``"dwds" / "goethe" / "schiller"``. The same idiom can land
      in both DWDS and a Goethe attestation; in that case the
      column carries ``"dwds,goethe"``. The DB column is loose
      String; the wire layer (and the seed-row validator) enforce
      the literal at the application boundary (PHASE-8.md
      gotcha #6 — same discipline as ``register`` / ``source_corpus``
      on Phase 7.1's tables). Indexed for Phase 9 attribution
      queries.
    - ``frequency_band`` — ``"high"`` / ``"mid"`` / ``"low"`` —
      hand-bucketed by the seed author. Top-100 most common =
      ``"high"``, next 100 = ``"mid"``, the rest = ``"low"``.
      Same loose-String-on-the-DB / Literal-at-the-wire-layer
      pattern. Indexed for the high-band-first cloze variant
      in Phase 8.4.
    - ``dwds_url`` — the source DWDS Idiome URL for the lemma
      (``NULL`` when the row was not sourced from DWDS).
    - ``attested_quote`` — optional Phase 8.2 attestation
      (Goethe/Schiller quotation). DB column is loose TEXT;
      the column is reserved here so 8.2 doesn't need an
      Alembic rewrite (it's an additive seed-step.
    - ``attested_source`` — optional Phase 8.2 source citation
      ("Faust I, Studierzimmer (1168-1186)"). Same dialect-agnostic
      pattern; LOOSE TEXT, Pydantic-bounded at the wire layer.
    - ``created_at`` — DB-side default to ``sa.func.now()`` so raw-
      SQL inserts (``psql -c "INSERT ..."``) get a sane value.

    Two indexes ship in the migration: ``ix_phrases_source_attribution``
    (Phase 9 attribution queries) and ``ix_phrases_frequency_band``
    (Phase 8.4 high-band-first cloze variant). The PK itself is an
    implicit index on ``id``.
    """

    __tablename__ = "phrases"

    # The DWDS ``<Lemma>`` slugified — e.g. ``"Tomaten auf den
    # Augen"`` → ``"tomaten-auf-den-augen"``. Used as the PK so the
    # idempotent seed (``ON CONFLICT (id) DO NOTHING`` /
    # INSERT-OR-IGNORE) can re-run cleanly. Slug is 80-char-capped
    # at the seed boundary (long lemmas are trimmed); the column
    # itself is String (no DB-side cap) because the corpus is
    # hand-curated and the bound lives at the application layer.
    id = Column(String(120), primary_key=True, index=True)
    phrase = Column(Text, nullable=False, unique=True)
    definition = Column(Text, nullable=False)
    # ``example_usage`` is the only nullable text column besides
    # the attestation fields — DWDS occasionally ships an idiom
    # without an example, and the schema must tolerate that.
    example_usage = Column(Text, nullable=True)
    # Comma-joined literal — e.g. ``"dwds"``, ``"dwds,goethe"``,
    # ``"goethe,schiller"``. Loose String on the DB; the wire layer
    # enforces the per-element literal at parse time. Indexed for
    # Phase 9 attribution queries.
    source_attribution = Column(String, nullable=False, index=True)
    # ``"high" / "mid" / "low"`` — hand-bucketed. Indexed for the
    # Phase 8.4 high-band-first cloze variant. Same dialect-agnostic
    # loose-String / Literal-at-the-wire-layer pattern as
    # ``register`` / ``source_corpus`` on the Phase 7.1 tables.
    frequency_band = Column(String, nullable=False, index=True)
    dwds_url = Column(Text, nullable=True)
    # Phase 8.2 attestation columns — reserved here (additive,
    # nullable) so the Goethe/Schiller seed script doesn't need an
    # Alembic rewrite. 8.2 lands as a SECOND seed-script PR, NOT a
    # schema migration; the columns are present from 8.1 onward.
    attested_quote = Column(Text, nullable=True)
    attested_source = Column(Text, nullable=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=sa.func.now(),
    )


class PhrasePair(Base):
    """Phase 10.1 (card t_18c90a68) — curated phrase-pair relationships.

    A *phrase pair* is a relation between two ``phrases`` rows (the
    Phase 8.1 / 8.2 ``phrases`` table — multi-word fixed
    expressions like ``ins Blaue hinein``). The four-way relation
    literal encodes the closeness of the pair::

        - "equivalent": near-synonymous in form AND meaning
          (``Tomaten auf den Augen`` ≈ ``Scheuklappen auf`` is too
          lossy to qualify; this band is reserved for the very
          closest paraphrase-pairs).
        - "paraphrase": same meaning, different surface form
          (``das geht mir auf den Keks`` ≈ ``das nervt mich``).
        - "related": theme- or register-adjacent, not interchangeable
          (``Tomaten auf den Augen`` ~ ``den inneren Schweinehund
          überwinden`` — both are figurative body-part idioms but
          the topic is unrelated).
        - "unrelated": included for the exercise generator's
          distractor surface; ``unrelated`` rows anchor the
          negative-cohort retrieval pull at request time.

    Like ``phrases``, this table is **read-only at runtime**
    (Hard rule #2 of PHASE-8.md, restated verbatim by the
    plan body for Phase 10). The Phase 10.3 ``/exercises/phrase_match``
    endpoint *reads* from it (e.g. for the ``enable_rag=True``
    nearest-neighbor pull); nothing writes to it at request time.

    Columns mirror the card body exactly:

    - ``id`` — autoincrement integer PK (NOT slug). ``phrase_pairs``
      rows are paired-pair identities, not human-readable
      handles — the FKs (``phrase_a_id`` / ``phrase_b_id``) carry
      the meaning. Index is implicit on the PK.
    - ``phrase_a_id`` — FK to ``phrases.id``
      (``ondelete="RESTRICT"`` — never cascade-delete a phrase just
      because its pair row goes away). Indexed
      (``ix_phrase_pairs_phrase_a_id``).
    - ``phrase_b_id`` — FK to ``phrases.id``, same
      ``ondelete="RESTRICT"``. Indexed
      (``ix_phrase_pairs_phrase_b_id``).
    - ``relation`` — VARCHAR, NOT NULL. Loose-String-on-the-DB /
      Literal-at-the-wire-layer (matches ``register`` /
      ``source_corpus`` on Phase 7.1, ``frequency_band`` on 8.1).
      Indexed (``ix_phrase_pairs_relation``) for the
      Phase 10.3 nearest-neighbor pull when the request
      pins a relation.
    - ``attested_pair`` — BOOLEAN, NOT NULL. ``True`` for the
      Goethe / Schiller attested-pair override rows that
      ``backend/data/attested_pairs.json`` declares; ``False``
      for the seed-script's bge-m3-bucketed rows. Indexed
      (``ix_phrase_pairs_attested_pair``) so the planner can
      quickly fetch the attested subset.
    - ``created_at`` — DB-side default ``sa.func.now()`` (mirrors
      Phase 8.1's idiom).

    Hard rules at the DB level:

    - ``phrase_a_id != phrase_b_id`` (CHECK constraint; the card
      body says "422 on a row whose ``phrase_a_id`` and
      ``phrase_b_id`` are the same"). The seed script enforces
      this at insert time; the Phase 10.3 wire endpoint enforces
      it at request time via the matching Pydantic validator
      on ``PhrasePairSeedRow``.
    - ``UNIQUE(phrase_a_id, phrase_b_id)`` — same pair can't
      appear twice with swapped order. The seed script sorts
      ``(a, b)`` lexicographically before insert so the (a, b)
      pair never collides with its (b, a) mirror.

    The migration in ``10a1_phrase_pairs_table`` ships the table
    itself plus all four indexes and the two constraints (UNIQUE
    + CHECK). The migration is ``inspect()``-guarded so re-running
    ``alembic upgrade head`` is a clean no-op.
    """

    __tablename__ = "phrase_pairs"

    id = Column(Integer, primary_key=True, index=True)
    # ``ondelete="RESTRICT"`` — same discipline as Phase 7.1
    # ``collocations.verb_lemma`` FK: a paired-pair row outlives
    # even the removal of its parent phrase (an audit trail is
    # more useful than a silent cascade).
    phrase_a_id = Column(
        String(120),
        ForeignKey("phrases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    phrase_b_id = Column(
        String(120),
        ForeignKey("phrases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    # Loose String on the DB; the Pydantic
    # ``PhrasePairRelation`` literal at the wire layer
    # (``app.schemas``) is the wire-level guardrail.
    relation = Column(String, nullable=False, index=True)
    attested_pair = Column(
        Boolean, nullable=False, default=False, index=True
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=sa.func.now(),
    )
