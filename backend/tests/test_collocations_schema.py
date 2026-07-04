"""Tests for Phase 7.1 — collocations + prepositional_objects schema.

Card: t_96ab949e.

Coverage map (mirrors the card body's "Acceptance" list):

1. ``alembic upgrade head`` runs cleanly on SQLite against a fresh
   temp DB. The chain lands on
   ``7a2_prepositional_objects_table`` (Phase 7.1 head).
2. ``alembic downgrade -1`` cleanly reverses the Phase 7.1 ops —
   both the ``collocations`` and ``prepositional_objects`` tables
   are dropped. The schema is back at the Phase 5.2 head
   (``p5a2_unique_fsrs_grade_logs``).
3. ``upgrade head`` is idempotent against an already-migrated DB
   (re-running it is a no-op — the ``inspect()`` guards short-circuit
   both Phase 7.1 ops).
4. ``Collocation`` and ``PrepositionalObject`` insert + select
   round-trip preserves every column.
5. The Pydantic ``CollocationOut`` + ``PrepositionalObjectOut``
   schemas enforce the ``Literal`` source_corpus / register / case
   enums at the wire layer (DB column is loose String; the type
   guard is the application boundary).
6. The FK relationship: ``collocations.headword_id`` references
   ``words.id`` with ``ondelete=SET NULL``. A delete of the parent
   word must NOT cascade-wipe the curated row (the corpus is
   read-only — Hard rule #2 of PHASE-7.md).

   Concretely: after deleting the parent word, the collocation
   row stays in place with ``headword_id=NULL``.

   Note: SQLite does NOT enforce FK constraints by default — each
   connection must issue ``PRAGMA foreign_keys=ON``. We listen for
   the engine's ``connect`` event so the test engine gets the
   pragma. This mirrors the production behavior on Postgres
   (where FKs are always enforced) and the implicit SQLite
   behavior (where FKs are silently ignored unless the pragma is
   set). Without this listener, the test would pass trivially on
   SQLite even if the SQLAlchemy model declared ``CASCADE`` —
   which is *not* what we want to verify.

Hermetic: a fresh temp SQLite DB + per-test JWT/decks env stubs.
The tests don't depend on the live Postgres / docker stack.

Run from ``backend/``::

    bash /tmp/runpytest.sh tests/test_collocations_schema.py
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.models import Base, Collocation, PrepositionalObject, Word


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB so the migration sees an empty namespace.

    Mirrors the ``test_fsrs_schema.py`` fixture: pin ``DATABASE_URL``
    to ``sqlite:///<tmp>`` so the alembic subprocess uses the right
    file, plus the JWT/decks env vars so ``app`` imports cleanly.
    """
    db_path = tmp_path / "collocations_schema.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


def _alembic_env(db_path: str, tmp_path: Path) -> dict:
    """Build the env dict the alembic subprocess inherits."""
    return {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }


def _run_alembic(
    db_path: str, tmp_path: Path, *args: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Run an alembic CLI call against ``db_path``.

    Returns the CompletedProcess; the test asserts on
    ``returncode`` + ``stderr`` so failures are debuggable.
    """
    backend_dir = Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(backend_dir),
        env=_alembic_env(db_path, tmp_path),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _inspect_tables(engine) -> dict:
    """Snapshot of {tables, collocation_indexes, prepositional_object_indexes}."""
    insp = inspect(engine)
    out = {
        "tables": set(insp.get_table_names()),
        "collocations_indexes": set(),
        "prepositional_objects_indexes": set(),
    }
    if "collocations" in out["tables"]:
        out["collocations_indexes"] = {
            ix["name"] for ix in insp.get_indexes("collocations")
        }
    if "prepositional_objects" in out["tables"]:
        out["prepositional_objects_indexes"] = {
            ix["name"] for ix in insp.get_indexes("prepositional_objects")
        }
    return out


def _bootstrap_sqlite_with_upgrade_head(db_path: str, tmp_path: Path):
    """Run ``alembic upgrade head`` against ``db_path`` and return a
    session factory.

    Mirrors the ``test_fsrs_schema.py`` helper. The schema-test
    scaffolding needs the schema applied + a live Python-level
    engine so we can drive ORM inserts.
    """
    _run_alembic(db_path, tmp_path, "upgrade", "head")
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return engine, Session


# ---------------------------------------------------------------------------
# Alembic round-trip — SQLite path
# ---------------------------------------------------------------------------


def test_alembic_upgrade_head_runs_cleanly_on_sqlite(sqlite_db_path, tmp_path):
    """``alembic upgrade head`` against a fresh SQLite DB exits 0 and
    applies every migration up to and including the Phase 7.1
    revision (``7a2_prepositional_objects_table``)."""
    result = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert result.returncode == 0, result.stderr

    log_blob = (result.stdout or "") + (result.stderr or "")
    assert "7a2_prepositional_objects_table" in log_blob


def test_alembic_upgrade_head_is_idempotent_on_sqlite(
    sqlite_db_path, tmp_path
):
    """Re-running ``alembic upgrade head`` against an already-migrated
    SQLite DB is a clean no-op — the ``inspect()`` guards short-circuit
    both Phase 7.1 ops. Tests "the migration can be re-applied without
    exploding" (which is what CI smoke and the
    ``downgrade -1 && upgrade head`` idempotency check rely on)."""
    first = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert first.returncode == 0, first.stderr

    second = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert second.returncode == 0, second.stderr


def test_alembic_downgrade_minus_one_reverses_phase_71_ops(
    sqlite_db_path, tmp_path
):
    """``alembic downgrade -1`` cleanly reverses the Phase 7.1 ops:

    - the ``prepositional_objects`` table is dropped
    - the ``collocations`` table is dropped

    The schema lands on ``7a1_collocations_table`` (the second
    downgrade step would also work, dropping both). A subsequent
    ``upgrade head`` re-applies the Phase 7.1 ops cleanly.

    This is the ``downgrade -1 && upgrade head`` smoke path from
    the card acceptance list.
    """
    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    # Inspect mid-state: both Phase 7.1 tables should be present
    # after upgrade.
    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    insp_pre_down = _inspect_tables(engine)
    assert "collocations" in insp_pre_down["tables"]
    assert "prepositional_objects" in insp_pre_down["tables"]

    down = _run_alembic(sqlite_db_path, tmp_path, "downgrade", "-1")
    assert down.returncode == 0, down.stderr

    # Inspect post-state: prepositional_objects should be gone, but
    # collocations should still be there (because ``-1`` only drops
    # the top of the chain — the second downgrade drops collocations).
    insp_post = _inspect_tables(engine)
    assert "prepositional_objects" not in insp_post["tables"]
    assert "collocations" in insp_post["tables"]

    # Second downgrade drops collocations.
    down2 = _run_alembic(sqlite_db_path, tmp_path, "downgrade", "-1")
    assert down2.returncode == 0, down2.stderr
    insp_after2 = _inspect_tables(engine)
    assert "collocations" not in insp_after2["tables"]

    # Re-apply — must be a clean re-run. The migrations'
    # ``inspect()`` guards see fresh namespace and apply both ops
    # again.
    re_up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert re_up.returncode == 0, re_up.stderr
    insp_final = _inspect_tables(
        create_engine(f"sqlite:///{sqlite_db_path}")
    )
    assert "collocations" in insp_final["tables"]
    assert "prepositional_objects" in insp_final["tables"]
    engine.dispose()


def test_postgres_upgrade_head_is_skipped_when_db_url_is_sqlite(
    sqlite_db_path, tmp_path, monkeypatch
):
    """The Phase 7.1 test matrix covers SQLite. Postgres is exercised
    in the live QA hook (the dev container on :25432). This guard
    makes the skip pattern explicit — ``DATABASE_URL`` left unset
    means the test runner is using the SQLite fall-back and we
    don't connect to a live Postgres cluster.

    (Mirrors the "live infra is a manual gate" pattern from
    ``test_fsrs_schema.py``.)
    """
    db_url = os.environ.get("DATABASE_URL", "")
    assert db_url.startswith("sqlite"), (
        f"Phase 7.1 SQLite tests expect DATABASE_URL to point at a "
        f"SQLite file. Got {db_url!r} — unset the env var or run on "
        f"a workspace where the dev Postgres is intentionally "
        f"pointed elsewhere."
    )


# ---------------------------------------------------------------------------
# Schema-level guardrails (run against an upgraded SQLite DB in-process)
# ---------------------------------------------------------------------------


def test_collocation_insert_and_select_round_trip(sqlite_db_path, tmp_path):
    """``Collocation`` insert + select preserves every column. The
    curated corpus row carries the full collocation snapshot at
    write-time; losing a column on round-trip would break the
    Phase 7.2 collocation-cloze generator."""
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        # Need a Word row for the FK — the baseline migration
        # creates the ``words`` table and the ``collocations``
        # table's ``headword_id`` references ``words.id``.
        with Session.begin() as s:
            s.add(
                Word(
                    word="Entscheidung",
                    word_type="noun",
                    frequency="high",
                    level="B1",
                )
            )

        # Drive a Collocation insert via the same session.
        with Session.begin() as s:
            s.add(
                Collocation(
                    headword_id=1,
                    partner_lemma="treffen",
                    frequency_score=0.95,
                    register="neutral",
                    source_corpus="dwds",
                )
            )

        # Select back — every column should match.
        from sqlalchemy import select

        with Session() as s:
            row = s.scalar(select(Collocation))
            assert row is not None
            assert row.headword_id == 1
            assert row.partner_lemma == "treffen"
            assert row.frequency_score == 0.95
            assert row.register == "neutral"
            assert row.source_corpus == "dwds"
            assert row.created_at is not None
    finally:
        engine.dispose()


def test_collocation_fk_to_words_uses_set_null_on_delete(
    sqlite_db_path, tmp_path
):
    """The ``collocations.headword_id`` FK is declared with
    ``ondelete=SET NULL``: deleting the parent ``words`` row
    must NOT cascade-wipe the curated row — the curated corpus
    is read-only and should survive the word's deletion (Hard
    rule #2 of PHASE-7.md).

    Concretely: after deleting the parent word, the collocation
    row stays in place with ``headword_id=NULL``.

    SQLite needs ``PRAGMA foreign_keys=ON`` per-connection for
    FK enforcement; we listen on the engine's ``connect`` event
    so the pragma fires for every new connection from the pool.
    This makes the test exercise the same FK-on-delete behavior
    Postgres enforces by default.
    """
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    # Enable FK enforcement on every new connection.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fks(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    try:
        # Seed: a Word + a Collocation pointing at it.
        with Session.begin() as s:
            s.add(
                Word(
                    word="Entscheidung",
                    word_type="noun",
                    frequency="high",
                    level="B1",
                )
            )
        with Session.begin() as s:
            s.add(
                Collocation(
                    headword_id=1,
                    partner_lemma="treffen",
                    frequency_score=0.95,
                    register="neutral",
                    source_corpus="dwds",
                )
            )

        # Delete the parent Word. SQLite enforces the FK with
        # ``ondelete=SET NULL`` (matching the SQLAlchemy model
        # declaration), so the Collocation row should survive
        # with ``headword_id=None``.
        with Session.begin() as s:
            from sqlalchemy import delete

            s.execute(delete(Word).where(Word.id == 1))

        # Inspect post-state: Collocation still exists, but its
        # ``headword_id`` is now NULL.
        from sqlalchemy import select

        with Session() as s:
            row = s.scalar(select(Collocation))
            assert row is not None, "collocation row was cascade-deleted"
            assert row.headword_id is None, (
                "headword_id should be NULL after parent delete "
                "(ondelete=SET NULL invariant)"
            )
            assert row.partner_lemma == "treffen"
    finally:
        engine.dispose()


def test_prepositional_object_insert_and_select_round_trip(
    sqlite_db_path, tmp_path
):
    """``PrepositionalObject`` insert + select preserves every
    column. ``verb_lemma`` is a free-form string (no FK), so no
    parent Word row is needed."""
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        with Session.begin() as s:
            s.add(
                PrepositionalObject(
                    verb_lemma="warten",
                    preposition="auf",
                    case="Akk",
                    example_sentence="Ich warte auf den Bus.",
                    frequency_score=0.95,
                    source_corpus="dwds",
                )
            )

        from sqlalchemy import select

        with Session() as s:
            row = s.scalar(select(PrepositionalObject))
            assert row is not None
            assert row.verb_lemma == "warten"
            assert row.preposition == "auf"
            assert row.case == "Akk"
            assert row.example_sentence == "Ich warte auf den Bus."
            assert row.frequency_score == 0.95
            assert row.source_corpus == "dwds"
            assert row.created_at is not None
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Pydantic schema-level guardrails (the wire layer)
# ---------------------------------------------------------------------------


def test_collocation_out_rejects_unknown_register_literal():
    """``CollocationOut`` enforces the ``Literal["formal",
    "neutral", "colloquial"]`` register enum at the wire layer
    (PHASE-7.md gotcha #12). A typo'd register like "Formal" (with
    a capital F) must raise ``ValidationError`` at the boundary, not
    silently propagate to the DB."""
    from app.schemas import CollocationOut

    with pytest.raises(ValidationError) as exc_info:
        CollocationOut(
            collocation_id=1,
            headword_id=1,
            partner_lemma="treffen",
            frequency_score=0.5,
            register="Formal",  # typo: capital F is not in the literal
            source_corpus="dwds",
            created_at="2026-07-04T00:00:00",
        )
    assert "register" in str(exc_info.value).lower()


def test_collocation_out_rejects_unknown_source_corpus_literal():
    """``CollocationOut`` enforces the ``Literal["dwds",
    "wiktionary", "manual"]`` source_corpus enum. A typo'd source
    like "google" must raise ``ValidationError``."""
    from app.schemas import CollocationOut

    with pytest.raises(ValidationError) as exc_info:
        CollocationOut(
            collocation_id=1,
            headword_id=1,
            partner_lemma="treffen",
            frequency_score=0.5,
            register="formal",
            source_corpus="google",  # not in the literal
            created_at="2026-07-04T00:00:00",
        )
    assert "source_corpus" in str(exc_info.value).lower()


def test_prepositional_object_out_rejects_unknown_case_literal():
    """``PrepositionalObjectOut`` enforces the ``Literal["Akk",
    "Dat", "Gen"]`` case enum. A typo'd case like "akk" (lowercase)
    must raise ``ValidationError``."""
    from app.schemas import PrepositionalObjectOut

    with pytest.raises(ValidationError) as exc_info:
        PrepositionalObjectOut(
            prepositional_object_id=1,
            verb_lemma="warten",
            preposition="auf",
            case="akk",  # lowercase typo
            example_sentence="Ich warte auf den Bus.",
            frequency_score=0.95,
            source_corpus="dwds",
            created_at="2026-07-04T00:00:00",
        )
    assert "case" in str(exc_info.value).lower()


def test_collocation_out_accepts_all_three_register_values():
    """All three valid register literals (``formal``, ``neutral``,
    ``colloquial``) round-trip through ``CollocationOut``. Negative
    coverage for the literal: every valid value passes."""
    from app.schemas import CollocationOut

    for reg in ("formal", "neutral", "colloquial"):
        out = CollocationOut(
            collocation_id=1,
            headword_id=1,
            partner_lemma="treffen",
            frequency_score=0.5,
            register=reg,
            source_corpus="dwds",
            created_at="2026-07-04T00:00:00",
        )
        # Wire-level JSON key must be ``register`` (alias), even
        # though the Python attribute is ``register_label`` (to
        # avoid shadowing Pydantic v2's ``BaseModel.register``
        # method). This is the seed-script round-trip contract.
        dumped = out.model_dump(by_alias=True)
        assert dumped["register"] == reg
        # And the Python attribute must be ``register_label``.
        assert out.register_label == reg


def test_collocation_out_accepts_all_three_source_corpus_values():
    """All three valid source_corpus literals (``dwds``,
    ``wiktionary``, ``manual``) round-trip through
    ``CollocationOut``."""
    from app.schemas import CollocationOut

    for src in ("dwds", "wiktionary", "manual"):
        out = CollocationOut(
            collocation_id=1,
            headword_id=1,
            partner_lemma="treffen",
            frequency_score=0.5,
            register="formal",
            source_corpus=src,
            created_at="2026-07-04T00:00:00",
        )
        assert out.source_corpus == src


def test_prepositional_object_out_accepts_all_three_case_values():
    """All three valid case literals (``Akk``, ``Dat``, ``Gen``)
    round-trip through ``PrepositionalObjectOut``."""
    from app.schemas import PrepositionalObjectOut

    for c in ("Akk", "Dat", "Gen"):
        out = PrepositionalObjectOut(
            prepositional_object_id=1,
            verb_lemma="warten",
            preposition="auf",
            case=c,
            example_sentence="Ich warte auf den Bus.",
            frequency_score=0.95,
            source_corpus="dwds",
            created_at="2026-07-04T00:00:00",
        )
        assert out.case == c


def test_collocation_out_serializes_with_register_alias():
    """``CollocationOut.model_dump(by_alias=True)`` emits the wire-
    level ``register`` key (matching the SQLAlchemy column name and
    the seed-row JSON contract). The Python attribute name is
    ``register_label`` (to avoid shadowing Pydantic v2's
    ``BaseModel.register`` method), but the *wire* JSON key is
    ``register`` for client compatibility."""
    from app.schemas import CollocationOut

    out = CollocationOut(
        collocation_id=1,
        headword_id=1,
        partner_lemma="treffen",
        frequency_score=0.5,
        register="neutral",
        source_corpus="dwds",
        created_at="2026-07-04T00:00:00",
    )

    # Alias on: wire-level key is ``register``.
    aliased = out.model_dump(by_alias=True)
    assert "register" in aliased
    assert aliased["register"] == "neutral"

    # Alias off: Python attribute name is ``register_label``.
    unaliased = out.model_dump(by_alias=False)
    assert "register_label" in unaliased
    assert unaliased["register_label"] == "neutral"
    # And ``register`` is NOT a key in the un-aliased dump.
    assert "register" not in unaliased


def test_collocation_list_out_envelope_shape():
    """``CollocationListOut`` wraps a list of ``CollocationOut``
    with a ``total`` count. The envelope shape matches the
    ``PrepositionalObjectListOut`` pattern so future endpoints
    can swap one for the other without changing client code."""
    from app.schemas import CollocationListOut, CollocationOut

    items = [
        CollocationOut(
            collocation_id=i,
            headword_id=1,
            partner_lemma=f"partner-{i}",
            frequency_score=0.5,
            register="neutral",
            source_corpus="dwds",
            created_at="2026-07-04T00:00:00",
        )
        for i in range(3)
    ]
    envelope = CollocationListOut(items=items, total=3)
    assert envelope.total == 3
    assert len(envelope.items) == 3
    assert envelope.items[0].collocation_id == 0


def test_prepositional_object_list_out_envelope_shape():
    """``PrepositionalObjectListOut`` wraps a list of
    ``PrepositionalObjectOut`` with a ``total`` count."""
    from app.schemas import PrepositionalObjectListOut, PrepositionalObjectOut

    items = [
        PrepositionalObjectOut(
            prepositional_object_id=i,
            verb_lemma=f"verb-{i}",
            preposition="auf",
            case="Akk",
            example_sentence=f"Example {i}.",
            frequency_score=0.5,
            source_corpus="wiktionary",
            created_at="2026-07-04T00:00:00",
        )
        for i in range(2)
    ]
    envelope = PrepositionalObjectListOut(items=items, total=2)
    assert envelope.total == 2
    assert len(envelope.items) == 2


# ---------------------------------------------------------------------------
# Seed-file structural guardrails
# ---------------------------------------------------------------------------


def test_collocation_seed_file_is_json_lines_with_at_least_200_rows():
    """The ``collocations_seed.json`` file is JSON-Lines (one row
    per line) with at least 200 hand-curated rows (card acceptance
    list). Every row has the five required keys, and every literal
    value is in the allowed enum."""
    backend_dir = Path(__file__).resolve().parent.parent
    seed_path = backend_dir / "app" / "seeds" / "collocations_seed.json"
    assert seed_path.exists(), f"seed file missing at {seed_path}"

    with open(seed_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) >= 200, (
        f"collocations seed file has {len(rows)} rows; "
        f"target >= 200 (card acceptance list)"
    )

    valid_regs = {"formal", "neutral", "colloquial"}
    valid_corpus = {"dwds", "wiktionary", "manual"}
    required_keys = {
        "headword_id",
        "partner_lemma",
        "register",
        "source_corpus",
        "frequency_score",
    }
    for i, r in enumerate(rows):
        missing = required_keys - set(r.keys())
        assert not missing, f"row {i} missing keys {missing}: {r}"
        assert r["register"] in valid_regs, (
            f"row {i} has bad register {r['register']!r}: {r}"
        )
        assert r["source_corpus"] in valid_corpus, (
            f"row {i} has bad source_corpus {r['source_corpus']!r}: {r}"
        )
        assert 0.0 <= r["frequency_score"] <= 1.0, (
            f"row {i} frequency_score out of [0,1]: {r['frequency_score']}"
        )


def test_prepositional_object_seed_file_is_json_lines_with_at_least_200_rows():
    """The ``prepositional_objects_seed.json`` file is JSON-Lines
    with at least 200 hand-curated rows. Every row has the six
    required keys, and every literal value is in the allowed enum."""
    backend_dir = Path(__file__).resolve().parent.parent
    seed_path = backend_dir / "app" / "seeds" / "prepositional_objects_seed.json"
    assert seed_path.exists(), f"seed file missing at {seed_path}"

    with open(seed_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) >= 200, (
        f"prepositional_objects seed file has {len(rows)} rows; "
        f"target >= 200 (card acceptance list)"
    )

    valid_case = {"Akk", "Dat", "Gen"}
    valid_corpus = {"dwds", "wiktionary", "manual"}
    required_keys = {
        "verb_lemma",
        "preposition",
        "case",
        "example_sentence",
        "frequency_score",
        "source_corpus",
    }
    for i, r in enumerate(rows):
        missing = required_keys - set(r.keys())
        assert not missing, f"row {i} missing keys {missing}: {r}"
        assert r["case"] in valid_case, (
            f"row {i} has bad case {r['case']!r}: {r}"
        )
        assert r["source_corpus"] in valid_corpus, (
            f"row {i} has bad source_corpus {r['source_corpus']!r}: {r}"
        )
        assert 0.0 <= r["frequency_score"] <= 1.0, (
            f"row {i} frequency_score out of [0,1]: {r['frequency_score']}"
        )


# ---------------------------------------------------------------------------
# SQLAlchemy model registry — both tables are part of Base.metadata
# ---------------------------------------------------------------------------


def test_phase_71_tables_present_in_base_metadata():
    """``Base.metadata.tables`` includes both Phase 7.1 tables.
    This is the schema-registration guardrail: if a future
    developer adds a new model to ``app/models.py`` but forgets
    to import it (or imports it after ``Base.metadata.create_all``
    has been called), Alembic's autogenerate won't see it. This
    test catches the ``import statement missing`` regression."""
    assert "collocations" in Base.metadata.tables
    assert "prepositional_objects" in Base.metadata.tables

    # And the columns match the spec exactly.
    coll_cols = {c.name for c in Base.metadata.tables["collocations"].columns}
    assert coll_cols == {
        "collocation_id",
        "headword_id",
        "partner_lemma",
        "frequency_score",
        "register",
        "source_corpus",
        "created_at",
    }

    prep_cols = {
        c.name
        for c in Base.metadata.tables["prepositional_objects"].columns
    }
    assert prep_cols == {
        "prepositional_object_id",
        "verb_lemma",
        "preposition",
        "case",
        "example_sentence",
        "frequency_score",
        "source_corpus",
        "created_at",
    }

    # And there is no ``updated_at`` — Hard rule #2 (immutability).
    assert "updated_at" not in coll_cols
    assert "updated_at" not in prep_cols


# ---------------------------------------------------------------------------
# Seed-row input models (Phase 7.1)
# ---------------------------------------------------------------------------


def test_collocation_seed_row_accepts_valid_payload():
    """``CollocationSeedRow`` accepts the seed-file shape (5
    required keys, no ``collocation_id`` / ``created_at`` — those
    are server-generated)."""
    from app.schemas import CollocationSeedRow

    row = CollocationSeedRow.model_validate(
        {
            "headword_id": 1,
            "partner_lemma": "treffen",
            "frequency_score": 0.5,
            "register": "neutral",
            "source_corpus": "dwds",
        }
    )
    assert row.headword_id == 1
    assert row.partner_lemma == "treffen"
    assert row.frequency_score == 0.5
    assert row.register_label == "neutral"
    assert row.source_corpus == "dwds"


def test_collocation_seed_row_rejects_out_of_range_frequency():
    """``CollocationSeedRow`` enforces ``frequency_score in [0, 1]``
    via ``Field(ge=0, le=1)``. The DB column is loose Float; this
    is the application boundary."""
    from app.schemas import CollocationSeedRow

    with pytest.raises(ValidationError):
        CollocationSeedRow.model_validate(
            {
                "headword_id": 1,
                "partner_lemma": "treffen",
                "frequency_score": 1.5,  # out of range
                "register": "neutral",
                "source_corpus": "dwds",
            }
        )


def test_collocation_seed_row_allows_optional_headword_id():
    """``CollocationSeedRow.headword_id`` is Optional (the FK may
    be NULL when the curated row's partner isn't in the
    ``words`` table)."""
    from app.schemas import CollocationSeedRow

    row = CollocationSeedRow.model_validate(
        {
            "headword_id": None,  # optional
            "partner_lemma": "treffen",
            "frequency_score": 0.5,
            "register": "neutral",
            "source_corpus": "dwds",
        }
    )
    assert row.headword_id is None


def test_prepositional_object_seed_row_accepts_valid_payload():
    """``PrepositionalObjectSeedRow`` accepts the seed-file shape
    (6 required keys, no PK / created_at)."""
    from app.schemas import PrepositionalObjectSeedRow

    row = PrepositionalObjectSeedRow.model_validate(
        {
            "verb_lemma": "warten",
            "preposition": "auf",
            "case": "Akk",
            "example_sentence": "Ich warte auf den Bus.",
            "frequency_score": 0.95,
            "source_corpus": "dwds",
        }
    )
    assert row.verb_lemma == "warten"
    assert row.preposition == "auf"
    assert row.case == "Akk"
    assert row.example_sentence == "Ich warte auf den Bus."
    assert row.frequency_score == 0.95
    assert row.source_corpus == "dwds"


# ---------------------------------------------------------------------------
# Seed-script end-to-end (against a fresh SQLite DB)
# ---------------------------------------------------------------------------


def test_seed_collocations_script_inserts_at_least_200_rows(
    sqlite_db_path, tmp_path, monkeypatch
):
    """End-to-end: run ``scripts.seed_collocations`` against a
    fresh migrated SQLite DB and verify >= 200 rows landed in
    the ``collocations`` table. The script is the single write
    path to ``collocations`` outside Alembic (Hard rule #2)."""
    import subprocess

    # Apply migrations first so the seed target table exists.
    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{sqlite_db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }
    backend_dir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "scripts.seed_collocations"],
        cwd=str(backend_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        from sqlalchemy import func, select

        with sessionmaker(bind=engine)() as s:
            n = s.scalar(select(func.count()).select_from(Collocation))
        assert n >= 200, f"expected >= 200 rows, got {n}"
    finally:
        engine.dispose()


def test_seed_prepositional_objects_script_inserts_at_least_200_rows(
    sqlite_db_path, tmp_path
):
    """End-to-end: run ``scripts.seed_prepositional_objects``
    against a fresh migrated SQLite DB and verify >= 200 rows
    landed in the ``prepositional_objects`` table."""
    import subprocess

    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{sqlite_db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }
    backend_dir = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "scripts.seed_prepositional_objects"],
        cwd=str(backend_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        from sqlalchemy import func, select

        with sessionmaker(bind=engine)() as s:
            n = s.scalar(
                select(func.count()).select_from(PrepositionalObject)
            )
        assert n >= 200, f"expected >= 200 rows, got {n}"
    finally:
        engine.dispose()


def test_seed_collocations_is_idempotent(
    sqlite_db_path, tmp_path
):
    """Re-running ``scripts.seed_collocations`` against an
    already-populated DB is a clean no-op — the script exits 0
    with a "skipping seed" message and the row count stays the
    same. No duplicate rows (Hard rule #2 — read-only contract,
    immutable once seeded)."""
    import subprocess

    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{sqlite_db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
    }
    backend_dir = Path(__file__).resolve().parent.parent

    # First run — populates the table.
    first = subprocess.run(
        [sys.executable, "-m", "scripts.seed_collocations"],
        cwd=str(backend_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert first.returncode == 0, first.stderr

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        from sqlalchemy import func, select

        with sessionmaker(bind=engine)() as s:
            n_after_first = s.scalar(
                select(func.count()).select_from(Collocation)
            )

        # Second run — should skip cleanly.
        second = subprocess.run(
            [sys.executable, "-m", "scripts.seed_collocations"],
            cwd=str(backend_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert second.returncode == 0, second.stderr
        assert "skipping seed" in (second.stdout + second.stderr).lower()

        # Row count unchanged (no duplicates).
        with sessionmaker(bind=engine)() as s:
            n_after_second = s.scalar(
                select(func.count()).select_from(Collocation)
            )
        assert n_after_first == n_after_second, (
            f"idempotency violated: first run had {n_after_first} rows, "
            f"second run has {n_after_second}"
        )
    finally:
        engine.dispose()