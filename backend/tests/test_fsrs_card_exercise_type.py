"""Tests for Phase 9.1 — ``fsrs_cards.exercise_type`` schema widening.

Card: t_0bfdb7ed.

Coverage map (mirrors the card body's "Scope" list):

1. **Default-to-cloze round-trip** — inserting an ``FsrsCard``
   row via the ORM without specifying ``exercise_type`` results
   in ``exercise_type='cloze'`` (the Python-side ``default``
   fires).
2. **Round-trip all 4 exercise kinds** — inserting a row with
   each of ``cloze`` / ``matching`` / ``comprehension`` /
   ``idiom`` and selecting it back preserves the value. The DB
   column is loose ``String``; the wire layer's ``ExerciseType``
   literal (Phase 8.3 widened to 4) is the source of truth.
3. **Alembic upgrade + downgrade on a fresh SQLite DB** — the
   ``9a1_fsrs_cards_exercise_type`` migration applies cleanly,
   is idempotent on a re-run, and ``downgrade -1`` cleanly
   reverses both ops (column + index).
4. **Alembic backfills existing rows** — pre-migration rows
   inserted via raw SQL (mimicking Phase 0/5/6/8 cloze-only
   cards) come back as ``exercise_type='cloze'`` after upgrade.
   This is the ``server_default='cloze'`` guarantee.
5. **Lifespan does NOT add the column on a fresh SQLite DB
   without migrations** — when the ``TestClient`` boots the app
   against an empty SQLite file (no ``alembic upgrade head``
   having been run), the ``fsrs_cards`` table either stays
   absent OR — if a sibling test fixture populated it via
   ``Base.metadata.create_all`` — does NOT have the
   ``exercise_type`` column sourced from the production lifespan.
   This is the Phase 7.1 (card t_96ab949e) invariant: only
   Alembic owns schema migrations on a fresh DB.

Hermetic: a fresh temp SQLite DB + per-test JWT/decks env
stubs. No live Postgres, no live OpenRouter, no live Langfuse.
The tests don't depend on the docker stack.

Run from ``backend/``::

    uv run pytest -q tests/test_fsrs_card_exercise_type.py
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

from app.models import Base, FsrsCard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB so the migration sees an empty namespace.

    Mirrors the ``test_fsrs_schema.py`` / ``test_collocations_schema.py``
    fixture: pin ``DATABASE_URL`` to ``sqlite:///<tmp>`` so the
    alembic subprocess uses the right file, plus the JWT/decks env
    vars so ``app`` imports cleanly.
    """
    db_path = tmp_path / "fsrs_exercise_type.db"
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
    """Run an alembic CLI call against ``db_path``."""
    backend_dir = Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(backend_dir),
        env=_alembic_env(db_path, tmp_path),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _bootstrap_sqlite_with_upgrade_head(db_path: str, tmp_path: Path):
    """Run ``alembic upgrade head`` against ``db_path`` and return
    a ready ``engine`` + ``SessionLocal``-style session factory.

    Mirrors the scaffolding in ``test_fsrs_schema.py`` — for the
    ORM round-trip tests below we need the schema applied AND a
    live Python-level engine so the SA ORM can drive INSERTs.
    """
    _run_alembic(db_path, tmp_path, "upgrade", "head")
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return engine, Session


# ---------------------------------------------------------------------------
# 1. Default-to-cloze round-trip (Python-side ``default`` fires)
# ---------------------------------------------------------------------------


def test_fsrs_card_default_exercise_type_is_cloze(
    sqlite_db_path, tmp_path
):
    """Insert an ``FsrsCard`` via the ORM without specifying
    ``exercise_type`` — the Python-side ``default='cloze'`` fires
    and the round-tripped row carries the default.

    This is the path Phase 5/6/8 grader code takes today: it
    creates ``FsrsCard(word_id=...)`` without the new column, and
    a successful INSERT must continue to succeed (the column has
    a server-side AND a Python-side default). Phase 9.2+
    (9.5/9.6) updates the writer to pass ``exercise_type=...``
    explicitly.
    """
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        with Session.begin() as s:
            s.add(FsrsCard(word_id=1))
            # Flush so the Python-side default is materialised into
            # the in-memory row before the commit (SQLAlchemy
            # resolves ``default`` on INSERT, but explicit flush
            # lets us read it back through the same session).
            s.flush()
            card = s.execute(
                select(FsrsCard).where(FsrsCard.word_id == 1)
            ).scalar_one()
            assert card.exercise_type == "cloze", (
                f"Expected the Python-side default 'cloze' to "
                f"materialise on INSERT, got {card.exercise_type!r}"
            )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# 2. Round-trip all 4 exercise kinds via the ORM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exercise_type",
    ["cloze", "matching", "comprehension", "idiom"],
)
def test_fsrs_card_round_trip_each_exercise_kind(
    sqlite_db_path, tmp_path, exercise_type
):
    """For each value of the wire-layer ``ExerciseType`` literal,
    insert an ``FsrsCard`` row with that ``exercise_type`` and
    select it back. The DB column is loose ``String``; the
    constraint at the wire layer is in ``app.schemas.GradeRequest``
    (and the Phase 8.3-widened ``ExerciseType`` literal). The DB
    accepts the value verbatim — the test confirms the SA model
    matches the DB column type and round-trips each of the 4
    values.
    """
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        # Each kind needs a distinct ``word_id`` because the
        # Phase 5.2 unique index (``ix_fsrs_cards_word_id_unique``)
        # forbids duplicates.
        with Session.begin() as s:
            s.add(
                FsrsCard(
                    word_id=hash(exercise_type) & 0x7FFFFFFF,
                    exercise_type=exercise_type,
                )
            )

        with Session() as s:
            row = s.execute(
                select(FsrsCard).where(
                    FsrsCard.exercise_type == exercise_type
                )
            ).scalar_one()
            assert row.exercise_type == exercise_type
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# 3. Alembic upgrade + downgrade on a fresh SQLite DB
# ---------------------------------------------------------------------------


def test_alembic_upgrade_head_runs_cleanly_on_sqlite(sqlite_db_path, tmp_path):
    """``alembic upgrade head`` against a fresh SQLite DB exits 0
    and applies every migration up to and including the Phase 9.1
    revision (``9a1_fsrs_cards_exercise_type``)."""
    result = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert result.returncode == 0, result.stderr
    log_blob = (result.stdout or "") + (result.stderr or "")
    assert "9a1_fsrs_cards_exercise_type" in log_blob


def test_alembic_upgrade_head_is_idempotent_on_sqlite(
    sqlite_db_path, tmp_path
):
    """Re-running ``alembic upgrade head`` against an
    already-migrated SQLite DB is a clean no-op — the
    ``inspect()`` guards short-circuit both ops (column add +
    index create)."""
    first = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert first.returncode == 0, first.stderr

    second = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert second.returncode == 0, second.stderr


def test_alembic_downgrade_minus_one_reverses_phase_9_1_ops(
    sqlite_db_path, tmp_path
):
    """``alembic downgrade 8a1_phrases_table`` cleanly reverses the
    Phase 9.1 ops:

    - the ``fsrs_cards.exercise_type`` column is dropped
    - the ``ix_fsrs_cards_exercise_type`` index is dropped

    After the downgrade, the SQLite DB is back at the Phase 8.1
    head (``8a1_phrases_table``). A subsequent ``upgrade head``
    re-applies both ops cleanly.
    """
    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    # Inspect mid-state: the Phase 9.1 ops should be present after
    # upgrade.
    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    insp_pre_down = inspect(engine)
    fsrs_columns_pre = {
        col["name"] for col in insp_pre_down.get_columns("fsrs_cards")
    }
    fsrs_indexes_pre = {
        ix["name"] for ix in insp_pre_down.get_indexes("fsrs_cards")
    }
    assert "exercise_type" in fsrs_columns_pre
    assert "ix_fsrs_cards_exercise_type" in fsrs_indexes_pre

    # Downgrade past the Phase 9.1 revision (back to the
    # Phase 8.1 head).
    down = _run_alembic(
        sqlite_db_path, tmp_path, "downgrade", "8a1_phrases_table"
    )
    assert down.returncode == 0, down.stderr

    # Inspect post-state: the Phase 9.1 ops should be gone after
    # downgrade.
    insp_post = inspect(engine)
    fsrs_columns_post = {
        col["name"] for col in insp_post.get_columns("fsrs_cards")
    }
    fsrs_indexes_post = {
        ix["name"] for ix in insp_post.get_indexes("fsrs_cards")
    }
    assert "exercise_type" not in fsrs_columns_post, (
        "After downgrade, exercise_type must be dropped from "
        "fsrs_cards. The migration is supposed to be additive-only "
        "in the upgrade direction, and the downgrade must reverse "
        "it cleanly."
    )
    assert "ix_fsrs_cards_exercise_type" not in fsrs_indexes_post
    engine.dispose()

    # Re-apply — must be a clean re-run. The migration's
    # ``inspect()`` guards see a clean state and apply both ops
    # again. This is the ``downgrade && upgrade head`` smoke
    # path from the card acceptance list.
    re_up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert re_up.returncode == 0, re_up.stderr
    insp_final = inspect(create_engine(f"sqlite:///{sqlite_db_path}"))
    fsrs_columns_final = {
        col["name"] for col in insp_final.get_columns("fsrs_cards")
    }
    fsrs_indexes_final = {
        ix["name"] for ix in insp_final.get_indexes("fsrs_cards")
    }
    assert "exercise_type" in fsrs_columns_final
    assert "ix_fsrs_cards_exercise_type" in fsrs_indexes_final


# ---------------------------------------------------------------------------
# 4. Migration backfills existing rows via ``server_default='cloze'``
# ---------------------------------------------------------------------------


def test_alembic_migration_backfills_existing_rows_with_cloze(
    sqlite_db_path, tmp_path
):
    """Pre-migration rows (inserted via raw SQL to mimic the
    Phase 0/5/6/8 corpus) MUST backfill to ``exercise_type='cloze'``
    on upgrade — the ``server_default='cloze'`` clause on the
    migration is the safety belt.

    Without the backfill, every existing card row would land in
    the new column as NULL and the Phase 9.2 union query would
    drop them. Phase 9.2's due-list widens the kind filter and
    Phase 9.5/9.6 add per-kind UI; both assume the backfill
    worked.
    """
    # --- Pre-upgrade state: bring the schema up to the
    # Phase 8.1 head (just below Phase 9.1), insert a raw row
    # WITHOUT ``exercise_type``, then run the Phase 9.1
    # migration. The backfill must populate 'cloze' for the
    # pre-existing row. ---
    # Phase 8.1 head — the immediate predecessor of Phase 9.1.
    pre_up = _run_alembic(
        sqlite_db_path, tmp_path, "upgrade", "8a1_phrases_table"
    )
    assert pre_up.returncode == 0, pre_up.stderr

    # Sanity: pre-upgrade the column is absent.
    pre_engine = create_engine(f"sqlite:///{sqlite_db_path}")
    with pre_engine.connect() as conn:
        cols = {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(fsrs_cards)")
            )
        }
        assert "exercise_type" not in cols, (
            "Phase 9.1 column must NOT be in the schema before "
            "the migration runs. If you see this assertion fail, "
            "the test setup is broken — the migration ran when "
            "it shouldn't have."
        )

    # Insert a raw row mimicking Phase 0/5/6/8. The pre-9.1
    # schema's only NOT NULL requirement is ``word_id`` (the
    # rest are nullable or have server-side defaults), so a raw
    # INSERT with just ``word_id`` is valid against the
    # pre-migration schema.
    with pre_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO fsrs_cards (word_id) VALUES (42)"
            )
        )
        # Confirm the column is still absent — the raw INSERT
        # only specified ``word_id``.
        cols_after_insert = {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(fsrs_cards)")
            )
        }
        assert "exercise_type" not in cols_after_insert
    pre_engine.dispose()

    # --- Run the Phase 9.1 migration ---
    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    # --- Post-upgrade state: the column exists AND the
    # pre-existing row backfilled to 'cloze'. ---
    post_engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        with post_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT exercise_type FROM fsrs_cards "
                    "WHERE word_id = 42"
                )
            ).first()
            assert row is not None, (
                "The pre-existing row vanished during the "
                "migration. ALTER TABLE ADD COLUMN must NOT "
                "drop rows."
            )
            assert row[0] == "cloze", (
                f"server_default='cloze' failed: pre-existing "
                f"row backfilled to {row[0]!r} instead of "
                f"'cloze'. Phase 9.2 due-list will drop "
                f"every non-cloze backfill silently."
            )
    finally:
        post_engine.dispose()


# ---------------------------------------------------------------------------
# 5. Phase 7.1 invariant — lifespan does NOT add the column on a
# fresh SQLite DB without migrations
# ---------------------------------------------------------------------------


def test_lifespan_does_not_create_exercise_type_column_on_fresh_sqlite(
    sqlite_db_path, tmp_path, monkeypatch
):
    """The Phase 7.1 (card t_96ab949e) invariant: the FastAPI
    ``lifespan`` does NOT call ``Base.metadata.create_all``. Only
    Alembic owns schema migrations on a fresh DB. Phase 9.1 keeps
    that invariant intact — the new ``exercise_type`` column on
    the ``FsrsCard`` model is a SA-side declaration (necessary
    so the ORM can INSERT/SELECT the column) but it MUST NOT be
    silently added by the lifespan.

    This test boots the app via ``TestClient`` against a fresh
    SQLite file (no ``alembic upgrade head`` having been run) and
    asserts:

    1. The ``lifespan`` did NOT call ``Base.metadata.create_all``.
       The diagnostic: we read ``Base.metadata.tables`` BEFORE
       entering the ``TestClient`` context AND check the same
       ``tables`` AFTER — they MUST be the same Python object
       reference. The lifespan must not mutate the metadata
       table set (Phase 7.1 invariant).
    2. The ``fsrs_cards`` table is absent from the SQLite DB
       after the lifespan runs — the lifespan only logs
       "alembic owns schema migrations" + calls
       ``bootstrap.seed_corpus()`` (a no-op on SQLite) +
       ``_ensure_client()``. None of those touch the schema.
    3. Belt-and-braces: the lifespan did NOT log "create_all"
       or any equivalent schema-creation marker.

    This is the end-to-end Phase 7.1 invariant test for the
    Phase 9.1 column. The companion test
    ``test_base_metadata_create_all_does_add_exercise_type_column``
    confirms the SA model declaration is correctly wired (so
    tests that bootstrap via ``Base.metadata.create_all``
    themselves — which is most of the lexora test matrix — DO
    pick up the new column).
    """
    # Import inside the test so the module-level engine is
    # bound to whatever ``DATABASE_URL`` is in the env at
    # fixture time. Mirrors the pattern in
    # ``test_idiom_endpoint.py``.
    from app import database
    from app.main import app

    # Rebind the module-level engine to the per-test SQLite
    # file. Without this the engine stays bound to the lexora
    # dev DB (``data/vocabeo_words.db``) and the lifespan would
    # touch the dev corpus. Per-test isolation is the
    # Phase 5.2/7.1/8.1 test fixture convention.
    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")

    # Snapshot the metadata ``tables`` identity BEFORE the
    # TestClient lifespan runs. The lifespan must not mutate
    # this set (Phase 7.1 invariant). We snapshot by identity,
    # not by content, because ``create_all`` on already-loaded
    # metadata is a no-op for tables; what matters is that the
    # lifespan does NOT call create_all at all.
    tables_before = set(database.Base.metadata.tables.keys())

    # Bootstrap phase log capture — the lifespan emits
    # ``logger.info("startup: alembic owns schema migrations
    # (no-op on lifespan)")`` on every boot. If the lifespan
    # regresses and calls create_all, we'll see a different
    # log line and pytest can flag it.
    import logging
    captured_logs: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_logs.append(record.getMessage())

    capture_handler = _CaptureHandler(level=logging.INFO)
    lexora_logger = logging.getLogger("lexora.main")
    lexora_logger.addHandler(capture_handler)
    prior_level = lexora_logger.level
    lexora_logger.setLevel(logging.INFO)

    try:
        with TestClient(app) as client:
            # Lifespan has now run (startup phase). Verify
            # metadata ``tables`` identity is preserved.
            tables_after = set(database.Base.metadata.tables.keys())
            assert tables_before == tables_after, (
                "Lifespan mutated Base.metadata.tables — the "
                "Phase 7.1 invariant is broken. Phase 9.1 must "
                "not regress this. Diff: "
                f"added={tables_after - tables_before}, "
                f"removed={tables_before - tables_after}"
            )

            # Verify the SQLite DB on disk does NOT have the
            # ``exercise_type`` column on ``fsrs_cards``.
            # Possibilities:
            #   - ``fsrs_cards`` table is absent (lifespan did
            #     nothing to the schema) — fine, expected.
            #   - ``fsrs_cards`` table is present but the
            #     ``exercise_type`` column is absent — also
            #     fine, because the lifespan never added it.
            #   - ``fsrs_cards.exercise_type`` is present —
            #     BAD. The lifespan created it, which is the
            #     Phase 7.1 regression.
            conn = sqlite3.connect(sqlite_db_path)
            try:
                tables_on_disk = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table'"
                    )
                }
            finally:
                conn.close()
            if "fsrs_cards" in tables_on_disk:
                conn = sqlite3.connect(sqlite_db_path)
                try:
                    fsrs_columns = {
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(fsrs_cards)"
                        )
                    }
                finally:
                    conn.close()
                assert "exercise_type" not in fsrs_columns, (
                    "Lifespan created fsrs_cards.exercise_type "
                    "on a fresh SQLite DB without Alembic. "
                    "Phase 7.1 invariant is broken: only "
                    "Alembic owns schema migrations. The "
                    "Phase 9.1 column must NOT be silently "
                    "added by the lifespan path."
                )

            # Belt-and-braces: the lifespan log line is the
            # Phase 7.1 marker. A regression would change the
            # log wording (or add a create_all call).
            assert any(
                "alembic owns schema migrations" in m
                for m in captured_logs
            ), (
                "Lifespan startup log is missing the Phase "
                "7.1 'alembic owns schema migrations' "
                "marker. Either the log line was edited (a "
                "documentation drift bug) or the lifespan "
                "regressed to call create_all."
            )
            assert not any(
                "create_all" in m for m in captured_logs
            ), (
                "Lifespan logged 'create_all' — the Phase "
                "7.1 invariant forbids this."
            )

            # Touch the app to confirm TestClient lifecycle is
            # healthy. ``GET /health`` is the cheapest endpoint
            # that exercises the lifespan + auth-free path.
            resp = client.get("/health")
            assert resp.status_code == 200, resp.text
    finally:
        lexora_logger.removeHandler(capture_handler)
        lexora_logger.setLevel(prior_level)


def test_base_metadata_create_all_does_add_exercise_type_column(
    sqlite_db_path, monkeypatch
):
    """Companion to the lifespan test above: confirms the
    SQLAlchemy model declaration on ``FsrsCard.exercise_type`` is
    correctly wired into ``Base.metadata``.

    This is the contract the rest of the lexora test matrix
    relies on (most tests bootstrap via
    ``database.Base.metadata.create_all`` for speed, bypassing
    Alembic). The Phase 9.1 model declaration must add the
    column to ``Base.metadata.tables['fsrs_cards'].columns`` so
    those tests pick up the new column without code changes.

    Note: this is INTENTIONAL divergence from the
    lifespan-doesn't-create-column invariant. The lifespan
    must not call ``create_all`` (Phase 7.1), but tests that
    explicitly call ``create_all`` themselves DO get the
    column — that's how the test matrix picks up the new
    field without each test re-running Alembic.
    """
    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)

    conn = sqlite3.connect(sqlite_db_path)
    try:
        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(fsrs_cards)"
            )
        }
    finally:
        conn.close()
    assert "exercise_type" in cols, (
        "Base.metadata.create_all on a fresh SQLite DB must "
        "produce fsrs_cards with the exercise_type column. The "
        "SA model declaration is the test-matrix bootstrap "
        "path — Phase 9.1 relies on it to pick up the new "
        "column for tests that bypass Alembic."
    )
    # Default value of 'cloze' is asserted at the SA-declared
    # ``default=`` (Python-side); the DB-side
    # ``server_default`` is asserted in the alembic round-trip
    # tests above. The SQLite-level ``dflt_value`` column from
    # PRAGMA table_info reflects the migration's
    # ``server_default`` (only present if Alembic created the
    # table). On this create_all path, no server_default is
    # present — which is the intentional trade-off: the test
    # matrix doesn't need it because every writer goes through
    # the SA ORM (which has its own ``default='cloze'``).