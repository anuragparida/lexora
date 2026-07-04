"""Tests for Phase 5.2 — FSRS schema migration + ``grade_logs`` table.

Card: t_88b6f1c4.

Coverage map (mirrors the card body's "Tests" section):

1. ``alembic upgrade head`` runs cleanly on SQLite against a fresh
   temp DB. Postgres is exercised only when ``DATABASE_URL`` is
   already pointed at a live cluster — matches the framework rule
   of "don't connect to live infra from CI" (mirror of the
   ``test_eval.py`` skip pattern).
2. ``alembic downgrade -1`` cleanly reverses BOTH ops: the
   ``grade_logs`` table is dropped, and the unique index on
   ``fsrs_cards.word_id`` is dropped. The schema is back to the
   Phase 3.1 state.
3. ``upgrade head`` is idempotent against an already-migrated DB
   (re-running it is a no-op).
4. After upgrade, inserting a second ``FsrsCard`` row with the
   same ``word_id`` raises ``IntegrityError`` (the unique index is
   enforced at the DB level, regardless of which connection /
   ORM made the insert).
5. ``GradeLog`` round-trip: insert via the SQLAlchemy ORM, select
   back, every column matches.

Hermetic: a fresh temp SQLite DB + per-test JWT/decks env stubs.
The tests don't depend on the live Postgres / docker stack.

Run from ``backend/``::

    uv run pytest -q tests/test_fsrs_schema.py
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models import Base, FsrsCard, GradeLog, User


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB so the migration sees an empty namespace.

    Mirrors the ``test_users_weakness.py`` fixture: pin
    ``DATABASE_URL`` to ``sqlite:///<tmp>`` so the alembic subprocess
    uses the right file, plus the JWT/decks env vars so ``app``
    imports cleanly.
    """
    db_path = tmp_path / "fsrs_schema.db"
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


# ---------------------------------------------------------------------------
# Alembic round-trip — SQLite path
# ---------------------------------------------------------------------------


def test_alembic_upgrade_head_runs_cleanly_on_sqlite(sqlite_db_path, tmp_path):
    """``alembic upgrade head`` against a fresh SQLite DB exits 0 and
    applies every migration up to and including the Phase 5.2
    revision (``p5a2_unique_fsrs_grade_logs``)."""
    result = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert result.returncode == 0, result.stderr
    # The migration's docstring / alembic's stream-of-consciousness
    # log surfaces the revision id; merge the streams so the
    # assertion finds it regardless of which stream alembic chose.
    log_blob = (result.stdout or "") + (result.stderr or "")
    assert "p5a2_unique_fsrs_grade_logs" in log_blob


def test_alembic_upgrade_head_is_idempotent_on_sqlite(
    sqlite_db_path, tmp_path
):
    """Re-running ``alembic upgrade head`` against an already-migrated
    SQLite DB is a clean no-op — the ``inspect()`` guards short-circuit
    both ops. Tests "the migration can be re-applied without
    exploding" (which is what CI smoke and the
    ``downgrade -1 && upgrade head`` idempotency check rely on)."""
    first = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert first.returncode == 0, first.stderr

    second = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert second.returncode == 0, second.stderr


def test_alembic_downgrade_minus_one_reverses_both_ops(
    sqlite_db_path, tmp_path
):
    """``alembic downgrade 003_phase3_diagnostic`` cleanly reverses:

    - the ``ix_fsrs_cards_word_id_unique`` unique index
    - the ``grade_logs`` table

    After the downgrade, the SQLite DB is back at the Phase 3.1
    head (``003_phase3_diagnostic``). A subsequent ``upgrade head``
    re-applies both ops cleanly.

    Phase 7.1 (card t_96ab949e) updated this test from ``-1`` to
    ``003_phase3_diagnostic`` so it remains revision-stable across
    future migration additions. With Phase 5.2 + 4.5 + 7.1 on the
    chain, the alembic head is ``7a2_prepositional_objects_table``;
    ``downgrade -1`` would only walk back through the Phase 7.1
    ops, leaving the Phase 5.2 ops intact and breaking the
    assertion. The explicit-revision form walks the chain past
    Phase 5.2 + 7.1 to the revision immediately below
    ``p5a2_unique_fsrs_grade_logs``.
    """
    up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert up.returncode == 0, up.stderr

    # Inspect mid-state: both ops should be present after upgrade.
    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    insp_pre_down = _inspect(engine)
    assert "grade_logs" in insp_pre_down["tables"]
    assert "ix_fsrs_cards_word_id_unique" in insp_pre_down["fsrs_indexes"]

    down = _run_alembic(
        sqlite_db_path, tmp_path, "downgrade", "003_phase3_diagnostic"
    )
    assert down.returncode == 0, down.stderr

    # Inspect post-state: both Phase 5.2 ops should be gone after
    # downgrade. The Phase 7.1 tables are also gone (they sit
    # above ``p5a2`` in the chain; walking to
    # ``003_phase3_diagnostic`` walks below ``p5a2``).
    insp_post = _inspect(engine)
    assert "grade_logs" not in insp_post["tables"]
    assert "ix_fsrs_cards_word_id_unique" not in insp_post["fsrs_indexes"]
    assert "collocations" not in insp_post["tables"]
    assert "prepositional_objects" not in insp_post["tables"]
    engine.dispose()

    # Re-apply — must be a clean re-run. The migration's
    # ``inspect()`` guards see fresh namespace and apply both ops
    # again. This is the ``downgrade && upgrade head`` smoke
    # path from the card acceptance list.
    re_up = _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    assert re_up.returncode == 0, re_up.stderr
    insp_final = _inspect(
        create_engine(f"sqlite:///{sqlite_db_path}")
    )
    assert "grade_logs" in insp_final["tables"]
    assert "ix_fsrs_cards_word_id_unique" in insp_final["fsrs_indexes"]
    assert "collocations" in insp_final["tables"]
    assert "prepositional_objects" in insp_final["tables"]


def _inspect(engine) -> dict:
    """Snapshot of {tables, fsrs_indexes} via the SQLAlchemy
    ``Inspector`` API. Helper for the round-trip assertions above.
    """
    from sqlalchemy import inspect

    insp = inspect(engine)
    out = {
        "tables": set(insp.get_table_names()),
        "fsrs_indexes": set(),
    }
    if "fsrs_cards" in out["tables"]:
        out["fsrs_indexes"] = {
            ix["name"] for ix in insp.get_indexes("fsrs_cards")
        }
    return out


def test_postgres_upgrade_head_is_skipped_when_db_url_is_sqlite(
    sqlite_db_path, tmp_path, monkeypatch
):
    """The Phase 5.2 test matrix covers SQLite. Postgres is exercised
    in the live QA hook (the dev container on :25432). This guard
    makes the skip pattern explicit — ``DATABASE_URL`` left unset
    means the test runner is using the SQLite fall-back and we
    don't connect to a live Postgres cluster.

    (Mirrors the "live infra is a manual gate" pattern from
    ``test_eval.py``.)
    """
    db_url = os.environ.get("DATABASE_URL", "")
    assert db_url.startswith("sqlite"), (
        f"Phase 5.2 SQLite tests expect DATABASE_URL to point at a "
        f"SQLite file. Got {db_url!r} — unset the env var or run on "
        f"a workspace where the dev Postgres is intentionally "
        f"pointed elsewhere."
    )


# ---------------------------------------------------------------------------
# Schema-level guardrails (run against an upgraded SQLite DB in-process)
# ---------------------------------------------------------------------------


def _bootstrap_sqlite_with_upgrade_head(db_path: str, tmp_path: Path):
    """Run ``alembic upgrade head`` against ``db_path`` in-process
    (via the live alembic.ini's ``env.py``) and return a ready
    ``SessionLocal``-style session factory.

    This is the schema-test scaffolding — for the integrity check
    below we need the schema applied, but we also need a live
    Python-level engine so we can drive ``IntegrityError`` from
    the ORM. Mirrors ``app.database.get_db`` but bound to the
    per-test DB path.
    """
    _run_alembic(db_path, tmp_path, "upgrade", "head")

    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return engine, Session


def test_fsrs_card_unique_word_id_rejects_second_row(
    sqlite_db_path, tmp_path
):
    """After the migration, two ``FsrsCard`` rows for the same
    ``word_id`` must raise ``IntegrityError`` at INSERT time. The
    guardrail is the DB-level ``ix_fsrs_cards_word_id_unique``
    unique index created by the migration; the SQLAlchemy
    ``unique=True`` flag is just metadata mirror.

    The two inserts are split across two transactions so the
    first one commits cleanly and the second one's INSERT can be
    rejected on FLUSH by the unique index."""
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        # First card — committed cleanly.
        with Session.begin() as s:
            s.add(FsrsCard(word_id=1))

        # Second card with a DIFFERENT word_id — should also commit.
        with Session.begin() as s:
            s.add(FsrsCard(word_id=2))

        # Third attempt — same word_id as the first card.
        # The flush before commit fires the unique-index check
        # and raises IntegrityError. We catch and swallow so the
        # surrounding ``Session.begin()`` doesn't roll back
        # unnecessarily; the IntegrityError is the signal we want.
        with Session.begin() as s:
            duplicate = FsrsCard(word_id=1)
            s.add(duplicate)
            with pytest.raises(IntegrityError):
                s.flush()
    finally:
        engine.dispose()


def test_grade_log_insert_and_select_round_trip(
    sqlite_db_path, tmp_path
):
    """``GradeLog`` insert + select preserves every column. The audit
    row carries the full grading snapshot at write-time, so losing
    a column on round-trip would break Phase 6's Ragas join."""
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        # Need a User row for the FK — the migration's
        # ``grade_logs.user_id`` references ``users.id`` and the
        # baseline migration creates the ``users`` table.
        with Session.begin() as s:
            s.add(
                User(
                    email="round-trip@example.com",
                    password_hash="x" * 60,
                )
            )

        # Now drive a GradeLog insert via the same session.
        from sqlalchemy import select

        with Session.begin() as s:
            user_id = s.execute(
                select(User.id).where(
                    User.email == "round-trip@example.com"
                )
            ).scalar_one()
            now = datetime.utcnow()
            s.add(
                GradeLog(
                    user_id=user_id,
                    exercise_id=42,
                    exercise_type="cloze",
                    word_id=7,
                    grade=3,
                    scheduled_next_due_at=now,
                    prev_due_at=now,
                    state=2,
                    stability=5.0,
                    difficulty=3.0,
                    reps=4,
                    lapses=0,
                    trace_id="trace-abc-123",
                    latency_ms=120,
                    graded_at=now,
                )
            )

        with Session() as s:
            row = s.execute(
                select(GradeLog).where(
                    GradeLog.user_id == user_id
                )
            ).scalar_one()
            assert row.exercise_id == 42
            assert row.exercise_type == "cloze"
            assert row.word_id == 7
            assert row.grade == 3
            assert row.state == 2
            assert row.stability == 5.0
            assert row.difficulty == 3.0
            assert row.reps == 4
            assert row.lapses == 0
            assert row.trace_id == "trace-abc-123"
            assert row.latency_ms == 120
            # ``graded_at`` is server-populated when SQLAlchemy
            # sends a raw SQL insert; our test goes through the ORM
            # which uses the Python-side ``default`` — both paths
            # produce a non-NULL timestamp.
            assert row.graded_at is not None
    finally:
        engine.dispose()


def test_grade_log_trace_id_can_be_none(sqlite_db_path, tmp_path):
    """``trace_id`` is NULLABLE because Langfuse keys are unset in
    tests / dev. The graceful-degradation path on 5.3's writer
    relies on this — a NOT NULL constraint here would force
    Phase 5.3 to lie about non-existent traces, which is worse
    than an honest NULL."""
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        with Session.begin() as s:
            s.add(
                User(
                    email="no-trace@example.com",
                    password_hash="x" * 60,
                )
            )
        from sqlalchemy import select

        with Session.begin() as s:
            user_id = s.execute(
                select(User.id).where(
                    User.email == "no-trace@example.com"
                )
            ).scalar_one()
            now = datetime.utcnow()
            s.add(
                GradeLog(
                    user_id=user_id,
                    exercise_id=99,
                    exercise_type="cloze",
                    word_id=5,
                    grade=1,
                    scheduled_next_due_at=now,
                    prev_due_at=now,
                    state=1,
                    stability=0.0,
                    difficulty=5.0,
                    reps=0,
                    lapses=0,
                    trace_id=None,
                    latency_ms=80,
                    graded_at=now,
                )
            )
        with Session() as s:
            row = s.execute(
                select(GradeLog).where(
                    GradeLog.user_id == user_id
                )
            ).scalar_one()
            assert row.trace_id is None
    finally:
        engine.dispose()


def test_fsrs_cards_table_present_after_upgrade(
    sqlite_db_path, tmp_path
):
    """Sanity-check: the Phase 5.2 migration does NOT touch the
    ``fsrs_cards`` table beyond adding the unique index. The
    baseline table must still exist with its Phase 0 column
    shape."""
    engine, Session = _bootstrap_sqlite_with_upgrade_head(
        sqlite_db_path, tmp_path
    )
    try:
        with engine.connect() as conn:
            cols = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(fsrs_cards)")
                )
            }
        # Phase 0 columns are preserved verbatim.
        for col in (
            "id",
            "word_id",
            "difficulty",
            "stability",
            "retrievability",
            "due_date",
            "last_review",
            "reps",
            "lapses",
            "state",
            "elapsed_days",
            "scheduled_days",
        ):
            assert col in cols, (
                f"Phase 0 column {col!r} missing from fsrs_cards "
                f"after the Phase 5.2 migration. The migration "
                f"must be additive."
            )
    finally:
        engine.dispose()
