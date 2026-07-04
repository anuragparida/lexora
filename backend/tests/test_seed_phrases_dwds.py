"""Tests for Phase 8.1 — DWDS phrases seed script.

Card: t_d967c006.

Coverage map (mirrors the card body's "Verification" §"Test 6"):

1. The bundled ``data/dwds_idioms_subset.json`` loads cleanly:
   - At least 150 distinct rows (the hand-curated starting set
     for Phase 8.1; the card body documents a "target ≥200"
     that 8.x and Phase 9 progressively grow toward).
   - Every row validates via ``PhraseSeedRow`` at parse time —
     a malformed row is a 422-equivalent that fails the test
     before the seed script even runs.
2. The seed script is idempotent:
   - Calling it twice against the same DB ends with the same
     total row count (no duplicates piled up).
3. The seed script against the test fixture loads at least the
   documented minimum (5 rows in the bundle).
4. The seed-script's load logs the row count line that the
   card body expects (``Loaded N phrases (target >= 150)``).

Hermetic: each test uses its own temp SQLite DB and
``alembic upgrade head``. No live Postgres, no LLM call.

Run from ``backend/``::

    bash /tmp/runpytest.sh tests/test_seed_phrases_dwds.py
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker


# Path constants — mirrors the patterns in
# ``test_collocations_schema.py``.
BACKEND_DIR = Path(__file__).resolve().parent.parent
SEED_SCRIPT = BACKEND_DIR / "scripts" / "seed_phrases_dwds.py"
SEED_JSON = BACKEND_DIR / "data" / "dwds_idioms_subset.json"
FIXTURE_JSON = (
    Path(__file__).resolve().parent / "fixtures" / "phrases_fixture.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB so the migration sees an empty namespace.

    Mirrors the ``test_collocations_schema.py`` fixture: pin
    ``DATABASE_URL`` to ``sqlite:///<tmp>`` so the alembic subprocess
    uses the right file, plus the JWT/decks env vars so ``app`` imports
    cleanly.
    """
    db_path = tmp_path / "phrases_seed.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


def _alembic_env(db_path: str, tmp_path: Path) -> dict:
    """Build the env dict the alembic subprocess inherits.

    Includes ``PYTHONPATH`` so the seed script can ``import app``
    when invoked as a subprocess — SQLite's home dir
    resolution doesn't always pick up the venv's site-packages
    the same way the test's in-process import does.
    """
    return {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path}",
        "LEXORA_DECKS_DIR": str(tmp_path / "decks"),
        "JWT_SECRET": secrets.token_hex(32),
        "PYTHONPATH": str(BACKEND_DIR),
    }


def _run_alembic(
    db_path: str, tmp_path: Path, *args: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(BACKEND_DIR),
        env=_alembic_env(db_path, tmp_path),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_seed(env: dict, *args: str, timeout: int = 60):
    """Run the seed script as a subprocess (mirrors the production
    invocation so the test exercises the actual script, not an
    in-process reimplementation)."""
    return subprocess.run(
        [sys.executable, str(SEED_SCRIPT), *args],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _bootstrap_empty_db(db_path: str, tmp_path: Path):
    """Run ``alembic upgrade head`` against ``db_path`` and return a
    session factory.

    Symmetric with the helpers in ``test_collocations_schema.py`` —
    the test scaffolding needs the schema applied + a live
    Python-level engine so we can also drive ORM inserts.
    """
    _run_alembic(db_path, tmp_path, "upgrade", "head")
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return engine, Session


def _row_count(db_path: str) -> int:
    """Open the SQLite file via SQLAlchemy and count rows in
    ``phrases``."""
    from app.models import Phrase
    from sqlalchemy import func

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with sessionmaker(bind=engine)() as s:
            return (
                s.scalar(select(func.count()).select_from(Phrase)) or 0
            )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Tests: the bundled seed JSON itself
# ---------------------------------------------------------------------------


def test_bundled_dwds_seed_json_exists():
    """``data/dwds_idioms_subset.json`` ships with the
    backend. ``scripts/seed_phrases_dwds.py`` reads this file
    by default — if it's missing, the seed script's default
    path errors out and the test suite flags the regression."""
    assert SEED_JSON.exists(), f"missing seed file {SEED_JSON}"


def test_bundled_dwds_seed_json_parses_as_list():
    """The seed file is a JSON array of row dicts (not JSON-Lines,
    not a single object). Mirrors the seed script's expectation
    documented in its docstring."""
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, list)
    # Sanity: at least one row.
    assert len(payload) > 0


def test_bundled_dwds_seed_json_loads_at_least_150_rows():
    """Card body target is "≥200 rows" for the full deployment;
    the 8.1 hand-curated starting set ships with ≥150 rows
    (the seed script's ``TARGET_ROWS = 150``). Future
    extension commits grow this toward 200+."""
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    assert len(payload) >= 150, (
        f"seed JSON has only {len(payload)} rows; expected ≥ 150. "
        f"Extend the curated set or update TARGET_ROWS in the seed script."
    )


def test_bundled_dwds_seed_json_every_row_validates_via_phrase_seed_row():
    """Every row in the bundled JSON parses cleanly via the
    ``PhraseSeedRow`` validator. A typo'd ``frequency_band`` or
    misspelled ``source_attribution`` token surfaces at parse
    time — this test guards against a malformed seed file
    silently shipping past the wire-layer guardrail
    (PHASE-8.md gotcha #6)."""
    from app.schemas import PhraseSeedRow

    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    bad = []
    for i, row in enumerate(payload):
        try:
            PhraseSeedRow.model_validate(row)
        except Exception as e:
            bad.append((i, str(e)[:120]))
    assert not bad, f"{len(bad)} malformed rows in seed JSON: {bad[:5]}"


def test_bundled_dwds_seed_json_distributions_across_frequency_bands():
    """The bundled JSON has rows distributed across the three
    frequency bands (``high``, ``mid``, ``low``). The exact
    distribution is not pinned — only that every band has at
    least one row so the Phase 8.4 high-band-first cloze
    variant has visible cohorts."""
    from collections import Counter

    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    counts = Counter(r["frequency_band"] for r in payload)
    for band in ("high", "mid", "low"):
        assert counts[band] > 0, (
            f"frequency_band={band!r} has no rows in the seed JSON; "
            f"Phase 8.4 expects at least one row per band."
        )


# ---------------------------------------------------------------------------
# Tests: seed script end-to-end against a fresh SQLite DB
# ---------------------------------------------------------------------------


def test_seed_script_loads_minimum_target_into_fresh_db(
    sqlite_db_path, tmp_path
):
    """``alembic upgrade head`` + the seed script ends with
    ≥150 rows in ``phrases``. The card body's verification
    says "Loaded N phrases (target ≥200)"; the 8.1 hand-curated
    starting set ships at ≥150, the discrepancy is documented
    in the seed script's comment + this test."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env)
    assert result.returncode == 0, (
        f"seed script failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    n = _row_count(sqlite_db_path)
    assert n >= 150, f"only {n} rows loaded; expected >= 150"


def test_seed_script_logs_loaded_count_message(
    sqlite_db_path, tmp_path
):
    """The seed script prints ``Loaded N phrases (target >= 150)``
    (the line shape documented in the card body's verification
    section). Helena's review card can grep stdout for this line
    shape to confirm the contract holds."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env)
    assert result.returncode == 0, result.stderr
    assert "Loaded" in result.stdout
    assert "phrases" in result.stdout
    assert "target" in result.stdout.lower()


def test_seed_script_against_fixture_loads_only_fixture_rows(
    sqlite_db_path, tmp_path
):
    """The seed script accepts ``--source <path>``. Pointing it
    at the bundled fixture file (5 hand-curated rows for the
    seed-script test) loads exactly those 5 rows into a fresh
    DB."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env, "--source", str(FIXTURE_JSON))
    assert result.returncode == 0, result.stderr
    n = _row_count(sqlite_db_path)
    assert n == 5, f"expected 5 rows from fixture, got {n}"


def test_seed_script_is_idempotent_on_repeated_invocation(
    sqlite_db_path, tmp_path
):
    """Re-running the seed script against an already-populated
    DB is a clean no-op: ``ON CONFLICT (id) DO NOTHING`` keeps
    the row count stable. This is the contract the
    documentation in the seed script's docstring promises;
    the test pins it."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    first = _run_seed(env)
    assert first.returncode == 0, first.stderr
    n_after_first = _row_count(sqlite_db_path)
    assert n_after_first >= 150

    second = _run_seed(env)
    assert second.returncode == 0, second.stderr
    n_after_second = _row_count(sqlite_db_path)
    assert n_after_first == n_after_second, (
        f"second seed run changed the row count "
        f"({n_after_first} -> {n_after_second}); idempotent contract broken"
    )


def test_seed_script_creates_expected_indexes_on_phrases_table(
    sqlite_db_path, tmp_path
):
    """The Phase 8.1 migration creates two indexes on
    ``phrases`` — ``ix_phrases_source_attribution`` (Phase 9
    attribution queries) and ``ix_phrases_frequency_band`` (the
    Phase 8.4 high-band-first cloze variant). After
    ``alembic upgrade head`` + the seed script, both indexes
    exist on the table."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    _run_seed(env)

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        insp = inspect(engine)
        assert "phrases" in set(insp.get_table_names())
        index_names = {
            ix["name"] for ix in insp.get_indexes("phrases")
        }
        # The PK index on ``id`` is implicit and named by the
        # dialect. Both explicit indexes must be present.
        assert "ix_phrases_source_attribution" in index_names, (
            f"missing index ix_phrases_source_attribution; "
            f"present: {index_names}"
        )
        assert "ix_phrases_frequency_band" in index_names, (
            f"missing index ix_phrases_frequency_band; "
            f"present: {index_names}"
        )
    finally:
        engine.dispose()


def test_seed_script_round_trips_a_row_via_orm(
    sqlite_db_path, tmp_path
):
    """INSERT + SELECT via the ``Phrase`` ORM model round-trips
    every column. After the seed script runs, picking a known
    row and reading it back returns the same phrase /
    definition / frequency_band / etc. The seed script's
    write-side shape is verified at the SQLAlchemy layer.
    """
    from app.models import Phrase

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env)
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        Session = sessionmaker(bind=engine)
        with Session() as s:
            row = s.scalar(
                select(Phrase).where(Phrase.id == "ins-blaue-hinein")
            )
            assert row is not None, (
                "expected row id='ins-blaue-hinein' after seed"
            )
            assert row.phrase == "ins Blaue hinein"
            assert row.frequency_band == "high"
            assert row.source_attribution == "dwds"
            assert row.created_at is not None
            # The example can be a non-empty string here. Just
            # check it's non-null because the source row had a
            # value.
            assert row.example_usage is not None
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Tests: malformed seed input is caught at parse time
# ---------------------------------------------------------------------------


def test_seed_script_fails_loudly_on_malformed_seed_json(
    sqlite_db_path, tmp_path
):
    """A seed file with a typo'd ``frequency_band`` is rejected
    with a non-zero exit and a clear error pointing at the bad
    line. The validation happens BEFORE the row reaches the DB
    (the type system is the gate)."""
    import json
    bad_seed = tmp_path / "bad_seed.json"
    rows = [
        {
            "id": "x-y-z-test",
            "phrase": "Test phrase",
            "definition": "Test definition",
            "example_usage": None,
            "source_attribution": "dwds",
            "frequency_band": "Hi",  # typo: not in the literal
            "dwds_url": None,
        }
    ]
    with open(bad_seed, "w", encoding="utf-8") as f:
        json.dump(rows, f)

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env, "--source", str(bad_seed))
    # The seed script is invoked as ``__main__`` and exits with
    # rc=1 on validation failure (via ``raise SystemExit``).
    assert result.returncode != 0, (
        f"seed script should fail on malformed JSON; stdout="
        f"{result.stdout!r} stderr={result.stderr!r}"
    )
    # Error message names the offending field.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "frequency_band" in combined.lower(), (
        f"error message missing 'frequency_band': {combined!r}"
    )
