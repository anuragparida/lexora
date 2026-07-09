"""Tests for Phase 10.1 — phrase_pairs seed script.

Card: t_18c90a68.

Coverage map (mirrors the card body's "Verification" §"Test 6"):

1. Migration applies: ``alembic upgrade head`` against a
   fresh SQLite produces the ``phrase_pairs`` table with the
   expected columns, indexes, UNIQUE constraint, and CHECK
   constraint.

2. The seed script is **deterministic**:
   - ``--seed 42`` produces byte-equal output on two
     consecutive runs against the same phrases table.
   - ``--seed 17`` produces a different bucket
     distribution than ``--seed 42`` (the seed propagates
     into the rank-based quartile assignment).

3. Re-running with the same seed is idempotent — the row
   count is unchanged.

4. Attested-pair rows land with ``attested_pair=True`` and
   the explicit ``relation`` from the JSON, AND override any
   bucketed row at the same (a, b) pair via ON CONFLICT
   DO UPDATE.

5. The self-pair hard rule is enforced at the DB level (CHECK
   constraint + Python pre-filter), so an attempt to insert
   ``a == b`` via the seed script fails loudly.

6. The bundled ``data/attested_pairs.json`` exists, is valid
   JSON, parses cleanly via ``PhrasePairSeedManifest``, and
   starts empty (``{"pairs": []}``) for the 10.1 hand-off —
   10.4 populates it.

Hermetic: each test uses its own temp SQLite DB and
``alembic upgrade head``. No live Postgres, no LLM call (a
deterministic stub similarity function is injected via
``--similarity-fn``).

Run from ``backend/``::

    uv run pytest -q tests/test_seed_phrase_pairs.py
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import Phrase, PhrasePair
from app.schemas import PhrasePairSeedManifest


# Path constants — mirror the patterns in
# ``test_seed_phrases_dwds.py``.
BACKEND_DIR = Path(__file__).resolve().parent.parent
SEED_SCRIPT = BACKEND_DIR / "scripts" / "seed_phrase_pairs.py"
SEED_JSON = BACKEND_DIR / "data" / "attested_pairs.json"
FIXTURE_JSON = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "phrase_pairs_fixture.json"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB so the migration sees an empty namespace.

    Mirrors the ``test_collocations_schema.py`` / ``test_seed_phrases_dwds.py``
    fixture: pin ``DATABASE_URL`` to ``sqlite:///<tmp>`` so the
    alembic subprocess uses the right file, plus the JWT / decks
    env vars so ``app`` imports cleanly.
    """
    db_path = tmp_path / "phrase_pairs_seed.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


def _alembic_env(db_path: str, tmp_path: Path) -> dict:
    """Build the env dict the alembic subprocess inherits.

    Includes ``PYTHONPATH`` so the seed script can ``import app``
    when invoked as a subprocess — SQLite's home dir resolution
    doesn't always pick up the venv's site-packages the same way
    the test's in-process import does.
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


def _bootstrap_empty_db(db_path: str, tmp_path: Path):
    """Run ``alembic upgrade head`` against ``db_path`` and
    return a (engine, Session) tuple.

    Symmetric with the helpers in ``test_collocations_schema.py``
    — the test scaffolding needs the schema applied + a live
    Python-level engine so we can also drive ORM inserts.
    """
    _run_alembic(db_path, tmp_path, "upgrade", "head")
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return engine, Session


def _row_counts(db_path: str) -> tuple[int, int]:
    """Open the SQLite file via SQLAlchemy and count rows in
    both ``phrases`` (for the test fixture) and ``phrase_pairs``.

    Returns ``(n_phrases, n_phrase_pairs)``."""
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with sessionmaker(bind=engine)() as s:
            from sqlalchemy import func

            n_phrases = (
                s.scalar(
                    select(func.count()).select_from(Phrase)
                )
                or 0
            )
            n_pp = (
                s.scalar(
                    select(func.count()).select_from(PhrasePair)
                )
                or 0
            )
            return n_phrases, n_pp
    finally:
        engine.dispose()


def _row_counts_raw(db_path: str) -> dict:
    """Read the relation-distribution counts directly from
    SQLite (no SQLAlchemy involvement) — used to assert the
    deterministic bucket assignment holds across runs.

    Returns ``{relation: count}`` for the ``phrase_pairs`` table.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT relation, COUNT(*) FROM phrase_pairs "
            "GROUP BY relation"
        )
        return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def _rows_raw(
    db_path: str,
) -> list[tuple[str, str, str]]:
    """Read the full ``phrase_pairs`` row set as
    ``(phrase_a_id, phrase_b_id, relation)`` triples sorted by
    ``(phrase_a_id, phrase_b_id)`` for byte-equal comparison.

    Used by the determinism tests: same-seed runs must produce
    the same set of triples; different-seed runs must produce a
    non-trivial shift in the triples.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT phrase_a_id, phrase_b_id, relation "
            "FROM phrase_pairs "
            "ORDER BY phrase_a_id, phrase_b_id"
        )
        return [tuple(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _insert_test_phrases(
    db_path: str, n: int = 12
) -> list[str]:
    """Insert ``n`` simple test phrases via raw SQL.

    Avoids the SA ``Phrases`` constructor so the test stays
    fast and the slug-IDs are predictable
    (``test-phrase-01`` ... ``test-phrase-NN``). Returns the
    inserted IDs as a list for downstream checks.
    """
    ids = [f"test-phrase-{i:02d}" for i in range(1, n + 1)]
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for i, slug in enumerate(ids, 1):
            cur.execute(
                "INSERT INTO phrases (id, phrase, definition, "
                "source_attribution, frequency_band, created_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    slug, f"Test Phrase {i}", f"def {i}",
                    "dwds", "high",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return ids


# ---------------------------------------------------------------------------
# Tests: the bundled seed JSON itself
# ---------------------------------------------------------------------------


def test_bundled_attested_pairs_json_exists():
    """``data/attested_pairs.json`` ships with the backend as
    the 10.4 hand-curated override source. The Phase 10.1
    hand-off starts it empty (``{"pairs": []}``); 10.4
    populates it."""
    assert SEED_JSON.exists(), f"missing seed file {SEED_JSON}"


def test_bundled_attested_pairs_json_is_valid_manifest():
    """The bundled JSON parses cleanly via
    ``PhrasePairSeedManifest``. A typo'd slug or invalid relation
    in any row would cause this test to fail (the manifest
    validator enforces both).

    Phase 10.4 (card ``t_f3d2a634``) shipped the populated
    Goethe/Schiller attested-pair list (12 curated rows sourced
    from the Phase 8.2 attestation corpus); the file is no
    longer the pre-10.4 ``{"pairs": []}`` placeholder. The
    comprehensive field + relation coverage is exercised by
    ``tests/test_phrase_match_eval.py``; this test only pins
    that the bundled JSON is a valid manifest.
    """
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    manifest = PhrasePairSeedManifest.model_validate(payload)
    assert isinstance(manifest.pairs, list)
    # Phase 10.4 deliverable: 12 curated Goethe/Schiller pairs
    # (card t_f3d2a634, fold card t_51289780). The seed script
    # asserts every row's slug exists in the planted phrases
    # table; the field-by-field coverage lives in
    # ``tests/test_phrase_match_eval.py``.
    assert len(manifest.pairs) == 12


# ---------------------------------------------------------------------------
# Tests: alembic migration shape
# ---------------------------------------------------------------------------


def test_alembic_upgrade_head_creates_phrase_pairs_table(
    sqlite_db_path, tmp_path
):
    """``alembic upgrade head`` against a fresh SQLite creates
    the ``phrase_pairs`` table with all expected columns and
    the four named indexes + the composite UNIQUE + the CHECK
    constraint."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")

    conn = sqlite3.connect(sqlite_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='phrase_pairs'"
        )
        sql = cur.fetchone()[0]
        assert "phrase_a_id" in sql
        assert "phrase_b_id" in sql
        assert "relation" in sql
        assert "attested_pair" in sql
        assert "created_at" in sql
        assert (
            "UNIQUE (phrase_a_id, phrase_b_id)"
            in sql.replace("CONSTRAINT uq_phrase_pairs_a_b ", "")
        )
        # CHECK constraint — the no-self-pair rule.
        assert "phrase_a_id <> phrase_b_id" in sql
        # FK to phrases.
        assert "REFERENCES phrases (id)" in sql
        assert "ON DELETE RESTRICT" in sql

        # Indexes — the four named indexes + the autoindex
        # for the PK + the autoindex for the composite UNIQUE.
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='phrase_pairs'"
        )
        idx_names = {row[0] for row in cur.fetchall()}
        assert {
            "ix_phrase_pairs_phrase_a_id",
            "ix_phrase_pairs_phrase_b_id",
            "ix_phrase_pairs_relation",
            "ix_phrase_pairs_attested_pair",
        }.issubset(idx_names)
    finally:
        conn.close()


def test_alembic_upgrade_head_is_idempotent_on_sqlite(
    sqlite_db_path, tmp_path
):
    """Re-running ``alembic upgrade head`` against an
    already-migrated DB is a clean no-op (Phase 7.1 / 8.1
    / 9.1 discipline)."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    # Second run — must exit 0 with no errors.
    result = _run_alembic(
        sqlite_db_path, tmp_path, "upgrade", "head"
    )
    assert result.returncode == 0, result.stderr
    # No row inserted twice — the schema is unchanged.
    n_phrases, n_pp = _row_counts(sqlite_db_path)
    assert n_phrases == 0
    assert n_pp == 0


def test_alembic_downgrade_then_upgrade_is_clean(
    sqlite_db_path, tmp_path
):
    """Downgrade then re-upgrade is the gold standard for the
    migration contract — verifies the downgrade drops the
    right things and the upgrade recreates them."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")

    # Downgrade to the previous head (9a1_fsrs_cards_exercise_type).
    down_result = _run_alembic(
        sqlite_db_path,
        tmp_path,
        "downgrade",
        "9a1_fsrs_cards_exercise_type",
    )
    assert down_result.returncode == 0, down_result.stderr

    # Verify the table is gone.
    conn = sqlite3.connect(sqlite_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='phrase_pairs'"
        )
        assert cur.fetchone()[0] == 0
    finally:
        conn.close()

    # Re-upgrade — must recreate the table cleanly.
    up_result = _run_alembic(
        sqlite_db_path, tmp_path, "upgrade", "head"
    )
    assert up_result.returncode == 0, up_result.stderr

    conn = sqlite3.connect(sqlite_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='phrase_pairs'"
        )
        assert cur.fetchone() is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests: the seed script's determinism
# ---------------------------------------------------------------------------


def _run_seed(env: dict, *args: str, timeout: int = 60):
    """Run the seed script as a subprocess (mirrors the
    production invocation so the test exercises the actual
    script, not an in-process reimplementation)."""
    return subprocess.run(
        [sys.executable, str(SEED_SCRIPT), *args],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _stub_similarity_phrase_pairs():
    """Inline stub similarity function for the seed-script
    test path.

    Returns 0.7 (above the 0.55 threshold) for any two
    strings, so every (i, j) phrase pair lands in the
    candidate pool. This is intentional — the test asserts
    the seed script's *bucket assignment* is deterministic,
    not the candidate-pool selection (which depends on the
    bge-m3 cosine path).

    The stub lives in a dedicated file under
    ``tests/fixtures/`` because the seed script imports it
    via ``--similarity-fn module:function``.
    """
    return _stub_similarity_phrase_pairs  # noqa: F401


def _seed_with_db(
    db_path: str,
    tmp_path: Path,
    *seed_args: str,
    bootstrap: bool = True,
    attested_pairs_path: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run the seed script against ``db_path``.

    On the first call in a test, ``bootstrap=True`` (the default)
    runs ``alembic upgrade head`` and seeds 12 test phrases. On
    subsequent calls against the same ``db_path`` (e.g. for the
    determinism tests that re-invoke the script), pass
    ``bootstrap=False`` so we don't trip the ``phrases.phrase``
    UNIQUE on a second INSERT of the same fixture phrases.

    ``attested_pairs_path`` overrides the bundled
    ``backend/data/attested_pairs.json`` (default = bundled). The
    seed-script determinism tests below pass a tmp empty-JSON
    (``{"pairs": []}``) because the bundled 12 Goethe/Schiller
    pairs (Phase 10.4 deliverable, card ``t_f3d2a634``) reference
    slugs that aren't in the test fixture's 12 ``test-phrase-NN``
    rows; loading the bundled file would error out with
    ``attested slug '<slug>' not found in phrases table`` before
    the test ever reaches its determinism assertions.

    Returns the subprocess result (stdout/stderr/returncode).
    """
    if bootstrap:
        _bootstrap_empty_db(db_path, tmp_path)
        _insert_test_phrases(db_path, n=12)

    # Default: use a tmp empty attested-pairs file so the
    # determinism tests aren't coupled to the bundled Phase 10.4
    # Goethe/Schiller attested-pair list. Callers can override
    # with ``attested_pairs_path`` to exercise the override path
    # explicitly (see ``test_seed_script_respects_attested_pair_override``
    # style tests, which pass a custom fixture).
    if attested_pairs_path is None:
        attested_pairs_path = tmp_path / "empty_attested_pairs.json"
        attested_pairs_path.write_text(
            json.dumps({"pairs": []}), encoding="utf-8"
        )

    # Drive the script with the stub similarity function
    # (CI / test path; production uses bge-m3 cosine when
    # the cache is warm). The stub lives in a test fixture
    # module imported via ``--similarity-fn module:function``.
    return _run_seed(
        _alembic_env(db_path, tmp_path),
        "--similarity-fn",
        "tests.stub_seed_similarity:stub_similarity",
        "--attested",
        str(attested_pairs_path),
        *seed_args,
    )


def test_seed_script_produces_deterministic_distribution(
    sqlite_db_path, tmp_path
):
    """``seed_phrase_pairs.py --seed 42`` produces identical
    bucket distributions on two consecutive runs against the
    same phrases table. Byte-equal reproducibility is the
    plan body's contract."""
    result1 = _seed_with_db(
        sqlite_db_path, tmp_path, "--seed", "42"
    )
    assert result1.returncode == 0, result1.stderr
    rows1 = _rows_raw(sqlite_db_path)

    # Re-run with the same seed against the same DB — skip
    # the bootstrap step so we don't trip the ``phrases.phrase``
    # UNIQUE on a second INSERT.
    result2 = _seed_with_db(
        sqlite_db_path,
        tmp_path,
        "--seed",
        "42",
        bootstrap=False,
    )
    assert result2.returncode == 0, result2.stderr
    rows2 = _rows_raw(sqlite_db_path)

    # Same row set, byte-equal (deterministic seed).
    assert rows1 == rows2
    # 12 phrases -> 12*11/2 = 66 unique ordered pairs.
    assert len(rows2) == 66


def test_seed_script_idempotent_on_same_seed(
    sqlite_db_path, tmp_path
):
    """Re-running with the same seed is a no-op at the row-
    count level (no PK collisions, no duplicates)."""
    result1 = _seed_with_db(
        sqlite_db_path, tmp_path, "--seed", "42"
    )
    assert result1.returncode == 0, result1.stderr
    _, n_pp_1 = _row_counts(sqlite_db_path)

    # Idempotent re-run — bootstrap=False so the phrases
    # fixture isn't double-inserted.
    result2 = _seed_with_db(
        sqlite_db_path,
        tmp_path,
        "--seed",
        "42",
        bootstrap=False,
    )
    assert result2.returncode == 0, result2.stderr
    _, n_pp_2 = _row_counts(sqlite_db_path)

    assert n_pp_1 == n_pp_2
    # 12 phrases -> 12*11/2 = 66 unique ordered pairs.
    assert n_pp_2 == 66


def test_seed_script_different_seed_produces_different_distribution(
    sqlite_db_path, tmp_path
):
    """Different seeds produce different per-pair bucket
    assignments (seed-derivation correctness). Two phrases
    tables, two seeds.

    Note: the *bucket count* is invariant under seed choice
    (the rank-quartile split on 66 pairs always yields
    ~17/16/17/16 — the seed only affects WHICH specific
    pairs land in WHICH bucket). The assertion therefore
    looks at the (a, b, relation) tuples directly."""
    result_a = _seed_with_db(
        sqlite_db_path, tmp_path, "--seed", "42"
    )
    assert result_a.returncode == 0, result_a.stderr
    rows_a = _rows_raw(sqlite_db_path)

    # Fresh DB, different seed.
    fresh_db = str(
        tmp_path / "fresh.db"
    )
    result_b = _seed_with_db(
        fresh_db, tmp_path, "--seed", "17"
    )
    assert result_b.returncode == 0, result_b.stderr
    rows_b = _rows_raw(fresh_db)

    # Same total — the candidate pool + bucket count are
    # both deterministic given the phrases table, so the
    # row counts must match.
    assert len(rows_a) == len(rows_b) == 66

    # The per-pair relation tuples must differ: the seed
    # drives the shuffle that precedes the rank-quartile
    # assignment, so a different seed puts different pairs
    # into the same bucket. With 66 pairs and 4 buckets,
    # the vast majority of pairs change their assigned
    # relation between two random seeds.
    differing_pairs = 0
    pairs_a = {(a, b): rel for a, b, rel in rows_a}
    pairs_b = {(a, b): rel for a, b, rel in rows_b}
    common = set(pairs_a) & set(pairs_b)
    for k in common:
        if pairs_a[k] != pairs_b[k]:
            differing_pairs += 1
    assert differing_pairs > 0, (
        f"seed=42 and seed=17 produced identical "
        f"per-pair assignments — seed didn't propagate"
    )
    # Sanity bound: at least 10% of pairs should differ
    # (conservative; in practice it's much higher).
    assert differing_pairs >= len(common) // 10, (
        f"only {differing_pairs}/{len(common)} pairs differ "
        f"between seed=42 and seed=17; expected a non-trivial "
        f"shift. rows_a={rows_a[:5]}... rows_b={rows_b[:5]}..."
    )


def test_seed_script_rejects_self_pair(
    sqlite_db_path, tmp_path, monkeypatch
):
    """A self-pair in ``attested_pairs.json`` is rejected at
    seed-script parse time via the Pydantic validator (the
    same guard catches it before INSERT would fail the DB
    CHECK constraint)."""
    # Build a temp attested_pair file containing a self-pair.
    fixture_file = tmp_path / "bad_attested.json"
    fixture_file.write_text(
        json.dumps({
            "pairs": [
                {
                    "phrase_a_id": "test-phrase-01",
                    "phrase_b_id": "test-phrase-01",
                    "relation": "equivalent",
                    "attested_pair": True,
                }
            ]
        })
    )

    _bootstrap_empty_db(sqlite_db_path, tmp_path)
    _insert_test_phrases(sqlite_db_path, n=4)

    result = _run_seed(
        _alembic_env(sqlite_db_path, tmp_path),
        "--similarity-fn",
        "tests.stub_seed_similarity:stub_similarity",
        "--seed", "42",
        "--attested", str(fixture_file),
    )

    # Self-pair rows fail at Pydantic parse time — the seed
    # script exits non-zero with a clear message.
    assert result.returncode != 0
    assert (
        "distinct" in result.stderr
        or "phrase_a_id" in result.stderr
    )


def test_seed_script_attested_pair_lands_with_override(
    sqlite_db_path, tmp_path
):
    """An attested-pair row lands with ``attested_pair=1``
    AND its explicit ``relation`` from the JSON, even when
    the bucketed run would have assigned a different relation
    to the same (a, b) pair (the ``ON CONFLICT DO UPDATE``
    path makes the attested row win).

    The fixture pairs (1, 2) and assigns it ``equivalent`` +
    ``attested_pair=true``. Without the override, seed=42 would
    have assigned a quartile-bucket relation. We don't pin
    which bucket — only that the attested row wins.
    """
    # Build a temp attested_pair file with one entry.
    fixture_file = tmp_path / "good_attested.json"
    fixture_file.write_text(
        json.dumps({
            "pairs": [
                {
                    "phrase_a_id": "test-phrase-01",
                    "phrase_b_id": "test-phrase-02",
                    "relation": "equivalent",
                    "attested_pair": True,
                }
            ]
        })
    )

    _bootstrap_empty_db(sqlite_db_path, tmp_path)
    _insert_test_phrases(sqlite_db_path, n=12)

    result = _run_seed(
        _alembic_env(sqlite_db_path, tmp_path),
        "--similarity-fn",
        "tests.stub_seed_similarity:stub_similarity",
        "--seed", "42",
        "--attested", str(fixture_file),
    )
    assert result.returncode == 0, result.stderr

    # Verify (1, 2) has attested_pair=1 and relation=equivalent.
    conn = sqlite3.connect(sqlite_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT phrase_a_id, phrase_b_id, relation, "
            "attested_pair FROM phrase_pairs "
            "WHERE phrase_a_id='test-phrase-01' "
            "AND phrase_b_id='test-phrase-02'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[2] == "equivalent"
        assert row[3] == 1

        # Re-running is a no-op.
        result2 = _run_seed(
            _alembic_env(sqlite_db_path, tmp_path),
            "--similarity-fn",
            "tests.stub_seed_similarity:stub_similarity",
            "--seed", "42",
            "--attested", str(fixture_file),
        )
        assert result2.returncode == 0, result2.stderr
    finally:
        conn.close()


def test_seed_script_rejects_unknown_phrase_in_attested(
    sqlite_db_path, tmp_path
):
    """An attested-row whose ``phrase_a_id`` doesn't match an
    existing ``phrases.id`` value is rejected loudly by the
    seed script (not silently skipped). Same for ``phrase_b_id``."""
    fixture_file = tmp_path / "bad_attested_unknown.json"
    fixture_file.write_text(
        json.dumps({
            "pairs": [
                {
                    "phrase_a_id": "this-phrase-does-not-exist",
                    "phrase_b_id": "test-phrase-02",
                    "relation": "equivalent",
                    "attested_pair": True,
                }
            ]
        })
    )

    _bootstrap_empty_db(sqlite_db_path, tmp_path)
    _insert_test_phrases(sqlite_db_path, n=4)

    result = _run_seed(
        _alembic_env(sqlite_db_path, tmp_path),
        "--similarity-fn",
        "tests.stub_seed_similarity:stub_similarity",
        "--seed", "42",
        "--attested", str(fixture_file),
    )

    # The seed script must fail loudly — a typo'd slug in the
    # attested manifest is an authoring error, not an
    # auto-skip.
    assert result.returncode != 0
    assert (
        "not found in phrases table" in result.stderr
    )


def test_seed_script_refuses_too_few_phrases(
    sqlite_db_path, tmp_path
):
    """The seed script refuses to run against a phrases table
    with fewer than 2 rows — pair-generation is undefined."""
    _bootstrap_empty_db(sqlite_db_path, tmp_path)
    # Insert exactly 1 phrase.
    _insert_test_phrases(sqlite_db_path, n=1)

    result = _run_seed(
        _alembic_env(sqlite_db_path, tmp_path),
        "--similarity-fn",
        "tests.stub_seed_similarity:stub_similarity",
        "--seed", "42",
    )
    assert result.returncode != 0
    assert "at least 2" in result.stderr


def test_seed_script_uses_dialect_specific_similarity_fn(
    sqlite_db_path, tmp_path, monkeypatch
):
    """The production similarity path requires
    ``sentence-transformers``. When the stub similarity_fn
    is missing AND --similarity-fn isn't passed, the script
    fails loudly with a clear error rather than trying to
    load bge-m3 in CI."""

    _bootstrap_empty_db(sqlite_db_path, tmp_path)
    _insert_test_phrases(sqlite_db_path, n=4)

    # Run the script WITHOUT --similarity-fn. The default
    # path tries to load bge-m3 via sentence-transformers,
    # which isn't installed in CI — the script fails loudly.
    result = _run_seed(
        _alembic_env(sqlite_db_path, tmp_path),
        "--seed", "42",
    )

    if result.returncode != 0:
        # Expected outcome in CI: the script refuses with a
        # clear error pointing at the alternative path.
        assert (
            "sentence-transformers" in result.stderr
            or "similarity-fn" in result.stderr
        )
    else:
        # Some CI environments DO have sentence-transformers
        # installed (caching bge-m3 locally). In that case the
        # production path runs cleanly. We just assert no
        # crash.
        assert "Loaded" in result.stdout or "phrase_pairs" in result.stdout
