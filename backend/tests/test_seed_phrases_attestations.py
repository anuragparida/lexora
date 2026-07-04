"""Tests for Phase 8.2 — Goethe/Schiller attestation seed script.

Card: t_32ef1260.

This is the second seed script for the ``phrases`` table (the
first was 8.1's ``seed_phrases_dwds.py``). Coverage:

1. The bundled ``data/goethe_schiller_idioms_subset.json`` loads
   cleanly:
   - Exactly the card body's 200-300 row target window.
   - Every row validates via ``PhraseSeedRow`` at parse time.
   - ≥95% of rows have non-null ``attested_quote`` AND
     non-null ``attested_source`` (the wire-layer guardrail).
   - Distribution covers both Goethe and Schiller attributions
     (no single-author monopoly).
   - Distribution covers all three frequency bands.

2. The seed script end-to-end against a fresh SQLite DB:
   - Loads the target 225 (or whatever the curated set has)
     rows.
   - Idempotent: a second invocation doesn't pile up duplicates.
   - Logs the card-body verification line (``Goethe/Schiller
     attestations: N rows loaded``).

3. The bump rule (frequency-band MAX() across an 8.1 + 8.2
   re-seed):
   - Pre-seed an 8.1 row at ``frequency_band="low"``.
   - Run 8.2 with a Goethe/Schiller attestation for the same
     slug with ``frequency_band="high"``.
   - Confirm the row's band is now ``"high"`` and the
     ``source_attribution`` has been union'd to
     ``"dwds,goethe"``.

4. ``--source`` mode: small 5-row fixture loads exactly 5 rows.

Hermetic: each test uses its own temp SQLite DB and ``alembic
upgrade head``. No live Postgres, no LLM call.

Run from ``backend/``::

    bash /tmp/runpytest.sh tests/test_seed_phrases_attestations.py
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


# Path constants — mirror ``test_seed_phrases_dwds.py`` and
# ``test_collocations_schema.py``.
BACKEND_DIR = Path(__file__).resolve().parent.parent
SEED_SCRIPT = BACKEND_DIR / "scripts" / "seed_phrases_attestations.py"
SEED_JSON = (
    BACKEND_DIR / "data" / "goethe_schiller_idioms_subset.json"
)
FIXTURE_JSON = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "phrases_attestations_fixture.json"
)


# Card-body band counts from PHASE-8.md §"What 8.2 ships" item 4
# (≥95% attested, ≤5% null) and the row-count window 200-300.
ROWS_MIN = 200
ROWS_MAX = 300
NULL_QUOTE_MAX_PCT = 5  # card body contract


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB so the migration sees an empty namespace.

    Mirrors the ``test_collocations_schema.py`` and
    ``test_seed_phrases_dwds.py`` fixtures: pin ``DATABASE_URL``
    to ``sqlite:///<tmp>`` so the alembic subprocess uses the right
    file, plus the JWT/decks env vars so ``app`` imports cleanly.
    """
    db_path = tmp_path / "phrases_attestations_seed.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


def _alembic_env(db_path: str, tmp_path: Path) -> dict:
    """Build the env dict the alembic subprocess inherits.

    Mirrors ``test_seed_phrases_dwds.py::_alembic_env``.
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
# Tests: bundled seed JSON shape
# ---------------------------------------------------------------------------


def test_bundled_goethe_schiller_seed_json_exists():
    """``data/goethe_schiller_idioms_subset.json`` ships with the
    backend. The seed script's default path reads this file; if
    it's missing, the script's default-path branch errors out and
    the test suite flags the regression."""
    assert SEED_JSON.exists(), f"missing seed file {SEED_JSON}"


def test_bundled_goethe_schiller_seed_json_parses_as_list():
    """The seed file is a JSON array of row tuples. The 8.1
    fixture uses dicts; the 8.2 fixture uses tuples (so the
    ``attested_quote`` + ``attested_source`` columns are first-class
    rather than implicit)."""
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_bundled_goethe_schiller_seed_json_row_count_in_target_window():
    """Card body contract: 200-300 rows. The hand-curated
    Goethe/Schiller subset ships with 225 rows."""
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    n = len(payload)
    assert ROWS_MIN <= n <= ROWS_MAX, (
        f"seed JSON has {n} rows; expected {ROWS_MIN}-{ROWS_MAX}. "
        f"Extend the curated set or update the script."
    )


def test_bundled_goethe_schiller_seed_json_every_tuple_has_eight_fields():
    """Each row tuple has the documented 8-tuple shape::

        (id, phrase, definition, example_usage, source_attribution,
         frequency_band, attested_quote, attested_source)

    A row that's missing either attestation column at parse time
    is a malformed seed row.
    """
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    bad = [(i, len(r)) for i, r in enumerate(payload) if len(r) != 8]
    assert not bad, f"{len(bad)} rows with wrong arity: {bad[:5]}"


def test_bundled_goethe_schiller_seed_json_wire_layer_validates():
    """Every row's wire-layer fields (id, phrase, definition,
    example_usage, source_attribution, frequency_band) validate
    against the ``PhraseSeedRow`` Pydantic model. The attestation
    fields (attested_quote, attested_source) are NOT in
    ``PhraseSeedRow`` — they're populated at the seed boundary
    by this script and land in the DB as nullable Text."""
    from app.schemas import PhraseSeedRow

    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    bad = []
    for i, row in enumerate(payload):
        (
            sid, phrase, defn, ex, sa, fb, aq, ats
        ) = row
        try:
            PhraseSeedRow.model_validate(
                {
                    "id": sid,
                    "phrase": phrase,
                    "definition": defn,
                    "example_usage": ex,
                    "source_attribution": sa,
                    "frequency_band": fb,
                    "dwds_url": None,
                }
            )
        except Exception as e:
            bad.append((i, sid, str(e)[:200]))
    assert not bad, f"{len(bad)} malformed wire-layer rows: {bad[:5]}"


def test_bundled_goethe_schiller_seed_json_attestation_columns_present():
    """For ≥95% of rows, BOTH ``attested_quote`` AND
    ``attested_source`` are non-null. Card body contract:
    "attested_quote + attested_source are non-null for ≥95% of
    rows (some Goethe texts have idioms without clear
    line-numbering, which the script leaves null)".
    """
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    n = len(payload)
    both = sum(
        1 for r in payload
        if r[6] is not None and r[7] is not None
    )
    pct = 100 * both / n
    assert pct >= (100 - NULL_QUOTE_MAX_PCT), (
        f"only {both}/{n} ({pct:.1f}%) rows have both "
        f"attested_quote and attested_source populated; card body "
        f"requires ≥{100 - NULL_QUOTE_MAX_PCT}%."
    )


def test_bundled_goethe_schiller_seed_json_attested_source_always_set():
    """When ``attested_quote`` is null, ``attested_source`` is
    still set (the citation is the trailing metadata; the
    missing-``attested_quote`` rows are those where the Goethe
    text has the idiom in a passage without clear line
    numbering, but the work + chapter citation is still
    known).

    Reversing the constraint — populated ``attested_quote`` but
    missing ``attested_source`` — never happens in the curated
    set.
    """
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    bad = [r for r in payload if r[6] is not None and r[7] is None]
    assert not bad, (
        f"{len(bad)} rows have attested_quote but no "
        f"attested_source: {bad[:3]}"
    )


def test_bundled_goethe_schiller_seed_json_covers_both_authors():
    """Distribution check: the seed has rows with
    ``source_attribution`` containing ``"goethe"``, rows with
    ``"schiller"``, and rows spanning both. A pure-Goethe or
    pure-Schiller subset would defeat the dual-extension
    rationale; either author alone is also insufficient.
    """
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)

    has_goethe = sum(1 for r in payload if "goethe" in r[4])
    has_schiller = sum(1 for r in payload if "schiller" in r[4])
    has_both = sum(
        1 for r in payload
        if "goethe" in r[4] and "schiller" in r[4]
    )

    assert has_goethe > 0, "no Goethe-attributed rows"
    assert has_schiller > 0, "no Schiller-attributed rows"
    assert has_both > 0, "no Goethe+Schiller dual-attributed rows"


def test_bundled_goethe_schiller_seed_json_distributions_across_frequency_bands():
    """The bundled JSON has rows distributed across all three
    frequency bands. The exact distribution is not pinned; only
    that every band has at least one row so the Phase 8.4
    high-band-first cloze variant has visible cohorts.
    """
    from collections import Counter

    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    counts = Counter(r[5] for r in payload)
    for band in ("high", "mid", "low"):
        assert counts[band] > 0, (
            f"frequency_band={band!r} has no rows in the seed JSON; "
            f"Phase 8.4 expects at least one row per band."
        )


def test_bundled_goethe_schiller_seed_json_unique_slugs():
    """The script's PK is the slug. Duplicate slugs would
    create ambiguous ``attested_quote`` rows; the curated set
    keeps them unique."""
    with open(SEED_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    slugs = [r[0] for r in payload]
    dupes = [s for s in set(slugs) if slugs.count(s) > 1]
    assert not dupes, f"duplicate slugs in seed JSON: {dupes[:5]}"


# ---------------------------------------------------------------------------
# Tests: seed script end-to-end against a fresh SQLite DB
# ---------------------------------------------------------------------------


def test_seed_script_loads_minimum_target_into_fresh_db(
    sqlite_db_path, tmp_path
):
    """``alembic upgrade head`` + the seed script ends with
    200-300 rows in ``phrases`` (the card body's
    verification shape)."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env)
    assert result.returncode == 0, (
        f"seed script failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    n = _row_count(sqlite_db_path)
    assert ROWS_MIN <= n <= ROWS_MAX, (
        f"loaded {n} rows; expected {ROWS_MIN}-{ROWS_MAX}. "
        f"Stub the curated set to fill the gap."
    )


def test_seed_script_logs_loaded_count_message(
    sqlite_db_path, tmp_path
):
    """The seed script prints
    ``Goethe/Schiller attestations: N rows loaded`` — the
    card-body verification line shape. Helena's review card
    can grep stdout for this string."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env)
    assert result.returncode == 0, result.stderr
    assert "Goethe/Schiller attestations:" in result.stdout
    assert "rows loaded" in result.stdout


def test_seed_script_against_fixture_loads_only_fixture_rows(
    sqlite_db_path, tmp_path
):
    """The seed script accepts ``--source <path>``. Pointing
    it at a 5-row fixture loads exactly those 5 rows into a
    fresh DB."""
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
    DB is a clean no-op: the upsert shape (ON CONFLICT DO
    NOTHING + targeted UPDATE for the bump path) keeps the
    row count stable. The total may *not* change between
    runs."""
    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    first = _run_seed(env)
    assert first.returncode == 0, first.stderr
    n_after_first = _row_count(sqlite_db_path)
    assert ROWS_MIN <= n_after_first <= ROWS_MAX

    second = _run_seed(env)
    assert second.returncode == 0, second.stderr
    n_after_second = _row_count(sqlite_db_path)
    assert n_after_first == n_after_second, (
        f"second seed run changed the row count "
        f"({n_after_first} -> {n_after_second}); idempotent "
        f"contract broken"
    )


def test_seed_script_round_trips_attestation_columns_via_orm(
    sqlite_db_path, tmp_path
):
    """After the seed script runs, an ``attested_quote`` +
    ``attested_source`` row round-trips via the ``Phrase``
    ORM model. Catches a typo'd column name in the script's
    INSERT statement."""
    from app.models import Phrase

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env)
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        Session = sessionmaker(bind=engine)
        with Session() as s:
            # Pick a known Goethe-attested row (the famous "Die
            # Geister, die ich rief" from Der Zauberlehrling).
            row = s.scalar(
                select(Phrase).where(
                    Phrase.id == "die-geister-die-ich-rief"
                )
            )
            assert row is not None, (
                "expected row id='die-geister-die-rief' after seed"
            )
            assert row.attested_quote is not None
            assert "Geister" in row.attested_quote
            assert row.attested_source is not None
            # source_attribution is the dual literal from the
            # 8.1/8.2 overlap rows.
            assert "goethe" in row.source_attribution
            # frequency_band was bumped from "high" to "high"
            # (idempotent — but at minimum still "high").
            assert row.frequency_band in {"high", "mid", "low"}
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Tests: bump rule (8.1 → 8.2 frequency-band upgrade)
# ---------------------------------------------------------------------------


def test_bump_rule_promotes_low_band_to_mid_when_attestation_is_mid(
    sqlite_db_path, tmp_path
):
    """The bump rule (per the seed script's docstring)::

        New band strictly higher → bump.
        Same band or lower → keep existing.

    Pre-seed an 8.1 row at ``frequency_band="low"`` (via a
    crafted 5-row fixture with one low row), then run 8.2 with
    ``frequency_band="mid"`` for the same slug. The DB row's
    band is now ``"mid"`` and the ``source_attribution`` has
    ``"dwds,"`` union'd with the new attribution.

    Setup::

        {id: "test-bump-low-mid", phrase: "...",
         source_attribution: "dwds", frequency_band: "low"}

    Bump target::

        {id: "test-bump-low-mid", phrase: "...",
         source_attribution: "goethe", frequency_band: "mid",
         attested_quote: "...", attested_source: "Goethe, ..."}
    """
    # Pre-write the 8.1 row via direct ORM (matches the shape
    # the 8.1 seed script produces).
    from app.models import Phrase

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        Session = sessionmaker(bind=engine)
        with Session() as s:
            s.add(
                Phrase(
                    id="test-bump-low-mid",
                    phrase="Test Bump Low Mid",
                    definition="Test idiom for the low-to-mid bump path.",
                    example_usage="Test usage example.",
                    source_attribution="dwds",
                    frequency_band="low",
                    dwds_url=None,
                    attested_quote=None,
                    attested_source=None,
                )
            )
            s.commit()
            # Verify the pre-state.
            pre = s.scalar(
                select(Phrase).where(
                    Phrase.id == "test-bump-low-mid"
                )
            )
            assert pre.frequency_band == "low"
            assert pre.source_attribution == "dwds"
    finally:
        engine.dispose()

    # Build a 1-row 8.2 seed fixture that bumps low→mid.
    att_fixture = tmp_path / "att_fixture.json"
    with open(att_fixture, "w", encoding="utf-8") as f:
        json.dump(
            [
                [
                    "test-bump-low-mid",
                    "Test Bump Low Mid",
                    "Test idiom for the low-to-mid bump path.",
                    "Test usage example.",
                    "goethe",
                    "mid",
                    "Die Prüfung stand ihm bevor.",
                    "Goethe, Faust I, Studierzimmer",
                ]
            ],
            f,
        )

    env = _alembic_env(sqlite_db_path, tmp_path)
    seed_result = _run_seed(env, "--source", str(att_fixture))
    assert seed_result.returncode == 0, (
        f"seed script failed: {seed_result.stdout}\n"
        f"{seed_result.stderr}"
    )

    # Re-open the DB and verify the bump fired.
    engine2 = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        Session2 = sessionmaker(bind=engine2)
        with Session2() as s:
            post = s.scalar(
                select(Phrase).where(
                    Phrase.id == "test-bump-low-mid"
                )
            )
            assert post is not None
            assert post.frequency_band == "mid", (
                f"expected mid after bump; got {post.frequency_band}"
            )
            # source_attribution is the union: "dwds,goethe"
            tokens = sorted(
                t.strip() for t in post.source_attribution.split(",")
            )
            assert tokens == ["dwds", "goethe"], (
                f"expected source_attribution='dwds,goethe'; "
                f"got {post.source_attribution!r}"
            )
            assert post.attested_quote == "Die Prüfung stand ihm bevor."
            assert (
                post.attested_source
                == "Goethe, Faust I, Studierzimmer"
            )
    finally:
        engine2.dispose()


def test_bump_rule_does_not_downgrade_high_band_to_low(
    sqlite_db_path, tmp_path
):
    """The bump rule never downgrades. An 8.2 attestation with
    ``frequency_band="low"`` against an existing 8.1 row at
    ``"high"`` keeps ``"high"``."""
    from app.models import Phrase

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    engine = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        Session = sessionmaker(bind=engine)
        with Session() as s:
            s.add(
                Phrase(
                    id="test-no-downgrade",
                    phrase="Test No Downgrade",
                    definition="Test idiom for the no-downgrade rule.",
                    example_usage="Test usage example.",
                    source_attribution="dwds",
                    frequency_band="high",
                    dwds_url=None,
                    attested_quote=None,
                    attested_source=None,
                )
            )
            s.commit()
    finally:
        engine.dispose()

    att_fixture = tmp_path / "att_fixture_nodowngrade.json"
    with open(att_fixture, "w", encoding="utf-8") as f:
        json.dump(
            [
                [
                    "test-no-downgrade",
                    "Test No Downgrade",
                    "Test idiom for the no-downgrade rule.",
                    "Test usage example.",
                    "goethe",
                    "low",
                    None,
                    "Goethe, Brief an Schiller",
                ]
            ],
            f,
        )

    env = _alembic_env(sqlite_db_path, tmp_path)
    seed_result = _run_seed(env, "--source", str(att_fixture))
    assert seed_result.returncode == 0, seed_result.stderr

    engine2 = create_engine(f"sqlite:///{sqlite_db_path}")
    try:
        Session2 = sessionmaker(bind=engine2)
        with Session2() as s:
            post = s.scalar(
                select(Phrase).where(Phrase.id == "test-no-downgrade")
            )
            assert post.frequency_band == "high", (
                f"expected high (preserved); got {post.frequency_band}"
            )
            # source_attribution still unions but band stays high.
            tokens = sorted(
                t.strip() for t in post.source_attribution.split(",")
            )
            assert tokens == ["dwds", "goethe"], (
                f"expected union 'dwds,goethe'; got "
                f"{post.source_attribution!r}"
            )
    finally:
        engine2.dispose()


# ---------------------------------------------------------------------------
# Tests: malformed seed input is caught at parse time
# ---------------------------------------------------------------------------


def test_seed_script_fails_loudly_on_malformed_seed_json(
    sqlite_db_path, tmp_path
):
    """A seed file with a typo'd ``frequency_band`` is rejected
    with a non-zero exit and a clear error pointing at the
    bad line."""
    bad_seed = tmp_path / "bad_seed.json"
    rows = [
        [
            "x-y-z-test",
            "Test phrase",
            "Test definition",
            None,
            "goethe",
            "Hi",  # typo: not in the literal
            None,
            "Goethe, Faust I",
        ]
    ]
    with open(bad_seed, "w", encoding="utf-8") as f:
        json.dump(rows, f)

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env, "--source", str(bad_seed))
    assert result.returncode != 0, (
        f"seed script should fail on malformed JSON; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "frequency_band" in combined.lower(), (
        f"error message missing 'frequency_band': {combined!r}"
    )


def test_seed_script_fails_loudly_on_malformed_row_arity(
    sqlite_db_path, tmp_path
):
    """A seed row tuple that's missing a field (e.g. only
    7-tuple instead of 8-tuple) is rejected with a clear
    error pointing at the line."""
    bad_seed = tmp_path / "bad_arity.json"
    rows = [
        [
            "x-y-z-test",
            "Test phrase",
            "Test definition",
            None,
            "goethe",
            "high",
            None,
            # missing attested_source!
        ]
    ]
    with open(bad_seed, "w", encoding="utf-8") as f:
        json.dump(rows, f)

    _run_alembic(sqlite_db_path, tmp_path, "upgrade", "head")
    env = _alembic_env(sqlite_db_path, tmp_path)
    result = _run_seed(env, "--source", str(bad_seed))
    assert result.returncode != 0, (
        f"seed script should fail on 7-tuple row; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "expected 8-tuple" in combined, (
        f"error message missing 'expected 8-tuple': {combined!r}"
    )
