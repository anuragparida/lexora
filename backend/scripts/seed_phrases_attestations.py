"""Seed the ``phrases`` table with Goethe/Schiller attestations.

Phase 8.2 (card t_32ef1260). This is the SECOND seed script for the
``phrases`` table — Phase 8.1 (card t_d967c006) ships the DWDS Idiome
subset via ``seed_phrases_dwds.py``; this script extends it with
200–300 hand-curated idioms attested in Goethe / Schiller texts.

Frequency-band bump rule
------------------------

For an idiom that lands in BOTH the 8.1 DWDS seed and the 8.2
Goethe/Schiller attestation seed (slugs match exactly), the
``frequency_band`` is bumped to whichever is higher::

    SQLite:    INSERT INTO phrases (...) VALUES (...) ON CONFLICT (id)
               DO UPDATE SET frequency_band = MAX(excluded.frequency_band,
                                                   phrases.frequency_band)
    Postgres:  same SQL compiled by ``pg_insert.on_conflict_do_update``.

A "high" attestation bumps a "mid" row to "high"; a "mid"
attestation against an existing "high" row keeps "high"; a "low"
attestation never downgrades. The bump fires ONLY on
``source_attribution`` — the seeded JSON columns for the other
fields follow a single canonical value per slug (idempotent
on re-run via the 8.1 ON CONFLICT DO NOTHING shape).

The bump rationale is recorded here, not in code:

- Goethe's *Faust* and Schiller's *Wilhelm Tell* are two of the
  most frequently cited classical texts in the German school
  canon — an idiom that surfaces in either corpus is, on
  correlation with the Goethe/Schiller reception, more
  likely encountered by a contemporary learner than a
  corpus-internal-only attestation would suggest.

- The bump is deterministic (idempotent on re-run via the
  MAX() UPDATE) and only ever moves UPWARD. It never erases
  the original DWDS attribution: the column is overwritten
  with a comma-joined ``"dwds,goethe"`` /
  ``"dwds,schiller"`` literal rather than replaced.

Idempotency
-----------

Idempotent on re-run. ``INSERT ... ON CONFLICT (id) DO UPDATE`` is
the dialect-portable shape (``INSERT OR ...`` on SQLite,
``ON CONFLICT DO UPDATE`` on Postgres). Pre-deduped in Python by
``id`` (the PK) and by ``phrase`` (the UNIQUE secondary); the
Postgres ``ON CONFLICT (id)`` clause covers the PK collision and
the Python pre-dedupe handles the phrase-UNIQUE case (SQLite's
``on_conflict_do_nothing`` only fires on named indices, not on
the secondary UNIQUE column).

Source corpus
-------------

Citations follow the canonical act/scene/line numbering of the
Project Gutenberg editions of:

- Goethe's *Faust I* and *Faust II* (Public Domain in the
  United States; pre-1928 author; both works also public-domain
  in Germany via § 64 UrhG / 70 years post mortem auctoris).
- Goethe's *Reineke Fuchs*, *Die Wahlverwandtschaften*, *Die
  Leiden des jungen Werthers*, *Egmont*, *Götz von
  Berlichingen*, *West-östlicher Divan*, *Hermann und Dorothea*,
  *Dichtung und Wahrheit*, *Campagne in Frankreich*, *Pandoras
  Wiederkunft*, *Die natürliche Tochter*, *Die Geschwister*,
  *Der Zauberlehrling*, *Die Aufgeregten*, *Reise in die
  Schweiz*, *Italienische Reise*, *Maximen und Reflexionen*,
  *Sprüche in Reimen*.
- Schiller's *Wilhelm Tell*, *Die Räuber*, *Wallensteins
  Lager / Wallensteins Tod*, *Maria Stuart*, *Don Carlos*,
  *Kabale und Liebe*, *Die Jungfrau von Orleans*, *Die Braut
  von Messina*, *Die Verschwörung des Fiesco zu Genua*,
  *Über die ästhetische Erziehung des Menschen* (the 27
  Briefe).

No 20th-century critical-edition commentary is cited.
``attested_source`` preserves the original work + act/scene/line
format without normalizing to a single citation style (Phase 9
may add a citation formatter; Phase 8 doesn't).

Usage::

    cd backend
    uv run python scripts/seed_phrases_attestations.py \\
        --source data/goethe_schiller_idioms_subset.json

Expected output::

    Goethe/Schiller attestations: 225 rows loaded (target 200-300).
    Frequency-band bumps applied: N (count rows whose band moved up).

The exact bump count depends on the state of the existing
``phrases`` table when the script runs. Re-running against a
fresh DB after a 8.1 seed (Phase 8.1 + 8.2 back-to-back)
bumps whatever the 8.1 bands left as ``"low"`` (the 9 low-band
rows in the 8.1 DWDS seed) wherever the 8.2 attestation is
``"high"`` or ``"mid"`` — that's the deterministic bump path.

The ``phrases`` table is **read-only at runtime** (Hard rule #2
of PHASE-8.md). The seed script is the only writer outside
Alembic. The 8.1 ``seed_phrases_dwds.py`` runs FIRST (per the
8.2 card body's hard rule "8.1's seed must have run first");
this script is downstream of 8.1 and extends.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import SessionLocal, DATABASE_URL
from app.models import Phrase

logger = logging.getLogger("lexora.seed_phrases_attestations")

# Default seed path (mirrors the 8.1 layout under ``backend/data/``).
DEFAULT_SEED_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "goethe_schiller_idioms_subset.json"
)

# Card body contract: target 200-300 rows. The hand-curated
# Goethe/Schiller subset in ``data/goethe_schiller_idioms_subset.json``
# ships with 225 rows; the test asserts 200-300.
TARGET_MIN_ROWS = 200
TARGET_MAX_ROWS = 300


def _is_pg() -> bool:
    """Dialect discriminator (mirrors ``app.models._is_pg``)."""
    return DATABASE_URL.startswith("postgresql")


def _load_seed_rows(path: Path) -> list[dict]:
    """Parse the JSON-array seed file into a list of dicts.

    Each row carries the same shape as the 8.1 fixture (Pydantic
    validation upstream), plus the two attestation columns that
    8.1 left null and 8.2 populates.

    The row shape on disk is a 7- or 8-tuple::

        (
            id, phrase, definition, example_usage, source_attribution,
            frequency_band, attested_quote, attested_source
        )

    ``attested_quote`` may be null; ``attested_source`` is the
    citation string (preserved verbatim — no normalization).

    Returns a list of dicts in the shape consumed by the
    SQLAlchemy Core ``insert`` statement below.
    """
    with open(path, encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"seed file {path}: invalid JSON: {e}")
    if not isinstance(payload, list):
        raise SystemExit(
            f"seed file {path}: expected JSON array of rows, got "
            f"{type(payload).__name__}"
        )

    out: list[dict] = []
    for lineno, row in enumerate(payload, 1):
        if not isinstance(row, list):
            raise SystemExit(
                f"seed file {path}:{lineno}: expected JSON array "
                f"tuple, got {type(row).__name__}"
            )
        if len(row) != 8:
            raise SystemExit(
                f"seed file {path}:{lineno}: expected 8-tuple "
                f"(id, phrase, definition, example_usage, "
                f"source_attribution, frequency_band, attested_quote, "
                f"attested_source), got {len(row)}-tuple"
            )
        (
            sid, phrase, definition, example_usage, source_attribution,
            frequency_band, attested_quote, attested_source,
        ) = row
        # Validate via the wire-layer Pydantic model so a typo'd
        # frequency_band / source_attribution is caught at the seed
        # boundary, NOT when the row silently propagates into the
        # Phase 8.3 idiom generator.
        from app.schemas import PhraseSeedRow  # local import — Pydantic runtime

        try:
            validated = PhraseSeedRow.model_validate(
                {
                    "id": sid,
                    "phrase": phrase,
                    "definition": definition,
                    "example_usage": example_usage,
                    "source_attribution": source_attribution,
                    "frequency_band": frequency_band,
                    "dwds_url": None,
                }
            )
        except Exception as e:
            raise SystemExit(
                f"seed file {path}:{lineno} (id={sid!r}) "
                f"PhraseSeedRow validation failed: {e}"
            )

        out.append(
            {
                "id": validated.id,
                "phrase": validated.phrase,
                "definition": validated.definition,
                "example_usage": validated.example_usage,
                "source_attribution": validated.source_attribution,
                "frequency_band": validated.frequency_band,
                "attested_quote": attested_quote,
                "attested_source": attested_source,
            }
        )
    return out


# Frequency-band ordering for the MAX() bump rule. The MAX() function
# over the literal Text columns would do a string MAX, not a band-rank
# MAX — so we need to do the bump manually via a CASE/SELECT roundtrip.
# The simpler portable shape: explicit Python-side ranking via the
# _BAND_RANK dict, then compare against the existing row to write
# only when the new band is strictly higher.
_BAND_RANK = {"low": 0, "mid": 1, "high": 2}


def _upsert_with_bump(
    session, rows: list[dict]
) -> tuple[int, int]:
    """Upsert rows into ``phrases`` with frequency-band bump.

    On a fresh DB or a row whose ``id`` isn't yet present, this is
    a plain ``INSERT ... ON CONFLICT (id) DO NOTHING`` (mirroring
    the 8.1 shape). When ``id`` is already populated (i.e. 8.1
    seeded it), the script UPDATEs the row to:

    - bump ``frequency_band`` if the new row is strictly higher
      (per ``_BAND_RANK``);
    - overwrite ``source_attribution`` with the comma-joined
      union of the existing + new literal tokens (so a row
      seeded by 8.1 as ``"dwds"`` and re-seeded by 8.2 with
      ``"goethe"`` becomes ``"dwds,goethe"``);
    - fill ``attested_quote`` and ``attested_source`` from 8.2's
      values (null on the first re-run if a row's 8.2 attestation
      fields are null).

    The dialect-portable shape is::

        INSERT INTO phrases (cols) VALUES (...)
        ON CONFLICT (id) DO UPDATE SET
          frequency_band = CASE
            WHEN excluded.frequency_band RANK > phrases.frequency_band RANK
              THEN excluded.frequency_band
            ELSE phrases.frequency_band
          END,
          ...

    SQLite doesn't expose a ``RANK()`` function in the dialect;
    the cleaner portable shape is to do the bump in two phases:

    1. INSERT ... ON CONFLICT (id) DO NOTHING for the rows
       whose ``id`` is fresh (lets SQLite skip silently).
    2. SELECT the existing rows by ``id``; in Python, compare
       ``_BAND_RANK`` to determine which existing rows need a
       higher-band UPDATE; issue the targeted UPDATE.

    Returns ``(total_rows, bump_count, union_count)`` where:

    - ``total_rows`` is the row count in ``phrases`` AFTER the run
      (a re-run against a populated DB returns the same count as
      the previous run — that's the idempotency contract).
    - ``bump_count`` is the number of existing rows whose
      ``frequency_band`` was *strictly promoted* (low→mid, low→high,
      or mid→high). Equal-rank bands don't count.
    - ``union_count`` is the number of existing rows whose
      ``source_attribution`` was union'd with the new
      attribution (e.g. ``"dwds"`` → ``"dwds,goethe"``).
    """
    if not rows:
        return (0, 0, 0)

    # Pre-dedupe by id AND phrase (mirrors the 8.1 dedupe).
    seen_id = set()
    seen_phrase = set()
    deduped = []
    for r in rows:
        if r["id"] in seen_id or r["phrase"] in seen_phrase:
            continue
        seen_id.add(r["id"])
        seen_phrase.add(r["phrase"])
        deduped.append(r)
    rows = deduped

    table = Phrase.__table__
    stmt_dicts = [
        {
            "id": r["id"],
            "phrase": r["phrase"],
            "definition": r["definition"],
            "example_usage": r["example_usage"],
            "source_attribution": r["source_attribution"],
            "frequency_band": r["frequency_band"],
            "dwds_url": None,
            "attested_quote": r["attested_quote"],
            "attested_source": r["attested_source"],
        }
        for r in rows
    ]

    # Phase 1: INSERT ... ON CONFLICT (id) DO NOTHING (handles the
    # fresh-row case, idempotent for re-runs on the same source set).
    if _is_pg():
        stmt = (
            pg_insert(table)
            .values(stmt_dicts)
            .on_conflict_do_nothing(index_elements=["id"])
        )
    else:
        stmt = (
            sqlite_insert(table)
            .values(stmt_dicts)
            .on_conflict_do_nothing(index_elements=["id"])
        )
    session.execute(stmt)
    session.commit()

    # Phase 2: read back existing rows; bump frequency_band, union
    # source_attribution, and fill attestation columns where the
    # 8.2 row carries new data the 8.1 row didn't have.
    bump_count = 0
    union_count = 0
    table_ids = [d["id"] for d in stmt_dicts]
    existing = session.execute(
        select(
            Phrase.id,
            Phrase.frequency_band,
            Phrase.source_attribution,
        ).where(Phrase.id.in_(table_ids))
    ).all()
    existing_map = {row[0]: (row[1], row[2]) for row in existing}

    for d in stmt_dicts:
        sid = d["id"]
        new_band = d["frequency_band"]
        new_sa = d["source_attribution"]
        if sid not in existing_map:
            continue
        old_band, old_sa = existing_map[sid]
        new_rank = _BAND_RANK.get(new_band, -1)
        old_rank = _BAND_RANK.get(old_band, -1)

        rank_bump = new_rank > old_rank
        sa_union = bool(new_sa) and new_sa not in old_sa
        aq_set = d["attested_quote"] is not None
        as_set = d["attested_source"] is not None

        if not (rank_bump or sa_union or aq_set or as_set):
            continue

        merged_sa = old_sa
        if sa_union:
            tokens = [t.strip() for t in old_sa.split(",") if t.strip()]
            for tok in new_sa.split(","):
                tok = tok.strip()
                if tok and tok not in tokens:
                    tokens.append(tok)
            merged_sa = ",".join(tokens)

        new_freq = new_band if rank_bump else old_band
        new_aq = d["attested_quote"] if aq_set else None
        new_as = d["attested_source"] if as_set else None
        target_sa = merged_sa if sa_union else old_sa

        session.execute(
            table.update()
            .where(Phrase.id == sid)
            .values(
                frequency_band=new_freq,
                source_attribution=target_sa,
                attested_quote=new_aq,
                attested_source=new_as,
            )
        )
        if rank_bump:
            bump_count += 1
        if sa_union:
            union_count += 1

    session.commit()

    from sqlalchemy import func

    n_after = session.scalar(select(func.count()).select_from(Phrase))
    return (n_after or 0, bump_count, union_count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SEED_PATH),
        help=(
            f"Path to the Goethe/Schiller idioms JSON file "
            f"(default: {DEFAULT_SEED_PATH})"
        ),
    )
    args = parser.parse_args()

    seed_path = Path(args.source)
    if not seed_path.exists():
        print(f"ERROR: seed file not found at {seed_path}", file=sys.stderr)
        return 1

    rows = _load_seed_rows(seed_path)
    n = len(rows)
    if n < TARGET_MIN_ROWS or n > TARGET_MAX_ROWS:
        print(
            f"WARNING: seed file has {n} rows; expected "
            f"{TARGET_MIN_ROWS}-{TARGET_MAX_ROWS}. The card body "
            f"target is 200-300; a follow-up commit can extend.",
            file=sys.stderr,
        )

    n_null = sum(1 for r in rows if r["attested_quote"] is None)
    if rows:
        pct = 100 * n_null / len(rows)
        if pct > 5:
            print(
                f"WARNING: {n_null}/{len(rows)} rows ({pct:.1f}%) "
                f"have null attested_quote; card body accepts ≤5%.",
                file=sys.stderr,
            )

    session = SessionLocal()
    try:
        total, bump_count, union_count = _upsert_with_bump(session, rows)
        print(
            f"Goethe/Schiller attestations: {n} rows loaded "
            f"(target {TARGET_MIN_ROWS}-{TARGET_MAX_ROWS}). "
            f"Frequency-band bumps applied: {bump_count} (of "
            f"{union_count} total touched existing rows). "
            f"Total phrases now: {total}."
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
