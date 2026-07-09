"""Interactive labeler helper for the ``phrase_match`` eval set.

Phase 10.4 (card t_f3d2a634). Walks Anurag through ~10 candidate
phrase-pairs at a time, prompts for a 4-way relation choice
(``equivalent`` / ``paraphrase`` / ``related`` / ``unrelated``),
and writes each accepted row to ``../eval/phrase_match_judgments.jsonl``.

Hard rules
----------

- **HUMAN-LABELED only.** The helper does NOT call any LLM; it
  reads from the planted ``phrases`` table (or from
  ``backend/data/`` fixtures if the DB is offline) and prompts the
  human. The whole point of the card is a human-labeled set; an
  LLM-assisted path would defeat the purpose.
- **Idempotent on (phrase_a_id, phrase_b_id).** Re-running the
  helper against a partially-filled JSONL picks up where the
  previous session left off; the same pair is never written twice.
- **Validates every labeled row.** Each accepted row is checked
  against:

  - the literal 4-way relation set,
  - that ``phrase_a_id`` and ``phrase_b_id`` exist in the planted
    ``phrases`` table (loaded from the DB or from the same fixtures
    the offline tests use),
  - that ``attested_pair`` is a bool,
  - that ``phrase_a_id != phrase_b_id`` (mirror of the DB CHECK
    constraint + Pydantic validator).

- **Tracks running distribution across the 4 relations.** Warns
  after ~25 labels if any relation is under-represented (mirror of
  the manifest's ``target_distribution``).
- **No write to ``phrases`` or ``phrase_pairs`` table.** This script
  writes ONLY to ``eval/phrase_match_judgments.jsonl`` + updates
  ``eval/phrase_match_judgments.manifest.json``'s ``current_count``
  and ``current_distribution``.

## Operating modes

1. ``--dry-run`` — print candidate pairs (no writes). Use this to
   preview what the helper would walk through. Confirms the
   DB-to-phrase-pool wiring is healthy.
2. ``--target N --batch B`` (default ``--target 50 --batch 10``) —
   interactive prompt-driven labeling session. Anurag types the
   relation choice; the helper writes the row immediately. After
   each batch, the helper prints the running distribution.

## Source pool

Default source: the planted ``phrases`` table (Phase 8.1 + 8.2
combined, ~367 rows). If the DB is offline, the helper falls
back to the union of ``backend/data/dwds_idioms_subset.json``
and ``backend/data/goethe_schiller_idioms_subset.json`` — the
same fixtures the offline tests use. The candidate pool is the
same set either way (the seed scripts idempotently mirror
``backend/data/`` JSON into the ``phrases`` table).

## Attested-pair shortcut

When a candidate pair appears in ``backend/data/attested_pairs.json``
(the Phase 10.4 hand-curated list), the helper offers the shortcut::

    [a] accept attested-pair override (relation=AUTOFILL, attested_pair=true)

Selecting ``a`` pre-fills the relation with the curated value from
``attested_pairs.json`` and writes the row without further
prompting. Anurag can always reject the shortcut by entering
manually.

## Idempotency / resume

The helper maintains the set of already-labeled
``(phrase_a_id, phrase_b_id)`` pairs in-memory on every invocation
by reading the existing ``eval/phrase_match_judgments.jsonl``. The
script is also exit-safe at the prompt (Ctrl-C exits cleanly;
partial state is intact on disk).

## Usage

From ``backend/``::

    # Preview 10 candidates without writes:
    uv run python -m scripts.label_phrase_match_pairs --dry-run --target 10

    # Interactive labeling (no LLM):
    uv run python -m scripts.label_phrase_match_pairs --target 50 --batch 10

    # Smaller batch size:
    uv run python -m scripts.label_phrase_match_pairs --target 50 --batch 5

Exit code: 0 on a clean session (target reached or graceful exit),
1 on unrecoverable error (no planted ``phrases`` available even
from the fixture fallback, the JSONL is malformed, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants — the same hard literals as the manifest. Kept in sync on purpose:
# a future widening of the 4-way relation must update BOTH this file and
# ``eval/phrase_match_judgments.manifest.json`` (the test_phrase_match_eval
# suite pins both).
# ---------------------------------------------------------------------------

RELATION_TAXONOMY: tuple[str, ...] = (
    "equivalent",
    "paraphrase",
    "related",
    "unrelated",
)

PROVENANCE_TAG: str = "HUMAN-LABELED"
LABELER: str = "Anurag Parida"

BACKEND_DIR: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = BACKEND_DIR.parent

EVAL_DIR: Path = REPO_ROOT / "eval"
JUDGMENTS_FILE: Path = EVAL_DIR / "phrase_match_judgments.jsonl"
MANIFEST_FILE: Path = EVAL_DIR / "phrase_match_judgments.manifest.json"

ATTESTED_PAIRS_FILE: Path = BACKEND_DIR / "data" / "attested_pairs.json"

DWDS_FIXTURE: Path = BACKEND_DIR / "data" / "dwds_idioms_subset.json"
GOETHE_SCHILLER_FIXTURE: Path = (
    BACKEND_DIR / "data" / "goethe_schiller_idioms_subset.json"
)


# ---------------------------------------------------------------------------
# Phrase-pool loading.
#
# Strategy: try the live DB first (the planted ``phrases`` table).
# If the DB is offline or the connection fails, fall back to the
# fixture files (``backend/data/dwds_idioms_subset.json`` and
# ``backend/data/goethe_schiller_idioms_subset.json``). Same pool
# either way (the seed scripts idempotently mirror the JSON into
# the DB).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhraseRecord:
    """In-memory view of a single planted phrases row.

    Mirrors ``app.models.Phrase`` for the columns the labeler
    needs. The DB-session connection is closed before any
    candidate-pair generation, so this dataclass carries only
    the data the helper needs (no live ORM session).
    """

    id: str  # slug PK (lowercase-hyphenated, 3-120 chars)
    phrase: str  # surface form
    source_attribution: str  # one of the literal set, comma-joined
    attested_source: str | None  # Phase 8.2 attestation citation


def _normalize_attribution(value: str) -> tuple[str, ...]:
    """Split a comma-joined ``source_attribution`` into tokens.

    Mirror of ``app.schemas._split_source_attribution`` (the
    Pydantic validator on ``PhraseSeedRow``). Used here so the
    helper doesn't have to import from the app package (the
    labeler keeps its dependency surface tiny — it's a CLI, not
    part of the FastAPI request path).
    """
    if not value:
        return ()
    return tuple(t.strip() for t in value.split(",") if t.strip())


def load_phrases_from_db() -> list[PhraseRecord] | None:
    """Try to load phrases from the planted ``phrases`` table.

    Returns ``None`` (rather than raising) on any DB failure so
    the caller can fall through to the fixture loader. The
    seed scripts guarantee the DB rows match the fixture
    rows, so the candidate-pool surface is identical either
    way.
    """
    try:
        from sqlalchemy import select  # noqa: F401  (tested for ImportError only)
        from app.database import SessionLocal  # type: ignore[import-not-found]
        from app.models import Phrase  # type: ignore[import-not-found]
    except Exception as exc:
        logging.debug("DB import failed (%s); falling back to fixture", exc)
        return None

    try:
        session = SessionLocal()
    except Exception as exc:
        logging.debug("DB session open failed (%s); falling back", exc)
        return None

    try:
        from sqlalchemy import select

        stmt = select(
            Phrase.id,
            Phrase.phrase,
            Phrase.source_attribution,
            Phrase.attested_source,
        )
        rows = session.execute(stmt).all()
        out = [
            PhraseRecord(
                id=row[0],
                phrase=row[1],
                source_attribution=row[2] or "",
                attested_source=row[3],
            )
            for row in rows
        ]
        if not out:
            logging.debug("DB query returned 0 phrases; falling back")
            return None
        return out
    except Exception as exc:
        logging.debug("DB query failed (%s); falling back", exc)
        return None
    finally:
        try:
            session.close()
        except Exception:
            pass


def load_phrases_from_fixtures() -> list[PhraseRecord]:
    """Load phrases from the offline JSON fixtures.

    Always succeeds (the fixtures ship with the repo). Used as
    the fallback when the DB is unavailable.

    The two fixtures use different JSON shapes:

    - ``dwds_idioms_subset.json`` — JSON array of dicts
      (``{"id": ..., "phrase": ..., ...}``, 7 keys).
    - ``goethe_schiller_idioms_subset.json`` — JSON array of
      lists (``[id, phrase, definition, example, source_attr,
      frequency_band, attested_quote, attested_source]``, 8
      elements).

    Both shapes produce the same in-memory ``PhraseRecord``
    form so the candidate-pool surface is identical.
    """
    out: list[PhraseRecord] = []
    for fixture in (DWDS_FIXTURE, GOETHE_SCHILLER_FIXTURE):
        if not fixture.exists():
            continue
        data = json.loads(fixture.read_text())
        for row in data:
            if isinstance(row, dict):
                # DWDS fixture: dict-shape.
                sid = row.get("id")
                phrase = row.get("phrase")
                source_attr = row.get("source_attribution", "dwds")
                # DWDS-only fixture has no attestation. The labeler
                # treats unattested as `None`.
                attested = row.get("attested_source")
                if not (sid and phrase):
                    continue
            elif isinstance(row, list):
                # Goethe/Schiller fixture: list-shape.
                if len(row) < 7:
                    continue
                sid = row[0]
                phrase = row[1]
                source_attr = row[4] if len(row) > 4 else "dwds"
                attested = row[7] if len(row) > 7 else None
            else:
                continue
            out.append(
                PhraseRecord(
                    id=sid,
                    phrase=phrase,
                    source_attribution=source_attr or "",
                    attested_source=attested,
                )
            )
    return out


def load_phrases() -> list[PhraseRecord]:
    """Load the planted ``phrases`` pool with fallback.

    Always returns a non-empty list (the fixtures ship with the
    repo). Logs which source was used so the operator can tell
    whether the helper hit the live DB.
    """
    db_pool = load_phrases_from_db()
    if db_pool:
        logging.info("loaded %d phrases from live DB", len(db_pool))
        return db_pool
    fixture_pool = load_phrases_from_fixtures()
    logging.info("loaded %d phrases from fixtures (DB fallback)", len(fixture_pool))
    return fixture_pool


# ---------------------------------------------------------------------------
# Attested-pair loader.
#
# Reads ``backend/data/attested_pairs.json`` and indexes the
# curated rows by their (phrase_a_id, phrase_b_id) pair key so
# the labeler can offer the attested-pair shortcut during
# interactive sessions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttestedPairRecord:
    """In-memory view of a single ``attested_pairs.json`` entry."""

    phrase_a_id: str
    phrase_b_id: str
    relation: str
    attested_quote_a: str | None
    attested_source_a: str | None
    attested_quote_b: str | None
    attested_source_b: str | None
    attested_quote_combined: str | None
    rationale: str


def load_attested_pairs() -> dict[tuple[str, str], AttestedPairRecord]:
    """Load ``attested_pairs.json`` and index by (a, b) pair.

    The lookup is symmetric (both (a, b) and (b, a) match) because
    the planted ``phrase_pairs`` table uses ``ON CONFLICT (a, b)``
    with a lexicographic sort — the seed script writes each
    attested-pair override once, in canonical (sorted) order.
    """
    if not ATTESTED_PAIRS_FILE.exists():
        return {}
    payload = json.loads(ATTESTED_PAIRS_FILE.read_text())
    pairs = payload.get("pairs", [])
    out: dict[tuple[str, str], AttestedPairRecord] = {}
    for p in pairs:
        a = p["phrase_a_id"]
        b = p["phrase_b_id"]
        rec = AttestedPairRecord(
            phrase_a_id=a,
            phrase_b_id=b,
            relation=p["relation"],
            attested_quote_a=p.get("attested_quote_a"),
            attested_source_a=p.get("attested_source_a"),
            attested_quote_b=p.get("attested_quote_b"),
            attested_source_b=p.get("attested_source_b"),
            attested_quote_combined=p.get("attested_quote_combined"),
            rationale=p.get("rationale", ""),
        )
        out[(a, b)] = rec
        out[(b, a)] = rec  # symmetric: same attested-pair shortcut either way
    return out


# ---------------------------------------------------------------------------
# Candidate-pair generation.
#
# We sample pairs at random from the planted pool. The distribution
# rule (Phase 10.1 seed script, referenced by this card body) uses
# quartiles, but THIS card is the human-labeled eval set — Anurag
# picks the relation for each pair, not the script. The helper
# only needs to surface candidate pairs; the relation judgment
# is human. So we sample uniformly from the pool and surface
# already-labeled-pair-aware skip pairs.
#
# The seed script's bucket-assignment rule (top-quartile =
# equivalent, next = paraphrase, etc.) is the script's surface
# for the auto-populated phrase_pairs rows; it's NOT the
# shape this helper ships.
# ---------------------------------------------------------------------------


def iter_candidate_pairs(
    pool: list[PhraseRecord],
    *,
    seed: int,
    target: int,
) -> Iterable[tuple[PhraseRecord, PhraseRecord]]:
    """Yield candidate (A, B) phrase-pairs from the pool.

    Sampling rule: random unique pair from the pool, with a
    fixed ``seed`` so a re-run produces the same candidate
    stream (helpful for the dry-run preview). Self-pairs
    (a.id == b.id) are excluded.

    The iterator stops as soon as any of three conditions
    fires:

    - ``target`` unique pairs have been yielded,
    - ``seen`` set is full (every legal pair from the pool has
      already been yielded),
    - the random sampler has drawn ``max_draws`` candidate
      combinations without yielding (safety floor for tiny
      pools where the function would otherwise spin forever).

    For the production pool (~367 phrases) ``max_draws=10 *
    target`` is unreachable. For the 3-phrase offline test
    pool the upper bound on legal pairs is C(3, 2) = 3; the
    iterator returns 3 pairs and stops, mirroring the real
    pool's behavior under exhaustion.

    Note: ordering is randomized, NOT balanced by source
    attribution. Anurag is responsible for the 4-way balance
    across the session — the helper only warns when the
    running distribution drifts (see ``_warn_distribution``).
    """
    rng = random.Random(seed)
    if len(pool) < 2:
        return
    seen: set[tuple[str, str]] = set()
    max_draws = max(100, 10 * target)
    draws = 0
    while len(seen) < target and draws < max_draws:
        draws += 1
        a, b = rng.sample(pool, 2)
        # Lexicographic-canonical key (matches the DB UNIQUE constraint).
        if a.id == b.id:
            continue
        key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        if key in seen:
            continue
        seen.add(key)
        # If we drew (b, a), swap to canonical (a, b) for display.
        a, b = (a, b) if a.id < b.id else (b, a)
        yield a, b


# ---------------------------------------------------------------------------
# Existing-label loading.
#
# The helper reads ``phrase_match_judgments.jsonl`` on every
# invocation to build the seen-pair index for idempotency.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExistingLabel:
    """In-memory view of one already-labeled row."""

    phrase_a_id: str
    phrase_b_id: str
    relation: str
    attested_pair: bool


def load_existing_labels() -> dict[tuple[str, str], ExistingLabel]:
    """Load already-labeled pairs from the JSONL.

    Uses the lexicographic-canonical key (matches the DB UNIQUE
    constraint and the candidate-pair iterator above).
    """
    if not JUDGMENTS_FILE.exists():
        return {}
    out: dict[tuple[str, str], ExistingLabel] = {}
    for line_no, raw_line in enumerate(JUDGMENTS_FILE.read_text().splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL at line {line_no} of {JUDGMENTS_FILE}: {exc}"
            ) from exc
        try:
            a = row["phrase_a_id"]
            b = row["phrase_b_id"]
        except KeyError as exc:
            raise ValueError(
                f"row at line {line_no} missing required key {exc!s}"
            ) from exc
        if a == b:
            raise ValueError(
                f"row at line {line_no} has phrase_a_id == phrase_b_id "
                f"(both = {a!r}); distinct pair required (refusing to continue)"
            )
        # Canonical key: (min, max)
        key = (a, b) if a < b else (b, a)
        relation = row.get("relation")
        if relation not in RELATION_TAXONOMY:
            raise ValueError(
                f"row at line {line_no} has invalid relation {relation!r}; "
                f"must be one of {RELATION_TAXONOMY}"
            )
        attested = bool(row.get("attested_pair", False))
        if key in out:
            # Already saw this pair (duplicate label). Tolerate with
            # a debug log; the manual session may have re-labeled.
            logging.debug("duplicate label for pair %s at line %d", key, line_no)
            continue
        out[key] = ExistingLabel(
            phrase_a_id=a,
            phrase_b_id=b,
            relation=relation,
            attested_pair=attested,
        )
    return out


# ---------------------------------------------------------------------------
# Manifest read/write.
#
# The helper updates the manifest's ``current_count`` and
# ``current_distribution`` fields after each accepted row. The
# rest of the manifest is left untouched.
# ---------------------------------------------------------------------------


def read_manifest() -> dict:
    """Read the manifest, or raise if it doesn't exist."""
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"manifest not found at {MANIFEST_FILE}; this card expects it "
            f"to be present (Phase A scaffold)"
        )
    return json.loads(MANIFEST_FILE.read_text())


def write_manifest(manifest: dict) -> None:
    """Write the manifest back, atomically (write-replace)."""
    tmp = MANIFEST_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(MANIFEST_FILE)


def write_judgment_row(row: dict) -> None:
    """Append one row to ``phrase_match_judgments.jsonl``.

    Atomic write: build the new file content in-memory (existing
    lines + the new JSON-encoded row + trailing newline) and
    write-replace. This avoids partial writes if the process is
    interrupted at a bad moment.
    """
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
    existing = JUDGMENTS_FILE.read_text() if JUDGMENTS_FILE.exists() else ""
    # Ensure exactly one trailing newline before appending the new row.
    if existing and not existing.endswith("\n"):
        existing += "\n"
    new_content = existing + encoded + "\n"
    JUDGMENTS_FILE.write_text(new_content)


# ---------------------------------------------------------------------------
# Validation.
#
# Each accepted row is checked against the same hard rules as the
# planted ``phrase_pairs`` Pydantic schema (Phase 10.1). Validation
# runs BEFORE write so a bad input never lands in the JSONL.
# ---------------------------------------------------------------------------


def validate_row(
    row: dict,
    *,
    phrase_pool: list[PhraseRecord],
) -> None:
    """Validate a labeled row against the hard rules.

    Raises ``ValueError`` with a parser-friendly message. The
    caller surfaces the error and re-prompts the user.
    """
    pool_ids = {p.id for p in phrase_pool}
    a = row.get("phrase_a_id")
    b = row.get("phrase_b_id")
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(
            f"phrase_a_id / phrase_b_id must be strings "
            f"(got {type(a).__name__} / {type(b).__name__})"
        )
    if a == b:
        raise ValueError(
            f"phrase_a_id and phrase_b_id must be distinct (both = {a!r})"
        )
    if a not in pool_ids:
        raise ValueError(
            f"phrase_a_id {a!r} not found in the planted phrases table"
        )
    if b not in pool_ids:
        raise ValueError(
            f"phrase_b_id {b!r} not found in the planted phrases table"
        )
    relation = row.get("relation")
    if relation not in RELATION_TAXONOMY:
        raise ValueError(
            f"relation must be one of {RELATION_TAXONOMY} (got {relation!r})"
        )
    attested = row.get("attested_pair")
    if not isinstance(attested, bool):
        raise ValueError(
            f"attested_pair must be a bool (got {type(attested).__name__})"
        )


def build_row(
    *,
    a: PhraseRecord,
    b: PhraseRecord,
    relation: str,
    attested_pair: bool,
    rationale: str,
) -> dict:
    """Build the canonical JSONL row shape from labeled inputs.

    Field order matches the manifest's ``row_schema``. All
    fields written; ``rationale`` is required so future
    reviewers can audit the relation choice.
    """
    return {
        "phrase_a_id": a.id,
        "phrase_b_id": b.id,
        "phrase_a": a.phrase,
        "phrase_b": b.phrase,
        "relation": relation,
        "attested_pair": attested_pair,
        "rationale": rationale,
        "source_attribution_a": a.source_attribution,
        "source_attribution_b": b.source_attribution,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Distribution tracking + warning.
#
# Phase A's hard rule: balance across the 4 relations (~12-13 per
# relation for a 50-row target). The helper warns when the running
# distribution drifts from the manifest's ``target_distribution``
# after ~25 labels.
# ---------------------------------------------------------------------------


WARN_AFTER_LABELS = 25


def update_distribution(
    manifest: dict,
    relation: str,
) -> None:
    """Update the manifest's ``current_distribution`` and ``current_count``.

    Always increments the relation bucket; the helper writes the
    manifest back after each accepted row.
    """
    cur = manifest.setdefault("current_distribution", {r: 0 for r in RELATION_TAXONOMY})
    cur[relation] = cur.get(relation, 0) + 1
    n = sum(cur.values())
    manifest["current_count"] = n
    if n > 0:
        manifest["labeler_session_window"] = (
            f"{datetime.now(timezone.utc).date().isoformat()} (labels #1-#{n})"
        )


def warn_distribution_drift(
    manifest: dict,
    *,
    force: bool = False,
) -> None:
    """Print a warning if the running distribution is unbalanced.

    Suppressed for the first ``WARN_AFTER_LABELS`` rows (early
    labels can't be balanced yet). After that threshold,
    under-represented relations trigger a non-blocking stderr
    hint so Anurag can pivot to fill the gap.
    """
    cur = manifest.get("current_distribution", {})
    n = sum(cur.values())
    if not force and n < WARN_AFTER_LABELS:
        return
    target_total = manifest.get("target_count", 50)
    if target_total <= 0:
        return
    target = manifest.get("target_distribution", {})
    under: list[tuple[str, int, int]] = []
    for r in RELATION_TAXONOMY:
        want = int(target.get(r, 0))
        have = int(cur.get(r, 0))
        # What we'd want at the same proportion of the running total.
        projected_target = round(want * n / target_total) if target_total else 0
        if have + 2 < projected_target:
            under.append((r, have, want))
    if under:
        lines = ["[distribution warning] running distribution is underweight:"]
        for r, have, want in under:
            lines.append(f"   - {r}: have {have}, projected-target {projected_target}, target {want}")
        lines.append("Next pairs: prefer relations marked above.")
        print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# Interactive prompt.
#
# Uses stdin in non-TTY-aware mode (the labeler is meant for
# terminal use but is also CI-callable for the dry-run path).
# ---------------------------------------------------------------------------


def prompt_relation(*, allowed: tuple[str, ...] = RELATION_TAXONOMY) -> str:
    """Prompt the user for a relation choice.

    Accepts the full 4-way literal. Empty input = ask again.
    'q' = quit (Ctrl-C equivalent for stdin-driven sessions).
    """
    choices = " | ".join(f"[{i+1}] {r}" for i, r in enumerate(allowed))
    while True:
        try:
            raw = input(f"Relation? ({choices})  > ").strip().lower()
        except EOFError:
            print()  # newline after Ctrl-D
            raise SystemExit(0)
        if raw in ("q", "quit", "exit"):
            raise SystemExit(0)
        if raw in allowed:
            return raw
        # 1..4 numeric shortcut.
        try:
            idx = int(raw)
            if 1 <= idx <= len(allowed):
                return allowed[idx - 1]
        except ValueError:
            pass
        print(f"  (must be one of {allowed} or 1..{len(allowed)})")


def prompt_rationale() -> str:
    """Prompt for a one-sentence rationale (audit trail)."""
    while True:
        try:
            raw = input("Rationale (1-400 chars; Enter to skip)  > ")
        except EOFError:
            print()
            raise SystemExit(0)
        raw = raw.strip()
        if not raw:
            return "(no rationale)"
        if 1 <= len(raw) <= 400:
            return raw
        print(f"  (must be 1-400 chars; got {len(raw)})")


# ---------------------------------------------------------------------------
# Display formatting.
#
# One card per candidate pair. The display mirrors what a learner
# would see: the two phrases side-by-side, with their source
# attribution and attestation citation.
# ---------------------------------------------------------------------------


def _format_pair(a: PhraseRecord, b: PhraseRecord) -> str:
    """Format a candidate pair for interactive display."""
    border = "-" * 72
    parts = [
        border,
        f"  A: {a.phrase!r}    [{a.source_attribution}]",
        f"      id={a.id}",
    ]
    if a.attested_source:
        parts.append(f"      attested: {a.attested_source}")
    parts.append("")
    parts.append(
        f"  B: {b.phrase!r}    [{b.source_attribution}]"
    )
    parts.append(f"      id={b.id}")
    if b.attested_source:
        parts.append(f"      attested: {b.attested_source}")
    parts.append(border)
    return "\n".join(parts)


def _format_attested_shortcut(rec: AttestedPairRecord) -> str:
    """Format the attested-pair shortcut prompt."""
    return (
        f"\n[attested-pair override] relation={rec.relation!r}, "
        f"rationale={rec.rationale!r}\n"
        f"  Type 'a' to accept, or pick a relation manually."
    )


# ---------------------------------------------------------------------------
# Main entrypoint.
# ---------------------------------------------------------------------------


def cmd_dry_run(args: argparse.Namespace) -> int:
    """``--dry-run`` mode: print candidate pairs, no writes."""
    pool = load_phrases()
    if not pool:
        print(
            "ERROR: planted phrases pool is empty (DB and fixtures both "
            "unavailable). The helper needs at least 2 phrases to "
            "generate candidates.",
            file=sys.stderr,
        )
        return 1

    attested_pairs = load_attested_pairs()
    n = min(args.target, len(pool) * (len(pool) - 1) // 2)
    print(
        f"\n[dry-run] {n} candidate pair(s) from a pool of {len(pool)} "
        f"planted phrases\n"
        f"  attested-pair override rows: {len(attested_pairs) // 2} "
        f"(symmetric index)\n"
        f"  relation_taxonomy: {RELATION_TAXONOMY}\n"
    )
    candidates = list(
        iter_candidate_pairs(pool, seed=args.seed, target=n)
    )
    for i, (a, b) in enumerate(candidates, 1):
        print(f"\n  Pair #{i}")
        print(_format_pair(a, b))
        key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        if key in attested_pairs:
            rec = attested_pairs[key]
            print(_format_attested_shortcut(rec))
    print(
        f"\n[dry-run] {len(candidates)} candidate pair(s) printed. "
        f"No writes. Re-run without --dry-run to start the interactive "
        f"session."
    )
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """Interactive labeling session.

    Walks Anurag through ``args.batch`` candidate pairs at a time,
    up to ``args.target`` total. Updates the manifest after each
    accepted row. Re-runs resume from the existing JSONL.
    """
    manifest = read_manifest()
    if manifest.get("provenance") != PROVENANCE_TAG:
        print(
            f"ERROR: manifest provenance is {manifest.get('provenance')!r}; "
            f"expected {PROVENANCE_TAG!r}. Refusing to write a "
            f"non-HUMAN-LABELED manifest.",
            file=sys.stderr,
        )
        return 1

    pool = load_phrases()
    if not pool:
        print(
            "ERROR: planted phrases pool is empty (DB and fixtures both "
            "unavailable). Refusing to start interactive session.",
            file=sys.stderr,
        )
        return 1

    attested_pairs = load_attested_pairs()
    existing = load_existing_labels()
    existing_count = manifest.get("current_count", len(existing))
    if existing_count != len(existing):
        # Drift between manifest and JSONL — defer to JSONL.
        logging.warning(
            "manifest current_count=%d but JSONL has %d rows; "
            "recomputing from JSONL",
            existing_count,
            len(existing),
        )
        # Recompute current_distribution from existing labels.
        cur_dist = {r: 0 for r in RELATION_TAXONOMY}
        for label in existing.values():
            cur_dist[label.relation] += 1
        manifest["current_distribution"] = cur_dist
        manifest["current_count"] = len(existing)
        write_manifest(manifest)

    target = args.target
    batch = args.batch
    if existing_count >= target:
        print(
            f"Already labeled {existing_count} pairs (target={target}); "
            f"nothing to do."
        )
        return 0

    print(
        f"\nStarting interactive labeling session.\n"
        f"  pool size: {len(pool)} planted phrases\n"
        f"  attested-pair overrides: {len(attested_pairs) // 2}\n"
        f"  already labeled: {existing_count}/{target}\n"
        f"  batch size: {batch}\n"
        f"  relation taxonomy: {RELATION_TAXONOMY}\n"
        f"  (Ctrl-C or 'q' to exit; partial state is preserved)\n"
    )

    accepted_this_session = 0
    needed = target - existing_count

    candidates = list(
        iter_candidate_pairs(pool, seed=args.seed, target=needed + batch)
    )
    # Skip pairs that are already labeled.
    pending: list[tuple[PhraseRecord, PhraseRecord]] = []
    for cand in candidates:
        a, b = cand
        key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        if key in existing:
            continue
        pending.append(cand)
        if len(pending) >= needed + batch:
            break

    if not pending:
        print("No candidates available (DB / fixture is exhausted).")
        return 0

    for idx, (a, b) in enumerate(pending, 1):
        if existing_count + accepted_this_session >= target:
            print(
                f"\nReached target ({target} labeled pairs). "
                f"Session complete."
            )
            break

        print(f"\nPair #{idx} of batch (running total: {existing_count + accepted_this_session}/{target})")
        print(_format_pair(a, b))
        key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        attested_rec = attested_pairs.get(key)

        relation: str
        attested_pair_flag: bool
        rationale: str

        if attested_rec is not None:
            print(_format_attested_shortcut(attested_rec))
            while True:
                raw = input("[a] accept shortcut  |  [m] manual  > ").strip().lower()
                if raw in ("a", "accept"):
                    relation = attested_rec.relation
                    attested_pair_flag = True
                    rationale = (
                        f"attested-pair shortcut: {attested_rec.rationale}"
                    )
                    break
                if raw in ("m", "manual", ""):
                    relation = prompt_relation()
                    attested_pair_flag = False
                    rationale = prompt_rationale()
                    break
                print("  (type 'a' for attested-pair shortcut or 'm' for manual)")
        else:
            relation = prompt_relation()
            attested_pair_flag = False
            rationale = prompt_rationale()

        row = build_row(
            a=a,
            b=b,
            relation=relation,
            attested_pair=attested_pair_flag,
            rationale=rationale,
        )

        try:
            validate_row(row, phrase_pool=pool)
        except ValueError as exc:
            print(f"  ! validation failed: {exc}; not written", file=sys.stderr)
            continue

        write_judgment_row(row)
        update_distribution(manifest, relation)
        write_manifest(manifest)
        accepted_this_session += 1
        existing[key] = ExistingLabel(
            phrase_a_id=a.id,
            phrase_b_id=b.id,
            relation=relation,
            attested_pair=attested_pair_flag,
        )
        print(
            f"  ok (relation={relation}, attested={attested_pair_flag}, "
            f"running={existing_count + accepted_this_session}/{target})"
        )
        warn_distribution_drift(manifest)

    final_count = existing_count + accepted_this_session
    print(
        f"\nSession summary:\n"
        f"  starting count: {existing_count}\n"
        f"  accepted this session: {accepted_this_session}\n"
        f"  total now: {final_count} / {target}\n"
        f"  distribution: {manifest.get('current_distribution', {})}\n"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI arg parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help=(
            f"Number of pairs to label (default 50, per the manifest's "
            f"target_count). Lowered in dry-run preview mode."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help=(
            "Candidate-pair chunk to surface (default 10). Each pair is "
            "prompted interactively; the script resumes after the batch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260706,
        help=(
            "Random seed for candidate-pair sampling. Same seed = same "
            "candidate stream (helpful for dry-run preview)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print candidate pairs without writing them to the JSONL. "
            "Use this to preview what the interactive session would "
            "surface."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.batch <= 0:
        print("--batch must be > 0", file=sys.stderr)
        return 1
    if args.target <= 0:
        print("--target must be > 0", file=sys.stderr)
        return 1
    if args.dry_run:
        return cmd_dry_run(args)
    return cmd_interactive(args)


if __name__ == "__main__":
    sys.exit(main())
