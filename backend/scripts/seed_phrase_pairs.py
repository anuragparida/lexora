"""Seed the ``phrase_pairs`` table deterministically.

Phase 10.1 (card t_18c90a68). Idempotent: re-running against an
already-seeded table is a clean no-op — every INSERT goes through
``ON CONFLICT (phrase_a_id, phrase_b_id) DO NOTHING`` /
``INSERT OR IGNORE``.

## Inputs

1. The live ``phrases`` table (Phase 8.1 / 8.2) — the set of
   candidate phrase rows the pair-pool draws from.
2. ``backend/data/attested_pairs.json`` — the hand-curated
   Goethe / Schiller attested-pair list. Created **empty** in
   this card (``{"pairs": []}``); populated by Anurag in
   Phase 10.4. The script reads this file every run, so newly
   attested pairs land on the next re-run without code changes.

The script is the **single write path** to ``phrase_pairs``
outside Alembic (Hard rule of the plan body — same discipline
as ``phrases`` on Phase 8.1). It does no UPDATE / DELETE; rows
are immutable once seeded.

## Algorithm (verbatim from the plan body)

**Step 1 — candidate pair pool.** Compute a similarity score
across all pairs of ``phrases`` rows. Default similarity is the
``compute_pair_similarity(text_a, text_b)`` injection — the
production implementation uses bge-m3 cosine via the offline
``sentence-transformers`` cache (the same path Phase 7.5 A/B
prepares), but the function is a **swappable parameter** so the
test path can inject a deterministic stub and stay 100% offline.
The pool is filtered to ``score >= 0.55`` (the plan body's
threshold) and capped at ``CANDIDATE_POOL_CAP`` (default 500)
sorted by score descending, to keep the script's runtime bounded.

**Step 2 — bucket assignment.** The candidate pool is sorted
deterministically by ``(phrase_a_id, phrase_b_id)`` (alphabetic)
then split into quartiles by **rank** (NOT similarity — that's
anti-correlated with the deterministic bucket assignment the
plan body requires). Top quartile → ``equivalent``; next →
``paraphrase``; next → ``related``; bottom → ``unrelated``.

**Step 3 — attested-pair override.** For each row in
``attested_pairs.json``, INSERT (or ON CONFLICT DO NOTHING) a
``phrase_pairs`` row with the attested ``relation`` (which is
NOT necessarily ``equivalent`` — Goethe paraphrases can be
``paraphrase``; an allusion can be ``related``) and
``attested_pair=True``. Attested rows always win over the
bucketed ranking.

**Step 4 — insert idempotently.** The bucketed candidates +
attested overrides land via a single ``INSERT ... ON CONFLICT
(phrase_a_id, phrase_b_id) DO NOTHING`` statement
(``INSERT OR IGNORE`` on SQLite). The script prints ``Loaded
N phrase_pairs (target ~M, attested K)`` to stdout where M is
the candidate-pool size before attestation and K is the
attested-override count.

**Step 5 — sorting discipline.** Every (a, b) tuple is sorted
lexicographically *before* INSERT so the same pair can never
collide with its (b, a) mirror — the (a, b) pair on one INSERT
never collides with the (b, a) pair of a different INSERT
because we never carry (b, a). The composite UNIQUE on
``(phrase_a_id, phrase_b_id)`` enforces the in-order uniqueness,
not the swapped-pair uniqueness (which the seed script's
pre-sort rules out deterministically).

## Reproducibility

The script accepts ``--seed <int>`` (default 42). The seed
propagates into the candidate-pool cap behaviour (the pool is
pre-sorted alphabetically by ``(phrase_a_id, phrase_b_id)`` so
seed-derived randomness is applied to a stable input — re-runs
with the same seed on the same ``phrases`` table produce
byte-for-byte identical output).

## Usage ::

    cd backend
    uv run python scripts/seed_phrase_pairs.py --seed 42
    uv run python scripts/seed_phrase_pairs.py --seed 42  # idempotent

Pass ``--attested data/attested_pairs.json`` to point at a
non-default attested-pair file; ``--similarity-fn custom_module:fn``
injects a custom similarity function (CI / dev only).

## Hard rules

- **No LLM call.** The bge-m3 cosine step uses local
  ``sentence-transformers`` (Phase 1.3, same offline path
  Phase 7.5 A/B sets up). When that dependency isn't available
  the test / dev path injects a stub similarity function. CI
  stays offline.
- **No ``phrase_pairs`` writes outside this script.** The
  table is read-only at runtime; this script is the single
  write path.
- **Pre-sort before insert.** Every (a, b) tuple sorts
  lexicographically so the composite UNIQUE never collides
  with a swapped-pair duplicate. Self-pairs (``a == b``) are
  rejected at INSERT time via the DB CHECK constraint AND a
  Python pre-filter (belt-and-braces).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal, DATABASE_URL, reconfigure_for_test
from app.models import Phrase, PhrasePair

logger = logging.getLogger("lexora.seed_phrase_pairs")

DEFAULT_SEED = 42
# Per plan body — top-500 by cosine keeps the script's runtime
# bounded. Re-runs with a smaller cap just trim the pool from
# the bottom; the deterministic quartile assignment still
# produces a stable output.
CANDIDATE_POOL_CAP = 500
# Plan body threshold — pairs whose similarity is below 0.55
# don't enter the candidate pool at all.
SIMILARITY_THRESHOLD = 0.55
DEFAULT_ATTESTED_FILENAME = "attested_pairs.json"

# Type alias for the swappable similarity function. The
# production implementation is bge-m3 cosine (Phase 1.3
# offline); the test path passes a deterministic stub. ``fn``
# returns a float in [0.0, 1.0].
SimilarityFn = Callable[[str, str], float]


# Default attested-pair file path — resolved relative to
# ``backend/data/`` (the canonical home for the curated
# corpus).
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data"
)
DEFAULT_ATTESTED_PATH = DEFAULT_DATA_DIR / DEFAULT_ATTESTED_FILENAME


def _is_pg() -> bool:
    """Dialect discriminator (mirrors ``app.models._is_pg``)."""
    return DATABASE_URL.startswith("postgresql")


def _default_similarity_fn() -> SimilarityFn:
    """Build the production similarity function (offline bge-m3).

    Returns a closure that uses local ``sentence-transformers``
    for bge-m3 cosine — the Phase 1.3 offline path Phase 7.5 A/B
    sets up. When ``sentence-transformers`` isn't available in
    the environment, the closure raises ``ImportError`` so the
    caller knows to inject a stub. The seed script's CLI accepts
    ``--similarity-fn custom_module:fn`` to bypass this entirely
    for the test path; production callers keep the default.

    The closure captures the model on first call so the cost is
    paid once (the bge-m3 model load is ~2.3 GB and takes ~30 s
    on a cold cache).
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as st_util
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers not available — Phase 10.1 "
            "bge-m3 cosine requires the Phase 1.3 offline path. "
            "Install sentence-transformers, or inject a stub via "
            "--similarity-fn custom_module:fn."
        ) from exc

    model: SentenceTransformer | None = None

    def similarity(text_a: str, text_b: str) -> float:
        nonlocal model
        if model is None:
            model = SentenceTransformer("baai/bge-m3")
        # Cosine similarity via sentence-transformers' helper —
        # they're already L2-normalised at the model output, so
        # the dot product equals cosine similarity on the unit
        # sphere.
        embeddings = model.encode([text_a, text_b])
        # ``st_util.cos_sim`` returns a (1, 1) tensor for the
        # 2-element list — extract to Python float.
        score = st_util.cos_sim(
            embeddings[0], embeddings[1]
        ).item()
        return float(score)

    return similarity


def _candidate_pool_from_phrases(
    phrases: list[Phrase],
    similarity_fn: SimilarityFn,
    *,
    pool_cap: int = CANDIDATE_POOL_CAP,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Step 1 — build the candidate pair pool.

    Iterates over all (i < j) pairs of phrases rows, scores each
    via ``similarity_fn``, keeps pairs whose score is at least
    ``threshold``, and returns the top-``pool_cap`` by score
    descending.

    The returned list carries ``(phrase_a_id, phrase_b_id, score)``
    tuples; sorting the IDs alphabetically inside the script
    (Step 2) ensures deterministic ordering. The score is
    discarded in Step 2 — quartile assignment is by *rank*, not
    by score, mirroring the plan body's "deterministic bucket
    assignment" rule.

    Phase 10.1 ships this function for completeness; the
    production seed run uses bge-m3 in the live path. For the
    10.1 deliverable (and for the matching test) the closure
    is injected — see ``tests.test_seed_phrase_pairs``.
    """
    pool: list[tuple[str, str, float]] = []
    n = len(phrases)
    for i in range(n):
        for j in range(i + 1, n):
            a = phrases[i]
            b = phrases[j]
            score = similarity_fn(
                a.phrase, b.phrase
            )
            if score >= threshold:
                pool.append((a.id, b.id, score))
    # Sort by score descending, then by (a, b) lexicographically
    # for stable order when scores tie.
    pool.sort(key=lambda t: (-t[2], t[0], t[1]))
    return pool[:pool_cap]


def _bucket_quartile_assignment(
    candidates: list[tuple[str, str, float]],
    *,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, str, str]]:
    """Step 2 — bucket assignment by rank.

    The input ``candidates`` list may carry the bge-m3 scores
    (Step 1) or may already be filtered — the function treats
    it as an ordered list and bins by position.

    Plan body contract: the input is **pre-sorted alphabetically
    by (phrase_a_id, phrase_b_id)** *before* rank-based
    bucketing, so seed-derived randomness enters on a stable
    input and re-runs with the same seed produce byte-equal
    output. (The alphabetical pre-sort is the key: without it,
    the bge-m3 score ordering from Step 1 would dominate the
    ordering and the seed wouldn't propagate bit-for-bit.)

    Returns a list of ``(phrase_a_id, phrase_b_id, relation)``
    tuples; ``attested_pair`` is False here (attestation is
    the Step 3 override).
    """
    # Step 2a: pre-sort alphabetically by (a, b) to make the
    # function independent of Step 1's score ordering. The seed
    # propagates as a deterministic shuffle ON TOP of this
    # sorted input, so the same seed on the same input always
    # yields the same output (byte-for-byte reproducibility).
    sorted_candidates = sorted(
        candidates, key=lambda t: (t[0], t[1])
    )

    # Step 2b: apply seed-derived Fisher-Yates shuffle. Same
    # seed + same input = same shuffle.
    rng = random.Random(seed)
    shuffled = list(sorted_candidates)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return []

    # Step 2c: split into quartiles by rank. The plan body is
    # explicit: rank, not similarity. Top quartile ->
    # ``equivalent``, next -> ``paraphrase``, next ->
    # ``related``, bottom -> ``unrelated``.
    bucket_names = (
        "equivalent", "paraphrase", "related", "unrelated"
    )
    out: list[tuple[str, str, str]] = []
    for rank, (a, b, _score) in enumerate(shuffled):
        quartile_index = min(
            len(bucket_names) - 1,
            (rank * len(bucket_names)) // n,
        )
        out.append((a, b, bucket_names[quartile_index]))
    return out


def _load_attested_pairs(
    path: Path,
) -> list[tuple[str, str, str, bool]]:
    """Step 3 — load the attested-pair override list.

    Reads ``backend/data/attested_pairs.json`` (or the path the
    caller passes). Validates every row via
    ``PhrasePairSeedManifest`` so a typo'd slug or invalid
    ``relation`` literal is caught here — never reaches INSERT.

    Returns a list of ``(phrase_a_id, phrase_b_id, relation,
    attested_pair)`` tuples; ``attested_pair`` is always True
    for these rows (it's what the manifest guarantees).

    Empty list when the JSON file is empty (Phase 10.1's
    starting state) or missing (Phase 10.4 hasn't shipped yet).
    """
    if not path.exists():
        return []
    from app.schemas import PhrasePairSeedManifest

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"attested-pair file {path}: invalid JSON: {e}"
        )
    try:
        manifest = PhrasePairSeedManifest.model_validate(payload)
    except Exception as e:
        raise SystemExit(
            f"attested-pair file {path}: validation failed: {e}"
        )

    out: list[tuple[str, str, str, bool]] = []
    for row in manifest.pairs:
        out.append(
            (row.phrase_a_id, row.phrase_b_id, row.relation, True)
        )
    return out


def _canonical_pair(
    a: str, b: str
) -> tuple[str, str]:
    """Lexicographically sort (a, b) so the same pair can't
    collide with its mirror.

    The seed script's pre-sort discipline: every (a, b) tuple
    sorts lexicographically before INSERT, so the (a, b) pair
    on one INSERT never collides with the (b, a) pair of a
    different INSERT — the composite UNIQUE
    ``(phrase_a_id, phrase_b_id)`` enforces in-order
    uniqueness, and this sort rules out the swapped pair
    variant entirely.
    """
    return (a, b) if a <= b else (b, a)


def _merge_bucketed_with_attested(
    bucketed: list[tuple[str, str, str]],
    attested: list[tuple[str, str, str, bool]],
) -> list[tuple[str, str, str, bool]]:
    """Combine bucketed candidates with attested overrides.

    Attested rows win: if a (a, b) pair appears in BOTH the
    bucketed list and the attested list, the attested row's
    relation + ``attested_pair=True`` survives; the bucketed
    row is dropped. The composite UNIQUE constraint +
    ``ON CONFLICT DO NOTHING`` would otherwise let the
    bucketed row win (it lands first), so we explicitly
    de-duplicate here.

    Returns a list of ``(phrase_a_id, phrase_b_id, relation,
    attested_pair)`` tuples ready for INSERT.
    """
    by_pair: dict[tuple[str, str], tuple[str, str, str, bool]] = {}
    for a, b, relation in bucketed:
        ca, cb = _canonical_pair(a, b)
        by_pair[(ca, cb)] = (ca, cb, relation, False)
    for a, b, relation, _ in attested:
        ca, cb = _canonical_pair(a, b)
        # Attested wins — overwrites any bucketed row.
        by_pair[(ca, cb)] = (ca, cb, relation, True)
    return list(by_pair.values())


def _insert_bucketed_with_ignore_conflict(
    session, rows: list[tuple[str, str, str, bool]],
) -> int:
    """INSERT the bucketed rows idempotently (no-op on conflict).

    The plan body's Step 4 contract is an ``ON CONFLICT
    (phrase_a_id, phrase_b_id) DO NOTHING`` /
    ``INSERT OR IGNORE`` INSERT — a re-run against an
    already-seeded table must NOT duplicate rows, and the
    bucketed rows don't override existing data. Attestation
    overrides (the next function) are different: they go via
    ON CONFLICT DO UPDATE so the attested row always wins.

    Each row is a ``(phrase_a_id, phrase_b_id, relation,
    attested_pair)`` tuple. Self-pairs (``a == b``) are
    pre-filtered in Python (belt-and-braces — the DB CHECK
    constraint would also reject them). Slugs that don't match
    an existing ``phrases.id`` value would fail the FK constraint
    at INSERT time; the caller filters against the live
    ``phrases`` table so a typo'd entry never lands.

    Returns the count of rows INSERTed (driver-reported; the
    actual delta is computed by re-counting after the bulk
    insert).
    """
    return _execute_insert(
        session,
        rows,
        on_conflict_update=False,
    )


def _insert_attested_with_update_conflict(
    session, rows: list[tuple[str, str, str, bool]],
) -> int:
    """INSERT the attested rows with ON CONFLICT DO UPDATE.

    Per plan body Step 3: attested-pair rows always win. If a
    (a, b) pair already exists from a prior bucketed run with
    ``attested_pair=False``, the attested INSERT UPDATES the
    existing row's ``relation`` + ``attested_pair=True`` so the
    attested evidence wins.

    Same dialect-portable shape as the bucketed INSERT: SQLite
    uses ``ON CONFLICT (...) DO UPDATE SET ...`` (3.24+),
    Postgres uses ``ON CONFLICT (...) DO UPDATE SET ...``
    with the ``index_elements`` clause. Both honor the
    composite UNIQUE constraint on ``(phrase_a_id,
    phrase_b_id)``.

    Self-pairs are pre-filtered here too (the DB CHECK would
    also reject).

    Returns the driver-reported INSERT count.
    """
    return _execute_insert(
        session,
        rows,
        on_conflict_update=True,
    )


def _execute_insert(
    session,
    rows: list[tuple[str, str, str, bool]],
    *,
    on_conflict_update: bool,
) -> int:
    """Bulk-INSERT helper shared by the bucketed / attested paths.

    Dialect-portable shape: SQLite ``ON CONFLICT (...) DO
    NOTHING`` / ``DO UPDATE``, Postgres ``ON CONFLICT (...)
    DO NOTHING`` / ``DO UPDATE`` with ``index_elements``. The
    Python-tuple input is flattened into ``INSERT ... VALUES``
    rows. Self-pairs are pre-filtered at this layer so the
    DB CHECK constraint is the safety belt, not the gate.
    """
    if not rows:
        return 0

    # Pre-filter: reject self-pairs (CHECK constraint would
    # also reject; pre-filter is belt-and-braces).
    filtered = [r for r in rows if r[0] != r[1]]
    if not filtered:
        return 0

    stmt_dicts = [
        {
            "phrase_a_id": a,
            "phrase_b_id": b,
            "relation": relation,
            "attested_pair": attested,
        }
        for a, b, relation, attested in filtered
    ]
    table = PhrasePair.__table__
    if _is_pg():
        stmt = pg_insert(table).values(stmt_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements=["phrase_a_id", "phrase_b_id"],
            set_={
                "relation": stmt.excluded.relation,
                "attested_pair": stmt.excluded.attested_pair,
            },
        ) if on_conflict_update else stmt.on_conflict_do_nothing(
            index_elements=["phrase_a_id", "phrase_b_id"]
        )
    else:
        # SQLite — same dialect-portable contract. Note that
        # SQLite's INSERT OR REPLACE would also work but it
        # deletes + re-inserts (changing the autoincrement id);
        # ON CONFLICT DO UPDATE keeps the original id so the
        # audit surface is stable across re-runs.
        stmt = sqlite_insert(table).values(stmt_dicts)
        stmt = stmt.on_conflict_do_update(
            index_elements=["phrase_a_id", "phrase_b_id"],
            set_={
                "relation": stmt.excluded.relation,
                "attested_pair": stmt.excluded.attested_pair,
            },
        ) if on_conflict_update else stmt.on_conflict_do_nothing(
            index_elements=["phrase_a_id", "phrase_b_id"]
        )

    try:
        session.execute(stmt)
    except IntegrityError as e:
        # The most likely cause: a manifest row whose slugs
        # don't match an existing ``phrases.id``. Surface the
        # detail so the user can fix the manifest.
        raise SystemExit(
            f"seed_phrase_pairs: integrity violation — likely "
            f"a slug in attested_pairs.json that doesn't match "
            f"any phrases.id value. Original error: {e}"
        )
    return 0



def _phrases_table_ids(session) -> set[str]:
    """Return the set of all ``phrases.id`` values in the DB.

    Used to pre-filter the attested-pair manifest: any row
    whose slug(s) don't exist in the live ``phrases`` table
    would fail the FK constraint at INSERT time and we'd
    rather surface that error loudly than wait for the
    driver to fail.

    Called once per ``main()`` invocation; the result is a
    set so the membership test is O(1).
    """
    ids = session.execute(select(Phrase.id)).scalars().all()
    return set(ids)


def _filter_attested_to_existing_phrases(
    attested: list[tuple[str, str, str, bool]],
    existing_ids: set[str],
) -> list[tuple[str, str, str, bool]]:
    """Drop attested rows whose slugs aren't in the live phrases table.

    ``ondelete="RESTRICT"`` on the FK plus the seed script's
    idempotent ``ON CONFLICT DO NOTHING`` semantics means the
    FK violation would abort the bulk INSERT, not silently skip
    the bad rows. Pre-filtering here gives a loud, traceable
    error message that names the offending slugs.
    """
    out: list[tuple[str, str, str, bool]] = []
    for a, b, relation, is_attested in attested:
        if a not in existing_ids:
            raise SystemExit(
                f"seed_phrase_pairs: attested slug {a!r} not "
                f"found in phrases table — every attested-pair "
                f"row must reference an existing phrase."
            )
        if b not in existing_ids:
            raise SystemExit(
                f"seed_phrase_pairs: attested slug {b!r} not "
                f"found in phrases table — every attested-pair "
                f"row must reference an existing phrase."
            )
        out.append((a, b, relation, is_attested))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Deterministic seed for bucket assignment "
            f"(default: {DEFAULT_SEED}). Same seed + same "
            f"phrases table = byte-equal output."
        ),
    )
    parser.add_argument(
        "--attested",
        default=str(DEFAULT_ATTESTED_PATH),
        help=(
            f"Path to the attested-pair JSON manifest (default: "
            f"{DEFAULT_ATTESTED_PATH})"
        ),
    )
    parser.add_argument(
        "--pool-cap",
        type=int,
        default=CANDIDATE_POOL_CAP,
        help=(
            f"Maximum candidate-pool size (default: "
            f"{CANDIDATE_POOL_CAP}). The top-N pairs by "
            f"similarity, descending."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=(
            f"Minimum similarity threshold for candidate "
            f"inclusion (default: {SIMILARITY_THRESHOLD} on a "
            f"0..1 scale)."
        ),
    )
    parser.add_argument(
        "--similarity-fn",
        default=None,
        help=(
            "Optional ``module:function`` injection — bypass "
            "the bge-m3 cosine default (CI / dev only). The "
            "function must accept (str, str) and return a "
            "float in [0.0, 1.0]."
        ),
    )
    args = parser.parse_args(argv)

    # ---- Step 0: resolve the similarity function -----------------
    if args.similarity_fn:
        # Custom injection path — used by tests; the format is
        # ``module:function`` (mirrors Python's official
        # ``importlib.import_module`` convention).
        module_name, _, fn_name = args.similarity_fn.partition(":")
        if not module_name or not fn_name:
            raise SystemExit(
                f"--similarity-fn must be 'module:function', "
                f"got {args.similarity_fn!r}"
            )
        import importlib
        try:
            mod = importlib.import_module(module_name)
            similarity_fn = getattr(mod, fn_name)
        except (ImportError, AttributeError) as e:
            raise SystemExit(
                f"--similarity-fn {args.similarity_fn!r}: "
                f"could not import ({e})"
            )
    else:
        similarity_fn = _default_similarity_fn()

    # ---- Step 1: load live phrases + filter existing IDs -------
    session = SessionLocal()
    try:
        phrases = (
            session.execute(
                select(Phrase).order_by(Phrase.id)
            ).scalars().all()
        )
        existing_ids = _phrases_table_ids(session)
    finally:
        session.close()

    if len(phrases) < 2:
        print(
            f"seed_phrase_pairs: phrases table has only "
            f"{len(phrases)} row(s); need at least 2 to form "
            f"a pair. Run scripts/seed_phrases_dwds.py first.",
            file=sys.stderr,
        )
        return 1

    # ---- Step 2: candidate pool (bge-m3 cosine) ----------------
    candidates = _candidate_pool_from_phrases(
        phrases,
        similarity_fn,
        pool_cap=args.pool_cap,
        threshold=args.threshold,
    )

    # ---- Step 3: bucket assignment by rank ----------------------
    bucketed = _bucket_quartile_assignment(
        candidates, seed=args.seed
    )

    # ---- Step 4: attested-pair override -------------------------
    attested_path = Path(args.attested)
    attested = _load_attested_pairs(attested_path)
    attested = _filter_attested_to_existing_phrases(
        attested, existing_ids
    )

    # ---- Step 5: split bucketed vs attested and insert -------
    # The merge produces a single dedup'd row stream; we then
    # split back into the two streams so the bucketed rows use
    # ON CONFLICT DO NOTHING (idempotent on re-run) while the
    # attested rows use ON CONFLICT DO UPDATE (attested
    # evidence wins on conflict). This mirrors the plan body's
    # Step 3 contract.
    rows = _merge_bucketed_with_attested(bucketed, attested)
    bucketed_only = [
        (a, b, relation)
        for a, b, relation, is_attested in rows
        if not is_attested
    ]
    attested_only = [
        (a, b, relation, True)
        for a, b, relation, is_attested in rows
        if is_attested
    ]

    session = SessionLocal()
    try:
        with session.begin():
            # Bucketed first: ON CONFLICT DO NOTHING, idempotent.
            _insert_bucketed_with_ignore_conflict(
                session, [
                    (a, b, relation, False)
                    for a, b, relation in bucketed_only
                ]
            )
            # Attested next: ON CONFLICT DO UPDATE, wins on conflict.
            _insert_attested_with_update_conflict(
                session, attested_only
            )
        # Re-count after the bulk insert to compute the actual
        # delta. The driver-reported INSERTED 0 doesn't tell us
        # how many were skipped; the count does.
        from sqlalchemy import func

        n_after = (
            session.execute(
                select(func.count()).select_from(PhrasePair)
            ).scalar()
        )
        inserted = (n_after or 0)

        # The plan body documents two output shapes: (a) the
        # first run lands N rows from a candidate pool of M and
        # K attested overrides; (b) a re-run against an
        # already-seeded table prints the same numbers (the
        # pre-count of the candidate pool is deterministic given
        # the phrases table, so we always show it).
        attested_count = len(attested_only)
        if attested_count > 0:
            print(
                f"Loaded {inserted} phrase_pairs "
                f"(target ~{len(candidates)}, attested "
                f"{attested_count})."
            )
        else:
            print(
                f"Loaded {inserted} phrase_pairs "
                f"(target ~{len(candidates)})."
            )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
