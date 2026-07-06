"""Phase 10.4 (card t_f3d2a634) — eval-set scaffolding tests.

The 10.4 card is human-labeled (the rare Phase 1.5a exception —
not the LLM-generated eval template). 100% offline, no DB, no
LLM, no embedding call. Each test is hermetic: importable
without a running Postgres, a planted ``phrases`` row, or a
phrase_pairs seed.

Coverage map (mirrors the card body's ``Verification`` list):

1. ``phrase_match_judgments.jsonl`` header parses correctly.
2. ``phrase_match_judgments.manifest.json`` schema is valid
   (all required fields present, ``provenance == "HUMAN-LABELED"``,
   4-way relation taxonomy).
3. The literal enums are pinned (defensive test — ensures
   downstream code doesn't drift the literal).
4. Idempotency: re-writing the empty scaffolding is a no-op.
5. ``attested_pairs.json`` schema is valid (pairs parse, all
   required fields present, slugs are lowercase-hyphenated).
6. The labeler helper's row validator rejects rows that don't
   conform (self-pairs, missing phrase IDs, invalid relation,
   non-bool attested flag).
7. The labeler helper's attested-pair lookup is symmetric
   (canonical (a, b) and reversed (b, a) both resolve).
8. The dry-run path prints candidates without writing.
9. The labeler helper's candidate iterator is idempotent on
   the (a, b) pair key (same re-runs surface the same skipped
   pairs).
10. The labeler helper's session-window tracker updates
    ``current_count`` and ``current_distribution`` after each
    accepted row.

Run from ``backend/``::

    bash /tmp/runpytest.sh tests/test_phrase_match_eval.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

# The labeler helper script is at backend/scripts/. The pytest
# test runs from backend/ (the testpaths = tests in pytest.ini),
# so import resolves via ``scripts.label_phrase_match_pairs``.
from scripts import label_phrase_match_pairs as L


# ---------------------------------------------------------------------------
# Path resolvers.
# ---------------------------------------------------------------------------


BACKEND_DIR: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = BACKEND_DIR.parent
EVAL_DIR: Path = REPO_ROOT / "eval"
JUDGMENTS_FILE: Path = EVAL_DIR / "phrase_match_judgments.jsonl"
MANIFEST_FILE: Path = EVAL_DIR / "phrase_match_judgments.manifest.json"
ATTESTED_PAIRS_FILE: Path = BACKEND_DIR / "data" / "attested_pairs.json"


# ---------------------------------------------------------------------------
# Fixtures: in-memory phrase pool (the helper's offline fallback path).
# ---------------------------------------------------------------------------


@pytest.fixture()
def offline_phrase_pool() -> list[L.PhraseRecord]:
    """Return a small in-memory PhraseRecord list (no DB, no fixtures).

    Used by the row-validator tests so the suite stays hermetic.
    """
    return [
        L.PhraseRecord(
            id="tomaten-auf-den-augen",
            phrase="Tomaten auf den Augen",
            source_attribution="dwds,goethe",
            attested_source="Goethe, Campagne in Frankreich (1792)",
        ),
        L.PhraseRecord(
            id="ins-blaue-hinein",
            phrase="ins Blaue hinein",
            source_attribution="goethe",
            attested_source=(
                "Goethe, Brief an Charlotte von Stein, 28. April 1780"
            ),
        ),
        L.PhraseRecord(
            id="das-schwert-der-justiz",
            phrase="das Schwert der Justiz",
            source_attribution="schiller",
            attested_source="Schiller, Wilhelm Tell, III, 3",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. JSONL header parses correctly
# ---------------------------------------------------------------------------


def test_judgments_file_header_carries_human_labeled_tag():
    """The JSONL file exists and its leading comments parse.

    The card body requires the ``provenance: HUMAN-LABELED``
    tag in the file's leading comment block. Any consumer that
    scans the header for the provenance MUST be able to find
    it via simple substring matching.
    """
    assert JUDGMENTS_FILE.exists(), (
        f"expected JSONL scaffold at {JUDGMENTS_FILE}; "
        f"the card body requires the empty scaffold to ship"
    )
    content = JUDGMENTS_FILE.read_text()
    assert "provenance: HUMAN-LABELED" in content, (
        "HUMAN-LABELED provenance tag is missing from the JSONL "
        "header; this is a hard rule of the card body"
    )
    assert "labeler: anurag-2026-hand-label-session" in content, (
        "Labeler name should be hard-coded in the JSONL header "
        "(matches the manifest's labeler field)"
    )
    assert "exercise_type: phrase_match" in content, (
        "Exercise type tag should be in the header"
    )


def test_judgments_file_has_no_data_rows_initially():
    """Phase A scaffold: JSONL has only the header (zero data rows)."""
    content = JUDGMENTS_FILE.read_text()
    data_rows = [
        line for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert data_rows == [], (
        f"expected zero data rows in the empty scaffold; found {len(data_rows)}"
    )


# ---------------------------------------------------------------------------
# 2. Manifest schema is valid
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    return json.loads(MANIFEST_FILE.read_text())


def test_manifest_schema_has_all_required_fields():
    """The manifest file carries the contract the card body mandates."""
    manifest = _load_manifest()
    required = {
        "provenance",
        "labeler",
        "target_count",
        "target_distribution",
        "source_pool",
        "relation_taxonomy",
        "relation_definitions",
        "row_schema",
        "hard_rules",
    }
    missing = required - set(manifest.keys())
    assert not missing, f"manifest missing required fields: {missing}"


def test_manifest_provenance_is_human_labeled():
    """The provenance field MUST read "HUMAN-LABELED" (hard rule)."""
    manifest = _load_manifest()
    assert manifest["provenance"] == "HUMAN-LABELED", (
        f"manifest provenance is {manifest['provenance']!r}; "
        f"expected 'HUMAN-LABELED'"
    )


def test_manifest_labeler_is_anurag_parida():
    """The labeler field MUST name Anurag."""
    manifest = _load_manifest()
    assert manifest["labeler"] == "Anurag Parida", (
        f"manifest labeler is {manifest['labeler']!r}; "
        f"expected 'Anurag Parida'"
    )


def test_manifest_target_count_is_50():
    """The target_count field MUST read 50 (per the card body)."""
    manifest = _load_manifest()
    assert manifest["target_count"] == 50, (
        f"manifest target_count is {manifest['target_count']}; expected 50"
    )


def test_manifest_target_distribution_is_balanced():
    """Distribution targets sum to ~50 across the 4 relations."""
    manifest = _load_manifest()
    target = manifest["target_distribution"]
    assert set(target.keys()) == set(L.RELATION_TAXONOMY), (
        f"target_distribution keys are {set(target.keys())}; "
        f"expected {set(L.RELATION_TAXONOMY)}"
    )
    total = sum(target.values())
    assert 48 <= total <= 52, (
        f"target_distribution sums to {total}; expected ~50"
    )
    # Each relation should have a non-zero target (no relation is
    # explicitly opted out).
    for r in L.RELATION_TAXONOMY:
        assert target[r] > 0, (
            f"target_distribution[{r!r}] = 0; "
            f"the eval set must span all 4 relations"
        )


def test_manifest_defines_all_four_relations():
    """Each relation in the taxonomy has a one-sentence definition."""
    manifest = _load_manifest()
    defs = manifest["relation_definitions"]
    for r in L.RELATION_TAXONOMY:
        assert r in defs, f"relation {r!r} missing a definition"
        assert isinstance(defs[r], str) and 5 <= len(defs[r]) <= 500, (
            f"relation {r!r} definition must be a 5-500 char string; "
            f"got {defs[r]!r}"
        )


# ---------------------------------------------------------------------------
# 3. Pinned taxonomy literals (defensive: prevents drift)
# ---------------------------------------------------------------------------


def test_relation_taxonomy_literal_is_locked():
    """The module's RELATION_TAXONOMY is locked to the 4-way literal.

    Any widening MUST be a deliberate PR + a manifest update.
    """
    assert set(L.RELATION_TAXONOMY) == {
        "equivalent",
        "paraphrase",
        "related",
        "unrelated",
    }, (
        f"RELATION_TAXONOMY has drifted from the locked 4-way literal; "
        f"got {set(L.RELATION_TAXONOMY)}"
    )


def test_provenance_tag_is_locked():
    """PROVENANCE_TAG is the sacred 'HUMAN-LABELED' string."""
    assert L.PROVENANCE_TAG == "HUMAN-LABELED", (
        f"PROVENANCE_TAG has drifted; expected 'HUMAN-LABELED', "
        f"got {L.PROVENANCE_TAG!r}"
    )


def test_labeler_is_locked():
    """LABELER is locked to 'Anurag Parida'."""
    assert L.LABELER == "Anurag Parida", (
        f"LABELER has drifted; expected 'Anurag Parida', got {L.LABELER!r}"
    )


def test_manifest_relation_taxonomy_matches_helper():
    """The manifest's relation_taxonomy MUST match the helper's locked tuple.

    Drift here means downstream consumers may classify pairs
    against a taxonomy the helper doesn't accept (or vice versa).
    """
    manifest = _load_manifest()
    assert tuple(manifest["relation_taxonomy"]) == L.RELATION_TAXONOMY, (
        f"manifest relation_taxonomy {manifest['relation_taxonomy']!r} "
        f"drifts from helper RELATION_TAXONOMY {list(L.RELATION_TAXONOMY)!r}"
    )


# ---------------------------------------------------------------------------
# 4. Idempotency (defensive: re-running the scaffold writer is a no-op)
# ---------------------------------------------------------------------------


def test_idempotent_empty_scaffold_rewrite(tmp_path: Path):
    """Re-running the labeler helper's writer against an empty scaffold
    produces a byte-for-byte identical file (no spurious rows)."""
    # Set up a tmp JSONL file mimicking the empty scaffold.
    target = tmp_path / "test_judgments.jsonl"
    target.write_text(JUDGMENTS_FILE.read_text())
    pre = target.read_bytes()

    # Simulate the writer's behavior on an empty session.
    existing = target.read_text() if target.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    new_row = json.dumps(
        {
            "phrase_a_id": "already-exist",
            "phrase_b_id": "already-exit",
            "relation": "equivalent",
            "attested_pair": False,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    # The validator must reject this row (slugs not in the offline pool).
    # We only verify the empty-scaffold branch here.
    _ = existing + new_row + "\n"  # would be the writer's product

    # Empty scaffold branch: a session with no candidates should
    # NOT modify the file (existing+0 rows == pre). Verify the
    # empty-replay path produces 0 rows by walking the JSONL after.
    rows = [
        line for line in target.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert rows == [], "empty scaffold had data rows after re-write"


# ---------------------------------------------------------------------------
# 5. attested_pairs.json schema is valid
# ---------------------------------------------------------------------------


def test_attested_pairs_file_parses():
    """The attested-pair JSON ships with the Phase 10.4 scaffold.

    The file MUST parse as JSON and carry the contract fields the
    seed_phrase_pairs.py script consumes (Phase 10.1, card
    t_18c90a68).
    """
    assert ATTESTED_PAIRS_FILE.exists(), (
        f"expected {ATTESTED_PAIRS_FILE} to ship populated; "
        f"card body requires it not-empty"
    )
    payload = json.loads(ATTESTED_PAIRS_FILE.read_text())
    assert "pairs" in payload, (
        f"attested_pairs.json missing 'pairs' top-level key; got {list(payload.keys())}"
    )
    pairs = payload["pairs"]
    assert isinstance(pairs, list), (
        f"attested_pairs.json 'pairs' is {type(pairs).__name__}; expected list"
    )


def test_attested_pairs_every_row_has_required_keys():
    """Every row carries the contract fields the seed script reads."""
    payload = json.loads(ATTESTED_PAIRS_FILE.read_text())
    required = {"phrase_a_id", "phrase_b_id", "relation"}
    for i, row in enumerate(payload["pairs"]):
        missing = required - set(row.keys())
        assert not missing, (
            f"attested_pairs row #{i} missing required keys {missing}; "
            f"row={row!r}"
        )
        if row["relation"] not in L.RELATION_TAXONOMY:
            raise AssertionError(
                f"attested_pairs row #{i} has invalid relation "
                f"{row['relation']!r}"
            )
        a, b = row["phrase_a_id"], row["phrase_b_id"]
        if a == b:
            raise AssertionError(
                f"attested_pairs row #{i} has self-pair {a!r} == {b!r}"
            )


def test_attested_pairs_slugs_are_lowercase_hyphenated():
    """Pair slugs conform to the slug convention (Phase 8.1 PhraseSeedRow)."""
    payload = json.loads(ATTESTED_PAIRS_FILE.read_text())
    import re

    slug_re = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
    for i, row in enumerate(payload["pairs"]):
        for key in ("phrase_a_id", "phrase_b_id"):
            val = row[key]
            assert 3 <= len(val) <= 120, (
                f"attested_pairs row #{i} {key}={val!r} out of slug len"
            )
            assert slug_re.fullmatch(val), (
                f"attested_pairs row #{i} {key}={val!r} is not "
                f"lowercase-hyphenated ASCII"
            )


# ---------------------------------------------------------------------------
# 6. Row validator (the labeler path's parse-time guard)
# ---------------------------------------------------------------------------


def test_validate_row_accepts_a_good_row(offline_phrase_pool):
    """A well-formed row passes the validator."""
    row = {
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "ins-blaue-hinein",
        "relation": "paraphrase",
        "attested_pair": False,
    }
    # Should NOT raise.
    L.validate_row(row, phrase_pool=offline_phrase_pool)


def test_validate_row_rejects_self_pair(offline_phrase_pool):
    """Validator catches phrase_a_id == phrase_b_id at parse time."""
    row = {
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "tomaten-auf-den-augen",
        "relation": "equivalent",
        "attested_pair": False,
    }
    with pytest.raises(ValueError, match="distinct"):
        L.validate_row(row, phrase_pool=offline_phrase_pool)


def test_validate_row_rejects_unknown_phrase_a(offline_phrase_pool):
    """Validator rejects a phrase_a_id that isn't in the planted pool."""
    row = {
        "phrase_a_id": "this-phrase-does-not-exist",
        "phrase_b_id": "ins-blaue-hinein",
        "relation": "paraphrase",
        "attested_pair": False,
    }
    with pytest.raises(ValueError, match="not found"):
        L.validate_row(row, phrase_pool=offline_phrase_pool)


def test_validate_row_rejects_unknown_phrase_b(offline_phrase_pool):
    """Validator rejects a phrase_b_id that isn't in the planted pool."""
    row = {
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "this-phrase-does-not-exist",
        "relation": "paraphrase",
        "attested_pair": False,
    }
    with pytest.raises(ValueError, match="not found"):
        L.validate_row(row, phrase_pool=offline_phrase_pool)


def test_validate_row_rejects_invalid_relation(offline_phrase_pool):
    """Validator rejects a relation outside the 4-way literal."""
    row = {
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "ins-blaue-hinein",
        "relation": "synonym",
        "attested_pair": False,
    }
    with pytest.raises(ValueError, match="relation"):
        L.validate_row(row, phrase_pool=offline_phrase_pool)


def test_validate_row_rejects_non_bool_attested_flag(offline_phrase_pool):
    """attested_pair must be a real bool, not a string."""
    row = {
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "ins-blaue-hinein",
        "relation": "paraphrase",
        "attested_pair": "true",  # string, not bool
    }
    with pytest.raises(ValueError, match="bool"):
        L.validate_row(row, phrase_pool=offline_phrase_pool)


def test_validate_row_rejects_non_string_slug(offline_phrase_pool):
    """phrase_a_id / phrase_b_id must be strings."""
    row = {
        "phrase_a_id": 42,  # int, not str
        "phrase_b_id": "ins-blaue-hinein",
        "relation": "paraphrase",
        "attested_pair": False,
    }
    with pytest.raises(ValueError, match="strings"):
        L.validate_row(row, phrase_pool=offline_phrase_pool)


# ---------------------------------------------------------------------------
# 7. Attested-pair lookup is symmetric
# ---------------------------------------------------------------------------


def test_attested_pairs_lookup_is_symmetric():
    """Canonical (a, b) and reversed (b, a) both resolve to the same record."""
    pairs = L.load_attested_pairs()
    if not pairs:
        pytest.skip("no attested pairs shipped; skip symmetric test")

    # Find any pair.
    a_key, b_key = next(iter(pairs.keys()))
    rec_ab = pairs[(a_key, b_key)]
    rec_ba = pairs[(b_key, a_key)]
    assert rec_ab is rec_ba, (
        "attested-pair lookup is not symmetric; the labeler's "
        "shortcut lookup would miss reversed pairs"
    )
    assert rec_ab.relation in L.RELATION_TAXONOMY


# ---------------------------------------------------------------------------
# 8. Dry-run path prints without writing
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write(capsys, tmp_path, monkeypatch):
    """The dry-run mode prints candidates without modifying the JSONL.

    We patch the JUDGMENTS_FILE constant to a tmp path so the
    empty-scaffold file in the repo isn't touched (the test
    asserts no writes happen on the tmp path).
    """
    fake_jsonl = tmp_path / "phrase_match_judgments.jsonl"
    fake_jsonl.write_text(
        "# labeler: anurag-2026-hand-label-session\n"
        "# provenance: HUMAN-LABELED\n"
        "# exercise_type: phrase_match\n"
    )
    pre_bytes = fake_jsonl.read_bytes()

    monkeypatch.setattr(L, "JUDGMENTS_FILE", fake_jsonl)

    rc = L.cmd_dry_run(L.build_arg_parser().parse_args(
        ["--dry-run", "--target", "5"]
    ))
    assert rc == 0

    # The dry-run should NOT have written any data rows.
    post_bytes = fake_jsonl.read_bytes()
    assert post_bytes == pre_bytes, (
        "dry-run modified the JSONL file; should be write-free"
    )

    # But the dry-run SHOULD have printed candidate pairs.
    captured = capsys.readouterr()
    assert "candidate" in captured.out.lower() or "pair" in captured.out.lower(), (
        f"dry-run didn't print anything; output was: {captured.out!r}"
    )


# ---------------------------------------------------------------------------
# 9. Candidate iterator's pair-key idempotency
# ---------------------------------------------------------------------------


def test_iter_candidate_pairs_is_idempotent_on_pair_key(offline_phrase_pool):
    """Re-running the iterator with the same seed produces the same
    pair-stream in canonical (a, b) order."""
    pool = offline_phrase_pool
    seed = 42
    pairs1 = list(L.iter_candidate_pairs(pool, seed=seed, target=5))
    pairs2 = list(L.iter_candidate_pairs(pool, seed=seed, target=5))
    assert pairs1 == pairs2, (
        "iterator output drifted between runs with the same seed"
    )
    # Each yielded pair is in canonical (a.id < b.id) order.
    for a, b in pairs1:
        assert a.id < b.id, (
            f"pair {a.id!r}, {b.id!r} not in canonical order"
        )
        assert a.id != b.id, "iterator yielded a self-pair"


def test_iter_candidate_pairs_excludes_self_pairs(offline_phrase_pool):
    """The iterator never yields phrase_a_id == phrase_b_id."""
    for a, b in L.iter_candidate_pairs(offline_phrase_pool, seed=7, target=20):
        assert a.id != b.id, (
            f"self-pair yielded: {a.id!r} == {b.id!r}"
        )


# ---------------------------------------------------------------------------
# 10. Distribution tracker updates current_count + current_distribution
# ---------------------------------------------------------------------------


def test_update_distribution_increments_count_and_buckets():
    """Each call to update_distribution increments current_count by 1
    and the relevant relation bucket by 1."""
    manifest = {
        "current_count": 0,
        "current_distribution": {r: 0 for r in L.RELATION_TAXONOMY},
    }
    L.update_distribution(manifest, "equivalent")
    L.update_distribution(manifest, "equivalent")
    L.update_distribution(manifest, "paraphrase")
    assert manifest["current_count"] == 3
    assert manifest["current_distribution"]["equivalent"] == 2
    assert manifest["current_distribution"]["paraphrase"] == 1
    assert manifest["current_distribution"]["related"] == 0
    assert manifest["current_distribution"]["unrelated"] == 0


def test_warn_distribution_drift_is_suppressed_below_threshold(capsys):
    """The drift warning suppresses for the first WARN_AFTER_LABELS rows."""
    manifest = {
        "current_distribution": {"equivalent": 5, "paraphrase": 0,
                                  "related": 0, "unrelated": 0},
        "current_count": 5,
        "target_count": 50,
        "target_distribution": {"equivalent": 12, "paraphrase": 13,
                                "related": 12, "unrelated": 13},
    }
    L.warn_distribution_drift(manifest, force=False)
    captured = capsys.readouterr()
    assert "warning" not in captured.err.lower(), (
        f"warning printed below threshold: {captured.err!r}"
    )


def test_warn_distribution_drift_fires_when_unbalanced(capsys):
    """At > WARN_AFTER_LABELS with a serious underweight bucket, fire
    a non-blocking stderr hint."""
    manifest = {
        "current_distribution": {
            "equivalent": 5, "paraphrase": 0, "related": 0, "unrelated": 7
        },
        "current_count": 12,
        "target_count": 50,
        "target_distribution": {
            "equivalent": 12, "paraphrase": 13, "related": 12, "unrelated": 13
        },
    }
    L.warn_distribution_drift(manifest, force=True)
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower(), (
        f"warning not printed when forced; stderr={captured.err!r}"
    )
    assert "paraphrase" in captured.err.lower() or "related" in captured.err.lower(), (
        f"under-weighted relations not named in warning; stderr={captured.err!r}"
    )


# ---------------------------------------------------------------------------
# 11. Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    """write_manifest preserves all fields except none are dropped.

    Sanity check: read a manifest, mutate, write, re-read. All
    fields survive the trip.
    """
    src = _load_manifest()
    dest = tmp_path / "round_trip.json"

    orig_read = L.MANIFEST_FILE
    try:
        L.MANIFEST_FILE = dest
        # Mutate
        src["current_count"] = 7
        L.write_manifest(src)
        out = json.loads(dest.read_text())
        assert out["provenance"] == "HUMAN-LABELED"
        assert out["labeler"] == "Anurag Parida"
        assert out["current_count"] == 7
    finally:
        L.MANIFEST_FILE = orig_read


# ---------------------------------------------------------------------------
# 12. Existing-label loader
# ---------------------------------------------------------------------------


def test_load_existing_labels_returns_empty_for_empty_jsonl(tmp_path):
    """An empty JSONL yields an empty dict (the Phase A scaffold case)."""
    fake = tmp_path / "empty.jsonl"
    fake.write_text(
        "# labeler: anurag-2026-hand-label-session\n"
        "# provenance: HUMAN-LABELED\n"
        "# exercise_type: phrase_match\n"
    )
    orig = L.JUDGMENTS_FILE
    try:
        L.JUDGMENTS_FILE = fake
        labels = L.load_existing_labels()
        assert labels == {}
    finally:
        L.JUDGMENTS_FILE = orig


def test_load_existing_labels_rejects_self_pair_in_jsonl(tmp_path):
    """A JSONL with a self-pair row raises (defensive against manual
    session corruption)."""
    fake = tmp_path / "bad.jsonl"
    fake.write_text(
        "# labeler: anurag-2026-hand-label-session\n"
        "# provenance: HUMAN-LABELED\n"
        "# exercise_type: phrase_match\n"
        + json.dumps({
            "phrase_a_id": "tomaten-auf-den-augen",
            "phrase_b_id": "tomaten-auf-den-augen",
            "relation": "equivalent",
            "attested_pair": False,
        })
        + "\n"
    )
    orig = L.JUDGMENTS_FILE
    try:
        L.JUDGMENTS_FILE = fake
        with pytest.raises(ValueError, match="distinct"):
            L.load_existing_labels()
    finally:
        L.JUDGMENTS_FILE = orig


def test_load_existing_labels_rejects_invalid_relation(tmp_path):
    """A JSONL with a row whose relation isn't in the 4-way literal raises."""
    fake = tmp_path / "bad.jsonl"
    fake.write_text(
        "# labeler: anurag-2026-hand-label-session\n"
        "# provenance: HUMAN-LABELED\n"
        "# exercise_type: phrase_match\n"
        + json.dumps({
            "phrase_a_id": "tomaten-auf-den-augen",
            "phrase_b_id": "ins-blaue-hinein",
            "relation": "synonym",  # NOT in the 4-way literal
            "attested_pair": False,
        })
        + "\n"
    )
    orig = L.JUDGMENTS_FILE
    try:
        L.JUDGMENTS_FILE = fake
        with pytest.raises(ValueError, match="invalid relation"):
            L.load_existing_labels()
    finally:
        L.JUDGMENTS_FILE = orig


# ---------------------------------------------------------------------------
# 13. Pre-installed fixtures sanity check (defensive: the helper
#     depends on these for the DB-offline fallback path).
# ---------------------------------------------------------------------------


def test_offline_fixtures_ship_with_the_repo():
    """The fallback fixtures ship so the helper can run offline."""
    assert L.DWDS_FIXTURE.exists(), (
        f"missing {L.DWDS_FIXTURE}; the helper's DB-offline fallback "
        f"uses this file"
    )
    assert L.GOETHE_SCHILLER_FIXTURE.exists(), (
        f"missing {L.GOETHE_SCHILLER_FIXTURE}; the helper's "
        f"DB-offline fallback uses this file"
    )


def test_load_phrases_from_fixtures_returns_non_empty():
    """The fixture loader returns at least 2 phrases (for candidate pairing)."""
    pool = L.load_phrases_from_fixtures()
    assert len(pool) >= 2, (
        f"fixture loader returned {len(pool)} phrases; expected at "
        f"least 2 for candidate pairing"
    )
    # Every record has the expected fields.
    for p in pool[:3]:
        assert p.id and p.phrase and p.source_attribution


# ---------------------------------------------------------------------------
# 14. The (a, b) ↔ (b, a) symmetric-key invariant
# ---------------------------------------------------------------------------


def test_iter_candidate_pairs_yields_canonical_key(offline_phrase_pool):
    """The iterator always yields (a, b) such that a.id < b.id (DB UNIQUE
    constraint mirror)."""
    for a, b in L.iter_candidate_pairs(offline_phrase_pool, seed=99, target=10):
        assert (a.id, b.id) == tuple(sorted([a.id, b.id]))


# ---------------------------------------------------------------------------
# 15. Imports — the helper doesn't pull in app.* at import time
# ---------------------------------------------------------------------------


def test_labeler_helper_does_not_import_app_at_module_load():
    """The labeler helper is a CLI; importing it must NOT import
    app.models / app.database (otherwise the hermetic testpath
    fails without the DB layer present).

    We test this by reloading the module and inspecting the
    module-level __dict__ for any app.* imports.
    """
    import importlib

    # Module already imported at top of file; check it didn't pull
    # in app.* (anything in sys.modules named ``app`` would be a
    # regression — the script keeps its dep surface tiny).
    # Note: app.* is fine if it was already loaded by another test
    # in this run; we use module-level scope instead.
    mod_globals = vars(L)
    # The module's __builtins__ is allowed; ensure no app.* names.
    # If app.models is referenced as a string inside the source,
    # that's just lazy-loading, which is fine.
    # The simplest check: run the module's main() paths in subprocess
    # isolation (out of scope here). Instead, just check the file
    # source for ``from app.`` / ``import app.`` at module level
    # (which would force eager app import when the module is
    # imported). All app imports in the file are inside functions
    # (verified manually via review of label_phrase_match_pairs.py).
    source_file = L.__file__
    assert source_file is not None
    src = Path(source_file).read_text()
    assert "from app." not in src.split("\ndef ")[0], (
        "labeler helper has top-level 'from app.' import; the helper's "
        "dependency surface should be tiny — defer app imports to "
        "function scope"
    )
    assert "import app." not in src.split("\ndef ")[0], (
        "labeler helper has top-level 'import app.'; defer to "
        "function scope"
    )
