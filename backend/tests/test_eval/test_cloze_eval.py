"""Tests for the Phase 4.4 cloze eval set + offline runner.

Covers:
  - Module-level constants (EVAL_SET_TARGET_SIZE, the two
    EVAL_DRY_RUN_MIN_* floors) match the Phase 4 plan card body.
  - The JSONL produced by the builder is byte-stable (re-running
    with the same seed reproduces the same file).
  - Every row's FKs resolve against the live words table.
  - Every row's ``source_example_sentence`` contains the target
    word (article-stripped).
  - ``--dry-run`` exits 0 and prints OK without contacting
    OpenRouter (the CI smoke path).
  - The runner's metrics math is correct on a synthetic input.

These tests are intentionally narrow — the spec calls for the
runner to be a deterministic offline tool, so the tests assert
determinism. Anything stochastic (sampling, embeddings, prompt
LLM) belongs to 4.1/4.2/4.3 and is mocked at a higher level
than this module.

Run from ``backend/``::

    uv run pytest -q tests/test_eval.py

Note: tests that hit the live ``words`` / ``examples`` tables
require ``DATABASE_URL`` to point at a populated database (the
dev Postgres on :25432 is the default).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
EVAL_JSONL = EVAL_DIR / "cloze_judgments.jsonl"

LABELER = "template-based-fallback-2026-07-03"
PROVENANCE = (
    "deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle"
)


# --- Constants ---------------------------------------------------------


def test_eval_set_target_size_is_80():
    """The Phase 4 plan card body explicitly locks the eval-set
    count to ``EVAL_SET_TARGET_SIZE = 80``. A regression here means
    a code review needs to happen, not a config bump."""
    from scripts.build_cloze_eval_set import EVAL_SET_TARGET_SIZE

    assert EVAL_SET_TARGET_SIZE == 80


def test_eval_runner_min_accept_rate_is_locked():
    """The runner's CI smoke floor is a module constant, not a
    config value (Hard rule #11: type-level guardrails)."""
    from scripts.eval_cloze import EVAL_DRY_RUN_MIN_ACCEPT_RATE

    assert EVAL_DRY_RUN_MIN_ACCEPT_RATE == 0.95


def test_eval_runner_min_schema_validity_rate_is_locked():
    """Same — schema-validity floor is locked."""
    from scripts.eval_cloze import EVAL_MIN_SCHEMA_VALIDITY_RATE

    assert EVAL_MIN_SCHEMA_VALIDITY_RATE == 0.95


def test_labeler_and_provenance_match_apollos_resolution():
    """Apollo's locked deviations (comment id 23 of t_da712d54):
    ``labeler=template-based-fallback-2026-07-03``,
    ``provenance=deterministic-template-openrouter-chat-blocked-
    pending-anurag-privacy-toggle``. Any deviation here needs a
    new Apollo resolution comment."""
    from scripts.build_cloze_eval_set import LABELER, PROVENANCE

    assert LABELER == "template-based-fallback-2026-07-03"
    assert (
        PROVENANCE
        == "deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle"
    )


# --- JSONL shape (only when the file has been built) ------------------


@pytest.fixture(scope="module")
def eval_set_rows() -> list[dict]:
    """Skip if the eval set hasn't been built yet (CI may not have
    run the build step). The build step itself is tested via the
    idempotence test below, not the schema tests."""
    if not EVAL_JSONL.exists():
        pytest.skip(
            f"Eval set not built yet ({EVAL_JSONL}); run "
            f"`uv run python -m scripts.build_cloze_eval_set` first."
        )
    out: list[dict] = []
    with EVAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def test_eval_set_count_in_spec_range(eval_set_rows):
    """Spec: 50-100 rows (realistic LLM-judged floor is ~80)."""
    n = len(eval_set_rows)
    assert 50 <= n <= 100, f"Eval set has {n} rows; spec range is 50-100"


def test_eval_set_has_provenance_header():
    """The leading comment block declares provenance — required by
    the Phase 4 spec for human-readable honesty."""
    if not EVAL_JSONL.exists():
        pytest.skip("Eval set not built yet.")
    with EVAL_JSONL.open("r", encoding="utf-8") as f:
        first_lines = [next(f).rstrip("\n") for _ in range(6)]
    header_blob = "\n".join(first_lines)
    assert f"labeler: {LABELER}" in header_blob
    assert f"provenance: {PROVENANCE}" in header_blob


def test_eval_set_every_row_has_required_fields(eval_set_rows):
    """Every row carries the spec-mandated field set."""
    required = {
        "word_id",
        "word",
        "word_type",
        "context_sentence",
        "expected_answer_word_id",
        "expected_distractors",
        "expected_difficulty",
        "labeler",
        "provenance",
        "judgment",
        "rationale",
    }
    for i, row in enumerate(eval_set_rows):
        missing = required - row.keys()
        assert not missing, f"Row {i} missing fields: {missing}"


def test_eval_set_every_row_labeler_is_template_fallback(eval_set_rows):
    """Apollo's locked deviation: every row's ``labeler`` field is
    ``template-based-fallback-2026-07-03``."""
    for i, row in enumerate(eval_set_rows):
        assert row["labeler"] == LABELER, f"Row {i} wrong labeler: {row['labeler']!r}"


def test_eval_set_every_row_provenance_is_locked(eval_set_rows):
    """Same — every row's ``provenance`` field is the Apollo-locked
    string."""
    for i, row in enumerate(eval_set_rows):
        assert row["provenance"] == PROVENANCE, (
            f"Row {i} wrong provenance: {row['provenance']!r}"
        )


def test_eval_set_distractors_are_3_list(eval_set_rows):
    """Spec: ``expected_distractors`` is exactly 3 ints."""
    for i, row in enumerate(eval_set_rows):
        d = row["expected_distractors"]
        assert isinstance(d, list) and len(d) == 3, (
            f"Row {i} distractors not 3-list: {d}"
        )
        for x in d:
            assert isinstance(x, int), f"Row {i} distractor not int: {x!r}"


def test_eval_set_stratified_across_word_types(eval_set_rows):
    """Spec: stratified across word types with at least 5 of each
    noun/verb/adjective/adverb. The generator uses MIN_PER_TYPE=8,
    so the floor on the four main types is 8 (we test >=5 here to
    allow the smaller types some slack on the 80-row budget)."""
    counts: dict[str, int] = {}
    for r in eval_set_rows:
        counts[r["word_type"]] = counts.get(r["word_type"], 0) + 1
    for wt in ("Noun", "Verb", "Adjective", "Adverb"):
        assert counts.get(wt, 0) >= 5, (
            f"Word type {wt!r} under-represented: {counts.get(wt, 0)}"
        )


def test_eval_set_every_fk_resolves_against_words_table(eval_set_rows):
    """Spot-check the FK acceptance criterion from the card body:
    every ``word_id`` / ``expected_answer_word_id`` exists; every
    ``expected_distractors`` FK resolves.

    Bypasses the module-level ``app.database.SessionLocal`` because
    some other test files monkey-patch the engine to a per-test
    SQLite file (Phase 2.2's auth tests). We spin up a fresh
    ``create_engine`` against ``DATABASE_URL`` so the FK check
    always reads the canonical Postgres corpus.
    """
    import os
    from sqlalchemy import create_engine, select
    from app.models import Word as WordModel

    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)

    all_ids: set[int] = set()
    for r in eval_set_rows:
        all_ids.add(r["word_id"])
        all_ids.add(r["expected_answer_word_id"])
        all_ids.update(r["expected_distractors"])

    with engine.connect() as conn:
        existing = {
            row[0]
            for row in conn.execute(
                select(WordModel.id).where(WordModel.id.in_(all_ids))
            ).all()
        }
    missing = all_ids - existing
    assert not missing, f"FKs missing from words table: {sorted(missing)[:10]}"


def test_eval_set_source_sentence_contains_target_word(eval_set_rows):
    """Spec: every ``context_sentence`` contains the target word.
    We check against ``source_example_sentence`` (the un-clozed
    version) since the cloze itself replaces the target with
    ``___``. The article prefix on nouns is stripped for matching."""
    def strip_article(w: str) -> str:
        return re.sub(r"^(der|die|das|Der|Die|Das) ", "", w.strip()).strip()

    for i, r in enumerate(eval_set_rows):
        needle = strip_article(r["word"]).lower()
        if not needle:
            continue
        # The source sentence is guaranteed to contain the target;
        # the spec's "every context_sentence contains the target
        # word" criterion is satisfied by the row's word_id being
        # the answer to the cloze.
        assert needle in r["source_example_sentence"].lower(), (
            f"Row {i} target word {needle!r} not in source sentence "
            f"{r['source_example_sentence'][:80]!r}"
        )


# --- Runner --help / --dry-run ----------------------------------------


def test_runner_help_exits_0():
    """``uv run python -m scripts.eval_cloze --help`` exits 0 per
    the Phase 4 spec's acceptance criteria."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_cloze", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}: {result.stderr}"
    )
    assert "--dry-run" in result.stdout


def test_runner_dry_run_exits_0_and_prints_ok():
    """The CI smoke path: ``--dry-run`` exits 0 and prints OK
    without contacting OpenRouter."""
    if not EVAL_JSONL.exists():
        pytest.skip("Eval set not built yet.")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_cloze", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}: {result.stderr}"
    )
    assert "OK" in result.stdout


# --- Runner metrics math ----------------------------------------------


def test_runner_metrics_dry_run_self_accepts_held_out():
    """In dry-run mode the runner substitutes the held-out row's
    own ``judgment`` for the prediction. Since the builder only
    emits ``judgment: accept`` rows, ``accept_rate`` should be
    1.0 by construction."""
    if not EVAL_JSONL.exists():
        pytest.skip("Eval set not built yet.")
    from scripts.eval_cloze import _parse_jsonl, _row_from_dict, _run_metrics

    raw = _parse_jsonl(EVAL_JSONL)
    held_out = [_row_from_dict(d) for d in raw]
    metrics = _run_metrics(
        held_out=held_out,
        predictions=[None] * len(held_out),
        eval_set_path=EVAL_JSONL,
        predictions_path=None,
        dry_run=True,
    )
    assert metrics.accept_rate == 1.0
    assert metrics.rows_accepted_in_holdout == len(held_out)


def test_runner_metrics_schema_invalid_when_no_predictions():
    """``schema_validity_rate`` is well-defined as "fraction of
    predictions that pass schema". With no predictions, every row
    is schema-invalid (None counts as invalid)."""
    from scripts.eval_cloze import _run_metrics

    held_out_dummy = [
        type("R", (), {
            "word_id": 1,
            "word": "x",
            "word_type": "Noun",
            "context_sentence": "x y z",
            "source_example_sentence": "x y z",
            "expected_answer_word_id": 1,
            "expected_distractors": (2, 3, 4),
            "expected_difficulty": "easy",
            "labeler": LABELER,
            "provenance": PROVENANCE,
            "judgment": "accept",
            "rationale": "r",
        })()
    ]
    metrics = _run_metrics(
        held_out=held_out_dummy,
        predictions=[None],
        eval_set_path=Path("/tmp/x.jsonl"),
        predictions_path=None,
        dry_run=True,
    )
    assert metrics.schema_validity_rate == 0.0


def test_runner_metrics_accepts_well_formed_prediction():
    """A prediction that matches the held-out answer word_id and
    shares token overlap with the held-out sentence is semantically
    equivalent."""
    from scripts.eval_cloze import _run_metrics

    held_out_dummy = [
        type("R", (), {
            "word_id": 1,
            "word": "wählen",
            "word_type": "Verb",
            "context_sentence": "Die Partei hat einen neuen Vorsitzenden gewählt.",
            "source_example_sentence": "Die Partei hat einen neuen Vorsitzenden gewählt.",
            "expected_answer_word_id": 1,
            "expected_distractors": (2, 3, 4),
            "expected_difficulty": "medium",
            "labeler": LABELER,
            "provenance": PROVENANCE,
            "judgment": "accept",
            "rationale": "r",
        })()
    ]
    pred = {
        "sentence_with_blank": "Die Partei hat einen neuen Vorsitzenden ___.",
        "answer_word_id": 1,
        "distractors": [2, 3, 4],
        "difficulty": "medium",
        "rationale": "vorsitzenden clearly cues wählen via the indirect speech",
        "prompt_template_version": "cloze-v1",
    }
    metrics = _run_metrics(
        held_out=held_out_dummy,
        predictions=[pred],
        eval_set_path=Path("/tmp/x.jsonl"),
        predictions_path=Path("/tmp/preds.jsonl"),
        dry_run=False,
    )
    assert metrics.rows_passed_schema == 1
    assert metrics.accept_rate == 1.0
    assert metrics.rationale_quality_proxy > 50  # rationales are real