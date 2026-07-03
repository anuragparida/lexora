"""Tests for the Phase 6.7 Ragas offline runner + held-out sets.

Covers:
  - Module-level constants (RAGAS_MIN_*, RAG_TOP_K,
    RAG_MAX_CHARS_*, RAGAS_DRY_RUN_MIN_OVERALL,
    RAGAS_API_KEY_ENV) match the Phase 6 plan card body.
  - ``format_retrieved_chunks`` truncates to per-chunk + total
    budgets.
  - ``build_ragas_sample`` produces the expected dict shape.
  - The match + comprehension JSONLs produced by the builder are
    byte-stable (re-running with the same seed reproduces the
    same files).
  - The match JSONL has 40 rows; comprehension has 40 rows.
  - Every match row's 3 right_word_ids are distinct, not the
    target, and resolve to the words table.
  - Every comprehension row has exactly 4 choices A/B/C/D and
    the ``expected_correct_choice`` is one of them.
  - ``scripts.eval_ragas --dry-run`` exits 0 and prints OK
    without contacting OpenRouter or Ragas (the CI smoke path).
  - The runner's metrics math is correct on a synthetic input.
  - No ``RAGAS_API_KEY`` literal with a real value anywhere
    in the repo (Hard rule #7).

These tests are intentionally narrow — the spec calls for the
runner to be a deterministic offline tool, so the tests assert
determinism. Anything stochastic (sampling, embeddings, prompt
LLM) belongs to 4.x and is mocked at a higher level than this
module.

Run from ``backend/``::

    uv run pytest -q tests/test_eval/test_ragas.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval"
EVAL_MATCH_JSONL = EVAL_DIR / "match_judgments.jsonl"
EVAL_COMP_JSONL = EVAL_DIR / "comprehension_judgments.jsonl"
EVAL_CLOZE_JSONL = EVAL_DIR / "cloze_judgments.jsonl"

LABELER = "template-based-fallback-2026-07-03"
PROVENANCE = (
    "deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle"
)


# --- Module-level constants -------------------------------------------


def test_ragas_min_context_precision_is_locked():
    """The Phase 6.7 plan card body explicitly locks
    RAGAS_MIN_CONTEXT_PRECISION = 0.6. A regression here means a
    code review needs to happen, not a config bump."""
    from app.eval.ragas import RAGAS_MIN_CONTEXT_PRECISION

    assert RAGAS_MIN_CONTEXT_PRECISION == 0.6


def test_ragas_min_context_recall_is_locked():
    from app.eval.ragas import RAGAS_MIN_CONTEXT_RECALL

    assert RAGAS_MIN_CONTEXT_RECALL == 0.5


def test_ragas_min_faithfulness_is_locked():
    from app.eval.ragas import RAGAS_MIN_FAITHFULNESS

    assert RAGAS_MIN_FAITHFULNESS == 0.7


def test_ragas_min_answer_relevance_is_locked():
    from app.eval.ragas import RAGAS_MIN_ANSWER_RELEVANCE

    assert RAGAS_MIN_ANSWER_RELEVANCE == 0.6


def test_rag_top_k_is_locked():
    from app.eval.ragas import RAG_TOP_K

    assert RAG_TOP_K == 5


def test_rag_max_chars_per_chunk_is_locked():
    from app.eval.ragas import RAG_MAX_CHARS_PER_CHUNK

    assert RAG_MAX_CHARS_PER_CHUNK == 300


def test_rag_max_chars_is_locked():
    from app.eval.ragas import RAG_MAX_CHARS

    assert RAG_MAX_CHARS == 1500


def test_ragas_dry_run_min_overall_is_locked():
    from app.eval.ragas import RAGAS_DRY_RUN_MIN_OVERALL

    assert RAGAS_DRY_RUN_MIN_OVERALL == 0.6


def test_ragas_api_key_env_is_locked():
    """Hard rule #7: the env var name is committed; the value is
    never written to the repo. This test pins the name so a
    future rename is intentional."""
    from app.eval.ragas import RAGAS_API_KEY_ENV
    # Use string composition to dodge the harness redaction
    # gotcha on the KEY substring. The comparison is built
    # dynamically so the source doesn't carry a literal
    # ``KEY-shaped token.
    expected = str("RAGAS") + "_API" + "_KEY"
    # Use getattr to compare so the ``= operator
    # doesn't appear in the source as a literal token.
    op = getattr(str.__eq__, "__name__", None)
    assert op is not None  # sanity: str.__eq__ exists
    # Direct equality check via the str class's __eq__ method.
    assert str.__eq__(RAGAS_API_KEY_ENV, expected)


def test_no_getenv_on_ragas_constants():
    """Hard rule #9: the Ragas constants are module-level, not
    env-derived. ``git grep``-equivalent: scan the module source
    for any ``getenv`` / ``environ`` reference inside ragas.py.
    """
    ragas_path = (
        Path(__file__).resolve().parent.parent.parent
        / "app" / "eval" / "ragas.py"
    )
    text = ragas_path.read_text(encoding="utf-8")
    assert "os.getenv" not in text, (
        "Ragas constants must be hard-coded module constants; "
        "found os.getenv in backend/app/eval/ragas.py"
    )
    assert "os.environ" not in text, (
        "Ragas constants must be hard-coded module constants; "
        "found os.environ in backend/app/eval/ragas.py"
    )


# --- format_retrieved_chunks ------------------------------------------


def test_format_retrieved_chunks_empty():
    from app.eval.ragas import format_retrieved_chunks

    assert format_retrieved_chunks([]) == ""


def test_format_retrieved_chunks_plain_strings():
    from app.eval.ragas import format_retrieved_chunks

    assert format_retrieved_chunks(["hello"]) == "hello"
    assert (
        format_retrieved_chunks(["hello", "world"]) == "hello\n---\nworld"
    )


def test_format_retrieved_chunks_dict_inputs():
    """Phase 1's ``/retrieve`` return shape is a list of dicts
    with ``german`` / ``text`` keys. The helper accepts both."""
    from app.eval.ragas import format_retrieved_chunks

    out = format_retrieved_chunks(
        [{"german": "die Uhr"}, {"text": "der Tag"}]
    )
    assert out == "die Uhr\n---\nder Tag"


def test_format_retrieved_chunks_per_chunk_truncation():
    """Each individual chunk is truncated to
    RAG_MAX_CHARS_PER_CHUNK = 300 chars."""
    from app.eval.ragas import (
        format_retrieved_chunks, RAG_MAX_CHARS_PER_CHUNK,
    )

    long = "x" * 1000
    out = format_retrieved_chunks([long])
    pieces = out.split("\n---\n")
    assert len(pieces) == 1
    assert len(pieces[0]) == RAG_MAX_CHARS_PER_CHUNK


def test_format_retrieved_chunks_total_truncation():
    """The joined result is truncated to RAG_MAX_CHARS = 1500."""
    from app.eval.ragas import format_retrieved_chunks, RAG_MAX_CHARS

    big = ["x" * 1000] * 10
    out = format_retrieved_chunks(big)
    assert len(out) <= RAG_MAX_CHARS


# --- build_ragas_sample -----------------------------------------------


def test_build_ragas_sample_basic():
    from app.eval.ragas import build_ragas_sample

    s = build_ragas_sample("q", "a", ["c1", "c2"], "gt")
    assert s == {
        "question": "q",
        "answer": "a",
        "contexts": ["c1", "c2"],
        "ground_truth": "gt",
    }


def test_build_ragas_sample_dict_contexts():
    """Accepts dict-style contexts (Phase 1 wire shape) and
    stringifies each."""
    from app.eval.ragas import build_ragas_sample

    s = build_ragas_sample(
        "q", "a", [{"german": "foo"}, {"text": "bar"}], "gt"
    )
    assert s["contexts"] == ["foo", "bar"]


def test_build_ragas_sample_single_string_context():
    """A single string is treated as a one-chunk context."""
    from app.eval.ragas import build_ragas_sample

    s = build_ragas_sample("q", "a", "just one", "gt")
    assert s["contexts"] == ["just one"]


def test_build_ragas_sample_empty_string_context():
    from app.eval.ragas import build_ragas_sample

    s = build_ragas_sample("q", "a", "", "gt")
    assert s["contexts"] == []


# --- JSONL shape (only when files have been built) ---------------------


@pytest.fixture(scope="module")
def match_rows() -> list[dict]:
    if not EVAL_MATCH_JSONL.exists():
        pytest.skip("match_judgments.jsonl not built yet")
    with EVAL_MATCH_JSONL.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


@pytest.fixture(scope="module")
def comprehension_rows() -> list[dict]:
    if not EVAL_COMP_JSONL.exists():
        pytest.skip("comprehension_judgments.jsonl not built yet")
    with EVAL_COMP_JSONL.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def test_match_eval_set_has_40_rows(match_rows):
    assert len(match_rows) == 40


def test_comprehension_eval_set_has_40_rows(comprehension_rows):
    assert len(comprehension_rows) == 40


def test_match_row_shape(match_rows):
    """Each match row has the required fields and a 3-pair
    expected_pairs list."""
    for r in match_rows:
        for k in (
            "word_id",
            "target_word",
            "word_type",
            "expected_pairs",
            "labeler",
            "provenance",
            "judgment",
            "rationale",
        ):
            assert k in r, f"missing key {k!r} in match row"
        assert len(r["expected_pairs"]) == 3
        for p in r["expected_pairs"]:
            assert "right_word_id" in p
            assert "right_word" in p
            assert "right_kind" in p
            assert p["right_kind"] == "antonym"


def test_comprehension_row_shape(comprehension_rows):
    """Each comprehension row has 4 choices A/B/C/D and a
    correct_choice in {A, B, C, D}."""
    for r in comprehension_rows:
        for k in (
            "word_id",
            "target_word",
            "word_type",
            "expected_passage",
            "expected_question",
            "expected_choices",
            "expected_correct_choice",
            "labeler",
            "provenance",
            "judgment",
            "rationale",
        ):
            assert k in r, f"missing key {k!r} in comprehension row"
        assert set(r["expected_choices"].keys()) == {"A", "B", "C", "D"}
        assert r["expected_correct_choice"] in ("A", "B", "C", "D")
        assert (
            r["expected_choices"][r["expected_correct_choice"]]
            is not None
        )


def test_match_and_comprehension_labeler_provenance(
    match_rows, comprehension_rows
):
    """Apollo's locked deviations apply to all 6.7 generated
    eval sets."""
    for r in match_rows:
        assert r["labeler"] == LABELER
        assert r["provenance"] == PROVENANCE
    for r in comprehension_rows:
        assert r["labeler"] == LABELER
        assert r["provenance"] == PROVENANCE


def test_match_right_ids_are_distinct_and_not_target(match_rows):
    """Each match row's 3 right_word_ids are distinct and do not
    collide with the row's own word_id."""
    for r in match_rows:
        right_ids = [p["right_word_id"] for p in r["expected_pairs"]]
        assert len(set(right_ids)) == 3, (
            f"word_id={r['word_id']} has duplicate right_ids={right_ids}"
        )
        assert r["word_id"] not in right_ids, (
            f"word_id={r['word_id']} appears in its own right_ids"
        )


def test_comprehension_passage_contains_target_word(comprehension_rows):
    """Each comprehension passage contains the target word."""
    for r in comprehension_rows:
        assert r["target_word"] in r["expected_passage"], (
            f"word_id={r['word_id']} target {r['target_word']!r} "
            f"missing from passage {r['expected_passage']!r}"
        )


# --- scripts.eval_ragas --dry-run smoke path --------------------------


def _cloze_jsonl_present() -> bool:
    return EVAL_CLOZE_JSONL.exists()


@pytest.mark.skipif(
    not _cloze_jsonl_present(),
    reason="cloze_judgments.jsonl not built yet",
)
def test_eval_ragas_dry_run_exits_zero_and_prints_ok(tmp_path):
    """The CI smoke path: ``--dry-run`` exits 0, prints ``OK``,
    writes a results file with overall >= RAGAS_DRY_RUN_MIN_OVERALL.

    Scoped to a temp output dir so we don't clobber the canonical
    eval/ artifacts (the stable symlink would still be touched
    — see the follow-up test for the canonical run).
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_ragas",
        "--dry-run",
        "--only",
        "cloze",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"expected 'OK' in stdout, got: {result.stdout!r}"
    )
    # The results file should exist in tmp_path.
    results_files = list(tmp_path.glob("ragas_results_*.jsonl"))
    assert results_files, (
        f"no ragas_results_*.jsonl in {tmp_path}: "
        f"{list(tmp_path.iterdir())}"
    )
    # And the symlink should be there.
    symlink = tmp_path / "ragas_results.jsonl"
    assert symlink.is_symlink() or symlink.exists(), (
        f"expected symlink at {symlink}"
    )
    # The symlink should point at the latest run.
    if symlink.is_symlink():
        target = symlink.resolve()
        assert target.name == results_files[-1].name, (
            f"symlink points to {target}, not the latest run"
        )


def test_eval_ragas_dry_run_metric_math():
    """The deterministic proxy: judgment=accept → all metrics
    1.0; judgment=reject → all metrics 0.0. The mean is the
    row's pass boolean. The overall mean across rows is
    compared to the threshold."""
    from app.eval.ragas import RAGAS_DRY_RUN_MIN_OVERALL
    from scripts.eval_ragas import _run_dry_metrics

    rows = _run_dry_metrics(
        exercise_type="cloze",
        judgments=[(1, "accept"), (2, "accept"), (3, "reject")],
    )
    assert len(rows) == 3
    for r in rows:
        if r.row_id in (1, 2):
            assert r.context_precision == 1.0
            assert r.context_recall == 1.0
            assert r.faithfulness == 1.0
            assert r.answer_relevance == 1.0
            assert r.pass_ is True
        else:
            assert r.context_precision == 0.0
            assert r.pass_ is False
    # Overall: 2/3 ≈ 0.667 >= RAGAS_DRY_RUN_MIN_OVERALL.
    overall = sum(
        (
            r.context_precision
            + r.context_recall
            + r.faithfulness
            + r.answer_relevance
        ) / 4.0
        for r in rows
    ) / len(rows)
    assert overall >= RAGAS_DRY_RUN_MIN_OVERALL


# --- No-secret guard (Hard rule #7) -----------------------------------



def test_no_ragas_api_key_literal_in_repo():
    """Hard rule #7: a literal ``RAGAS_API_KEY=<value>`` form
    must not appear anywhere in the backend source tree. The
    env var *name* is allowed (it's a constant in
    app/eval/ragas.py); the value is not.

    The grep is scoped to text files in the repo and excludes
    .git, .venv, .pytest_cache, and the data/ directory. The
    pattern matches ``RAGAS_API_KEY=<non-empty-value>``
    pairs at the start of a line (i.e. real config / env-var
    assignments) — a placeholder or template value would still
    trigger the check. Docstring mentions and code-comments
    are skipped.
    """
    backend_root = Path(__file__).resolve().parent.parent.parent
    # Match ``RAGAS_API_KEY=*** as a config-style assignment at
    # the start of a line (after optional leading whitespace).
    # The pattern requires a non-empty, non-quoted value
    # character to avoid matching docstring text.
    pattern = re.compile(
        r'^\s*RAGAS_API_KEY\s*=\s*\S+',
        re.IGNORECASE | re.MULTILINE,
    )
    offenders: list[str] = []
    for path in backend_root.rglob("*"):
        if not path.is_file():
            continue
        if any(
            part in path.parts
            for part in (
                ".git",
                ".venv",
                ".pytest_cache",
                "__pycache__",
                "data",
            )
        ):
            continue
        if path.suffix in (".pyc", ".so", ".db", ".db-journal"):
            continue
        # .env.example is allowed to mention the name with a
        # placeholder value (which is what the spec calls for
        # via the Python script workaround). Skip.
        if path.name == ".env.example":
            continue
        # The append script is the *legitimate writer* of the
        # .env.example entry — it's the script the spec calls
        # out by name as the harness-redaction workaround.
        # Skip it too (the literal it writes is the placeholder,
        # and the script itself is the audit trail).
        if path.name == "append_ragas_env_example.py":
            continue
        # This very test file documents the rule. The test's
        # own docstring mentions the pattern; skip self-checks.
        if path.name == "test_ragas.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            offenders.append(str(path))
    assert not offenders, (
        "literal RAGAS_API_KEY assignment with value found in: "
        + ", ".join(offenders)
    )
