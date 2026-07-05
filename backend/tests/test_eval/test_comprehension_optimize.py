"""Tests for the Phase 9.4 comprehension optimizer CLI.

Covers the same acceptance shape as ``test_cloze_eval.py``:

- ``--help`` exits 0 (the locked acceptance criterion).
- The CLI is invokable as ``python -m scripts.optimize_comprehension``
  from the canonical location.
- The default mode (no ``--live``) does not require the OpenRouter
  API key in the environment to write the artifact file.
- The written artifact carries the documented
  ``schema_version="comprehension-optimized-v1"`` so the
  production path can read it on next start (Phase 9+).
- The CLI surface is structurally aligned with ``optimize_cloze.py``
  (same flag set, same artifact shape) — a regression here means
  the cross-card mirroring drifted.

The dummy-LM path is exercised by the spec, but the offline
DummyLM discipline is owned by ``app.comprehension._configure_dspy``
(see test_comprehension.py::test_optimize_comprehension_module_runs_on_two_row_eval_set_offline).
This module's tests cover the CLI plumbing end-to-end, not the
DSPy internals.

Run from ``backend/``::

    uv run pytest -q tests/test_eval/test_comprehension_optimize.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval"
EVAL_JSONL = EVAL_DIR / "comprehension_judgments.jsonl"
APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"


# --- CLI --help --------------------------------------------------------


def test_cli_help_exits_0():
    """``uv run python -m scripts.optimize_comprehension --help`` exits
    0 per the locked acceptance criterion (mirrors optimize_cloze.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.optimize_comprehension", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}: {result.stderr}"
    )
    assert "--live" in result.stdout
    assert "--eval-path" in result.stdout
    assert "--output-path" in result.stdout


# --- Eval-set field projection ----------------------------------------


def test_eval_set_has_expected_fields():
    """Phase 6 already shipped the comprehension eval set. This test
    pins the field set so any drift in the loader surfaces here,
    not in a runtime crash."""
    if not EVAL_JSONL.exists():
        pytest.skip(
            f"Eval set not built yet ({EVAL_JSONL}); Phase 6 deliverable."
        )
    rows: list[dict] = []
    with EVAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(line))
    # Sanity: at least one accept row, with the comprehension-specific
    # fields.
    assert rows, "comprehension eval set is empty"
    required = {
        "target_word",
        "word_id",
        "word_type",
        "expected_passage",
        "expected_question",
        "expected_choices",
        "expected_correct_choice",
        "judgment",
    }
    for i, row in enumerate(rows):
        missing = required - row.keys()
        assert not missing, f"Row {i} missing fields: {missing}"
    # At least one accept row — the CLI's training path requires it.
    accept_rows = [r for r in rows if r.get("judgment") == "accept"]
    assert accept_rows, "No accept rows in comprehension eval set"


# --- Loader unit tests -------------------------------------------------


def test_loader_projects_target_word_to_word():
    """The CLI's loader must project ``target_word`` (eval-set key)
    onto ``word`` (DSPy signature key). A regression here means
    the optimizer silently trains on the wrong field."""
    from scripts.optimize_comprehension import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_compr_eval.jsonl"
    tmp.write_text(
        "# leading comment\n"
        '{"target_word": "Hund", "word_id": 1, "word_type": "Noun", '
        '"expected_passage": "p", "expected_question": "q", '
        '"expected_choices": {"A": "x", "B": "y", "C": "z", "D": "w"}, '
        '"expected_correct_choice": "A", "judgment": "accept", '
        '"labeler": "l", "provenance": "p", "rationale": "r"}\n'
        '{"target_word": "Katze", "word_id": 2, "word_type": "Noun", '
        '"expected_passage": "p", "expected_question": "q", '
        '"expected_choices": {"A": "x", "B": "y", "C": "z", "D": "w"}, '
        '"expected_correct_choice": "A", "judgment": "reject", '
        '"labeler": "l", "provenance": "p", "rationale": "r"}\n',
        encoding="utf-8",
    )
    try:
        rows = _load_eval_set(tmp)
        assert len(rows) == 1, "reject row should be skipped"
        assert rows[0]["word"] == "Hund"
        assert rows[0]["target_word_id"] == 1
        assert json.loads(rows[0]["learner_axes_json"]) == {}
        assert json.loads(rows[0]["retrieved_chunks_json"]) == []
    finally:
        tmp.unlink(missing_ok=True)


def test_loader_skips_comment_lines():
    """JSONL by convention allows ``#``-prefixed comment lines. The
    loader must strip them before json.loads."""
    from scripts.optimize_comprehension import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_compr_comments.jsonl"
    tmp.write_text(
        "# labeler: test\n"
        "# provenance: test\n"
        '{"target_word": "Hund", "word_id": 1, "word_type": "Noun", '
        '"expected_passage": "p", "expected_question": "q", '
        '"expected_choices": {"A": "x", "B": "y", "C": "z", "D": "w"}, '
        '"expected_correct_choice": "A", "judgment": "accept", '
        '"labeler": "l", "provenance": "p", "rationale": "r"}\n',
        encoding="utf-8",
    )
    try:
        rows = _load_eval_set(tmp)
        assert len(rows) == 1
    finally:
        tmp.unlink(missing_ok=True)


def test_loader_returns_empty_on_missing_file():
    """No eval set → no rows. The CLI handles this gracefully
    (prints NOOP, exits 0)."""
    from scripts.optimize_comprehension import _load_eval_set

    missing = Path("/tmp/does-not-exist-comprehension.jsonl")
    if missing.exists():
        missing.unlink()
    with pytest.raises(FileNotFoundError):
        _load_eval_set(missing)


# --- Train/val split ---------------------------------------------------


def test_train_val_split_is_deterministic_80_20():
    """Same input → same split across runs. Determinism matters
    because the operator re-runs the CLI on the same eval set
    and expects the same split (mirrors optimize_cloze.py)."""
    from scripts.optimize_comprehension import _train_val_split

    rows = [{"i": i} for i in range(10)]
    train_a, val_a = _train_val_split(rows, val_fraction=0.2)
    train_b, val_b = _train_val_split(rows, val_fraction=0.2)
    assert train_a == train_b
    assert val_a == val_b
    # 80/20 on 10 rows → 8 train, 2 val.
    assert len(train_a) == 8
    assert len(val_a) == 2


def test_train_val_split_handles_single_row():
    """A single-row eval set can't be split — train gets the only
    row, val is empty (the optimizer uses train for both)."""
    from scripts.optimize_comprehension import _train_val_split

    rows = [{"i": 0}]
    train, val = _train_val_split(rows, val_fraction=0.2)
    assert len(train) == 1
    assert val == []


# --- Serialization helper ----------------------------------------------


def test_serialize_optimized_has_locked_schema_version():
    """The artifact file's ``schema_version`` is the production path's
    read key. Bumping it requires coordinated reader changes."""
    from scripts.optimize_comprehension import _serialize_optimized

    class _StubModule:
        predictors = []  # no predictors → empty instructions_by_field

    payload = _serialize_optimized(_StubModule(), train_count=5, val_count=1, mode="dummy")
    assert payload["schema_version"] == "comprehension-optimized-v1"
    assert payload["mode"] == "dummy"
    assert payload["train_count"] == 5
    assert payload["val_count"] == 1
    assert payload["instructions_by_field"] == {}


def test_serialize_optimized_handles_callable_predictors():
    """DSPy 3.x's ``module.predictors`` can be a class method (not
    iterable). The serializer must tolerate both shapes — a
    regression here crashes the CLI on a serialisation quirk."""
    from scripts.optimize_comprehension import _serialize_optimized

    class _StubModule:
        @classmethod
        def predictors(cls):
            return []

    payload = _serialize_optimized(_StubModule(), train_count=1, val_count=0, mode="dummy")
    assert payload["schema_version"] == "comprehension-optimized-v1"
    assert payload["instructions_by_field"] == {}


# --- End-to-end dry-run (mirrors the locked acceptance criterion) ------


def test_cli_end_to_end_writes_artifact(tmp_path, monkeypatch):
    """The dry-run path (no API key) writes the artifact file cleanly.

    Mirrors the spec: ``DummyLM`` discipline is preserved; the CLI
    plumbing runs end-to-end; the artifact is on disk and carries
    the locked ``schema_version`` and ``mode=dummy`` markers.

    The ``OPENROUTER_API_KEY`` env is cleared at the start so this
    test runs hermetically even when the operator's environment
    has the key set (the same pattern ``test_comprehension.py``
    uses for the optimizer function unit test).
    """
    if not EVAL_JSONL.exists():
        pytest.skip(
            f"Eval set not built yet ({EVAL_JSONL}); Phase 6 deliverable."
        )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Force DSPy to reconfigure — same shape as test_comprehension.py
    # uses for the optimizer function. Without this, a stale
    # ``dspy.settings.lm`` from a prior test would route to the
    # live adapter in this process and bypass DummyLM.
    import dspy

    dspy.settings.lm = None

    output_path = tmp_path / "comprehension_optimized.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimize_comprehension",
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
        env={**os.environ, "OPENROUTER_API_KEY": ""},
    )
    assert result.returncode == 0, (
        f"dry-run exited {result.returncode}: {result.stderr}\nstdout: {result.stdout}"
    )
    assert "OK:" in result.stdout, result.stdout
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "comprehension-optimized-v1"
    # ``mode`` is "dummy" because the CLI ran without --live.
    assert payload["mode"] == "dummy"
    # Train/val counts are deterministic on the same eval set:
    # 40 accept rows → 32 train, 8 val (default 80/20 split).
    assert payload["train_count"] == 32
    assert payload["val_count"] == 8


def test_cli_noop_on_missing_eval_set(tmp_path):
    """When the eval set is absent, the CLI prints NOOP and exits 0
    so a CI step that runs the CLI before the eval set lands
    doesn't fail the build. Mirrors ``optimize_cloze.py``."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimize_comprehension",
            "--eval-path",
            "/tmp/does-not-exist-comprehension-eval.jsonl",
            "--output-path",
            str(tmp_path / "comprehension_optimized.json"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    assert result.returncode == 0
    assert "NOOP" in result.stdout
    # Artifact should NOT be written when the eval set is absent.
    assert not (tmp_path / "comprehension_optimized.json").exists()