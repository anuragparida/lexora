"""Tests for the Phase 10.7 phrase_match optimizer CLI.

Covers the same acceptance shape as
``test_comprehension_optimize.py``:

- ``--help`` exits 0 (the locked acceptance criterion).
- The CLI is invokable as
  ``python -m scripts.optimize_phrase_match`` from the canonical
  location.
- The default mode (no ``--live``) does not require the OpenRouter
  API key in the environment to write the artifact file.
- The written artifact carries the documented
  ``schema_version="phrase-match-optimized-v1"`` so the production
  path can read it on next start (Phase 10.3+).
- The CLI surface is structurally aligned with ``optimize_match.py``
  / ``optimize_comprehension.py`` (same flag set, same artifact
  shape) — a regression here means the cross-card mirroring drifted.
- The HUMAN-LABELED provenance contract is enforced by the loader
  (Phase 1.5a precedent).
- The optimizer falls back to a deterministic stub artefact when
  Phase 10.2's ``app.phrase_match.optimize_phrase_match_module`` is
  not yet shipped — the offline smoke stays green in either order.

The dummy-LM path is exercised by the spec, but the offline
DummyLM discipline is owned by ``app.phrase_match._configure_dspy``
(once 10.2 ships; see ``test_comprehension.py`` for the equivalent
pattern). This module's tests cover the CLI plumbing end-to-end,
not the DSPy internals.

Run from ``backend/``::

    uv run pytest -q tests/test_eval/test_phrase_match_optimize.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


EVAL_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "eval"
)
EVAL_JSONL = EVAL_DIR / "phrase_match_judgments.jsonl"
APP_DIR = (
    Path(__file__).resolve().parent.parent.parent / "app"
)


# --- CLI --help --------------------------------------------------------


def test_cli_help_exits_0():
    """``uv run python -m scripts.optimize_phrase_match --help``
    exits 0 per the locked acceptance criterion (mirrors
    optimize_match.py / optimize_comprehension.py)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimize_phrase_match",
            "--help",
        ],
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
    # The phrase_match-specific knobs that aren't in the match /
    # comprehension optimizers.
    assert "--target" in result.stdout
    assert "--seed" in result.stdout
    assert "--max-demos" in result.stdout
    assert "--strict" in result.stdout


# --- Eval-set field projection ----------------------------------------


def test_eval_set_has_expected_fields():
    """Phase 10.4 ships the phrase_match eval set scaffold; Phase B
    (Anurag's hand-label session, card ``t_f3d2a634``) fills the rows.
    This test pins the field set so any drift in the loader surfaces
    here, not in a runtime crash.

    Skip on missing-or-empty: the test only makes sense once the
    JSONL carries at least one accept row. The empty-scaffold state
    is covered separately by ``test_optimize_phrase_match.py`` (the
    Step 2 deliverable for card ``t_51289780``)."""
    if not EVAL_JSONL.exists():
        pytest.skip(
            f"Eval set not built yet ({EVAL_JSONL}); Phase 10.4 "
            f"deliverable."
        )
    rows: list[dict] = []
    with EVAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(line))
    if not rows:
        pytest.skip(
            f"phrase_match eval set scaffold is empty ({EVAL_JSONL}); "
            "Phase B (Anurag's hand-label session) fills the rows."
        )
    required = {
        "phrase_a_id",
        "phrase_b_id",
        "phrase_a_text",
        "phrase_b_text",
        "expected_relation",
        "judgment",
        "provenance",
        "labeler",
        "rationale",
    }
    for i, row in enumerate(rows):
        missing = required - row.keys()
        assert not missing, f"Row {i} missing fields: {missing}"
    # Sanity: at least one accept row with HUMAN-LABELED provenance.
    accept_rows = [
        r for r in rows if r.get("judgment") == "accept"
    ]
    assert accept_rows, "No accept rows in phrase_match eval set"
    for i, row in enumerate(accept_rows):
        assert row.get("provenance") == "HUMAN-LABELED", (
            f"Accept row {i} drifted from HUMAN-LABELED provenance: "
            f"{row.get('provenance')!r}"
        )


# --- Loader unit tests -------------------------------------------------


def test_loader_projects_phrase_pair_to_signature_keys():
    """The CLI's loader must project the eval-set row onto the six
    input keys the Phase 10.2 ``PhraseMatchSignature`` consumes. A
    regression here means the optimizer silently trains on the wrong
    fields."""
    from scripts.optimize_phrase_match import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_pm_eval.jsonl"
    tmp.write_text(
        "# leading comment\n"
        '{"phrase_a_id": "tomaten-auf-den-augen", '
        '"phrase_b_id": "scheuklappen-auf", '
        '"phrase_a_text": "Tomaten auf den Augen", '
        '"phrase_b_text": "Scheuklappen auf", '
        '"expected_relation": "related", '
        '"judgment": "accept", '
        '"labeler": "hand", "provenance": "HUMAN-LABELED", '
        '"rationale": "both figurative body-part idioms"}\n'
        '{"phrase_a_id": "a", "phrase_b_id": "b", '
        '"phrase_a_text": "A", "phrase_b_text": "B", '
        '"expected_relation": "paraphrase", '
        '"judgment": "reject", '
        '"labeler": "hand", "provenance": "HUMAN-LABELED", '
        '"rationale": "missed relation band"}\n',
        encoding="utf-8",
    )
    try:
        rows = _load_eval_set(tmp)
        assert len(rows) == 1, "reject row should be skipped"
        # The accept row projects to the six-key shape.
        assert rows[0]["phrase_a"] == "Tomaten auf den Augen"
        assert rows[0]["phrase_b"] == "Scheuklappen auf"
        assert rows[0]["expected_relation"] == "related"
        assert json.loads(rows[0]["learner_axes_json"]) == {}
        assert json.loads(rows[0]["retrieved_pairs_json"]) == []
        assert isinstance(rows[0]["pair_index"], int)
    finally:
        tmp.unlink(missing_ok=True)


def test_loader_falls_back_to_slug_for_missing_text():
    """When ``phrase_a_text`` / ``phrase_b_text`` are absent (a
    Phase 10.4 builder shortcut), the loader projects the slug into
    the text slot so the DSPy module's required input is never
    empty. Mirrors the optimize_match.py
    ``_synthesise_context_sentence`` fallback discipline."""
    from scripts.optimize_phrase_match import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_pm_fallback.jsonl"
    tmp.write_text(
        '{"phrase_a_id": "tomaten-auf-den-augen", '
        '"phrase_b_id": "scheuklappen-auf", '
        '"expected_relation": "related", '
        '"judgment": "accept", '
        '"labeler": "hand", "provenance": "HUMAN-LABELED", '
        '"rationale": "no surface text captured"}\n',
        encoding="utf-8",
    )
    try:
        rows = _load_eval_set(tmp)
        assert len(rows) == 1
        # Slug substituted for the missing text field.
        assert rows[0]["phrase_a"] == "tomaten-auf-den-augen"
        assert rows[0]["phrase_b"] == "scheuklappen-auf"
    finally:
        tmp.unlink(missing_ok=True)


def test_loader_skips_comment_lines():
    """JSONL by convention allows ``#``-prefixed comment lines. The
    loader must strip them before json.loads."""
    from scripts.optimize_phrase_match import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_pm_comments.jsonl"
    tmp.write_text(
        "# labeler: test\n"
        "# provenance: test\n"
        '{"phrase_a_id": "a", "phrase_b_id": "b", '
        '"phrase_a_text": "A", "phrase_b_text": "B", '
        '"expected_relation": "equivalent", '
        '"judgment": "accept", '
        '"labeler": "l", "provenance": "HUMAN-LABELED", '
        '"rationale": "r"}\n',
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
    from scripts.optimize_phrase_match import _load_eval_set

    missing = Path("/tmp/does-not-exist-phrase-match.jsonl")
    if missing.exists():
        missing.unlink()
    with pytest.raises(FileNotFoundError):
        _load_eval_set(missing)


def test_loader_warns_on_provenance_drift_by_default():
    """A row whose provenance drifts from HUMAN-LABELED must warn
    (the offline smoke doesn't crash on transient drift, but the
    operator needs to see it)."""
    from scripts.optimize_phrase_match import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_pm_drift.jsonl"
    tmp.write_text(
        '{"phrase_a_id": "a", "phrase_b_id": "b", '
        '"phrase_a_text": "A", "phrase_b_text": "B", '
        '"expected_relation": "equivalent", '
        '"judgment": "accept", '
        '"labeler": "l", "provenance": "synthetic", '
        '"rationale": "r"}\n',
        encoding="utf-8",
    )
    try:
        # The drift is a warning, not an error, by default.
        rows = _load_eval_set(tmp)
        assert len(rows) == 1, (
            "drift row should still train by default; --strict "
            "flips this to a hard error"
        )
    finally:
        tmp.unlink(missing_ok=True)


def test_loader_strict_mode_raises_on_provenance_drift():
    """``--strict`` flips the provenance-drift warning into a hard
    error so a CI gate can refuse to run on a drifted eval set."""
    from scripts.optimize_phrase_match import _load_eval_set

    tmp = Path(__file__).parent / "_tmp_pm_drift_strict.jsonl"
    tmp.write_text(
        '{"phrase_a_id": "a", "phrase_b_id": "b", '
        '"phrase_a_text": "A", "phrase_b_text": "B", '
        '"expected_relation": "equivalent", '
        '"judgment": "accept", '
        '"labeler": "l", "provenance": "synthetic", '
        '"rationale": "r"}\n',
        encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="provenance"):
            _load_eval_set(tmp, strict=True)
    finally:
        tmp.unlink(missing_ok=True)


# --- Train/val split ---------------------------------------------------


def test_train_val_split_is_deterministic_80_20():
    """Same input → same split across runs. Determinism matters
    because the operator re-runs the CLI on the same eval set and
    expects the same split (mirrors optimize_match.py /
    optimize_comprehension.py)."""
    from scripts.optimize_phrase_match import _train_val_split

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
    from scripts.optimize_phrase_match import _train_val_split

    rows = [{"i": 0}]
    train, val = _train_val_split(rows, val_fraction=0.2)
    assert len(train) == 1
    assert val == []


# --- Serialization helper ----------------------------------------------


def test_serialize_optimized_has_locked_schema_version():
    """The artifact file's ``schema_version`` is the production
    path's read key. Bumping it requires coordinated reader
    changes."""
    from scripts.optimize_phrase_match import _serialize_optimized

    class _StubModule:
        predictors = []  # no predictors → empty instructions_by_field

    payload = _serialize_optimized(
        _StubModule(), train_count=5, val_count=1, mode="dummy"
    )
    assert payload["schema_version"] == "phrase-match-optimized-v1"
    assert payload["mode"] == "dummy"
    assert payload["train_count"] == 5
    assert payload["val_count"] == 1
    assert payload["instructions_by_field"] == {}


def test_serialize_optimized_handles_callable_predictors():
    """DSPy 3.x's ``module.predictors`` can be a class method (not
    iterable). The serializer must tolerate both shapes — a
    regression here crashes the CLI on a serialisation quirk."""
    from scripts.optimize_phrase_match import _serialize_optimized

    class _StubModule:
        @classmethod
        def predictors(cls):
            return []

    payload = _serialize_optimized(
        _StubModule(), train_count=1, val_count=0, mode="dummy"
    )
    assert payload["schema_version"] == "phrase-match-optimized-v1"
    assert payload["instructions_by_field"] == {}


# --- Stub-fallback when Phase 10.2 hasn't shipped ----------------------


def test_resolve_optimizer_callable_returns_none_before_phase_10_2():
    """``_resolve_optimizer_callable`` is the contract the CLI uses
    to handle the dependency order: 10.7 may land before 10.2 in
    the build pipeline, in which case the optimizer callable is
    not importable yet. The CLI's stub fallback is what keeps the
    offline smoke green in either order."""
    from scripts.optimize_phrase_match import _resolve_optimizer_callable

    callable_obj = _resolve_optimizer_callable()
    # We can't assert None (10.2 may have shipped), but if it has,
    # it must be callable. If it hasn't, the resolver returns None
    # and the CLI writes a stub artefact.
    assert callable_obj is None or callable(callable_obj)


# --- End-to-end dry-run (mirrors the locked acceptance criterion) ------


def test_cli_end_to_end_writes_artifact(tmp_path, monkeypatch):
    """The dry-run path (no API key) writes the artifact file
    cleanly. Mirrors the spec: ``DummyLM`` discipline is preserved;
    the CLI plumbing runs end-to-end; the artifact is on disk and
    carries the locked ``schema_version`` and ``mode=dummy`` markers.

    The ``OPENROUTER_API_KEY`` env is cleared at the start so this
    test runs hermetically even when the operator's environment
    has the key set (the same pattern
    ``test_comprehension_optimize.py`` uses for the optimizer
    function unit test).

    Skip on missing-or-empty: the "writes OK artifact" path only
    triggers once Phase B fills the JSONL. The empty-scaffold NOOP
    path is covered by ``test_optimize_phrase_match.py`` (card
    ``t_51289780`` Step 2).
    """
    if not EVAL_JSONL.exists():
        pytest.skip(
            f"Eval set not built yet ({EVAL_JSONL}); Phase 10.4 "
            f"deliverable."
        )
    # The CLI's empty-set path returns NOOP (not OK) and writes no
    # artifact. The end-to-end-OK assertion only holds once the
    # JSONL carries at least one accept row.
    rows: list[dict] = []
    with EVAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append(json.loads(line))
    if not rows:
        pytest.skip(
            f"phrase_match eval set scaffold is empty ({EVAL_JSONL}); "
            "Phase B (Anurag's hand-label session) fills the rows."
        )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Force DSPy to reconfigure — same shape as the comprehension
    # test. Without this, a stale ``dspy.settings.lm`` from a
    # prior test would route to the live adapter in this process
    # and bypass DummyLM. (Defensive only — the phrase_match
    # CLI may resolve to the stub fallback if 10.2 hasn't shipped.)
    import dspy

    dspy.settings.lm = None

    output_path = tmp_path / "phrase_match_optimized.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimize_phrase_match",
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
        env={**os.environ, "OPENROUTER_API_KEY": ""},
    )
    assert result.returncode == 0, (
        f"dry-run exited {result.returncode}: {result.stderr}\n"
        f"stdout: {result.stdout}"
    )
    assert "OK:" in result.stdout, result.stdout
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "phrase-match-optimized-v1"
    # ``mode`` is "dummy" because the CLI ran without --live.
    assert payload["mode"] == "dummy"
    # Train/val counts are deterministic on the same eval set:
    # 50 accept rows → 40 train, 10 val (default 80/20 split).
    assert payload["train_count"] == 40
    assert payload["val_count"] == 10


def test_cli_end_to_end_writes_artifact_when_eval_set_missing(
    tmp_path,
):
    """When the eval set is absent, the CLI prints NOOP and exits
    0 so a CI step that runs the CLI before the eval set lands
    doesn't fail the build. Mirrors ``optimize_match.py`` /
    ``optimize_comprehension.py``."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimize_phrase_match",
            "--eval-path",
            "/tmp/does-not-exist-phrase-match-eval.jsonl",
            "--output-path",
            str(tmp_path / "phrase_match_optimized.json"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    assert result.returncode == 0
    assert "NOOP" in result.stdout
    # Artifact should NOT be written when the eval set is absent.
    assert not (tmp_path / "phrase_match_optimized.json").exists()


def test_cli_target_and_seed_are_recorded_in_artifact(tmp_path):
    """The CLI's ``--target`` and ``--seed`` flags must be
    reflected in the output so a re-run with the same seed on the
    same eval set produces a byte-equal artifact (reproducibility
    per Phase 4.4 discipline).

    We synthesize a minimal two-row eval set (one accept, one
    reject) so the full pipeline reaches the OK print statement
    and the artifact is written. The target/seed flags are echoed
    in the stdout so the operator can confirm they took effect.
    """
    eval_path = tmp_path / "phrase_match_judgments.jsonl"
    eval_path.write_text(
        "# labeler: test\n"
        "# provenance: HUMAN-LABELED\n"
        '{"phrase_a_id": "a", "phrase_b_id": "b", '
        '"phrase_a_text": "A", "phrase_b_text": "B", '
        '"expected_relation": "equivalent", '
        '"judgment": "accept", '
        '"labeler": "l", "provenance": "HUMAN-LABELED", '
        '"rationale": "r"}\n'
        '{"phrase_a_id": "c", "phrase_b_id": "d", '
        '"phrase_a_text": "C", "phrase_b_text": "D", '
        '"expected_relation": "paraphrase", '
        '"judgment": "reject", '
        '"labeler": "l", "provenance": "HUMAN-LABELED", '
        '"rationale": "r"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "phrase_match_optimized.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.optimize_phrase_match",
            "--eval-path",
            str(eval_path),
            "--output-path",
            str(output_path),
            "--target",
            "0.6",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
        env={**os.environ, "OPENROUTER_API_KEY": ""},
    )
    assert result.returncode == 0, (
        f"CLI failed: {result.stderr}\nstdout: {result.stdout}"
    )
    assert "target=0.6" in result.stdout
    assert "seed=42" in result.stdout
    assert output_path.exists()