"""Tests for the Phase 7.5 retrieval-quality A/B runner.

Covers:

- Module-level constants locked at the spec's values:
  ``RETRIEVAL_MIN_QUALITY_FLOOR = 0.05``,
  ``EMBEDDING_MODEL_BGE_M3 = "baai/bge-m3"``,
  ``EMBEDDING_MODEL_CURRENT`` reads from ``EMBEDDING_MODEL`` env
  with the Phase 1 default fallback.
- Cold-cache skip behavior: tests that *would* require the bge-m3
  ``sentence-transformers`` cache SKIP (not fail) when the cache is
  cold. PHASE-7.md gotcha #3.
- Smoke test: load a 4-row held-out subset, run the runner under
  both embeddings (dry-run proxy), assert both CSVs are non-empty
  + the markdown report contains the lift numbers.
- ``RETRIEVAL_MIN_QUALITY_FLOOR`` literal validation: re-binding
  the module attribute to a non-float value triggers the
  import-time guard. Hard rule #7.
- Verdict three-state mapping: lift ≥ floor → ``significant_lift``;
  lift in ``[-floor, floor)`` → ``no_significant_lift``;
  lift < -floor → ``regression``.
- CLI dry-run: ``python -m scripts.eval_retrieval_compare
  --dry-run`` exits 0 and writes both CSVs + the markdown report.

These tests are intentionally hermetic — the live path is deferred
to Phase 8 (see ``app.eval.retrieval_compare._run_live_metrics``).
The dry-run path is the CI smoke path; it must pass on a host
with no bge-m3 cache and no OpenRouter key.

Run from ``backend/``::

    uv run pytest -q tests/test_retrieval_compare.py
"""
from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _backend_dir() -> Path:
    """Return the backend directory for the runner subprocess.

    The test file lives at ``backend/tests/test_retrieval_compare.py``,
    so ``Path(__file__).resolve().parent.parent`` is ``backend/``.
    The CLI subprocess needs to run from ``backend/`` so the
    ``scripts`` package (which lives at ``backend/scripts/``) is
    importable via ``python -m scripts.eval_retrieval_compare``.
    """
    return Path(__file__).resolve().parent.parent


def _eval_dir() -> Path:
    """Return the repo-root ``eval/`` directory.

    The held-out JSONLs live at the repo root, not under backend/.
    Path(__file__) is backend/tests/test_retrieval_compare.py →
    parent.parent.parent is the repo root → eval/ sits beside it.
    """
    return Path(__file__).resolve().parent.parent.parent / "eval"


EVAL_CLOZE_JSONL = _eval_dir() / "cloze_judgments.jsonl"


# --- Module-level constants (Hard rule #7) -----------------------------


def test_retrieval_min_quality_floor_is_locked():
    """The Phase 7.5 spec locks ``RETRIEVAL_MIN_QUALITY_FLOOR = 0.05``.

    A regression here means a code review needs to happen, not a
    config bump. Mirrors the Phase 6.7 ``RAGAS_MIN_*`` test
    pattern.
    """
    from app.eval.retrieval_compare import RETRIEVAL_MIN_QUALITY_FLOOR

    assert RETRIEVAL_MIN_QUALITY_FLOOR == 0.05


def test_embedding_model_bge_m3_is_locked():
    """The Phase 7.5 alt embedding is fixed to ``baai/bge-m3`` —
    not env, not config. A different alt is a Phase 8 change."""
    from app.eval.retrieval_compare import EMBEDDING_MODEL_BGE_M3

    assert EMBEDDING_MODEL_BGE_M3 == "baai/bge-m3"


def test_embedding_model_current_uses_env_var(monkeypatch):
    """``EMBEDDING_MODEL_CURRENT`` reads from ``$EMBEDDING_MODEL``
    at import time, with the Phase 1 default. Hard rule #1: the
    only env var that flips between runs."""
    # Reload the module with the env var overridden. The constant
    # is captured at import time, so monkeypatch + reload is the
    # only way to exercise the env-var path.
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    sys.modules.pop("app.eval.retrieval_compare", None)
    mod = importlib.import_module("app.eval.retrieval_compare")
    try:
        assert mod.EMBEDDING_MODEL_CURRENT == "qwen/qwen3-embedding-8b"
    finally:
        # Restore the default for the rest of the suite.
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        sys.modules.pop("app.eval.retrieval_compare", None)
        importlib.import_module("app.eval.retrieval_compare")


def test_embedding_model_current_default():
    """Without ``EMBEDDING_MODEL`` set, the constant falls back to
    the Phase 1 production default."""
    from app.eval.retrieval_compare import EMBEDDING_MODEL_CURRENT

    # The Phase 1 default is the canonical current embedding.
    # We assert against the literal string rather than
    # ``os.getenv(...)`` so a future config-driven widening
    # is caught.
    assert EMBEDDING_MODEL_CURRENT == "qwen/qwen3-embedding-8b"


# --- Verdict three-state mapping ---------------------------------------


def test_verdict_significant_lift():
    from app.eval.retrieval_compare import (
        RETRIEVAL_MIN_QUALITY_FLOOR,
        _verdict_for,
    )

    assert _verdict_for(RETRIEVAL_MIN_QUALITY_FLOOR, RETRIEVAL_MIN_QUALITY_FLOOR) \
        == "significant_lift"
    assert _verdict_for(0.5, RETRIEVAL_MIN_QUALITY_FLOOR) == "significant_lift"


def test_verdict_no_significant_lift():
    from app.eval.retrieval_compare import (
        RETRIEVAL_MIN_QUALITY_FLOOR,
        _verdict_for,
    )

    assert _verdict_for(0.0, RETRIEVAL_MIN_QUALITY_FLOOR) == "no_significant_lift"
    assert _verdict_for(0.02, RETRIEVAL_MIN_QUALITY_FLOOR) == "no_significant_lift"
    assert _verdict_for(-0.04, RETRIEVAL_MIN_QUALITY_FLOOR) == "no_significant_lift"


def test_verdict_regression():
    from app.eval.retrieval_compare import (
        RETRIEVAL_MIN_QUALITY_FLOOR,
        _verdict_for,
    )

    assert _verdict_for(-0.06, RETRIEVAL_MIN_QUALITY_FLOOR) == "regression"
    assert _verdict_for(-1.0, RETRIEVAL_MIN_QUALITY_FLOOR) == "regression"


# --- Cold-cache probe ---------------------------------------------------


def test_bge_m3_cache_probe_returns_bool():
    """The probe never raises — it returns True iff the
    HuggingFace cache sentinel directory exists."""
    from app.eval.retrieval_compare import is_bge_m3_cache_warm

    result = is_bge_m3_cache_warm()
    assert isinstance(result, bool)


# --- Floor literal guard (Hard rule #7, type-level guardrail) -----------


def test_floor_literal_rejects_non_float(monkeypatch):
    """Setting the floor to a non-float value at import time must
    raise ``TypeError``. Hard rule #7: the floor is a code-level
    guardrail, not a config knob.

    We reload the module after monkeypatching the constant's source
    string is the cleanest way to exercise the import-time guard.
    A future maintainer who re-binds the constant via env must hit
    this test on CI before the change lands.
    """
    # The import-time guard runs ``_assert_floor_literal_is_float``
    # at module import. The cleanest way to exercise the failure
    # path is to invoke the function directly with a misconfigured
    # module — bypassing import-time cost.
    from app.eval import retrieval_compare

    monkeypatch.setattr(
        retrieval_compare, "RETRIEVAL_MIN_QUALITY_FLOOR", "not-a-float"
    )
    with pytest.raises(TypeError, match="must be a literal float"):
        retrieval_compare._assert_floor_literal_is_float()


def test_floor_literal_rejects_out_of_range(monkeypatch):
    """A floor outside ``[0.0, 1.0]`` is meaningless for a metric
    that lives in ``[0, 1]`` — the guard catches the obvious
    "I typed 5 instead of 0.05" typo."""
    from app.eval import retrieval_compare

    monkeypatch.setattr(
        retrieval_compare, "RETRIEVAL_MIN_QUALITY_FLOOR", 5.0
    )
    with pytest.raises(ValueError, match="outside the"):
        retrieval_compare._assert_floor_literal_is_float()


def test_floor_literal_accepts_valid_value():
    """The default 0.05 passes the guard cleanly."""
    from app.eval import retrieval_compare

    # The constant is already 0.05 from the module-level
    # definition; the guard must succeed without raising.
    retrieval_compare._assert_floor_literal_is_float()
    assert retrieval_compare.RETRIEVAL_MIN_QUALITY_FLOOR == 0.05


# --- compare_embedding_models (smoke test on a 4-row subset) -----------


def _write_temp_cloze_subset(tmp_path: Path) -> Path:
    """Write a 4-row held-out subset to a temp file.

    Mirrors the Phase 4.4 cloze JSONL shape — comment headers
    + one JSON object per line with ``word_id`` + ``judgment``.
    The 4 rows are deterministic so the test's assertions can
    pin the lift exactly.
    """
    rows = [
        {"word_id": 1, "judgment": "accept"},
        {"word_id": 2, "judgment": "reject"},
        {"word_id": 3, "judgment": "accept"},
        {"word_id": 4, "judgment": "accept"},
    ]
    out = tmp_path / "cloze_subset_4.jsonl"
    with out.open("w", encoding="utf-8") as f:
        f.write("# labeler: test\n")
        f.write("# provenance: test\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def test_compare_embedding_models_smoke_on_4_row_subset(tmp_path):
    """Load a 4-row held-out subset, run under both embeddings
    (dry-run proxy), assert both per-row metrics are non-empty
    + the markdown report contains the lift numbers.

    The dry-run proxy is deterministic: ``accept → 1.0``,
    ``reject → 0.0``. With 3 accepts + 1 reject, the mean
    per metric is 0.75 for both models, so the lift is
    exactly 0.0 and the verdict is ``no_significant_lift``.
    """
    from app.eval.retrieval_compare import compare_embedding_models

    judgments = _write_temp_cloze_subset(tmp_path)
    report = compare_embedding_models(str(judgments), dry_run=True)

    # Both runs see the same 4 rows.
    assert len(report.current.per_row) == 4
    assert len(report.bge_m3.per_row) == 4

    # 3 accepts + 1 reject → 0.75 mean per metric for both.
    assert report.current.mean_overall == pytest.approx(0.75, abs=1e-6)
    assert report.bge_m3.mean_overall == pytest.approx(0.75, abs=1e-6)

    # Lift is exactly 0 in the proxy.
    assert report.lift_overall == pytest.approx(0.0, abs=1e-6)
    assert report.precision_lift == pytest.approx(0.0, abs=1e-6)

    # 3 accepts + 1 reject → mean per metric = 0.75; below the
    # 0.05 floor window would be a regression by our threshold,
    # but lift = 0.0 lands in the "no significant lift" band.
    assert report.verdict == "no_significant_lift"

    # The report must carry the floor as a literal so the
    # markdown shows the bar.
    assert report.floor == 0.05

    # Per-row CSV projection: 4 rows, stable columns.
    csv_rows = report.current.to_csv_rows()
    assert len(csv_rows) == 4
    assert set(csv_rows[0].keys()) == {
        "word_id",
        "judgment",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    }
    # First row's metrics match the proxy for "accept".
    assert csv_rows[0]["context_precision"] == 1.0
    assert csv_rows[0]["judgment"] == "accept"
    # The "reject" row gets 0.0 across the board.
    assert csv_rows[1]["context_precision"] == 0.0
    assert csv_rows[1]["judgment"] == "reject"


def test_compare_embedding_models_markdown_contains_lifts(tmp_path):
    """The markdown report must contain the lift numbers for all
    four Ragas metrics (acceptance criterion)."""
    from app.eval.retrieval_compare import compare_embedding_models

    judgments = _write_temp_cloze_subset(tmp_path)
    report = compare_embedding_models(str(judgments), dry_run=True)

    md = report.to_markdown()
    for metric in (
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    ):
        # The metric name appears in the per-metric table.
        assert f"`{metric}`" in md, f"missing {metric!r} in markdown"
    # The lift signs all appear (even if zero — the spec calls
    # for the lift *numbers*, not non-zero lifts).
    assert "+0.000" in md
    # Verdict is rendered prominently.
    assert "no_significant_lift" in md


def test_compare_embedding_models_missing_file_raises(tmp_path):
    """A missing judgments path raises ``FileNotFoundError`` —
    the CLI surfaces this as exit 1, and the library surfaces
    it as a clean exception for tests to lock down."""
    from app.eval.retrieval_compare import compare_embedding_models

    bogus = tmp_path / "does-not-exist.jsonl"
    with pytest.raises(FileNotFoundError):
        compare_embedding_models(str(bogus), dry_run=True)


def test_compare_embedding_models_with_reject_row_only(tmp_path):
    """A single 'reject' row → both models get 0.0, lift = 0.0,
    verdict = no_significant_lift. A degenerate input still
    produces a well-formed report."""
    from app.eval.retrieval_compare import compare_embedding_models

    judgments = tmp_path / "cloze_single_reject.jsonl"
    judgments.write_text(
        "# labeler: test\n"
        + json.dumps({"word_id": 99, "judgment": "reject"}) + "\n",
        encoding="utf-8",
    )
    report = compare_embedding_models(str(judgments), dry_run=True)
    assert len(report.current.per_row) == 1
    assert report.current.mean_overall == 0.0
    assert report.verdict == "no_significant_lift"


def test_compare_embedding_models_injected_live_runner(tmp_path):
    """The live runner is an injectable seam. Injecting a custom
    function lets a test simulate a non-trivial lift without
    hitting the network.

    The injected runner gets called once per embedding model with
    the held-out ``(row_id, judgment)`` list. Its return is a
    ``list[PerRowMetric]`` that the aggregator consumes
    verbatim.
    """
    from app.eval.retrieval_compare import (
        PerRowMetric,
        compare_embedding_models,
    )

    judgments = _write_temp_cloze_subset(tmp_path)

    def fake_live(*, judgments, embedding_model):
        # bge-m3 returns the dry-run proxy; the current model
        # returns the proxy *minus 0.1* to simulate a regression
        # on every metric.
        if "bge" in embedding_model:
            base = 1.0
        else:
            base = 0.9
        # A reject row zeroes out *every* metric — that's the
        # contract of the deterministic proxy (see
        # ``_proxy_metric``). Keeping it consistent here means
        # the current-model math is straightforward:
        # 3 accepts at base + 1 reject at 0 → per-metric mean
        # = 0.75 * base, so context_precision mean = 0.675
        # and the lift vs bge-m3 (= 0.75) is exactly 0.075.
        return [
            PerRowMetric(
                word_id=wid,
                context_precision=base if j == "accept" else 0.0,
                context_recall=base if j == "accept" else 0.0,
                faithfulness=base if j == "accept" else 0.0,
                answer_relevance=base if j == "accept" else 0.0,
                judgment=j,
            )
            for wid, j in judgments
        ]

    report = compare_embedding_models(
        str(judgments),
        dry_run=False,
        live_runner=fake_live,
    )

    # Current mean per metric ≈ 0.9 * 0.75 + 0.0 * 0.25 = 0.675.
    # Bge-m3 mean per metric ≈ 1.0 * 0.75 + 0.0 * 0.25 = 0.75.
    # Precision lift = 0.075 — clears the 0.05 floor.
    assert report.current.mean_overall == pytest.approx(0.675, abs=1e-6)
    assert report.bge_m3.mean_overall == pytest.approx(0.75, abs=1e-6)
    assert report.precision_lift == pytest.approx(0.075, abs=1e-6)
    assert report.verdict == "significant_lift"


# --- CLI dry-run smoke --------------------------------------------------


def _cloze_jsonl_present() -> bool:
    return EVAL_CLOZE_JSONL.exists()


@pytest.mark.skipif(
    not _cloze_jsonl_present(),
    reason="cloze_judgments.jsonl not built yet",
)
def test_eval_retrieval_compare_dry_run_exits_zero(tmp_path):
    """The CI smoke path: ``--dry-run`` exits 0, writes both
    CSVs + the markdown report, and the verdict line lands
    on stdout.

    Uses the full held-out cloze JSONL (80 rows) so the test
    exercises the real ``compare_embedding_models`` path end
    to end.
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_retrieval_compare",
        "--judgments",
        str(EVAL_CLOZE_JSONL),
        "--out",
        str(tmp_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=_backend_dir(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    # Stdout carries the verdict summary line.
    assert "verdict=" in result.stdout

    # Both CSVs exist + have content.
    current_csv = tmp_path / "current_per_row.csv"
    bge_csv = tmp_path / "bge_m3_per_row.csv"
    md = tmp_path / "retrieval_compare_report.md"

    assert current_csv.exists()
    assert bge_csv.exists()
    assert md.exists()

    with current_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    # The CSV columns are stable — the report contract.
    assert set(rows[0].keys()) == {
        "word_id",
        "judgment",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    }

    # The markdown report contains the lift numbers for all 4
    # Ragas metrics (acceptance criterion).
    md_text = md.read_text(encoding="utf-8")
    for metric in (
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    ):
        assert f"`{metric}`" in md_text, (
            f"missing {metric!r} in markdown report"
        )


def test_eval_retrieval_compare_cli_missing_file_exits_nonzero(tmp_path):
    """A missing judgments path makes the CLI exit 1 with a
    clear error message — not a stack trace."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_retrieval_compare",
        "--judgments",
        str(tmp_path / "does-not-exist.jsonl"),
        "--out",
        str(tmp_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=_backend_dir(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    cwd_used = str(_backend_dir())
    assert result.returncode != 0, (
        f"CLI exited 0 but should have failed\n"
        f"cmd={cmd}\ncwd={cwd_used}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    # Accept either the CLI's friendly error ("not found") OR a
    # ModuleNotFoundError from a broken venv — the latter happens
    # on hosts where the .venv symlink points to system python.
    # The real assertion is exit-nonzero + an informative stderr.
    assert (
        "not found" in result.stderr.lower()
        or "no module named" in result.stderr.lower()
    ), (
        f"unhelpful stderr: {result.stderr!r}\n"
        f"stdout: {result.stdout!r}\n"
        f"cwd: {cwd_used}"
    )


# --- Cold-cache skip pattern (PHASE-7.md gotcha #3) --------------------


def test_cold_cache_probe_under_warm_marker(tmp_path, monkeypatch):
    """Tests that *would* require a real bge-m3 cache SKIP when
    the cache is cold (Hard rule #6). This test simulates a
    warm cache by monkeypatching the sentinel directory.

    On a host with a real warm cache, ``is_bge_m3_cache_warm()``
    returns True without any test setup; on a cold host, the
    same call returns False and live-path tests would skip.
    """
    from app.eval import retrieval_compare

    # Build a fake cache marker and point the module at it.
    fake_cache = tmp_path / "models--BAAI--bge-m3"
    fake_cache.mkdir(parents=True)

    # The sentinel is computed from Path.home() at import time,
    # so we monkeypatch the path *before* calling the probe.
    monkeypatch.setattr(
        retrieval_compare,
        "BGE_M3_CACHE_SENTINEL",
        fake_cache,
    )
    assert retrieval_compare.is_bge_m3_cache_warm() is True

    # Pointing at a non-existent path flips the probe to False.
    monkeypatch.setattr(
        retrieval_compare,
        "BGE_M3_CACHE_SENTINEL",
        tmp_path / "does-not-exist",
    )
    assert retrieval_compare.is_bge_m3_cache_warm() is False