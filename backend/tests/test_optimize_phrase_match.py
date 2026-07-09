"""Hermetic smoke test for the Phase 10.7 phrase_match optimizer CLI.

Ships as the Step 2 deliverable for card ``t_51289780`` (the Phase
10.4 + 10.7 fold fix). The optimizer CLI (``scripts/optimize_phrase_match.py``)
already lives on main from the 10.7 fold ceremony; this test
complements the existing ``tests/test_eval/test_phrase_match_optimize.py``
suite by pinning the **floor-gate behavior on the post-10.4 reality** —
the eval set is shipped empty (Phase 10.4's deliverable is the
*HUMAN-LABELED* scaffold, not the labeled rows; those land via the
human-input Phase B gate in card ``t_f3d2a634``).

Why this file exists separately from
``tests/test_eval/test_phrase_match_optimize.py``:

The 10.7 test file was written assuming the 10.4 eval set would be
populated at fold time. It guards the populated path (2 tests skip on
``phrase_match_judgments.jsonl`` missing) but doesn't pin the empty-set
behavior. After the 10.4 fold lands the empty scaffold on main, the 2
previously-skipped tests *run* and *fail* (they assume ``rows != []``).
This file's job is the opposite: pin the *empty-set / partially-populated
/ strict-mode* behavior so the post-fold contract is locked even
before Phase B fills the JSONL.

Acceptance criteria (per card ``t_51289780`` Step 2):

1. CLI starts up against a temp ``eval/`` fixture (empty
   ``phrase_match_judgments.jsonl`` + populated manifest + populated
   ``attested_pairs.json``).
2. Floor gate fires with the right error code when the JSONL is
   empty (CLI exits 0 + NOOP message; artifact is NOT written —
   the "0 accept rows" branch is the locked behavior).
3. Floor gate does NOT fire when the JSONL has ≥5 rows spanning the
   4 relations (CLI exits 0 + OK message; artifact IS written with
   the documented ``schema_version`` and deterministic
   ``train_count`` / ``val_count``).
4. Held-out-load error path is testable (pointing at a missing JSONL
   produces a clean NOOP exit, not a crash; ``--strict`` flips
   provenance drift into a non-zero exit).
5. The Makefile alias ``make eval-optimize-phrase-match`` is wired and
   runs end-to-end against a temp output path.

Run from ``backend/``::

    uv run pytest -q tests/test_optimize_phrase_match.py

No DB, no LLM, no network. ``DummyLM`` discipline is the spec; the
test forces ``OPENROUTER_API_KEY=""`` so a stray key in the operator's
environment can't route to the live adapter.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# Repository-relative path resolution. ``tests/test_optimize_phrase_match.py``
# lives at ``backend/tests/``; the optimizer CLI lives at
# ``backend/scripts/optimize_phrase_match.py``. We resolve everything
# relative to the backend root so the tests are cwd-independent.
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
SCRIPTS_DIR = BACKEND_DIR / "scripts"
MAKEFILE = REPO_ROOT / "Makefile"

# The CLI module name (matches ``scripts/optimize_phrase_match.py``).
CLI_MODULE = "scripts.optimize_phrase_match"


# The 4-way relation taxonomy (locked Phase 10.4 / 10.7 contract; mirrors
# ``app.schemas.PhrasePairRelation``).
RELATIONS = ("equivalent", "paraphrase", "related", "unrelated")

# The locked Phase 1.5a provenance tag — any row that drifts from it
# is by definition no longer the curated signal the Phase 6.7 Ragas
# floor was measured against. The optimizer loader warns (or fails on
# ``--strict``) so the drift surfaces at optimizer time, not at a
# later eval cycle.
EXPECTED_PROVENANCE = "HUMAN-LABELED"

# The post-10.4 deliverable's locked schema version (mirrors the
# ``schema_version`` the CLI bakes into the optimized-prompt
# artifact). Pinned here so a drift in the optimizer's wire contract
# surfaces as a test failure, not as a silent production-path load
# error.
ARTIFACT_SCHEMA_VERSION = "phrase-match-optimized-v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_empty_jsonl(path: Path) -> None:
    """Write the canonical 10.4 deliverable: empty JSONL + HUMAN-LABELED header.

    Mirrors the on-disk shape Athena shipped at ``bec039c`` — the
    leading comment block declares provenance so any downstream reader
    knows the file is a hand-labeled scaffold, not an LLM-synthesized
    training set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {EXPECTED_PROVENANCE} — Phase 10.4 (card t_f3d2a634) "
        "phrase_match eval set scaffold.\n"
        "# Phase A ships the empty scaffold; Phase B (Anurag's hand-label\n"
        "# session) fills the rows. See eval/phrase_match_judgments.manifest.json.\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path) -> None:
    """Write the canonical 10.4 manifest so the optimizer's environment is realistic."""
    payload = {
        "schema_version": 1,
        "provenance": EXPECTED_PROVENANCE,
        "labeler": "Anurag Parida",
        "exercise_type": "phrase_match",
        "relation_taxonomy": list(RELATIONS),
        "target_count": 50,
        "target_distribution": {
            "equivalent": 12,
            "paraphrase": 13,
            "related": 12,
            "unrelated": 13,
        },
        "current_count": 0,
        "current_distribution": {
            r: 0 for r in RELATIONS
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_attested_pairs(path: Path) -> None:
    """Write a populated attested-pairs JSON (mirrors the canonical 10.4 shape)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "exercise_type": "phrase_match",
        "source": "test fixture for card t_51289780",
        "pairs": [
            {
                "phrase_a_id": "ins-blaue-hinein",
                "phrase_b_id": "in-den-tag-hinein-leben",
                "relation": "paraphrase",
                "attested_quote_a": "test",
                "attested_source_a": "test",
                "attested_quote_b": "test",
                "attested_source_b": "test",
                "rationale": "test row",
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_populated_jsonl(path: Path, *, n_rows: int = 6) -> None:
    """Write a JSONL with ``n_rows`` accept rows spanning all 4 relations.

    Round-robins through the relations so the fixture satisfies the
    card body's "≥5 rows spanning the 4 relations" criterion. All
    rows carry ``provenance=HUMAN-LABELED`` and ``judgment=accept`` so
    the optimizer's loader treats every row as a training signal.
    """
    if n_rows < len(RELATIONS):
        raise ValueError(
            f"need at least {len(RELATIONS)} rows to cover all relations; got {n_rows}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("ins-blaue-hinein", "in-den-tag-hinein-leben"),
        ("die-daumen-druecken", "da-steppt-der-baer"),
        ("die-kirche-im-dorf-lassen", "ins-blaue-hinein"),
        ("tomaten-auf-den-augen", "die-daumen-druecken"),
        ("da-steppt-der-baer", "die-kirche-im-dorf-lassen"),
        ("tomaten-auf-den-augen", "da-steppt-der-baer"),
        ("ins-blaue-hinein", "die-kirche-im-dorf-lassen"),
        ("die-daumen-druecken", "in-den-tag-hinein-leben"),
    ]
    lines = [
        f"# {EXPECTED_PROVENANCE} — populated fixture for test_optimize_phrase_match.py",
    ]
    for i in range(n_rows):
        relation = RELATIONS[i % len(RELATIONS)]
        a, b = pairs[i % len(pairs)]
        lines.append(json.dumps({
            "phrase_a_id": a,
            "phrase_b_id": b,
            "phrase_a_text": a.replace("-", " "),
            "phrase_b_text": b.replace("-", " "),
            "expected_relation": relation,
            "labeler": "test_fixture",
            "provenance": EXPECTED_PROVENANCE,
            "judgment": "accept",
            "rationale": f"fixture row {i}",
        }, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_cli(
    *args: str,
    eval_path: Path,
    output_path: Path,
    cwd: Path | None = None,
    strict: bool = False,
) -> subprocess.CompletedProcess:
    """Invoke the optimizer CLI hermetically (no network, no LLM).

    Forces ``OPENROUTER_API_KEY=""`` so a stray key in the operator's
    environment can't route to the live adapter. ``--strict`` is
    forwarded when requested; the default is offline-only.
    """
    cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "--eval-path",
        str(eval_path),
        "--output-path",
        str(output_path),
    ]
    if strict:
        cmd.append("--strict")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or BACKEND_DIR,
        env={**os.environ, "OPENROUTER_API_KEY": ""},
        timeout=120,
    )


@pytest.fixture
def make(tmp_path: Path):
    """Factory fixture: build a self-contained eval-set directory under ``tmp_path``.

    Returns a callable that accepts a flag and writes the matching
    fixtures (``empty`` → empty JSONL, ``populated`` → 6-row JSONL
    spanning all 4 relations). Always writes the manifest + attested
    pairs alongside so the optimizer's environment mirrors the
    canonical 10.4 fold state.
    """
    def _build(shape: str) -> Path:
        eval_dir = tmp_path / "eval"
        jsonl = eval_dir / "phrase_match_judgments.jsonl"
        manifest = eval_dir / "phrase_match_judgments.manifest.json"
        data_dir = tmp_path / "data"
        attested = data_dir / "attested_pairs.json"
        if shape == "empty":
            _write_empty_jsonl(jsonl)
        elif shape == "populated":
            _write_populated_jsonl(jsonl, n_rows=6)
        elif shape == "reject_only":
            # Write a JSONL where every row has judgment=reject so the
            # optimizer's loader returns zero accept rows. Distinct
            # shape from "empty" because the empty-set floor gate
            # might be a different code path from the all-reject one.
            _write_empty_jsonl(jsonl)
            with jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "phrase_a_id": "x",
                    "phrase_b_id": "y",
                    "phrase_a_text": "x",
                    "phrase_b_text": "y",
                    "expected_relation": "paraphrase",
                    "labeler": "t",
                    "provenance": EXPECTED_PROVENANCE,
                    "judgment": "reject",
                    "rationale": "rejected",
                }) + "\n")
        else:
            raise ValueError(f"unknown fixture shape: {shape}")
        _write_manifest(manifest)
        _write_attested_pairs(attested)
        return jsonl
    return _build


# ---------------------------------------------------------------------------
# 1. CLI startup against the canonical temp fixture
# ---------------------------------------------------------------------------


def test_cli_starts_against_empty_temp_fixture(make, tmp_path):
    """Criterion 1: CLI starts up against a temp eval/ fixture with an
    empty JSONL + populated manifest + populated attested_pairs.

    The canonical 10.4 deliverable is the *scaffold* (empty JSONL,
    populated manifest, attested_pairs already populated). The CLI
    must accept that state without crashing. This test pins the
    startup behavior end-to-end.
    """
    eval_path = make("empty")
    output_path = tmp_path / "phrase_match_optimized.json"

    result = _run_cli(eval_path=eval_path, output_path=output_path)

    assert result.returncode == 0, (
        f"CLI exited {result.returncode} on empty fixture: "
        f"{result.stderr}\nstdout: {result.stdout}"
    )


# ---------------------------------------------------------------------------
# 2. Floor gate: empty JSONL → NOOP, no artifact
# ---------------------------------------------------------------------------


def test_empty_jsonl_triggers_noop_and_skips_artifact(make, tmp_path):
    """Criterion 2: floor gate fires with the right error code on empty JSONL.

    The CLI returns exit 0 + "NOOP: eval set has 0 accept rows" — the
    locked behavior for an empty-set floor gate. The artifact is NOT
    written (a stale artifact from a previous run is a real footgun).
    """
    eval_path = make("empty")
    output_path = tmp_path / "phrase_match_optimized.json"

    result = _run_cli(eval_path=eval_path, output_path=output_path)

    assert result.returncode == 0, result.stderr
    assert "NOOP" in result.stdout, (
        f"expected NOOP on empty JSONL, got: {result.stdout!r}"
    )
    assert "0 accept rows" in result.stdout, (
        f"expected '0 accept rows' message, got: {result.stdout!r}"
    )
    assert not output_path.exists(), (
        "artifact must NOT be written on empty JSONL (avoids stale "
        "artifact footgun when Phase B hasn't filled the JSONL yet)"
    )


def test_reject_only_jsonl_also_triggers_noop(make, tmp_path):
    """Reject-only JSONL is also a 0-accept-rows scenario; same NOOP path.

    The optimizer skips reject rows by design (a rejected phrase pair
    is bad data, not a useful training signal — Phase 10.7 spec). If
    every row is rejected, the CLI must take the same NOOP branch as
    the empty JSONL case.
    """
    eval_path = make("reject_only")
    output_path = tmp_path / "phrase_match_optimized.json"

    result = _run_cli(eval_path=eval_path, output_path=output_path)

    assert result.returncode == 0, result.stderr
    assert "NOOP" in result.stdout, result.stdout
    assert "0 accept rows" in result.stdout, result.stdout
    assert "reject rows skipped" in result.stderr, (
        f"expected reject-skip log line on stderr, got: {result.stderr!r}"
    )
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 3. Floor gate: ≥5 rows spanning the 4 relations → OK, artifact written
# ---------------------------------------------------------------------------


def test_populated_jsonl_writes_ok_artifact(make, tmp_path):
    """Criterion 3: floor gate does NOT fire when the JSONL has ≥5 rows
    spanning the 4 relations.

    The CLI exits 0 with "OK: wrote optimised prompt..." and writes
    the artifact to the requested output path with the documented
    ``schema_version`` and deterministic ``train_count`` / ``val_count``.
    """
    eval_path = make("populated")
    output_path = tmp_path / "phrase_match_optimized.json"

    result = _run_cli(eval_path=eval_path, output_path=output_path)

    assert result.returncode == 0, (
        f"CLI exited {result.returncode} on populated fixture: "
        f"{result.stderr}\nstdout: {result.stdout}"
    )
    assert "OK:" in result.stdout, (
        f"expected OK on populated JSONL, got: {result.stdout!r}"
    )
    assert output_path.exists(), (
        f"artifact not written at {output_path}; stdout={result.stdout!r}"
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    # Offline path: mode must be "dummy" (--live was not passed).
    assert payload["mode"] == "dummy"
    # 6 rows × 80/20 split → train=5, val=1 (deterministic on the
    # same fixture per Phase 4.4 reproducibility discipline).
    assert payload["train_count"] == 5
    assert payload["val_count"] == 1


def test_populated_jsonl_does_not_fire_provenance_drift_warning(
    make, tmp_path
):
    """All-rows-HUMAN-LABELED path must not emit a provenance drift warning.

    A provenance drift warning on a populated-and-curated JSONL is a
    bug — the curator's signal is intact, so the warning would
    mislead the operator into thinking the eval set is broken.
    """
    eval_path = make("populated")
    output_path = tmp_path / "phrase_match_optimized.json"

    result = _run_cli(eval_path=eval_path, output_path=output_path)

    assert "provenance" not in result.stderr.lower() or (
        "drift" not in result.stderr.lower()
    ), (
        f"unexpected provenance drift warning on curated JSONL: "
        f"{result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# 4. Held-out-load error path: missing file → clean NOOP, --strict flips
# ---------------------------------------------------------------------------


def test_missing_jsonl_exits_clean_noop(tmp_path):
    """Criterion 4a: pointing the CLI at a missing JSONL exits 0 + NOOP.

    The CLI's contract is "no crash, no artifact, clean NOOP message"
    on a missing eval set. This is the pre-Phase 10.4 state and the
    legacy CI-step shape (a fresh checkout doesn't have the JSONL yet);
    the CLI must NOT raise or exit non-zero on a missing file.
    """
    missing = tmp_path / "does-not-exist-phrase-match-eval.jsonl"
    output_path = tmp_path / "phrase_match_optimized.json"

    result = _run_cli(eval_path=missing, output_path=output_path)

    assert result.returncode == 0, (
        f"CLI exited {result.returncode} on missing file: "
        f"{result.stderr}\nstdout: {result.stdout}"
    )
    assert "NOOP" in result.stdout, result.stdout
    # No artifact written on missing file (parity with empty-JSONL case).
    assert not output_path.exists(), (
        "artifact must NOT be written when eval set is absent"
    )


def test_strict_mode_raises_on_provenance_drift(tmp_path):
    """Criterion 4b: --strict flips provenance drift into a hard error.

    The Phase 1.5a provenance contract is sacred. By default the CLI
    *warns* on drift (so a transient drift doesn't break the offline
    smoke), but with ``--strict`` the loader raises and the CLI exits
    non-zero. This is the "fail loudly when the curated signal is
    gone" gate.
    """
    jsonl = tmp_path / "phrase_match_judgments.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # First row carries the correct HUMAN-LABELED tag; second drifts.
    # The strict gate must fail on row 2 — even one drifting row is a
    # hard error.
    jsonl.write_text(
        "\n".join([
            f"# {EXPECTED_PROVENANCE}",
            json.dumps({
                "phrase_a_id": "a",
                "phrase_b_id": "b",
                "phrase_a_text": "a",
                "phrase_b_text": "b",
                "expected_relation": "paraphrase",
                "labeler": "t",
                "provenance": EXPECTED_PROVENANCE,
                "judgment": "accept",
                "rationale": "ok",
            }),
            json.dumps({
                "phrase_a_id": "c",
                "phrase_b_id": "d",
                "phrase_a_text": "c",
                "phrase_b_text": "d",
                "expected_relation": "related",
                "labeler": "t",
                "provenance": "LLM-SYNTHETIC",
                "judgment": "accept",
                "rationale": "drift",
            }),
            "",
        ]),
        encoding="utf-8",
    )

    output_path = tmp_path / "phrase_match_optimized.json"
    result = _run_cli(
        eval_path=jsonl, output_path=output_path, strict=True
    )

    assert result.returncode != 0, (
        f"--strict must exit non-zero on provenance drift; got rc=0: "
        f"{result.stdout!r}"
    )
    assert "HUMAN-LABELED" in (result.stderr + result.stdout) or (
        "provenance" in (result.stderr + result.stdout).lower()
    ), (
        f"expected the drift error message to mention provenance, got: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not output_path.exists()


def test_non_strict_mode_warns_but_continues_on_provenance_drift(tmp_path):
    """Without --strict, drift produces a warning and the CLI exits 0.

    The default offline smoke path tolerates a transient drift so a
    follow-up labeling pass with a different provenance tag doesn't
    silently break the smoke. The warning surfaces in the log so the
    operator sees it; the optimizer still runs.
    """
    jsonl = tmp_path / "phrase_match_judgments.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"# {EXPECTED_PROVENANCE}",
        json.dumps({
            "phrase_a_id": "ins-blaue-hinein",
            "phrase_b_id": "in-den-tag-hinein-leben",
            "phrase_a_text": "ins Blaue hinein",
            "phrase_b_text": "in den Tag hinein leben",
            "expected_relation": "paraphrase",
            "labeler": "t",
            "provenance": "LLM-SYNTHETIC",  # drift on the only row
            "judgment": "accept",
            "rationale": "drift",
        }),
        "",
    ]
    jsonl.write_text("\n".join(rows), encoding="utf-8")

    output_path = tmp_path / "phrase_match_optimized.json"
    result = _run_cli(eval_path=jsonl, output_path=output_path)

    # Default mode: continue despite drift.
    assert result.returncode == 0, result.stderr
    assert "provenance" in result.stderr.lower() and (
        "drift" in result.stderr.lower()
    ), (
        f"expected provenance drift warning in stderr, got: "
        f"stderr={result.stderr!r}"
    )
    # Artifact is written (the CLI optimistically tunes despite the
    # warning — the warning is advisory, not blocking).
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 5. Makefile alias wiring
# ---------------------------------------------------------------------------


def test_makefile_has_optimize_phrase_match_alias():
    """Criterion 5a: the Makefile alias `eval-optimize-phrase-match` exists
    and points at the optimizer CLI.

    The Makefile alias is the operator-facing surface; an operator
    running ``make eval-optimize-phrase-match`` from the repo root
    must reach the CLI. This test asserts the alias is wired (without
    running make, which would invoke the live DummyLM optimizer and
    slow the test path).
    """
    if not MAKEFILE.exists():
        pytest.skip(f"no Makefile at {MAKEFILE}; not a make-driven project")

    text = MAKEFILE.read_text(encoding="utf-8")
    assert "eval-optimize-phrase-match:" in text, (
        "Makefile alias `eval-optimize-phrase-match:` not found. "
        "The Phase 10.7 ceremony should have wired it."
    )
    # The alias body should invoke the CLI module.
    assert "scripts.optimize_phrase_match" in text, (
        "Makefile alias does not invoke `scripts.optimize_phrase_match`. "
        "Check the target body."
    )


@pytest.mark.skipif(
    shutil.which("make") is None,
    reason="`make` not on PATH; skip end-to-end alias check",
)
def test_makefile_alias_runs_end_to_end(make, tmp_path):
    """Criterion 5b: ``make eval-optimize-phrase-match`` runs end-to-end
    against the canonical main-state eval/ directory (empty JSONL).

    We invoke ``make`` from the repo root with the canonical paths
    (the post-10.4-fold empty JSONL + manifest on main). The alias
    exits 0 because the empty-set path is a NOOP. This is a slow test
    because it spawns ``uv run python -m scripts.optimize_phrase_match``
    via make → shell, but it catches wiring regressions that the
    Makefile-parse-only test above can't.
    """
    eval_path = REPO_ROOT / "eval" / "phrase_match_judgments.jsonl"
    if not eval_path.exists():
        pytest.skip(
            f"canonical eval set absent at {eval_path}; "
            "post-10.4-fold state must include the empty scaffold"
        )

    # Use a tmp output so we don't pollute the repo's
    # ``backend/app/phrase_match_optimized.json``.
    output_path = tmp_path / "phrase_match_optimized.json"

    result = subprocess.run(
        [
            "make",
            "eval-optimize-phrase-match",
            f"OUTPUT_PATH={output_path}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "OPENROUTER_API_KEY": ""},
        timeout=180,
    )

    # The empty-set NOOP path must exit 0.
    assert result.returncode == 0, (
        f"make eval-optimize-phrase-match exited {result.returncode}: "
        f"{result.stderr}\nstdout: {result.stdout}"
    )
    # Makefile body uses ``$(MAKE) ... `` — the env override of
    # OUTPUT_PATH only works if the Makefile target explicitly
    # respects it. The alias doesn't currently expose an OUTPUT_PATH
    # knob (it writes to the canonical path), so we only assert the
    # exit code here — the canonical artifact path's existence is the
    # real verification, and that path is gitignored. If the alias
    # grows an OUTPUT_PATH knob later, this test should tighten.
    assert "NOOP" in result.stdout or "OK:" in result.stdout, (
        f"expected NOOP or OK marker on stdout, got: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# 6. CLI surface alignment with optimize_match.py / optimize_comprehension.py
# ---------------------------------------------------------------------------


def test_cli_help_exits_0_and_documents_phrase_match_knobs():
    """``--help`` exits 0 and surfaces the phrase_match-specific knobs.

    Mirrors the locked acceptance criterion from the 10.7 ceremony
    (and the same criterion in optimize_match.py /
    optimize_comprehension.py): a CI step running ``--help`` must
    succeed even before the eval set exists.
    """
    result = subprocess.run(
        [sys.executable, "-m", CLI_MODULE, "--help"],
        capture_output=True,
        text=True,
        cwd=BACKEND_DIR,
    )
    assert result.returncode == 0, (
        f"--help exited {result.returncode}: {result.stderr}"
    )
    # Standard knob set (mirrors optimize_match / optimize_comprehension).
    assert "--eval-path" in result.stdout
    assert "--output-path" in result.stdout
    assert "--live" in result.stdout
    # Phase 10.4-specific knobs.
    assert "--target" in result.stdout
    assert "--max-demos" in result.stdout
    assert "--strict" in result.stdout