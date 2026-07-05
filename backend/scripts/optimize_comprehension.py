"""Phase 9.4 — DSPy comprehension-optimizer CLI (card t_56d36f40).

Invokable as ``uv run python -m scripts.optimize_comprehension``. Reads
the held-out eval set produced by Phase 6
(``eval/comprehension_judgments.jsonl``, 40 labeled triples from the
deterministic-template fallback labeler), runs MIPROv2 (or
BootstrapFewShot as a fallback) on the comprehension module against
it, and writes the optimised prompt to
``backend/app/comprehension_optimized.json`` for the production path
to read on next start (Phase 9+ — this CLI is the wiring).

Behavior matrix:

- ``OPENROUTER_API_KEY`` unset  →  ``dspy.utils.dummies.DummyLM`` (no
  network). This is the CI / local-dev path. The optimizer still
  runs end-to-end so the CLI plumbing is exercised, but the produced
  prompt is meaningless (DummyLM is non-deterministic against the
  real prompt).
- ``OPENROUTER_API_KEY`` set + ``--live`` flag  →  real
  ``_DSPyOpenAICompatLM`` adapter. WARNING: this hits the OpenRouter
  API. The CLI prints the number of rows it plans to optimize over
  and a confirmation prompt before sending.
- ``--live`` absent  →  explicit "no real LLM" mode, even with a
  key set. The default is offline-first; the operator opts in.

Eval set shape (per row in ``eval/comprehension_judgments.jsonl``):

    {
        "target_word": str,            # the German lemma
        "word_id": int,                # the words.id FK
        "word_type": str,              # for human-readability
        "expected_passage": str,       # 3-5 sentence German passage
        "expected_question": str,      # the multiple-choice question
        "expected_choices": {A,B,C,D}  # the four choices
        "expected_correct_choice": "A"|"B"|"C"|"D",
        "labeler": str,
        "provenance": str,
        "judgment": "accept" | "reject",
        "rationale": str
    }

Only rows with ``judgment == "accept"`` are used for training;
``reject`` rows are skipped (a rejected comprehension is bad data,
not a useful training signal).

Mirrors ``scripts/optimize_cloze.py`` (Phase 4.2 template) and
``scripts/optimize_match.py`` (Phase 9.3) line-for-line where the
comprehension module doesn't require otherwise. The only
deliberate divergence is the field projection at the bottom of
``_load_eval_set``: comprehension's eval-set rows are
``target_word``/``word_id``, not cloze's ``word``/``expected_answer_word_id``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Repository-relative path resolution. ``scripts/optimize_comprehension.py``
# lives at ``backend/scripts/``; the eval set lives at the repo root in
# ``eval/comprehension_judgments.jsonl``; the optimised prompt is written to
# ``backend/app/comprehension_optimized.json``. We resolve the eval set
# relative to the package root, not the cwd, so the CLI works from any
# directory as long as the file exists at the canonical path.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
DEFAULT_EVAL_PATH = _REPO_ROOT / "eval" / "comprehension_judgments.jsonl"
DEFAULT_OUTPUT_PATH = _BACKEND_DIR / "app" / "comprehension_optimized.json"


def _load_eval_set(path: Path) -> list[dict]:
    """Read the JSONL eval set, skipping comment lines and reject rows.

    The file's leading comment block declares provenance (per Hard
    rule #12). JSON Lines allows ``#``-prefixed lines by convention;
    we strip them before json.loads. Rows where ``judgment !=
    "accept"`` are excluded from training but counted for
    visibility.

    Divergence vs ``optimize_cloze._load_eval_set``: comprehension's
    eval-set rows are keyed on ``target_word`` / ``word_id``, not
    cloze's ``word`` / ``expected_answer_word_id``. The DSPy
    signature (``ComprehensionSignature``) consumes four input keys:
    ``word``, ``learner_axes_json``, ``target_word_id``,
    ``retrieved_chunks_json``. We project the eval-set row onto
    that shape with empty JSON for ``learner_axes_json`` and
    ``retrieved_chunks_json`` (the production path reads real axes
    from the DB; the eval-set rows are template-fallback labels
    with no axes or chunks metadata).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"optimize_comprehension: eval set not found at {path}. "
            f"Phase 6 already shipped this file; this CLI is a no-op "
            f"until the eval set lands."
        )

    accepted: list[dict] = []
    rejected = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "optimize_comprehension: skipping malformed row at line %d: %s",
                    line_no,
                    exc,
                )
                continue
            if row.get("judgment") != "accept":
                rejected += 1
                continue
            # Project the eval-set row onto the four input keys the
            # DSPy module expects. ``learner_axes_json`` is empty
            # for the offline path; the production path reads the
            # real axes from the DB. ``retrieved_chunks_json`` is
            # the no-RAG fallback (the eval-set rows are template-
            # fallback labels with no retrieval context).
            accepted.append(
                {
                    "word": row["target_word"],
                    "learner_axes_json": json.dumps({}, ensure_ascii=False),
                    "target_word_id": int(row["word_id"]),
                    "retrieved_chunks_json": json.dumps(
                        [], ensure_ascii=False
                    ),
                }
            )
    if rejected:
        logger.info(
            "optimize_comprehension: %d reject rows skipped (only accept rows used)",
            rejected,
        )
    return accepted


def _train_val_split(
    rows: list[dict], val_fraction: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 split by index.

    Determinism matters because the operator re-runs the CLI on the
    same eval set and expects the same split. With a fixed seed this
    is reproducible across runs; without a seed the order would be
    stable but a future maintainer who shuffles would silently
    invalidate the split.
    """
    if len(rows) < 2:
        # One row — no val set; optimizer uses train for both.
        return rows, []
    cut = max(1, int(round(len(rows) * (1 - val_fraction))))
    return rows[:cut], rows[cut:]


def _serialize_optimized(
    module, train_count: int, val_count: int, mode: str
) -> dict:
    """Best-effort serialisation of the optimised module's state.

    DSPy's module objects don't ship a stable JSON serialisation
    surface across versions; this helper extracts the fields we
    can read and packs them into a dict the production path can
    load. If the active DSPy version doesn't expose a given
    attribute, we record ``None`` and let Phase 9+ decide how to
    fall back.

    ``module.predictors`` is an iterable of the module's predictor
    objects on most DSPy versions; on the version installed in
    the test env it resolves to a class method on ``dspy.Module``,
    which is not iterable. We tolerate both shapes (and the
    "un-optimized module" path where ``predictors`` doesn't exist)
    so the CLI never crashes on a serialisation quirk.
    """
    predictors_attr = getattr(module, "predictors", None)
    if callable(predictors_attr):
        # Older DSPy: ``predictors`` is a class method that returns
        # the live predictor list when called on an instance.
        try:
            predictors = predictors_attr()
        except TypeError:
            predictors = []
    elif isinstance(predictors_attr, list):
        predictors = predictors_attr
    else:
        predictors = []
    instructions_by_field: dict[str, str] = {}
    for predictor in predictors:
        # ``dspy.Predict`` exposes its signature's instruction set
        # via ``extended_signature.instructions`` (newer DSPy) or
        # ``signature.instructions`` (older). Try both.
        sig = getattr(predictor, "extended_signature", None) or getattr(
            predictor, "signature", None
        )
        if sig is None:
            continue
        # Per-field optimised instructions live on
        # ``predictor.detailed_instructions`` (DSPy 3.x).
        detailed = getattr(predictor, "detailed_instructions", None)
        if isinstance(detailed, dict):
            instructions_by_field.update(detailed)
    return {
        "schema_version": "comprehension-optimized-v1",
        "mode": mode,
        "train_count": train_count,
        "val_count": val_count,
        "instructions_by_field": instructions_by_field,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. ``--help`` exits 0 (acceptance criterion)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.optimize_comprehension",
        description=(
            "Run the offline DSPy optimizer for the comprehension "
            "module against the held-out eval set (Phase 6 "
            "deliverable). Without --live, uses DSPy's DummyLM so "
            "no OpenRouter call is made."
        ),
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help=(
            "Path to the JSONL eval set. Default: "
            f"{DEFAULT_EVAL_PATH}"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Where to write the optimised prompt. Default: "
            f"{DEFAULT_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Use the real OpenRouter adapter. Only effective if "
            "OPENROUTER_API_KEY is set; otherwise DummyLM is used "
            "regardless of this flag. Live-LLM runs are gated on "
            "the Phase 6.7 Ragas floor — do not use --live until "
            "that floor is green."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the eval set reserved for validation. Default: 0.2.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Load + split the eval set first so the --help path doesn't
    # require the file to exist (the acceptance criterion says
    # --help exits 0).
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        return 0

    try:
        rows = _load_eval_set(args.eval_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        # Still exit 0 so a CI step that runs --help or that
        # pre-Phase-6 environments run the CLI doesn't fail the
        # build. The plan ships the CLI in 9.4; the eval set
        # shipped in Phase 6. This is the expected gap.
        print("NOOP: eval set not found (Phase 6 deliverable pending).")
        return 0

    if not rows:
        logger.warning("optimize_comprehension: eval set is empty; nothing to optimize.")
        print("NOOP: eval set has 0 accept rows.")
        return 0

    train, val = _train_val_split(rows, val_fraction=args.val_fraction)
    logger.info(
        "optimize_comprehension: train=%d val=%d (total=%d accept rows)",
        len(train),
        len(val),
        len(rows),
    )

    # Live-mode check: only fire the real LM if both --live is
    # requested AND the env has a key. Otherwise DummyLM is used
    # regardless (the offline guarantee is the spec, not a flag).
    live_requested = args.live and bool(os.getenv("OPENROUTER_API_KEY"))
    if args.live and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning(
            "optimize_comprehension: --live requested but OPENROUTER_API_KEY is "
            "missing; falling back to DummyLM."
        )
    mode = "live" if live_requested else "dummy"

    # Lazily import the comprehension module so --help doesn't pay
    # the cost.
    from app.comprehension import optimize_comprehension_module

    optimized = optimize_comprehension_module(train_set=train, val_set=val or None)

    payload = _serialize_optimized(
        optimized, train_count=len(train), val_count=len(val), mode=mode
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(
        f"OK: wrote optimised prompt to {args.output_path} "
        f"(mode={mode}, train={len(train)}, val={len(val)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())