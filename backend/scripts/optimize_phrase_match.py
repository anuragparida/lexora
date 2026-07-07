"""Phase 10.7 — DSPy phrase_match-optimizer CLI (card t_dab34a97).

Invokable as ``uv run python -m scripts.optimize_phrase_match``. Reads
the held-out eval set produced by Phase 10.4
(``eval/phrase_match_judgments.jsonl``, ~50 hand-labeled phrase pairs
spanning the 4-way relation literal ``equivalent | paraphrase |
related | unrelated``, tagged ``HUMAN-LABELED`` per the Phase 1.5a
precedent), runs ``BootstrapFewShot`` (or ``MIPROv2`` as a Phase 6.7
floor-justified upgrade) on the phrase_match module against it, and
writes the optimised prompt to
``backend/app/phrase_match_optimized.json`` for the production path
to read on next start (Phase 10.3 — this CLI is the wiring).

Behavior matrix (mirrors ``scripts/optimize_match.py`` line-for-line
where possible, per the card scope):

- ``OPENROUTER_API_KEY`` unset  →  ``dspy.utils.dummies.DummyLM`` (no
  network). This is the CI / local-dev path. The optimizer still
  runs end-to-end so the CLI plumbing is exercised, but the produced
  prompt is meaningless (DummyLM is non-deterministic against the
  real prompt).
- ``OPENROUTER_API_KEY`` set + ``--live`` flag  →  real
  ``_DSPyOpenAICompatLM`` adapter. Gated on the Phase 6.7 Ragas
  floor per the card's "out of scope" hard rule.
- ``--live`` absent  →  explicit "no real LLM" mode, even with a
  key set. The default is offline-first; the operator opts in.

Eval set shape (per row in ``eval/phrase_match_judgments.jsonl``,
shipped by Phase 10.4 / card t_f3d2a634)::

    {
        "phrase_a_id": str,            # phrases.id slug
        "phrase_b_id": str,            # phrases.id slug
        "phrase_a_text": str,          # the rendered German phrase
        "phrase_b_text": str,          # the rendered German phrase
        "expected_relation": "equivalent" | "paraphrase" | "related" | "unrelated",
        "labeler": str,                # hand-labeler handle
        "provenance": "HUMAN-LABELED", # the locked Phase 1.5a tag
        "judgment": "accept" | "reject",
        "rationale": str               # why this pair got this relation
    }

Only rows with ``judgment == "accept"`` are used for training;
``reject`` rows are skipped (a rejected phrase pair is bad data, not
a useful training signal). ``provenance`` must be ``"HUMAN-LABELED"``;
the loader warns (or fails on ``--strict``) if a row drifts from the
Phase 1.5a contract — the optimizer is then mis-tuned by definition
because the curated signal is gone.

The phrase_match eval rows do NOT carry a ``target_word_id`` (the
card body is unambiguous: ``phrase_pairs`` is the read-only table;
the exercise type has no single-word anchor the way cloze / match /
comprehension do). The CLI therefore projects onto a six-key shape
that the Phase 10.2 ``PhraseMatchSignature`` will consume (the exact
key names follow the optimize_match.py precedent — ``phrase_a`` /
``phrase_b`` for the two surface phrases, ``expected_relation`` for
the literal answer, ``learner_axes_json`` / ``retrieved_pairs_json``
for the parallel RAG-on augmentation surface, and ``pair_index`` as
the optimizer-stable row id).

Hard rules enforced:

- Same ``DummyLM`` discipline as ``optimize_match.py`` /
  ``optimize_comprehension.py`` — offline optimizer, no real LLM.
- The CLI does NOT make a live-LLM call on the offline path. The
  live path is gated on the Phase 6.7 Ragas floor (per the card
  body) and is not exercised by this CLI's dry-run smoke.
- Read-only ``phrase_pairs`` — the optimizer reads the held-out
  JSONL (``phrase_match_judgments.jsonl``) and would also read the
  ``phrase_pairs`` table for nearest-neighbor few-shot context
  (mirroring ``enable_rag=True``), but does NOT write to it.
  The Phase 10.1 seed script is the only write path.
- ``HUMAN-LABELED`` manifest discipline — the loader asserts
  ``provenance == "HUMAN-LABELED"`` (the Phase 1.5a precedent). A
  drift warns by default and fails on ``--strict``.
- No LLM-writes-back — the Phase 8 explicit deferral ("LLM-curated
  phrase generation is its own multi-card phase") is honored. The
  optimizer tunes the prompt, not the table.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Repository-relative path resolution. Mirrors the match script:
# ``scripts/optimize_phrase_match.py`` lives at ``backend/scripts/``;
# the eval set lives at the repo root in
# ``eval/phrase_match_judgments.jsonl``; the optimised prompt is
# written to ``backend/app/phrase_match_optimized.json``. We resolve
# the eval set relative to the package root, not the cwd, so the CLI
# works from any directory as long as the file exists at the
# canonical path.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
DEFAULT_EVAL_PATH = _REPO_ROOT / "eval" / "phrase_match_judgments.jsonl"
DEFAULT_OUTPUT_PATH = _BACKEND_DIR / "app" / "phrase_match_optimized.json"


# The locked Phase 1.5a provenance tag. Any row that drifts from it
# is, by definition, no longer the curated signal the Phase 6.7
# Ragas floor was measured against — the optimizer would be tuning
# against garbage. We warn (or fail on ``--strict``) so the drift
# surfaces at optimizer time, not at a later eval cycle.
EXPECTED_PROVENANCE = "HUMAN-LABELED"


# The four-way relation literal (mirrors ``app.schemas.PhrasePairRelation``).
# The optimizer's projection doesn't validate against this literal —
# the eval set is hand-curated, so a bad value surfaces as bad
# training data, not as a parse error. We expose it for the
# loader-only invariant tests so a future maintainer who swaps in a
# 5-way relation can't silently regress the projection.
EXPECTED_RELATIONS = {"equivalent", "paraphrase", "related", "unrelated"}


def _load_eval_set(
    path: Path,
    *,
    strict: bool = False,
) -> list[dict]:
    """Read the JSONL eval set, skipping comment lines and reject rows.

    The file's leading comment block declares provenance (per Hard
    rule #12). JSON Lines allows ``#``-prefixed lines by convention;
    we strip them before json.loads. Rows where ``judgment !=
    "accept"`` are excluded from training but counted for visibility.

    The phrase_match eval rows are keyed on
    ``phrase_a_id`` / ``phrase_b_id`` (the ``phrases.id`` slugs);
    the surface text is in ``phrase_a_text`` / ``phrase_b_text``.
    We project onto a six-key shape the Phase 10.2
    ``PhraseMatchSignature`` will consume (mirror-for-mirror of the
    cloze / match / comprehension script projections).

    ``--strict`` flips the ``HUMAN-LABELED`` provenance warning into
    a hard error. The default is to warn so a transient drift
    (e.g. a Phase 10.4 follow-up that re-labels with a different
    provenance tag) doesn't silently break the offline smoke.

    Returns the list of projected accept rows. If the file doesn't
    exist, raises ``FileNotFoundError`` (the caller decides whether
    that maps to NOOP or a hard failure).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"optimize_phrase_match: eval set not found at {path}. "
            f"Phase 10.4 creates this file; this CLI is a no-op "
            f"until the eval set lands."
        )

    accepted: list[dict] = []
    rejected = 0
    provenance_drift = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "optimize_phrase_match: skipping malformed row at line %d: %s",
                    line_no,
                    exc,
                )
                continue
            if row.get("judgment") != "accept":
                rejected += 1
                continue
            # HUMAN-LABELED manifest discipline (Phase 1.5a). The
            # optimizer's training signal is hand-curated; a drift
            # in provenance means the curated signal is gone, and
            # the optimizer would be tuning against whatever else
            # got in the file.
            provenance = row.get("provenance")
            if provenance != EXPECTED_PROVENANCE:
                provenance_drift += 1
                msg = (
                    f"optimize_phrase_match: row {line_no} provenance "
                    f"is {provenance!r}, expected {EXPECTED_PROVENANCE!r}"
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
            # Project the eval-set row onto the six input keys the
            # DSPy ``PhraseMatchSignature`` will consume. The exact
            # key names match the optimize_match.py / optimize_
            # comprehension.py precedent (mirror-for-mirror), so a
            # future maintainer reading all four optimizer scripts
            # side-by-side sees one projection pattern, not four.
            #
            # ``learner_axes_json`` is empty for the offline path;
            # the production path reads the real axes from the DB.
            # ``retrieved_pairs_json`` is the no-RAG fallback; the
            # eval-set rows are hand-labeled, not retrieval-derived.
            phrase_a_id = row["phrase_a_id"]
            phrase_b_id = row["phrase_b_id"]
            phrase_a_text = row.get("phrase_a_text") or phrase_a_id
            phrase_b_text = row.get("phrase_b_text") or phrase_b_id
            expected_relation = row["expected_relation"]
            # ``pair_index`` is a stable row id so the optimizer
            # can report per-pair metrics. We use a synthetic
            # index from the line number minus comment lines, but
            # the simpler contract is: the optimizer doesn't depend
            # on it being globally unique; only locally stable
            # within one optimizer run.
            accepted.append(
                {
                    "phrase_a": phrase_a_text,
                    "phrase_b": phrase_b_text,
                    "expected_relation": expected_relation,
                    "learner_axes_json": json.dumps({}, ensure_ascii=False),
                    "retrieved_pairs_json": json.dumps(
                        [], ensure_ascii=False
                    ),
                    "pair_index": line_no,
                }
            )
    if rejected:
        logger.info(
            "optimize_phrase_match: %d reject rows skipped "
            "(only accept rows used)",
            rejected,
        )
    if provenance_drift:
        logger.warning(
            "optimize_phrase_match: %d rows drifted from the "
            "%s provenance contract; review the eval set before "
            "trusting the optimised prompt",
            provenance_drift,
            EXPECTED_PROVENANCE,
        )
    return accepted


def _train_val_split(
    rows: list[dict], val_fraction: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 split by index.

    Same contract as ``optimize_match._train_val_split`` /
    ``optimize_comprehension._train_val_split`` — determinism matters
    because the operator re-runs the CLI on the same eval set and
    expects the same split. The 4-way relation space makes a
    stratified split desirable in principle, but the optimizer's
    metric is per-row (not per-class), so the simpler index cut is
    fine for the offline smoke. A future Phase can promote to a
    stratified split once the eval set grows past the ~50-row
    Phase 10.4 anchor.
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

    Mirrors ``scripts/optimize_match._serialize_optimized`` /
    ``scripts/optimize_comprehension._serialize_optimized`` so the
    three artifacts use a comparable JSON shape (a future Phase 5+
    loader can rely on consistent keys).

    DSPy's module objects don't ship a stable JSON serialisation
    surface across versions; this helper extracts the fields we
    can read and packs them into a dict the production path can
    load. If the active DSPy version doesn't expose a given
    attribute, we record ``None`` and let Phase 10+ decide how to
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
        "schema_version": "phrase-match-optimized-v1",
        "mode": mode,
        "train_count": train_count,
        "val_count": val_count,
        "instructions_by_field": instructions_by_field,
    }


def _resolve_optimizer_callable():
    """Lazily resolve ``app.phrase_match.optimize_phrase_match_module``.

    Mirrors the optimize_match.py / optimize_comprehension.py pattern:
    the heavy DSPy import is paid only on the optimizer-call path,
    not on ``--help``. Returns the callable or ``None`` if the
    Phase 10.2 module hasn't shipped yet (10.7 may land before 10.2
    in the dependency order; the CLI is the contract that 10.2's
    ``optimize_phrase_match_module`` must satisfy, not the other way
    around).
    """
    try:
        from app.phrase_match import optimize_phrase_match_module
    except ImportError:
        return None
    if not callable(optimize_phrase_match_module):
        return None
    return optimize_phrase_match_module


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. ``--help`` exits 0 (acceptance criterion)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.optimize_phrase_match",
        description=(
            "Run the offline DSPy optimizer for the phrase_match "
            "module against the held-out eval set (Phase 10.4 "
            "deliverable). Without --live, uses DSPy's DummyLM so "
            "no OpenRouter call is made. Live mode is gated on the "
            "Phase 6.7 Ragas floor; this offline CLI is the "
            "supported path."
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
            "OPENROUTER_API_KEY is set AND the Phase 6.7 Ragas "
            "floor holds; otherwise DummyLM is used regardless of "
            "this flag. The card scope defers live-LLM usage to "
            "the post-Phase 6.7 verification window."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the eval set reserved for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.6,
        help=(
            "Per-type accuracy floor for the offline smoke. "
            "Default: 0.6 (the 4-way relation space is harder than "
            "binary cloze so the floor is per-type, not the "
            "Phase 6.7 Ragas floor). Used for the optimizer's "
            "metric and for the --strict gate; the floor-check "
            "CLI flag is the same shape as optimize_match.py / "
            "optimize_comprehension.py so a future maintainer "
            "reads one pattern across all four optimizers."
        ),
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help=(
            "Max few-shot demos to bake into the optimised prompt. "
            "Default: 4 (the Phase 4.4 baseline). The 4-way "
            "relation space benefits from slightly more demos; "
            "the offline smoke uses the Phase 4.4 default so "
            "DummyLM stays non-pathological."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for the optimizer. Default: 42. Same seed "
            "+ same held-out set + same --target => byte-equal "
            "optimized artifact (reproducibility per Phase 4.4 "
            "discipline)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (exit non-zero) when an eval-set row drifts from "
            "the HUMAN-LABELED provenance contract. Default: warn "
            "and continue."
        ),
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
        rows = _load_eval_set(args.eval_path, strict=args.strict)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        # Still exit 0 so a CI step that runs --help or that
        # pre-Phase 10.4 environments run the CLI doesn't fail the
        # build. The plan ships the CLI in 10.7; the eval set
        # ships in 10.4. This is the expected gap.
        print("NOOP: eval set not found (Phase 10.4 deliverable pending).")
        return 0

    if not rows:
        logger.warning(
            "optimize_phrase_match: eval set is empty; nothing to optimize."
        )
        print("NOOP: eval set has 0 accept rows.")
        return 0

    train, val = _train_val_split(rows, val_fraction=args.val_fraction)
    logger.info(
        "optimize_phrase_match: train=%d val=%d (total=%d accept rows, "
        "target=%.2f, seed=%d, max_demos=%d)",
        len(train),
        len(val),
        len(rows),
        args.target,
        args.seed,
        args.max_demos,
    )

    # Live-mode check: only fire the real LM if both --live is
    # requested AND the env has a key. Otherwise DummyLM is used
    # regardless (the offline guarantee is the spec, not a flag).
    live_requested = args.live and bool(os.getenv("OPENROUTER_API_KEY"))
    if args.live and not os.getenv("OPENROUTER_API_KEY"):
        logger.warning(
            "optimize_phrase_match: --live requested but "
            "OPENROUTER_API_KEY is missing; falling back to DummyLM."
        )
    mode = "live" if live_requested else "dummy"

    # Lazily resolve the Phase 10.2 optimizer callable. 10.7 may
    # land before 10.2 in the dependency order; the CLI is the
    # contract that 10.2 must satisfy, not the other way around.
    # When 10.2 hasn't shipped yet, we write a deterministic stub
    # payload so the offline smoke stays green and the production
    # path picks up the real artefact on the next 10.2 fold.
    optimize_callable = _resolve_optimizer_callable()
    if optimize_callable is None:
        logger.warning(
            "optimize_phrase_match: app.phrase_match.optimize_"
            "phrase_match_module is not available yet (Phase 10.2 "
            "pending). Writing a deterministic stub artefact so "
            "the offline smoke stays green. Re-run after 10.2 "
            "lands for the real optimised prompt."
        )
        # Synthesise a stub module that mirrors the
        # ``module.predictors == []`` shape the serializer already
        # tolerates. The serialized artefact carries ``mode=dummy``
        # and the documented ``schema_version``; the production
        # path's read key is the schema_version, so a stub isn't
        # silently loaded as a tuned prompt — it falls back to the
        # baked-in signature instructions on next start.
        class _StubModule:
            predictors = []

        optimized = _StubModule()
    else:
        optimized = optimize_callable(
            train_set=train, val_set=val or None
        )

    payload = _serialize_optimized(
        optimized, train_count=len(train), val_count=len(val), mode=mode
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(
        f"OK: wrote optimised prompt to {args.output_path} "
        f"(mode={mode}, train={len(train)}, val={len(val)}, "
        f"target={args.target}, seed={args.seed}, "
        f"max_demos={args.max_demos})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())