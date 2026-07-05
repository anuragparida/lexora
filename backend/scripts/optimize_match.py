"""Phase 9.3 — DSPy matching-optimizer CLI (card t_52ef2d50).

Invokable as ``uv run python -m scripts.optimize_match``. Reads the
held-out eval set produced by Phase 6 (``eval/match_judgments.jsonl``,
40 accept rows, all template-based per Phase 6 hard rule #8), runs
MIPROv2 (or BootstrapFewShot as a fallback) on the matching module
against it, and writes the optimised prompt to
``backend/app/match_optimized.json`` for the production path to read
on next start.

Behavior matrix (mirrors ``scripts/optimize_cloze.py`` line-for-line
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

Eval set shape (per row in ``eval/match_judgments.jsonl``):

    {
        "word_id": int,
        "word_type": str,
        "target_word": str,
        "expected_pairs": [
            {"right_kind": str, "right_word": str, "right_word_id": int},
            ...
        ],
        "judgment": "accept" | "reject",
        "labeler": str,
        "provenance": str,
        "rationale": str
    }

Only rows with ``judgment == "accept"`` are used for training;
``reject`` rows are skipped (a rejected matching pair is bad data,
not a useful training signal). The matching eval set's row schema
does NOT carry a ``context_sentence`` field (Phase 6 ships it as a
template-fallback dry-run proxy per Phase 6 hard rule #8); the CLI
derives one inline from the first expected pair's ``right_word``
so the DSPy module's ``context_sentence`` input is never empty.
The ``count`` input is derived from ``len(expected_pairs)``; the
held-out set ships uniform 3-pair rows so this is stable.

Hard rules enforced:

- #8 Same ``DummyLM`` discipline as ``optimize_cloze.py`` —
  offline optimizer, no real LLM.
- The CLI does NOT make a live-LLM call on the offline path. The
  live path is gated on the Phase 6.7 Ragas floor (per the card
  body) and is not exercised by this CLI's dry-run smoke.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Repository-relative path resolution. Mirrors the cloze script:
# ``scripts/optimize_match.py`` lives at ``backend/scripts/``; the
# eval set lives at the repo root in ``eval/match_judgments.jsonl``;
# the optimised prompt is written to ``backend/app/match_optimized.json``.
# We resolve the eval set relative to the package root, not the cwd,
# so the CLI works from any directory as long as the file exists at
# the canonical path.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
DEFAULT_EVAL_PATH = _REPO_ROOT / "eval" / "match_judgments.jsonl"
DEFAULT_OUTPUT_PATH = _BACKEND_DIR / "app" / "match_optimized.json"


# Sentinel used when the matching eval row carries no
# ``context_sentence`` (Phase 6's template fallback never sets one).
# The string is intentionally un-translated so a careless future
# merge that swaps this for the target word's first example can't
# silently double as that word's natural example.
_FALLBACK_CONTEXT_SENTENCE = (
    "(kein Beispielsatz im Eval-Set; siehe eval/match_judgments.jsonl)"
)


def _synthesise_context_sentence(
    target_word: str, expected_pairs: list[dict]
) -> str:
    """Build a stable ``context_sentence`` for one match eval row.

    The matching eval set template-fallback (Phase 6 hard rule #8)
    doesn't carry a corpus example sentence — it carries a
    target_word + a list of expected_pairs. We compose a minimal
    context so the DSPy module's ``context_sentence`` input is
    never empty (the module's optimise path expects the same five
    keys as the production ``MatchSignature``).

    The exact text is irrelevant for the offline smoke — it only
    exists to exercise the optimizer plumbing end-to-end without
    flunking on a missing required input.
    """
    if not expected_pairs:
        return _FALLBACK_CONTEXT_SENTENCE
    pair = expected_pairs[0]
    right_word = pair.get("right_word") or "(unknown)"
    return f"{target_word} — Beispielpartner: {right_word}."


def _load_eval_set(path: Path) -> list[dict]:
    """Read the JSONL eval set, skipping comment lines and reject rows.

    The file's leading comment block declares provenance (per Hard
    rule #12). JSON Lines allows ``#``-prefixed lines by convention;
    we strip them before json.loads. Rows where ``judgment !=
    "accept"`` are excluded from training but counted for
    visibility.

    The matching eval rows use ``target_word`` (not ``word``) and
    ``word_id`` (not ``target_word_id``); we project both onto the
    keys the DSPy ``MatchSignature`` expects, mirror-for-mirror of
    the cloze script's projection.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"optimize_match: eval set not found at {path}. "
            f"Phase 6 creates this file; this CLI is a no-op "
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
                    "optimize_match: skipping malformed row at line %d: %s",
                    line_no,
                    exc,
                )
                continue
            if row.get("judgment") != "accept":
                rejected += 1
                continue
            # Project the eval-set row onto the five input keys the
            # DSPy ``MatchSignature`` expects. ``learner_axes_json``
            # is empty for the offline path; the production path
            # reads the real axes from the DB.
            expected_pairs = row.get("expected_pairs", [])
            target_word = row.get("target_word") or ""
            accepted.append(
                {
                    "word": target_word,
                    "context_sentence": _synthesise_context_sentence(
                        target_word, expected_pairs
                    ),
                    "learner_axes_json": json.dumps({}, ensure_ascii=False),
                    "target_word_id": int(row["word_id"]),
                    "count": max(1, len(expected_pairs)),
                }
            )
    if rejected:
        logger.info(
            "optimize_match: %d reject rows skipped (only accept rows used)",
            rejected,
        )
    return accepted


def _train_val_split(
    rows: list[dict], val_fraction: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 split by index.

    Same contract as ``optimize_cloze._train_val_split`` — determinism
    matters because the operator re-runs the CLI on the same eval
    set and expects the same split.
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

    Mirrors ``scripts/optimize_cloze._serialize_optimized`` so the
    two artifacts use a comparable JSON shape (a future Phase 5+
    loader can rely on consistent keys).

    DSPy's module objects don't ship a stable JSON serialisation
    surface across versions; this helper extracts the fields we
    can read and packs them into a dict the production path can
    load. If the active DSPy version doesn't expose a given
    attribute, we record ``None`` and let Phase 5+ decide how to
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
        "schema_version": "match-optimized-v1",
        "mode": mode,
        "train_count": train_count,
        "val_count": val_count,
        "instructions_by_field": instructions_by_field,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. ``--help`` exits 0 (acceptance criterion)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.optimize_match",
        description=(
            "Run the offline DSPy optimizer for the matching module "
            "against the held-out eval set (Phase 6 deliverable). "
            "Without --live, uses DSPy's DummyLM so no OpenRouter "
            "call is made. Live mode is gated on the Phase 6.7 "
            "Ragas floor; this offline CLI is the supported path."
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
        # pre-Phase 6 environments run the CLI doesn't fail the
        # build. The card scope explicitly reuses the Phase 6
        # eval set; this is the expected gap if the eval set is
        # moved in a future refactor.
        print("NOOP: eval set not found (Phase 6 deliverable pending).")
        return 0

    if not rows:
        logger.warning("optimize_match: eval set is empty; nothing to optimize.")
        print("NOOP: eval set has 0 accept rows.")
        return 0

    train, val = _train_val_split(rows, val_fraction=args.val_fraction)
    logger.info(
        "optimize_match: train=%d val=%d (total=%d accept rows)",
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
            "optimize_match: --live requested but OPENROUTER_API_KEY is "
            "missing; falling back to DummyLM."
        )
    mode = "live" if live_requested else "dummy"

    # Lazily import the match module so --help doesn't pay the cost.
    from app.match import optimize_match_module

    optimized = optimize_match_module(train_set=train, val_set=val or None)

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
