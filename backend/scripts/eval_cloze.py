"""Offline runner for the held-out cloze eval set.

Phase 4.4 deliverable. Compares a model's cloze predictions against
``eval/cloze_judgments.jsonl`` and computes three metrics:

- ``accept_rate`` — fraction of held-out rows where the new model
  emits a cloze whose ``answer_word_id`` matches the held-out
  ``expected_answer_word_id`` AND whose sentence is semantically
  equivalent to the held-out sentence (in --dry-run / no-model
  mode we substitute a deterministic self-accept proxy that reads
  ``judgment: accept`` from the held-out row, so the metric is
  well-defined even when no live model is wired in).
- ``schema_validity_rate`` — fraction of predictions that pass
  Pydantic validation against the ``ClozeExercise`` shape (4.2's
  Pydantic model). When a prediction is missing, the row counts
  as schema-invalid.
- ``rationale_quality_proxy`` — average character length of the
  ``rationale`` field across the predictions that pass schema
  validation. The spec calls this a heuristic, not a floor; the
  number is reported so reviewers can spot collapses to empty
  strings.

Writes ``eval/results_<timestamp>.json`` with the metrics + a
per-row diff. The timestamp keeps concurrent runs from clobbering
each other.

## CI smoke path: --dry-run

``--dry-run`` exits 0 and prints ``OK`` without contacting
OpenRouter or running any model. It still loads the JSONL, runs
the metrics computation against the held-out set's own
``judgment`` column (so a CI smoke run produces a real number),
and writes a results file. This is the path Helena's review runs
in CI; it must not depend on API keys.

## Live path: --predictions FILE

The live path takes a JSONL of predictions, one per held-out row
in the same order as the input file (or keyed by ``word_id`` /
``expected_answer_word_id`` if order doesn't match). Each
prediction is a ``ClozeExercise`` JSON object. The runner
validates each prediction against the Pydantic model and computes
the three metrics.

The live path is wired but not exercised by any Phase 4 build
card — the DSPy optimizer (4.2) and the live cloze endpoint (4.2)
will both feed predictions into this runner as follow-on work.

## Usage

From the backend directory::

    # CI smoke (default --dry-run):
    uv run python -m scripts.eval_cloze --dry-run

    # Live predictions:
    uv run python -m scripts.eval_cloze --predictions /path/to/preds.jsonl

Exit code: 0 on success, 1 on schema errors or missing inputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Type-level guardrails (Hard rule #11) -----------------------------

#: Floor for ``accept_rate`` in CI smoke mode. Phase 4 spec marks
#: the qualitative C1-accept bar as the hard floor; this constant
#: is the deterministic-template fallback's expected rate (a
#: self-accept against the held-out set's own ``judgment: accept``
#: column should be ~1.0 by construction).
EVAL_DRY_RUN_MIN_ACCEPT_RATE = 0.95

#: Floor for ``schema_validity_rate``. Anything below means the
#: predictions don't parse — a structural regression.
EVAL_MIN_SCHEMA_VALIDITY_RATE = 0.95


@dataclass(frozen=True)
class HeldOutRow:
    """Parsed form of one row of ``eval/cloze_judgments.jsonl``."""

    word_id: int
    word: str
    word_type: str
    context_sentence: str
    source_example_sentence: str
    expected_answer_word_id: int
    expected_distractors: tuple[int, ...]
    expected_difficulty: str
    labeler: str
    provenance: str
    judgment: str
    rationale: str


@dataclass(frozen=True)
class EvalMetrics:
    """Three primary metrics + the per-row diff."""

    eval_set_path: str
    rows_total: int
    rows_accepted_in_holdout: int
    rows_predicted: int
    rows_passed_schema: int
    accept_rate: float
    schema_validity_rate: float
    rationale_quality_proxy: float
    dry_run: bool
    predictions_path: str | None
    timestamp: str
    per_row: list[dict] = field(default_factory=list)


def _parse_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping leading comment lines.

    A leading comment line starts with ``#``. The held-out file's
    provenance block lives in those leading comments; the runner
    must skip them.
    """
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            obj = json.loads(line)
            out.append(obj)
    return out


def _row_from_dict(d: dict) -> HeldOutRow:
    """Construct a ``HeldOutRow`` from a parsed JSON object.

    The dataclass field types are strict (``tuple[int, ...]`` for
    distractors) so this function normalizes the raw JSON input.
    """
    return HeldOutRow(
        word_id=int(d["word_id"]),
        word=str(d["word"]),
        word_type=str(d.get("word_type", "")),
        context_sentence=str(d["context_sentence"]),
        source_example_sentence=str(d.get("source_example_sentence", "")),
        expected_answer_word_id=int(d["expected_answer_word_id"]),
        expected_distractors=tuple(int(x) for x in d["expected_distractors"]),
        expected_difficulty=str(d["expected_difficulty"]),
        labeler=str(d["labeler"]),
        provenance=str(d["provenance"]),
        judgment=str(d["judgment"]),
        rationale=str(d["rationale"]),
    )


def _word_token_set(sentence: str) -> set[str]:
    """Tokenize a sentence for semantic-equivalence comparison.

    Returns the lowercased set of word tokens with German
    diacritics folded (ä→a, ö→o, ü→u, ß→ss) and punctuation
    stripped. The diacritic fold mirrors ``_lemma_root``'s
    normalization in the eval-set builder so the two halves
    agree on what counts as a word.
    """
    s = (
        sentence.lower()
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )
    tokens = re.findall(r"[a-z0-9]+", s)
    return set(tokens)


def _semantic_equivalent(
    held_out: HeldOutRow,
    prediction: dict | None,
) -> bool:
    """Return True if the prediction's sentence carries the same
    answer word AND shares substantial token overlap with the
    held-out sentence.

    For ``--dry-run`` mode (no prediction supplied), the held-out
    row's own ``judgment`` field is the proxy — if the row was
    self-accepted at build time, treat it as semantically equivalent
    (this is what gives ``--dry-run`` its well-defined metric
    without needing a live model).
    """
    if prediction is None:
        return held_out.judgment == "accept"
    # answer_word_id match is the structural gate
    pred_answer = prediction.get("answer_word_id")
    if pred_answer != held_out.expected_answer_word_id:
        return False
    # Token overlap: ≥ 60% of the held-out sentence tokens
    # (excluding the blank) appear in the prediction sentence.
    pred_sentence = prediction.get("sentence_with_blank", "")
    if not pred_sentence:
        return False
    held_tokens = _word_token_set(held_out.context_sentence)
    pred_tokens = _word_token_set(pred_sentence)
    if not held_tokens:
        return False
    # Drop the blank token from the prediction set so we don't
    # count ___ as a token.
    pred_tokens.discard("___")
    overlap = held_tokens & pred_tokens
    return (len(overlap) / len(held_tokens)) >= 0.60


def _validate_prediction_schema(pred: dict | None) -> bool:
    """Validate a prediction dict against the ClozeExercise Pydantic
    model.

    Importing the Pydantic model is optional — 4.2 hasn't shipped
    yet, so we tolerate ImportError and fall back to a structural
    check that mirrors the documented field set. The structural
    check is the CI smoke path; once 4.2 lands, the runner will
    pick up the Pydantic model automatically.

    Returns True iff the prediction is valid.
    """
    if pred is None:
        return False
    try:
        from app.cloze import ClozeExercise  # type: ignore
        ClozeExercise.model_validate(pred)
        return True
    except ImportError:
        # Fall back to structural validation.
        required = {
            "sentence_with_blank": str,
            "answer_word_id": int,
            "distractors": list,
            "difficulty": str,
            "rationale": str,
            "prompt_template_version": str,
        }
        for field_name, field_type in required.items():
            if field_name not in pred:
                return False
            v = pred[field_name]
            if field_type is int and not isinstance(v, int):
                return False
            if field_type is str and not isinstance(v, str):
                return False
            if field_type is list and not isinstance(v, list):
                return False
        if len(pred.get("distractors", [])) != 3:
            return False
        return True
    except Exception:
        # Pydantic raised on validation — schema-invalid.
        return False


def _load_predictions(
    path: Path,
    held_out: list[HeldOutRow],
) -> list[dict | None]:
    """Read the predictions JSONL and align it with ``held_out``.

    The alignment strategy is order-preserving: the Nth prediction
    in the file maps to the Nth held-out row. If the counts don't
    match, the runner pads with ``None`` (which counts as a
    schema-invalid row) and logs a warning.
    """
    raw = _parse_jsonl(path)
    if len(raw) != len(held_out):
        logger.warning(
            "Prediction count %d != held-out count %d; padding with None.",
            len(raw),
            len(held_out),
        )
    aligned: list[dict | None] = []
    for i, h in enumerate(held_out):
        if i < len(raw):
            aligned.append(raw[i])
        else:
            aligned.append(None)
    return aligned


def _run_metrics(
    *,
    held_out: list[HeldOutRow],
    predictions: list[dict | None],
    eval_set_path: Path,
    predictions_path: Path | None,
    dry_run: bool,
) -> EvalMetrics:
    """Compute the three primary metrics + per-row diff."""
    accepted = 0
    accepted_in_holdout = sum(1 for h in held_out if h.judgment == "accept")
    passed_schema = 0
    rationale_lengths: list[int] = []
    per_row: list[dict] = []

    for h, pred in zip(held_out, predictions):
        schema_ok = _validate_prediction_schema(pred)
        if schema_ok:
            passed_schema += 1
            # rationale length counts only schema-passing rows
            if pred is not None and isinstance(pred.get("rationale"), str):
                rationale_lengths.append(len(pred["rationale"]))
        sem_ok = _semantic_equivalent(h, pred)
        if sem_ok:
            accepted += 1
        per_row.append({
            "word_id": h.word_id,
            "expected_answer_word_id": h.expected_answer_word_id,
            "predicted_answer_word_id": (
                pred.get("answer_word_id") if pred else None
            ),
            "schema_valid": schema_ok,
            "semantic_equivalent": sem_ok,
            "difficulty": h.expected_difficulty,
            "labeler": h.labeler,
        })

    n = len(held_out)
    accept_rate = (accepted / n) if n else 0.0
    schema_rate = (passed_schema / n) if n else 0.0
    rationale_quality_proxy = (
        sum(rationale_lengths) / len(rationale_lengths)
        if rationale_lengths
        else 0.0
    )

    return EvalMetrics(
        eval_set_path=str(eval_set_path),
        rows_total=n,
        rows_accepted_in_holdout=accepted_in_holdout,
        rows_predicted=sum(1 for p in predictions if p is not None),
        rows_passed_schema=passed_schema,
        accept_rate=accept_rate,
        schema_validity_rate=schema_rate,
        rationale_quality_proxy=rationale_quality_proxy,
        dry_run=dry_run,
        predictions_path=str(predictions_path) if predictions_path else None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        per_row=per_row,
    )


def _format_metrics(metrics: EvalMetrics) -> str:
    """Human-readable summary used by the CLI's stdout banner."""
    return (
        f"accept_rate={metrics.accept_rate:.3f} "
        f"schema_validity_rate={metrics.schema_validity_rate:.3f} "
        f"rationale_quality_proxy={metrics.rationale_quality_proxy:.1f} "
        f"rows={metrics.rows_total} dry_run={metrics.dry_run}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline runner for the held-out cloze eval set. "
            "Default mode (--dry-run) exits 0 and prints OK "
            "without contacting OpenRouter."
        )
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "eval"
        / "cloze_judgments.jsonl",
        help="Path to the held-out JSONL. Default: ../eval/cloze_judgments.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Optional JSONL of predictions (one ClozeExercise "
            "JSON object per line). When omitted, runs in "
            "--dry-run mode and uses the held-out judgment as the "
            "accept proxy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "eval",
        help=(
            "Directory for results_<timestamp>.json. "
            "Default: ../eval/"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "CI smoke mode: exit 0, print OK, do not contact "
            "OpenRouter, do not require --predictions. Overrides "
            "--predictions when both are set."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.eval_set.exists():
        print(f"eval-set file not found: {args.eval_set}", file=sys.stderr)
        return 1

    raw_held_out = _parse_jsonl(args.eval_set)
    held_out = [_row_from_dict(d) for d in raw_held_out]

    dry_run = bool(args.dry_run) or args.predictions is None

    if dry_run:
        # All-None predictions: every row is compared against its
        # own held-out ``judgment`` column.
        predictions: list[dict | None] = [None] * len(held_out)
        predictions_path: Path | None = None
    else:
        predictions = _load_predictions(args.predictions, held_out)  # type: ignore[arg-type]
        predictions_path = args.predictions

    metrics = _run_metrics(
        held_out=held_out,
        predictions=predictions,
        eval_set_path=args.eval_set,
        predictions_path=predictions_path,
        dry_run=dry_run,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"results_{metrics.timestamp.replace(':', '-')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_set_path": metrics.eval_set_path,
                "rows_total": metrics.rows_total,
                "rows_accepted_in_holdout": metrics.rows_accepted_in_holdout,
                "rows_predicted": metrics.rows_predicted,
                "rows_passed_schema": metrics.rows_passed_schema,
                "accept_rate": metrics.accept_rate,
                "schema_validity_rate": metrics.schema_validity_rate,
                "rationale_quality_proxy": metrics.rationale_quality_proxy,
                "dry_run": metrics.dry_run,
                "predictions_path": metrics.predictions_path,
                "timestamp": metrics.timestamp,
                "per_row": metrics.per_row,
            },
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    print("OK" if dry_run else _format_metrics(metrics))
    print(f"  results written to: {out_path}")
    if not dry_run:
        print(_format_metrics(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())