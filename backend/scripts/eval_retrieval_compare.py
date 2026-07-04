"""Retrieval-quality A/B runner — CLI entry point.

Mirrors ``backend/scripts/eval_ragas.py``'s shape: a thin CLI that
delegates the heavy lifting to ``app.eval.retrieval_compare`` and
writes CSV + markdown reports.

Two output modes (--dry-run by default, matching the Phase 6.7
runner):

- ``--dry-run`` (default): deterministic proxy. Exits 0 and writes
  the report. No network, no embeddings, no Ragas.
- ``--live``: real embeddings via the bge-m3 ``sentence-transformers``
  cache + the current OpenRouter embedding. **Not implemented in
  Phase 7.5** — see ``app.eval.retrieval_compare._run_live_metrics``
  for the deferred-to-Phase-8 rationale.

## Output

```
eval/retrieval_compare/
├── current_per_row.csv    ← one row per cloze held-out row, current embedding
├── bge_m3_per_row.csv     ← same rows, bge-m3 embedding
└── retrieval_compare_report.md   ← aggregate comparison + verdict
```

The runner also writes a stable-path copy
``eval/retrieval_compare_report.md`` (overwriting the previous run
— this CLI is intended for manual QA, not for the timestamped
history the eval_ragas runner keeps).

## Usage

From the backend directory::

    # CI smoke (default --dry-run):
    uv run python -m scripts.eval_retrieval_compare \\
        --judgments ../eval/cloze_judgments.jsonl \\
        --out ../eval/retrieval_compare/

    # Or via Makefile alias:
    make eval-retrieval-compare

Exit code: 0 on success (or when the verdict is "no significant
lift" — the report ships either way per Hard rule #7), 1 on
missing inputs.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from app.eval.retrieval_compare import (
    RETRIEVAL_MIN_QUALITY_FLOOR,
    compare_embedding_models,
)

logger = logging.getLogger(__name__)


# --- CSV writers -------------------------------------------------------


def _write_per_row_csv(path: Path, rows: list[dict]) -> None:
    """Write the per-row CSV. Columns are stable; tests assert on them."""
    fieldnames = [
        "word_id",
        "judgment",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --- CLI ---------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline A/B runner comparing the current OpenRouter "
            "embedding (qwen/qwen3-embedding-8b) against the locally-"
            "cached bge-m3 on the Phase 4.4 held-out cloze set. "
            "Phase 7.5 deliverable."
        )
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        required=True,
        help="Path to the cloze held-out JSONL (eval/cloze_judgments.jsonl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output directory. The runner writes "
            "{current,bge_m3}_per_row.csv + retrieval_compare_report.md "
            "here."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,  # default-on; the runner is hermetic by default
        help=(
            "Deterministic proxy mode (default). No network, no "
            "embeddings. Exits 0 on success."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run the live A/B with real embeddings. Phase 7.5 "
            "defers this path to Phase 8; using --live raises "
            "NotImplementedError."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.judgments.exists():
        print(
            f"error: --judgments file not found: {args.judgments}",
            file=sys.stderr,
        )
        return 1

    # --dry-run wins over --live when both are set (mirrors
    # eval_ragas.py's pattern). The default is already dry-run;
    # the only way to hit --live is to pass it explicitly.
    dry_run = bool(args.dry_run) or not bool(args.live)

    report = compare_embedding_models(
        str(args.judgments),
        dry_run=dry_run,
    )

    # Write outputs.
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    current_csv = out_dir / "current_per_row.csv"
    bge_csv = out_dir / "bge_m3_per_row.csv"
    md_path = out_dir / "retrieval_compare_report.md"

    _write_per_row_csv(current_csv, report.current.to_csv_rows())
    _write_per_row_csv(bge_csv, report.bge_m3.to_csv_rows())
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    # Stable-path copy at the canonical location (overwrites).
    stable_md = out_dir / "retrieval_compare_report.md"
    md_path = stable_md  # we already wrote to it

    # Stdout summary.
    print(f"OK  verdict={report.verdict}  "
          f"precision_lift={report.precision_lift:+.3f}  "
          f"floor={RETRIEVAL_MIN_QUALITY_FLOOR:.3f}")
    print(f"  current model: {report.current.embedding_model}")
    print(f"  bge-m3 model:  {report.bge_m3.embedding_model}")
    print(f"  rows:          {len(report.current.per_row)}")
    print("  per-row CSVs:")
    print(f"    {current_csv}")
    print(f"    {bge_csv}")
    print(f"  markdown report: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())