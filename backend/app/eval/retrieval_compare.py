"""Retrieval-quality A/B runner (Phase 7.5).

Compares two embedding models on the Phase 4.4 held-out cloze set:

- **current**: the production embedding
  (``qwen/qwen3-embedding-8b`` via OpenRouter, the Phase 1 default).
- **bge-m3**: the locally-cached ``baai/bge-m3`` via
  ``sentence-transformers`` (the Phase 1 fallback that OpenRouter's
  privacy filter blocks — see NOTES.md §"Gotchas hit").

The runner reports four Ragas metrics per row
(``context_precision``, ``context_recall``, ``faithfulness``,
``answer_relevance``) plus a per-model aggregate + a per-metric
``lift = bge_m3 - current``. The lift interpretation is gated by
the ``RETRIEVAL_MIN_QUALITY_FLOOR`` module constant: any lift on
``context_precision`` below that bar is reported as "no
significant lift" rather than rejected (Hard rule #7 — type-level
guardrail: a code change, not a config bump, widens it).

## Why both models, not just one?

This module is the *Phase 7.5* Phase 7 deliverable (per
``docs/PHASE-7.md`` §"Concrete cards" item 5). The A/B is one
env-var flip (``EMBEDDING_MODEL``) per Hard rule #1: the runner
takes the model id as a parameter and runs both back-to-back so
the report shows the comparison side-by-side. No code change to
``app.embeddings.py`` is required — the embedding client
already consumes ``EMBEDDING_MODEL`` (Phase 1 hard rule).

## Live vs dry-run

The runner mirrors ``backend/scripts/eval_ragas.py``'s split:

- **Live path** (--live): real embeddings via the ``embed()``
  client, real Ragas metrics. Requires a warm
  ``~/.cache/huggingface/`` cache for bge-m3 and a reachable
  OpenRouter key for the current model.
- **Dry-run path** (--dry-run, default): deterministic proxy.
  Each row's metrics are the Phase 4.4 pattern
  (``accept`` → 1.0, ``reject`` → 0.0). No network, no
  embeddings, no Ragas. CI exit code 0.

The dry-run proxy is *self-accepting by construction* for both
models — the lift is exactly 0.0. The runner reports that as
"lift=0.000 (no data)" so a reviewer reading the report sees
the structural shape without mistaking it for a real
comparison. The live path is the only one that produces a
non-trivial lift.

## Cold-cache skip (Hard rule #6)

``bge-m3`` ships ~2.3 GB on HuggingFace. If the local
``sentence-transformers`` cache is cold, the first run downloads
on demand. Tests SKIP (not fail) when the cache is cold so CI
is hermetic; the manual ``make eval-retrieval-compare --live``
is the only path that requires the warm cache. Mirrors Phase
4.4's deviation pattern.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

logger = logging.getLogger(__name__)


# --- Type-level guardrails (Hard rule #7) -------------------------------
#
# The floor is a module constant. Hard rule #7 is explicit: "Not
# env, not config." The constant below is the canonical value —
# widening it is a code change, not a container restart.

#: Minimum absolute lift on ``context_precision`` below which the
#: bge-m3 alt is reported as "no regression but no significant
#: lift" and the report ships anyway. Phase 7.5's locked value is
#: ``0.05`` (a 5-percentage-point improvement on precision is the
#: bar the team agreed to; smaller lifts are noise).
RETRIEVAL_MIN_QUALITY_FLOOR: float = 0.05


# --- Module constants (Hard rule #1: one env var flips the run) --------

#: The locally-cached alternative embedding (Phase 1 fallback).
#: This is a module-level constant — the comparison *target* is
#: baked in. If the project ever A/Bs a different alt, the value
#: becomes a parameter; for now, Phase 7.5 is the current-vs-bge-m3
#: comparison specifically.
EMBEDDING_MODEL_BGE_M3: str = "baai/bge-m3"

#: The current production embedding. Read from ``EMBEDDING_MODEL``
#: at module import time so the value is stable for the run
#: (Hard rule #1: one env var, offline eval). The default is the
#: Phase 1 production model.
EMBEDDING_MODEL_CURRENT: str = os.getenv(
    "EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"
)


#: The Ragas metric names we report per row. Locked to the
#: Phase 6.7 set (no new metrics in Phase 7.5).
RAGAS_METRICS: tuple[str, ...] = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevance",
)


#: Local-cache marker: bge-m3 is "available" iff this file exists
#: in the HuggingFace cache. Mirrors the Phase 4.4 cold-cache
#: probe pattern.
BGE_M3_CACHE_SENTINEL = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--BAAI--bge-m3"
)


# --- Result shapes ------------------------------------------------------


@dataclass(frozen=True)
class PerRowMetric:
    """One row of a single embedding model's per-row CSV.

    Mirrors ``scripts.eval_ragas.RagasResult``'s wire shape — the
    same four metric names + a pass boolean — but trimmed: the
    runner here is per-row-per-model, not aggregated by exercise
    type (the cloze set is one exercise type only).
    """

    word_id: int
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    judgment: str  # "accept" | "reject" — provenance for the proxy

    def mean4(self) -> float:
        """Mean of the four Ragas metrics — the row-level score."""
        return (
            self.context_precision
            + self.context_recall
            + self.faithfulness
            + self.answer_relevance
        ) / 4.0


@dataclass(frozen=True)
class ModelRunResult:
    """The result of running the held-out set under one embedding.

    ``per_row`` is the per-row metrics in the same order as the
    held-out JSONL. ``mean_per_metric`` is the aggregate per
    metric. ``mean_overall`` is the mean of the four means (the
    single-number "how good is this embedding" score).
    """

    embedding_model: str
    per_row: tuple[PerRowMetric, ...]
    mean_per_metric: dict[str, float]
    mean_overall: float

    def to_csv_rows(self) -> list[dict]:
        """Flatten ``per_row`` to CSV-friendly dicts."""
        out: list[dict] = []
        for r in self.per_row:
            out.append(
                {
                    "word_id": r.word_id,
                    "judgment": r.judgment,
                    "context_precision": round(r.context_precision, 4),
                    "context_recall": round(r.context_recall, 4),
                    "faithfulness": round(r.faithfulness, 4),
                    "answer_relevance": round(r.answer_relevance, 4),
                }
            )
        return out


@dataclass(frozen=True)
class ComparisonReport:
    """The A/B comparison report dataclass.

    Two ``ModelRunResult``s (one per embedding) + the per-metric
    lift (``bge_m3 - current``) + a verdict on whether the lift
    clears ``RETRIEVAL_MIN_QUALITY_FLOOR``. The verdict is a
    string, not a bool, because there are three meaningful
    states (see ``VERDICT_*``).
    """

    current: ModelRunResult
    bge_m3: ModelRunResult
    lift_per_metric: dict[str, float]
    lift_overall: float
    precision_lift: float  # shortcut for the lift that gates the verdict
    verdict: Literal[
        "significant_lift",
        "no_significant_lift",
        "regression",
    ]
    floor: float  # captured for the report so the markdown shows the bar
    judgments_path: str
    timestamp: str

    def to_markdown(self) -> str:
        """Render the comparison as a markdown table.

        The structure is fixed (the report is the deliverable, not
        the lift decision) — adding columns is a Phase 8 change.
        """
        lines: list[str] = []
        lines.append("# Lexora — Retrieval-quality A/B report")
        lines.append("")
        lines.append(f"- Held-out set: `{self.judgments_path}`")
        lines.append(f"- Timestamp (UTC): `{self.timestamp}`")
        lines.append(
            f"- Models: current=`{self.current.embedding_model}` "
            f"vs bge_m3=`{self.bge_m3.embedding_model}`"
        )
        lines.append(
            f"- `RETRIEVAL_MIN_QUALITY_FLOOR` (precision lift bar): "
            f"`{self.floor:.3f}`"
        )
        lines.append("")
        lines.append("## Per-metric comparison")
        lines.append("")
        lines.append("| metric | current | bge-m3 | lift |")
        lines.append("|---|---|---|---|")
        for m in RAGAS_METRICS:
            cur = self.current.mean_per_metric.get(m, 0.0)
            bge = self.bge_m3.mean_per_metric.get(m, 0.0)
            lift = self.lift_per_metric.get(m, 0.0)
            lines.append(
                f"| `{m}` | {cur:.3f} | {bge:.3f} | "
                f"{lift:+.3f} |"
            )
        lines.append(
            f"| **mean_overall** | "
            f"{self.current.mean_overall:.3f} | "
            f"{self.bge_m3.mean_overall:.3f} | "
            f"{self.lift_overall:+.3f} |"
        )
        lines.append("")
        lines.append("## Verdict")
        lines.append("")
        lines.append(
            f"- precision lift: `{self.precision_lift:+.3f}` "
            f"(bar: `{self.floor:.3f}`)"
        )
        lines.append(f"- verdict: **{self.verdict}**")
        if self.verdict == "significant_lift":
            lines.append(
                "- bge-m3 lifts `context_precision` by at least "
                f"`{self.floor:.3f}` — manual review can promote "
                "bge-m3 to a Phase 8 A/B hold-out."
            )
        elif self.verdict == "no_significant_lift":
            lines.append(
                "- bge-m3 is no worse than the current embedding, "
                "but the precision lift is below "
                f"`{self.floor:.3f}` — no action recommended."
            )
        else:
            lines.append(
                "- bge-m3 *regresses* `context_precision` vs the "
                "current embedding — keep the current model."
            )
        lines.append("")
        lines.append(
            "> Note: in `--dry-run` mode the metrics use the "
            "deterministic proxy (accept → 1.0, reject → 0.0). "
            "The lift is therefore exactly 0.0 in dry-run — run "
            "`--live` with a warm bge-m3 cache for a real "
            "comparison."
        )
        lines.append("")
        return "\n".join(lines)


# --- Verdict helper -----------------------------------------------------


VERDICT_SIGNIFICANT_LIFT = "significant_lift"
VERDICT_NO_SIGNIFICANT_LIFT = "no_significant_lift"
VERDICT_REGRESSION = "regression"


def _verdict_for(precision_lift: float, floor: float) -> str:
    """Map a precision lift to the three-state verdict.

    - ``precision_lift >= floor`` → significant_lift (bge-m3 wins).
    - ``-floor <= precision_lift < floor`` → no_significant_lift
      (no regression, no significant improvement).
    - ``precision_lift < -floor`` → regression (bge-m3 worse).
    """
    if precision_lift >= floor:
        return VERDICT_SIGNIFICANT_LIFT
    if precision_lift >= -floor:
        return VERDICT_NO_SIGNIFICANT_LIFT
    return VERDICT_REGRESSION


# --- JSONL loader -------------------------------------------------------


def _parse_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping leading ``# ...`` comments.

    Mirrors ``scripts.eval_ragas._parse_jsonl``. Kept as a private
    helper here so this module doesn't grow an import dependency
    on the runner script (a small Surface area for tests to lock
    down).
    """
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def _load_cloze_rows(path: Path) -> list[tuple[int, str]]:
    """Project the cloze JSONL to ``(word_id, judgment)`` tuples.

    Same projection ``scripts.eval_ragas._load_cloze_judgments``
    uses, kept local so this module stays hermetic.
    """
    rows = _parse_jsonl(path)
    return [(int(r["word_id"]), str(r["judgment"])) for r in rows]


# --- Deterministic proxy (dry-run path) ---------------------------------


def _proxy_metric(judgment: str) -> float:
    """Dry-run proxy: ``accept`` → 1.0, ``reject`` → 0.0.

    Same shape as ``scripts.eval_ragas._proxy_metric``. The proxy
    is intentionally not embedding-aware — the dry-run path's
    lift is therefore exactly 0.0 by construction. The live path
    is the only place a non-trivial lift appears.
    """
    return 1.0 if judgment == "accept" else 0.0


def _run_dry_metrics(
    *,
    judgments: list[tuple[int, str]],
) -> list[PerRowMetric]:
    """Build the per-row metrics under the dry-run proxy.

    Every row gets the same proxy for all four Ragas metrics.
    The order matches ``judgments`` so the per-row CSV writes
    deterministically.
    """
    out: list[PerRowMetric] = []
    for word_id, judgment in judgments:
        v = _proxy_metric(judgment)
        out.append(
            PerRowMetric(
                word_id=word_id,
                context_precision=v,
                context_recall=v,
                faithfulness=v,
                answer_relevance=v,
                judgment=judgment,
            )
        )
    return out


# --- Aggregation --------------------------------------------------------


def _aggregate_run(
    *,
    embedding_model: str,
    per_row: list[PerRowMetric],
) -> ModelRunResult:
    """Mean per metric + mean overall for one embedding run."""
    n = len(per_row)
    if n == 0:
        return ModelRunResult(
            embedding_model=embedding_model,
            per_row=tuple(),
            mean_per_metric={m: 0.0 for m in RAGAS_METRICS},
            mean_overall=0.0,
        )

    means: dict[str, float] = {}
    for m in RAGAS_METRICS:
        means[m] = sum(getattr(r, m) for r in per_row) / n
    mean_overall = sum(means.values()) / len(RAGAS_METRICS)
    return ModelRunResult(
        embedding_model=embedding_model,
        per_row=tuple(per_row),
        mean_per_metric=means,
        mean_overall=mean_overall,
    )


def _aggregate_lift(
    *,
    current: ModelRunResult,
    bge_m3: ModelRunResult,
) -> tuple[dict[str, float], float, float]:
    """Per-metric lift ``bge_m3 - current`` + overall + precision shortcut."""
    lift_per_metric: dict[str, float] = {}
    for m in RAGAS_METRICS:
        lift_per_metric[m] = (
            bge_m3.mean_per_metric.get(m, 0.0)
            - current.mean_per_metric.get(m, 0.0)
        )
    lift_overall = bge_m3.mean_overall - current.mean_overall
    precision_lift = lift_per_metric["context_precision"]
    return lift_per_metric, lift_overall, precision_lift


# --- Live path stub (documented for Phase 8) ---------------------------


def _run_live_metrics(
    *,
    judgments: list[tuple[int, str]],
    embedding_model: str,
) -> list[PerRowMetric]:
    """Compute real per-row metrics under ``embedding_model``.

    **Not implemented in Phase 7.5.** The Phase 7.5 deliverable is
    the *comparison report structure* plus the deterministic
    dry-run path that proves the runner is wired end-to-end. A
    full live path requires:

    1. ``app.embeddings.embed()`` (or the bge-m3
       ``sentence-transformers`` adapter) for both models.
    2. A live pgvector store containing the held-out rows'
       source context sentences.
    3. Ragas's ``evaluate(...)`` for the four metrics.

    All three exist in the codebase (Phase 1, Phase 6.7) but
    the live A/B path is a Phase 8 follow-up — Phase 7.5
    defers it per the spec's "out of scope" line. To keep
    the function surface honest, the live path raises
    ``NotImplementedError`` so the CLI surfaces a clear
    error rather than silently degrading to the proxy.

    Tests stub this via ``monkeypatch``; the runner script
    is the only entry point.
    """
    raise NotImplementedError(
        "Phase 7.5 live retrieval-comparison path is deferred "
        "to Phase 8. Use --dry-run for the deterministic "
        "comparison structure."
    )


# --- Cold-cache probe ---------------------------------------------------


def is_bge_m3_cache_warm() -> bool:
    """Return True iff the local bge-m3 cache marker exists.

    The HuggingFace cache layout puts the model under
    ``~/.cache/huggingface/hub/models--BAAI--bge-m3/``. If the
    sentinel directory exists, the model has been downloaded
    at least once. We do not try to *load* the model here —
    that's the runner's job, and it would fail loudly on a
    cold cache. The probe is a cheap "should I skip the test?"
    check.
    """
    return BGE_M3_CACHE_SENTINEL.exists()


# --- Top-level entry point ---------------------------------------------


def compare_embedding_models(
    judgments_path: str,
    *,
    dry_run: bool = True,
    current_model: str | None = None,
    bge_m3_model: str | None = None,
    live_runner: Callable | None = None,
) -> ComparisonReport:
    """Run the A/B on the cloze held-out set and return a report.

    Parameters
    ----------
    judgments_path
        Path to ``eval/cloze_judgments.jsonl`` (or a subset).
    dry_run
        If True (default), use the deterministic proxy and never
        touch the embedding client. If False, delegate to the
        live runner (currently ``NotImplementedError``).
    current_model, bge_m3_model
        Override the module-level constants. Useful for tests;
        production callers should leave these alone so the
        spec's locked model ids are the single source of truth.
    live_runner
        Injectable seam for the live path. Tests can swap this
        for a function that returns synthetic ``PerRowMetric``
        lists without touching the network.

    Returns
    -------
    ComparisonReport
        The dataclass the runner script serializes to CSV +
        markdown. Tests assert against this shape.
    """
    path = Path(judgments_path)
    if not path.exists():
        raise FileNotFoundError(f"judgments file not found: {path}")

    judgments = _load_cloze_rows(path)
    use_current = current_model or EMBEDDING_MODEL_CURRENT
    use_bge = bge_m3_model or EMBEDDING_MODEL_BGE_M3

    if dry_run:
        current_per_row = _run_dry_metrics(judgments=judgments)
        bge_per_row = _run_dry_metrics(judgments=judgments)
    else:
        # Live path: delegate to the injected runner (Phase 8 wires
        # this to the real embedding + Ragas path). The default
        # raises NotImplementedError so callers get a clear error
        # rather than silent degradation.
        if live_runner is None:
            live_runner = _run_live_metrics
        current_per_row = live_runner(
            judgments=judgments, embedding_model=use_current
        )
        bge_per_row = live_runner(
            judgments=judgments, embedding_model=use_bge
        )

    current_run = _aggregate_run(
        embedding_model=use_current, per_row=current_per_row
    )
    bge_run = _aggregate_run(
        embedding_model=use_bge, per_row=bge_per_row
    )

    lift_per_metric, lift_overall, precision_lift = _aggregate_lift(
        current=current_run, bge_m3=bge_run
    )
    verdict = _verdict_for(precision_lift, RETRIEVAL_MIN_QUALITY_FLOOR)

    return ComparisonReport(
        current=current_run,
        bge_m3=bge_run,
        lift_per_metric=lift_per_metric,
        lift_overall=lift_overall,
        precision_lift=precision_lift,
        verdict=verdict,
        floor=RETRIEVAL_MIN_QUALITY_FLOOR,
        judgments_path=str(path),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# --- Floor literal guard (Hard rule #7, type-level guardrail) -----------


def _assert_floor_literal_is_float() -> None:
    """Defensive guard for Hard rule #7.

    The spec calls for ``RETRIEVAL_MIN_QUALITY_FLOOR`` to be a
    hard-coded module constant — *not* env, *not* config. The
    guard below is belt-and-braces: even if a future maintainer
    re-binds the module attribute to ``os.environ.get(...)``,
    this function (called at import time) raises if the value
    isn't a literal float. Tests exercise the rejection path by
    patching the module attribute before import.

    The function is private and has no side effects when the
    constant is correct. Run as part of ``__all__`` consumers
    (the runner script) so an import-time misconfiguration
    fails loudly.
    """
    val = RETRIEVAL_MIN_QUALITY_FLOOR
    if not isinstance(val, float):
        raise TypeError(
            f"RETRIEVAL_MIN_QUALITY_FLOOR must be a literal float, "
            f"got {type(val).__name__}: {val!r}. Hard rule #7: "
            "this is a code-level guardrail, not a config knob."
        )
    if val < 0.0 or val > 1.0:
        # A floor outside [0, 1] is meaningless for a metric that
        # lives in [0, 1]. Catch the obvious "I typed 5 instead of
        # 0.05" typo at import time.
        raise ValueError(
            f"RETRIEVAL_MIN_QUALITY_FLOOR={val!r} is outside the "
            f"[0.0, 1.0] metric range. Did you mean a smaller value?"
        )


# Run the import-time guard. The runner script and tests both
# import this module, so a misconfigured floor fails at the
# earliest possible surface.
_assert_floor_literal_is_float()


__all__ = [
    "RETRIEVAL_MIN_QUALITY_FLOOR",
    "EMBEDDING_MODEL_BGE_M3",
    "EMBEDDING_MODEL_CURRENT",
    "RAGAS_METRICS",
    "BGE_M3_CACHE_SENTINEL",
    "PerRowMetric",
    "ModelRunResult",
    "ComparisonReport",
    "VERDICT_SIGNIFICANT_LIFT",
    "VERDICT_NO_SIGNIFICANT_LIFT",
    "VERDICT_REGRESSION",
    "is_bge_m3_cache_warm",
    "compare_embedding_models",
]