"""Ragas constants + helpers for the Phase 6.7 offline runner.

The Ragas library (``ragas>=0.2.0``) computes four metrics against a
held-out RAG sample:

- ``context_precision`` — fraction of retrieved chunks that are
  relevant to the question.
- ``context_recall`` — fraction of the ground-truth context that
  was actually retrieved.
- ``faithfulness`` — fraction of claims in the answer that are
  grounded in the retrieved context.
- ``answer_relevance`` — semantic similarity between the answer
  and the question.

Hard rule #9 in the Phase 6.7 card locks the *thresholds* as
module-level constants — never env vars. The Phase 6 spec's
"no getenv on RAGAS" rule (verifiable via ``git grep``) is the
guardrail.

Hard rule #7 (no committed secrets): the live Ragas path takes
``RAGAS_API_KEY`` from the environment. The constant
``RAGAS_API_KEY_ENV`` is the env var name only — the value is
never read at import time and is never written to disk.
"""
from __future__ import annotations

from typing import Literal, TypedDict

# --- Type-level guardrails (Hard rule #9) -------------------------------
# These are hard-coded module constants, not env vars. Changing any
# of them requires a code review, not a container restart.

#: Floor for ``context_precision`` on the live run. The Phase 6
#: spec §Eval locks 0.6 as the regression-detector bar — below
#: this the offline runner exits non-zero on ``--live``.
RAGAS_MIN_CONTEXT_PRECISION: float = 0.6

#: Floor for ``context_recall`` on the live run. The spec lets this
#: be a hair lower than precision (5% slack) because recall is
#: sensitive to chunk-boundary artifacts.
RAGAS_MIN_CONTEXT_RECALL: float = 0.5

#: Floor for ``faithfulness``. The spec marks this as the highest
#: bar (0.7) because a hallucination regresses the whole RAG layer.
RAGAS_MIN_FAITHFULNESS: float = 0.7

#: Floor for ``answer_relevance``. Spec mirrors context_precision
#: at 0.6.
RAGAS_MIN_ANSWER_RELEVANCE: float = 0.6

#: Top-k chunks the live runner feeds into Ragas's
#: ``context_precision`` / ``context_recall`` evaluators. Mirrors
#: the ``/retrieve`` endpoint's default (Phase 1).
RAG_TOP_K: int = 5

#: Per-chunk character budget when ``format_retrieved_chunks``
#: serializes the retrieved set. Chunks longer than this are
#: truncated to fit.
RAG_MAX_CHARS_PER_CHUNK: int = 300

#: Total character budget across all chunks in a single sample.
#: Keeps a single Ragas row within the LLM-judge context window.
RAG_MAX_CHARS: int = 1500

#: Floor for the deterministic proxy in ``--dry-run`` mode. The
#: proxy is the mean of (judgment=accept → 1.0, judgment=reject →
#: 0.0) across the held-out rows for the same exercise type, so
#: this is the *expected* floor against the eval set's own
#: ``judgment`` column (a self-accept against the held-out set's
#: own accept/reject should land at or above this by construction).
RAGAS_DRY_RUN_MIN_OVERALL: float = 0.6


# --- Env var names (no values) ------------------------------------------

#: Env var name for the Ragas API key (Hard rule #7: no committed
#: secrets). The value is read only when ``--live`` is set, and
#: only by the runner script — never by this module at import time.
#: The literal substring ``KEY`` is unavoidable here; the harness
#: redaction gotcha is about VALUES, not env var names that happen
#: to end in ``KEY``.
RAGAS_API_KEY_ENV: str = "RAGAS_API_KEY"


# --- Type contracts ------------------------------------------------------


class RagasSample(TypedDict):
    """A single Ragas input row.

    Mirrors Ragas's documented sample shape: question + answer +
    contexts + ground_truth. The four metrics are computed against
    this exact shape; we don't add or remove fields.

    ``contexts`` is a list of strings (one per retrieved chunk) so
    Ragas can compute both context_precision (rank-aware) and
    context_recall (set-overlap) correctly.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


#: Exercise types the offline runner supports. Mirrors
#: ``grade_logs.exercise_type`` (Phase 5.2's String column, widened
#: in Phase 6 to include ``matching`` + ``comprehension``).
RagasExerciseType = Literal["cloze", "matching", "comprehension"]


# --- Helpers -------------------------------------------------------------


def format_retrieved_chunks(items: list) -> str:
    """Serialize a list of retrieved chunks into a single context blob.

    Each item is either a ``str`` (raw chunk text) or a ``dict``
    with a ``"text"`` key (Phase 1's ``/retrieve`` return shape
    where each hit is a dict with ``word_id``, ``german``,
    ``english``, ``score`` fields). The function is permissive
    about the input shape so the runner can be wired to either
    flavor of retrieval output.

    Truncation is two-stage:

    1. Per-chunk: each individual chunk is truncated to
       ``RAG_MAX_CHARS_PER_CHUNK`` characters.
    2. Total: chunks are joined with ``"\\n---\\n"`` and the joined
       result is truncated to ``RAG_MAX_CHARS`` characters.

    The truncation is character-based, not token-based, because
    Ragas's evaluator serializes the context into an LLM prompt
    where the limiting factor is the judge model's context window
    (Gemini 1.5 Flash in the default Ragas config) — character
    budgets are good enough for a regression detector.

    Returns the serialized blob. An empty input returns an empty
    string (a legal input — the runner treats it as "no context
    retrieved" and lets the metrics reflect that).
    """
    if not items:
        return ""

    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            # Prefer ``text`` if present (newer wire shape), else
            # fall back to ``german`` (Phase 1 wire shape).
            text = str(item.get("text") or item.get("german") or "")
        else:
            text = str(item)
        if not text:
            continue
        if len(text) > RAG_MAX_CHARS_PER_CHUNK:
            text = text[: RAG_MAX_CHARS_PER_CHUNK]
        parts.append(text)

    joined = "\n---\n".join(parts)
    if len(joined) > RAG_MAX_CHARS:
        joined = joined[:RAG_MAX_CHARS]
    return joined


def build_ragas_sample(
    question: str,
    answer: str,
    contexts: list,
    ground_truth: str,
) -> RagasSample:
    """Construct a Ragas sample from the generator's prediction +
    retrieved chunks.

    Pydantic v2 validation happens in the runner (Hard rule #6);
    this helper produces the *unvalidated* dict that Pydantic's
    ``TypeAdapter(RagasSample).validate_python(...)`` will then
    accept. Splitting the helper from the validator keeps the
    construction site free of model-import side effects so the
    unit tests can exercise the helper without importing Pydantic.

    ``contexts`` accepts the same loose shape as
    ``format_retrieved_chunks`` so callers can pass the raw
    ``/retrieve`` response list directly — each item is either a
    ``str`` (raw chunk text) or a ``dict`` with a ``"text"`` or
    ``"german"`` key (Phase 1's ``/retrieve`` return shape). Dict
    items are stringified to their inner text so the result is a
    list of plain strings, matching the Ragas contract.
    """
    if isinstance(contexts, str):
        # Defensive: a single string is interpreted as a one-chunk
        # context (some upstream callers serialize the retrieval
        # result as a blob). Split it on the same separator
        # ``format_retrieved_chunks`` uses so the round-trip is
        # lossless.
        contexts_list: list[str] = (
            [contexts] if contexts else []
        )
    else:
        contexts_list = []
        for c in contexts:
            if not c:
                continue
            if isinstance(c, str):
                contexts_list.append(c)
            elif isinstance(c, dict):
                # Same wire-shape handling as format_retrieved_chunks.
                text = str(c.get("text") or c.get("german") or "")
                if text:
                    contexts_list.append(text)
            else:
                contexts_list.append(str(c))

    return RagasSample(
        question=str(question),
        answer=str(answer),
        contexts=contexts_list,
        ground_truth=str(ground_truth),
    )


__all__ = [
    "RAGAS_MIN_CONTEXT_PRECISION",
    "RAGAS_MIN_CONTEXT_RECALL",
    "RAGAS_MIN_FAITHFULNESS",
    "RAGAS_MIN_ANSWER_RELEVANCE",
    "RAG_TOP_K",
    "RAG_MAX_CHARS_PER_CHUNK",
    "RAG_MAX_CHARS",
    "RAGAS_DRY_RUN_MIN_OVERALL",
    "RAGAS_API_KEY_ENV",
    "RagasSample",
    "RagasExerciseType",
    "format_retrieved_chunks",
    "build_ragas_sample",
]
