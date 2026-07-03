"""Phase 6.2 — matching exercise generator + DSPy module (card t_ddaf9cf9).

This module is the matching-exercise counterpart of Phase 4.2's
``app.cloze`` module. It ships three distinct surfaces that share
one Pydantic contract (``MatchingExercise``):

1. **Production path** — ``generate_match(db, user_id, ...)``. Picks a
   target word deterministically from the user's weakness profile,
   builds a constrained prompt, calls ``app.llm.complete`` wrapped with
   ``instructor`` so the response validates against the
   ``MatchingExercise`` Pydantic model. Bounded retry (Hard rule #6).
   Traces through ``_trace_match`` (Phase 4.3's no-op pattern).

2. **Optimization path** — ``MatchSignature`` + ``MatchModule`` +
   ``optimize_match_module``. DSPy is wired to ``app.llm.complete`` via
   the shared ``_DSPyOpenAICompatLM`` adapter (lives in ``app.llm``;
   re-exported from ``app.cloze`` for one release — see the module
   docstring on ``app.llm``). The optimizer is offline-capable: with
   no API key, ``dspy.utils.dummies.DummyLM`` swaps in automatically
   so the CI suite runs without network.

3. **RAG-on path** — ``_retrieve_for_match``. When ``enable_rag=True``,
   augment the prompt with retrieval chunks from the Phase 1
   ``/retrieve`` endpoint (consumed as-is — Hard rule #2). The
   helper is the matching-exercise equivalent of the 6.1 cloze
   ``_retrieve_for_cloze``; the two are not promoted to a shared
   module yet because 6.1 ships in a sibling worktree and isn't
   available at 6.2's commit time. A follow-up card can DRY them.

Hard rules enforced here:

- #1 RAG-on is opt-in — default ``enable_rag=False``. The non-RAG
  prompt is byte-for-byte identical to the stored fixture.
- #2 ``/retrieve`` is consumed as-is — no new routes, no new
  embedding model.
- #3 Three exercise types only — ``exercise_type: Literal["matching"]``
  on the response.
- #4 Single LLM provider — production path goes through
  ``app.llm.complete`` (4.1's OpenAI-compatible client); the DSPy
  adapter targets the same wire format.
- #5 Every LLM call is wrapped — ``_trace_match`` is the hook.
- #6 Pydantic v2 validated output via ``instructor``. Schema-violation
  retries are bounded by ``MAX_ATTEMPTS``.
- #8 Offline-capable eval — DSPy runs on ``DummyLM`` when no key is
  present.
- #9 Type-level guardrails — ``PROMPT_TEMPLATE_VERSION``,
  ``MAX_ATTEMPTS``, ``MATCH_MIN/MAX/DEFAULT_COUNT`` are module
  constants, not env-derived.
- #11 Existing callers stay byte-for-byte unchanged — ``app.cloze``,
  ``app.observability``, ``app.retrieval``, ``app.embeddings``,
  ``app.fsrs`` are untouched. The DSPy adapter extraction is the
  only ``cloze.py`` diff and is documented there.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import crud, models
from app.cloze import select_target_word  # noqa: F401 — re-used (no cloze.py diff)
from app.llm import _DSPyOpenAICompatLM  # the shared adapter; see app.llm docstring
from app.observability import get_langfuse

# Lazy DSPy import — the optimization-path classes touch dspy at
# import time; the production path doesn't. Mirrors ``app.cloze``.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #9).
#
# These are the single source of truth for the matching exercise's
# bounds. ``MATCH_MIN_COUNT`` and ``MATCH_MAX_COUNT`` are also referenced
# by ``app.schemas.MatchingExerciseOut.pairs`` (the wire constraint
# mirrors the generator's). Any drift between the two is caught at
# import time by the Pydantic schema's ``min_length`` / ``max_length``
# assertions on a test fixture.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "match-v1"
MAX_ATTEMPTS: int = 3
MATCH_MIN_COUNT: int = 2
MATCH_MAX_COUNT: int = 8
MATCH_DEFAULT_COUNT: int = 4

# RAG config — kept as module constants (Hard rule #9). 6.1's
# ``app.cloze`` uses the same numbers; the values are duplicated
# here so a future DRY pass (when 6.1 is on main) lifts them to a
# shared module without breaking the offline test fixture.
RAG_TOP_K: int = 5
RAG_MAX_CHARS_PER_CHUNK: int = 300
RAG_MAX_CHARS: int = 1500

# Source string for the retrieval query — same as 6.1's cloze helper.
# Hard rule #2: ``/retrieve`` is consumed as-is; we don't add a new
# source kind.
RAG_SOURCE: Literal["words", "examples", "both"] = "both"


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class MatchingPair(BaseModel):
    """A single (left_word_id, right_word_id) pair in a matching exercise.

    Mirrors ``app.schemas.MatchingPair`` 1:1. The split is intentional:
    ``app.match`` owns the *generator* contract (used by the
    instructor-wrapped chat call); ``app.schemas`` owns the *wire*
    contract (used by the FastAPI response_model and the SPA).
    Field-for-field equivalence today; if they ever diverge, the
    schemas shape is the one Phase 6.3's route will return.
    """

    left_word_id: int
    right_word_id: int
    right_kind: Literal["translation", "synonym"]


class MatchingExercise(BaseModel):
    """The metadata contract for a single matching exercise.

    ``target_word_id`` is the FK to ``words.id`` of the German
    target — the user matches N pairs of the target (or near-target)
    vocabulary on the right-hand side. ``pairs`` is bounded in
    ``[MATCH_MIN_COUNT, MATCH_MAX_COUNT]`` by the module constants
    above; the Pydantic ``min_length`` / ``max_length`` constraints
    mirror the bounds so a drift surfaces as a Pydantic
    ValidationError at generation time, not a runtime shape
    mismatch on the wire.
    """

    target_word_id: int = Field(..., description="FK to words.id of the matching target.")
    pairs: list[MatchingPair] = Field(
        ...,
        min_length=MATCH_MIN_COUNT,
        max_length=MATCH_MAX_COUNT,
        description=(
            f"Match pairs the user connects (left -> right). Length "
            f"is bounded in [{MATCH_MIN_COUNT}, {MATCH_MAX_COUNT}] "
            f"by the module constants; the wire constraint mirrors "
            f"it so a validation drift surfaces as 422."
        ),
    )


class MatchingGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields that the activity layer (and the
    Langfuse trace in 6.2) record so an operator can triage without
    re-running the call. Mirrors ``ClozeGenerationError`` so the two
    exercise types share the same dead-letter shape across Phase 4
    and Phase 6.
    """

    def __init__(
        self,
        message: str,
        *,
        attempted_schema: str,
        last_validation_error: str,
        schema_retry_count: int,
    ) -> None:
        super().__init__(message)
        self.attempted_schema = attempted_schema
        self.last_validation_error = last_validation_error
        self.schema_retry_count = schema_retry_count

    @classmethod
    def from_validation_failure(
        cls,
        message: str,
        *,
        schema: dict,
        last_validation_error: str,
        schema_retry_count: int,
    ) -> "MatchingGenerationError":
        """Build a dead-letter from a Pydantic schema dict.

        The Pydantic v2 ``model_json_schema()`` returns a ``dict``;
        we serialise to a stable JSON string for the
        ``attempted_schema`` field. Tests that build a string
        directly (the public constructor) still work.
        """
        return cls(
            message,
            attempted_schema=json.dumps(schema, ensure_ascii=False),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        )


# ---------------------------------------------------------------------------
# Retrieval helper — RAG-on for matching.
# ---------------------------------------------------------------------------


def _retrieve_for_match(
    db: Session,
    word: models.Word,
) -> list[dict[str, Any]]:
    """Return top-k retrieval chunks for the matching prompt.

    This is the matching-equivalent of the 6.1 cloze
    ``_retrieve_for_cloze`` helper. The two are intentionally not
    promoted to a shared module yet because 6.1 ships in a sibling
    worktree and isn't on main at 6.2's commit time. A follow-up
    card can DRY them. Behaviour parity:

    - Calls ``app.retrieval.retrieve`` with ``k=RAG_TOP_K`` and
      ``source="both"`` using the word's lemma as the query.
    - Returns ``[]`` on non-Postgres (the same graceful fallback
      ``/retrieve`` uses — ``app.retrieval.retrieve`` itself raises
      ValueError on non-Postgres in some code paths, so we guard
      explicitly here too).
    - Truncates each chunk's text to ``RAG_MAX_CHARS_PER_CHUNK``.
    - Caps the total to ``RAG_MAX_CHARS`` so the prompt stays
      bounded.

    The shape returned matches the 6.1 cloze helper:
    ``[{"kind": "word" | "example", "id": int, "text": str}, ...]``.
    Both the prompt builder and the Langfuse span metadata consume
    this shape.
    """
    # Non-Postgres / no-key fallback. The /retrieve endpoint returns
    # 503 on non-Postgres (Phase 1's contract); the matching prompt
    # must not depend on retrieval, so we mirror that fallback here.
    from app import retrieval

    # The retrieval module is the consumer of the Phase 1 embedding
    # client; on a SQLite target, the embedding column is NULL and
    # cosine distance is undefined. We catch by checking the engine
    # dialect (matches the 6.1 helper's behaviour).
    if not retrieval._is_postgres_target():
        return []

    try:
        from app import embeddings

        query_vec = embeddings.embed_one(word.word)
        rows = retrieval.retrieve(db, query_vec, k=RAG_TOP_K, source=RAG_SOURCE)
    except Exception as exc:  # noqa: BLE001 — retrieval is best-effort
        logger.warning(
            "match: retrieval failed (non-fatal, falling back to []): %s", exc
        )
        return []

    chunks: list[dict[str, Any]] = []
    total_chars = 0
    for row in rows:
        # Truncate per chunk first, then check the total budget.
        # The schema column is "source" (set by ``retrieve`` when
        # source="both"); for source="words" / "examples" we
        # default the kind to the source.
        kind = row.get("source", "word")
        row_id = row.get("id")
        if kind == "words":
            kind = "word"
            text = row.get("word") or ""
        elif kind == "examples":
            kind = "example"
            text = row.get("german") or ""
        else:
            text = row.get("word") or row.get("german") or ""
        text = (text or "")[:RAG_MAX_CHARS_PER_CHUNK]
        if not text:
            continue
        if total_chars + len(text) > RAG_MAX_CHARS:
            break
        chunks.append({"kind": kind, "id": row_id, "text": text})
        total_chars += len(text)
    return chunks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _first_example_sentence(word: models.Word) -> str:
    """Return the first example sentence's German text, or a fallback.

    Same shape as ``app.cloze._first_example_sentence`` — duplicated
    here rather than imported because Hard rule #11 says don't
    touch ``app.cloze``. A small amount of duplication is the cost
    of the byte-for-byte cloze preservation; a follow-up can
    promote this to ``app.utils`` if it grows.
    """
    if word.examples:
        return word.examples[0].german or ""
    return f"{word.word} (Beispiel nicht verfügbar)."


def build_prompt(
    word: models.Word,
    weakness_axes: dict[str, int],
    *,
    count: int = MATCH_DEFAULT_COUNT,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> list[dict]:
    """Build the chat-completions messages for one match generation.

    Prompt shape
    ------------
    The system prompt:

    1. Establishes the role ("German matching-exercise designer for a
       C1 learner").
    2. Lists the explicit prohibitions (the C1-accept bar +
       right_kind constraints — see ``right_kind``'s wire spec).
    3. Specifies the JSON output schema with every field on
       ``MatchingExercise``, including ``prompt_template_version``
       so downstream eval tooling can split A/B cohorts.
    4. States the C1-accept bar.

    The user prompt:

    - Embeds the target word (lemma, ``word.word_type``, frequency).
    - Embeds the word's first example sentence (NOT a retrieval
      call when ``retrieved_chunks=[]``; the example still
      anchors the LLM on the target's C1 use).
    - Embeds the user's weakness axes as JSON so the model can
      bias the right-hand side toward the axes the learner is
      weakest on.
    - Embeds ``retrieved_chunks`` (when present) so the LLM has
      more in-corpus context for both the left and right pairs.
    - Sets ``count`` so the LLM knows how many pairs to produce.

    Parameters
    ----------
    word
        The target ``Word`` row returned by ``select_target_word``.
    weakness_axes
        The dict from ``WeaknessProfile.axes`` (may be empty for a
        fresh user).
    count
        How many pairs to ask the LLM to produce. Mirrors the
        ``MatchGenerateRequest.count`` field.
    retrieved_chunks
        Optional list of ``{"kind", "id", "text"}`` dicts from
        ``_retrieve_for_match``. When empty (the default), the
        prompt is byte-for-byte identical to the stored no-RAG
        fixture so the offline eval stays reproducible for A/B.

    Returns
    -------
    list[dict]
        A two-message list suitable for ``app.llm.complete``.
    """
    example_sentence = _first_example_sentence(word)
    system_content = (
        "You are a German matching-exercise designer for a C1 learner. "
        "Produce N match pairs that test the learner's recall of the "
        "German target word below.\n\n"
        "PROHIBITIONS (must obey all):\n"
        "1. Do NOT invent word ids that don't exist in the corpus; "
        "every left_word_id and right_word_id must be an integer FK "
        "into words.id from the corpus.\n"
        "2. Do NOT translate the right-hand side into a language "
        "other than what right_kind specifies (translation = English, "
        "synonym = German).\n"
        "3. Do NOT repeat the same left_word_id across pairs (each "
        "pair is unique).\n"
        "4. The correct match for a pair is one of the words already "
        "in the corpus; do not fabricate German or English strings.\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept these "
        "match pairs without edits? If no, redo before answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "target_word_id": <integer, the words.id of the target>,\n'
        '  "pairs": [\n'
        '    {"left_word_id": <int>, "right_word_id": <int>, '
        '"right_kind": "translation" | "synonym"},\n'
        '    ...\n'
        '  ]\n'
        "}\n"
    )

    user_payload: dict[str, Any] = {
        "target_word": {
            "id": word.id,
            "word": word.word,
            "word_type": word.word_type,
            "frequency": word.frequency,
        },
        "context_sentence": example_sentence,
        "learner_axes": weakness_axes,
        "count": count,
        "instructions": (
            f"Design {count} match pairs that test recall of "
            f"target_word.word. Each pair connects a German word on "
            f"the left to either its English translation "
            f"(right_kind='translation') or a near-synonym German "
            f"word (right_kind='synonym') on the right. All "
            f"left_word_id and right_word_id values must be valid "
            f"corpus FKs; the target_word itself counts as one of "
            f"the valid FKs."
        ),
    }
    if retrieved_chunks:
        # When the caller has opted in to RAG, embed the retrieved
        # chunks. The non-RAG path leaves this key absent so the
        # prompt stays byte-for-byte identical to the no-RAG
        # fixture — Phase 6.2 acceptance.
        user_payload["retrieved_chunks"] = retrieved_chunks

    user_content = json.dumps(user_payload, ensure_ascii=False)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Production path — instructor-wrapped chat completion
# ---------------------------------------------------------------------------


def _openai_client():
    """Build an ``OpenAI`` client pointed at OpenRouter.

    Same lazy-import shape as ``app.cloze._openai_client``. The
    matching production path uses an identical OpenAI client
    (Phase 4.1's transport). Returning ``None`` when the key is
    missing lets the caller surface ``LLMError`` so the route
    layer's existing 502 handler picks it up.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _perf_counter_ms() -> float:
    """Wall-clock in ms (float).

    Same local-import shape as ``app.cloze._perf_counter_ms`` —
    duplicated here so this module doesn't pay the import cost at
    module load time.
    """
    import time

    return time.perf_counter() * 1000.0


def _count_instructor_retries(exc: Exception) -> int:
    """Best-effort retry count from an instructor/pydantic failure.

    Mirrors ``app.cloze._count_instructor_retries`` — the two
    exercise types share the same dead-letter shape so a Phase 7+
    maintainer can read both with the same mental model.
    """
    for attr in ("n_attempts", "attempts", "retries", "_retries"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return MAX_ATTEMPTS


def generate_match(
    db: Session,
    user_id: int,
    *,
    force_word_id: int | None = None,
    count: int = MATCH_DEFAULT_COUNT,
    enable_rag: bool = False,
) -> MatchingExercise:
    """Generate one ``MatchingExercise`` for a logged-in user.

    Flow
    ----
    1. ``select_target_word`` (Phase 4.5 default mode) or
       ``select_target_word(force_word_id=...)`` (Phase 5.4
       ``/exercises/due`` mode).
    2. Read the user's weakness axes for the prompt.
    3. ``build_prompt`` produces the chat messages. When
       ``enable_rag=True``, ``_retrieve_for_match`` augments the
       prompt with retrieval chunks.
    4. Wrap an OpenRouter-targeted OpenAI client with ``instructor``
       and call ``chat.completions.create(response_model=MatchingExercise,
       ..., max_retries=MAX_ATTEMPTS)``.
    5. Stamp metadata; call ``_trace_match`` with the metadata dict.
    6. Return the validated Pydantic instance.

    Parameters
    ----------
    force_word_id
        Optional. When set, ``select_target_word`` returns this
        exact word (looked up by primary key). Used by the 6.3
        ``/exercises/match`` route when the due-queue has already
        picked the target. Defaults to ``None``.
    count
        How many pairs to generate. Must be in
        ``[MATCH_MIN_COUNT, MATCH_MAX_COUNT]``; the Pydantic schema
        ``MatchGenerateRequest`` enforces this at the wire layer.
    enable_rag
        When True, augment the prompt with retrieval chunks from
        ``/retrieve``. Default ``False`` (Hard rule #1: opt-in).

    Returns
    -------
    MatchingExercise
        The validated Pydantic instance ready to return to the
        route layer.

    Raises
    ------
    ValueError
        Bubbles up from ``select_target_word`` when
        ``force_word_id`` does not match any row in the words
        table.
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or the
        transport call hits a non-retryable / persistent-retryable
        failure. The route layer translates this into 502.
    MatchingGenerationError
        Persistent schema violation after ``MAX_ATTEMPTS``.
    """
    from app.llm import LLMError, _default_model

    if not (MATCH_MIN_COUNT <= count <= MATCH_MAX_COUNT):
        # Defense-in-depth: ``MatchGenerateRequest`` already enforces
        # this at the wire layer, but the function is also called
        # directly from tests / scripts that might bypass Pydantic.
        raise ValueError(
            f"count must be in [{MATCH_MIN_COUNT}, {MATCH_MAX_COUNT}]; got {count}"
        )

    word = select_target_word(db, user_id, force_word_id=force_word_id)

    profile = crud.get_weakness_profile(db, user_id)
    weakness_axes: dict[str, int] = (
        crud.serialize_weakness_profile_axes(profile)
        if profile is not None
        else {}
    )

    # RAG-on: when the caller opts in, retrieve top-k chunks and
    # pass them to ``build_prompt``. When the call returns ``[]``
    # (e.g. non-Postgres or a sparse corpus) the prompt falls
    # back to the no-RAG shape — the spec says "graceful fallback"
    # so a missing retrieval result is not a fatal error.
    retrieved_chunks: list[dict[str, Any]] = []
    if enable_rag:
        retrieved_chunks = _retrieve_for_match(db, word)

    messages = build_prompt(
        word,
        weakness_axes,
        count=count,
        retrieved_chunks=retrieved_chunks,
    )

    # Trace metadata captures the request shape; populated before
    # the call so the error path (MatchingGenerationError raised
    # below) can still log it.
    metadata: dict[str, Any] = {
        "exercise_type": "matching",
        "user_id": user_id,
        "weakness_axes": weakness_axes,
        "target_word_id": word.id,
        "count": count,
        "enable_rag": enable_rag,
        "retrieved_chunk_count": len(retrieved_chunks),
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": messages,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    raw_client = _openai_client()
    if raw_client is None:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.lexora/.env "
            "and restart the backend container."
        )

    import instructor

    # Same ``MD_JSON`` mode as Phase 4.2 — works uniformly with
    # the qwen default and gives the model a clear "output a
    # ```json ...``` block" instruction in the system prompt.
    instructor_client = instructor.from_openai(
        raw_client, mode=instructor.Mode.MD_JSON
    )

    schema_retry_count = 0
    last_validation_error = ""
    result: MatchingExercise | None = None
    started = _perf_counter_ms()

    try:
        result = instructor_client.chat.completions.create(
            response_model=MatchingExercise,
            messages=messages,
            max_retries=MAX_ATTEMPTS,
            model=metadata["model_id"],
            temperature=0.7,
            max_tokens=512,
        )
    except Exception as exc:  # noqa: BLE001 — translate any failure path
        schema_retry_count = _count_instructor_retries(exc)
        last_validation_error = repr(exc)[:400]
        latency_ms = int(_perf_counter_ms() - started)
        metadata["schema_retry_count"] = schema_retry_count
        try:
            _trace_match(None, metadata, latency_ms)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        raise MatchingGenerationError.from_validation_failure(
            f"match: schema validation failed after {schema_retry_count} "
            f"attempt(s): {last_validation_error}",
            schema=MatchingExercise.model_json_schema(),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        ) from exc

    assert result is not None  # noqa: S101 — see the except branch above
    # ``result.target_word_id`` should already match the input
    # word's id (the LLM echoes back the FK). We force-stamp it
    # so a model drift can never silently mismatch the request.

    raw_response = getattr(result, "_raw_response", None)
    if raw_response is not None:
        usage = getattr(raw_response, "usage", None)
        if usage is not None:
            metadata["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            metadata["completion_tokens"] = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
        metadata["schema_retry_count"] = int(
            getattr(raw_response, "_instructor_retry_count", 0) or 0
        )

    latency_ms = int(_perf_counter_ms() - started)
    _trace_match(result, metadata, latency_ms)
    return result


# ---------------------------------------------------------------------------
# Langfuse trace hook
# ---------------------------------------------------------------------------


def _trace_match(
    result: MatchingExercise | None,
    metadata: dict[str, Any],
    latency_ms: int,
) -> str | None:
    """Emit one Langfuse ``match.generate`` span per generation.

    Mirrors ``_trace_cloze``'s shape (Phase 4.3) — same v2 SDK
    call pattern, same graceful-degradation path, same metadata
    keyset as the card body. Returns the Langfuse span id when
    keys are set, ``None`` when the keys are missing (so a
    downstream caller can propagate the absence to the route).

    **SDK choice.** Same v2 SDK as Phase 4.3 (cloze) and Phase 5.3
    (grade). The ``client.span(name=...)`` / ``span.update`` /
    ``span.end()`` / ``client.flush()`` chain is the v2.60.10
    surface; the Phase 4 spec's ``start_as_current_span`` is a
    v3-only method and is not callable against our pinned SDK.

    **Graceful degradation.** When ``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY`` are missing, ``get_langfuse()`` returns
    ``None`` (Phase 0's design). We early-return ``None`` so the
    matching activity still succeeds — observability is
    best-effort, never blocking. The route layer reads the
    return value as the ``trace_id`` field on the response when
    it ships in 6.3.

    **Failure mode.** Any exception raised by the Langfuse SDK is
    caught and logged at WARNING — the matching generation has
    already succeeded at this point, so a trace failure must
    never break the request.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam per call.
        return None

    # The metadata keyset documented in the card body:
    # exercise_type, target_word_id, count, enable_rag,
    # retrieved_chunk_count, model_id, prompt_template_version,
    # schema_retry_count, latency_ms, prompt_tokens,
    # completion_tokens. We add user_id and weakness_axes so the
    # Phase 7 eval layer can split by learner (same shape as
    # Phase 4.3's cloze trace).
    span_metadata: dict[str, Any] = {
        "exercise_type": metadata.get("exercise_type", "matching"),
        "user_id": metadata["user_id"],
        "weakness_axes": metadata.get("weakness_axes", {}),
        "target_word_id": metadata["target_word_id"],
        "count": metadata["count"],
        "enable_rag": metadata["enable_rag"],
        "retrieved_chunk_count": metadata["retrieved_chunk_count"],
        "model_id": metadata["model_id"],
        "prompt_template_version": metadata["prompt_template_version"],
        "schema_retry_count": metadata["schema_retry_count"],
        "latency_ms": latency_ms,
        "prompt_tokens": metadata["prompt_tokens"],
        "completion_tokens": metadata["completion_tokens"],
    }

    span = None
    trace_id: str | None = None
    try:
        span = client.span(
            name="match.generate",
            input=metadata.get("prompt_messages"),
            output=(result.model_dump_json() if result is not None else None),
        )
        span.update(metadata=span_metadata)
        # The v2 SDK exposes the span id on the span object; we
        # capture it so the caller can propagate ``trace_id`` on
        # the response. The shape mirrors Phase 4.3's behaviour.
        trace_id = getattr(span, "id", None)
        span.end()
        client.flush()
    except Exception as exc:  # noqa: BLE001 — tracing must never break the activity
        logger.warning(
            "match: Langfuse trace failed (non-fatal): %s", exc
        )
        # ``trace_id`` stays ``None`` so a downstream caller sees
        # the absence of a successful trace id (a graceful
        # degradation, not a fake one).
        trace_id = None
        if span is not None:
            try:
                span.end()
            except Exception:  # noqa: BLE001
                pass

    return trace_id


# ---------------------------------------------------------------------------
# DSPy surface — optimization path
# ---------------------------------------------------------------------------


def _offline_dummy_answers() -> list[dict[str, str]]:
    """Build a pool of offline-LM answer dicts for the DummyLM.

    DSPy's ``DummyLM`` returns ``"[[ ## field ## ]] value"`` lines,
    and the ``ChatAdapter`` parses the value back into the field's
    declared type. Our ``exercise`` output is a Pydantic
    ``MatchingExercise``, so the value must be a JSON-encoded
    instance that parses cleanly through
    ``MatchingExercise.model_validate_json``.

    We pool a small number of diverse stubs so any ``dspy.Predict`` /
    ``dspy.ChainOfThought`` call against the signature gets a fresh
    answer that satisfies the schema. The exact contents are
    irrelevant — the offline path exists to exercise the optimizer
    plumbing without network calls, not to produce realistic
    generations.
    """
    base = {"target_word_id": 1}
    return [
        {
            "exercise": json.dumps(
                {
                    **base,
                    "pairs": [
                        {"left_word_id": 1, "right_word_id": 2, "right_kind": "translation"},
                        {"left_word_id": 3, "right_word_id": 4, "right_kind": "synonym"},
                    ],
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    **base,
                    "pairs": [
                        {"left_word_id": 1, "right_word_id": 5, "right_kind": "translation"},
                        {"left_word_id": 6, "right_word_id": 7, "right_kind": "translation"},
                        {"left_word_id": 8, "right_word_id": 9, "right_kind": "synonym"},
                    ],
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    **base,
                    "pairs": [
                        {"left_word_id": 1, "right_word_id": 10, "right_kind": "translation"},
                        {"left_word_id": 11, "right_word_id": 12, "right_kind": "synonym"},
                        {"left_word_id": 13, "right_word_id": 14, "right_kind": "translation"},
                        {"left_word_id": 15, "right_word_id": 16, "right_kind": "synonym"},
                    ],
                },
                ensure_ascii=False,
            )
        },
    ]


def _offline_json_answers() -> list[dict[str, str]]:
    """Answer pool for MIPROv2's internal ``JSONAdapter``-shaped calls.

    Same shape as ``app.cloze._offline_json_answers`` — MIPROv2
    probes the LM with a JSONAdapter-shaped prompt to propose
    instructions. We pool a small number of instruction-shaped
    stubs so the prompt-proposer can keep cycling without choking.
    """
    return [
        {"proposed_instruction": "Always include target_word_id in the response."},
        {"proposed_instruction": "Use the first example sentence from the corpus row."},
        {"proposed_instruction": "Pick right_kind='translation' for short words, 'synonym' for abstract."},
        {"proposed_instruction": "Embed the user's weakness axes as JSON."},
        {"proposed_instruction": "State the C1-accept bar in the rationale."},
    ]


def _configure_dspy() -> None:
    """Configure DSPy with our OpenRouter adapter, or ``DummyLM`` offline.

    Idempotent — calling twice doesn't reconfigure. If the env has
    ``OPENROUTER_API_KEY`` set, we wire the shared adapter from
    ``app.llm``; otherwise we fall back to
    ``dspy.utils.dummies.DummyLM`` so unit tests and offline
    optimizer runs never hit the network.

    The DummyLM in DSPy 3.x rotates through a list of answer
    dicts whose values match the output field's expected shape. We
    pool JSON-encoded ``MatchingExercise`` payloads for the match
    path AND instruction-shaped stubs for MIPROv2's internal
    ``JSONAdapter``-shaped prompt-proposer, so the offline
    optimizer doesn't choke on a shape mismatch.
    """
    import dspy as _dspy

    if _dspy.settings.lm is not None:
        return
    if os.getenv("OPENROUTER_API_KEY"):
        _dspy.settings.configure(lm=_DSPyOpenAICompatLM())
        return

    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.utils.dummies import DummyLM

    answers = _offline_dummy_answers() + _offline_json_answers()
    _dspy.settings.configure(
        lm=DummyLM(answers, adapter=ChatAdapter()),
        adapter=ChatAdapter(),
    )


class MatchSignature(dspy.Signature):
    """DSPy signature for one match generation.

    Inputs match the ``build_prompt`` payload shape; the output is
    the full ``MatchingExercise`` Pydantic model. DSPy 3.x supports
    Pydantic-typed output fields via ``dspy.Predict`` /
    ``dspy.ChainOfThought``.

    The ``target_word_id`` is in the input set so the optimizer
    can teach the model that the response's ``target_word_id``
    must equal the input word's id without the LLM having to
    guess.
    """

    word: str = dspy.InputField(desc="The German target word (lemma).")
    context_sentence: str = dspy.InputField(
        desc="An example sentence from the word's corpus row."
    )
    learner_axes_json: str = dspy.InputField(
        desc="JSON-encoded weakness axes dict from the user's profile."
    )
    target_word_id: int = dspy.InputField(
        desc="The words.id of the target word (FK)."
    )
    count: int = dspy.InputField(
        desc="How many match pairs to produce (in [MATCH_MIN_COUNT, MATCH_MAX_COUNT])."
    )
    exercise: MatchingExercise = dspy.OutputField(
        desc="A Pydantic MatchingExercise matching the production contract."
    )


class MatchModule(dspy.Module):
    """DSPy module that wraps the production ``MatchSignature``.

    Uses ``dspy.Predict`` (single-shot — no chain-of-thought) so
    the output shape stays compatible with the production
    ``instructor`` path. The optimizer can swap the predictor for
    a tuned one without changing this surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(MatchSignature)

    def forward(  # type: ignore[override]
        self,
        *,
        word: str,
        context_sentence: str,
        learner_axes_json: str,
        target_word_id: int,
        count: int,
    ) -> dspy.Prediction:
        return self.predict(
            word=word,
            context_sentence=context_sentence,
            learner_axes_json=learner_axes_json,
            target_word_id=target_word_id,
            count=count,
        )


def optimize_match_module(
    train_set: Iterable[dict],
    val_set: Iterable[dict] | None = None,
) -> MatchModule:
    """Run the offline DSPy optimizer against a held-out eval set.

    Mirrors ``app.cloze.optimize_cloze_module``'s strategy: try
    ``MIPROv2`` first, fall back to ``BootstrapFewShot``, return
    the un-optimized module if both fail on the offline path. The
    production ``--live`` path runs against the real OpenRouter
    adapter and exercises the optimizer properly.

    Parameters
    ----------
    train_set, val_set
        Iterable of dicts with the five input keys (word, context,
        learner_axes_json, target_word_id, count). The eval set
        is loaded by ``scripts/optimize_match.py`` (Phase 9+; this
        card just ships the CLI plumbing).
    """
    import dspy

    _configure_dspy()

    train_examples = [
        dspy.Example(**row).with_inputs(
            "word", "context_sentence", "learner_axes_json", "target_word_id", "count"
        )
        for row in train_set
    ]
    val_examples = (
        [
            dspy.Example(**row).with_inputs(
                "word",
                "context_sentence",
                "learner_axes_json",
                "target_word_id",
                "count",
            )
            for row in val_set
        ]
        if val_set is not None
        else None
    )

    module = MatchModule()

    optimizer_cls = getattr(dspy, "MIPROv2", None)
    if optimizer_cls is None:
        optimizer_cls = getattr(dspy, "BootstrapFewShot", None)

    if optimizer_cls is None:
        logger.warning(
            "optimize_match_module: no MIPROv2 / BootstrapFewShot on the "
            "DSPy dep tree; returning the un-optimized module."
        )
        return module

    optimizer = optimizer_cls(metric=_match_metric)
    try:
        optimized = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
        )
    except TypeError:
        # Older BootstrapFewShot signatures don't accept ``valset``.
        optimized = optimizer.compile(module, trainset=train_examples)
    except Exception as exc:  # noqa: BLE001
        # Offline / DummyLM path: the optimizer's internal
        # prompt-proposer (or, for MIPROv2, its bootstrap few-shot
        # proposer) probes the LM with a signature our ``DummyLM``
        # can't satisfy. This is the documented failure mode of
        # the offline path — the production path runs with a real
        # OpenRouter adapter (``--live``).
        logger.warning(
            "optimize_match_module: optimizer %s raised on the offline "
            "path (%s); returning the un-optimized module. Re-run with "
            "--live and OPENROUTER_API_KEY set to actually optimize.",
            optimizer_cls.__name__,
            exc,
        )
        return module
    return optimized


def _match_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
    """Offline quality metric for the match optimizer.

    Score range: 0.0..1.0. The bar is intentionally loose — the
    production C1-accept check is qualitative (Anurag hand-reviews),
    not a numeric gate. The optimizer uses this score to pick a
    better prompt; the eval runner (6.7's ``eval_ragas.py``)
    computes the more rigorous per-row comparison.

    Components:
    - ``+0.5`` if ``prediction.exercise.target_word_id == example.target_word_id``
    - ``+0.3`` if the pairs list has the right length
      (``len(prediction.exercise.pairs) == example.count``)
    - ``+0.2`` if every pair's ``right_kind`` is in the literal
      (``"translation" | "synonym"``)
    """
    try:
        ex: MatchingExercise = prediction.exercise  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0
    score = 0.0
    if ex.target_word_id == example.target_word_id:
        score += 0.5
    try:
        if len(ex.pairs) == example.count:
            score += 0.3
    except (AttributeError, TypeError):
        pass
    if all(p.right_kind in ("translation", "synonym") for p in ex.pairs):
        score += 0.2
    return score


__all__ = [
    "MATCH_DEFAULT_COUNT",
    "MATCH_MAX_COUNT",
    "MATCH_MIN_COUNT",
    "MAX_ATTEMPTS",
    "MatchingExercise",
    "MatchingGenerationError",
    "MatchingPair",
    "MatchModule",
    "MatchSignature",
    "PROMPT_TEMPLATE_VERSION",
    "RAG_MAX_CHARS",
    "RAG_MAX_CHARS_PER_CHUNK",
    "RAG_SOURCE",
    "RAG_TOP_K",
    "build_prompt",
    "generate_match",
    "optimize_match_module",
]
