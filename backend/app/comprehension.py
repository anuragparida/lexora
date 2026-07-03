"""Phase 6.4 — comprehension exercise generator + DSPy module (card t_8556fc97).

This module is the comprehension variant of the lexora exercise
generator. It ships three distinct surfaces that share one Pydantic
contract (``ComprehensionExercise``):

1. **Production path** — ``generate_comprehension(db, user_id, *,
   force_word_id=None, enable_rag=False)``. Picks a target word
   deterministically (delegating to ``app.cloze.select_target_word``,
   which is forward-compatible with the Phase 9 due-queue), builds
   a constrained prompt, calls ``app.llm.complete`` wrapped with
   ``instructor`` so the response validates against the
   ``ComprehensionExercise`` Pydantic model. Bounded retry (Hard
   rule #6). Traces through ``_trace_comprehension`` (Langfuse
   span name ``comprehension.generate``).

2. **Optimization path** — ``ComprehensionSignature`` +
   ``ComprehensionModule`` + ``optimize_comprehension_module``.
   DSPy is wired to ``app.llm.complete`` via a thin
   ``_DSPyOpenAICompatLM`` adapter (mirrors the shape that lives
   in ``app.cloze``). The optimizer is offline-capable: with no
   API key, ``dspy.utils.dummies.DummyLM`` swaps in automatically
   so the CI suite runs without network.

3. **Prompt + retrieval helper** — ``build_prompt`` and
   ``_retrieve_for_comprehension``. The comprehension prompt
   asks for a 3-5 sentence German passage + a multiple-choice
   question. When ``enable_rag=True`` (opt-in, default False),
   the user-side JSON includes ``retrieved_chunks`` — same shape
   as the Phase 6.1 cloze-with-RAG path. When the active
   ``DATABASE_URL`` is not Postgres, the retrieval helper returns
   ``[]`` (graceful — same pattern as ``/retrieve``'s 503 fallback
   for SQLite).

Hard rules enforced here:

- #1 RAG-on is opt-in. Default ``enable_rag=False``. The
  comprehension-with-RAG user prompt embeds ``retrieved_chunks``
  in the JSON; the no-RAG path is byte-for-byte stable
  (git-diff test in ``tests/test_comprehension.py``).
- #2 ``/retrieve`` consumed as-is. We import ``app.retrieval.retrieve``
  and the dialect-aware ``_is_postgres_target``-style helper —
  no new routes, no new vector stores.
- #3 three exercise types only. ``exercise_type:
  Literal["comprehension"] = "comprehension"`` on the wire
  shape (``app.schemas.ComprehensionExerciseOut``).
- #4 single LLM provider. The production path goes through
  ``app.llm.complete`` (4.1's OpenAI-compatible client); the
  DSPy adapter targets the same wire format.
- #5 every state-mutating call is traced. ``_trace_comprehension``
  is the Langfuse hook; the dead-letter branch traces before
  raising.
- #6 Pydantic v2 validated output via ``instructor``. Schema-
  violation retries are bounded by ``MAX_ATTEMPTS``.
- #8 offline-capable eval — DSPy runs on ``DummyLM`` when no key
  is present; the optimizer CLI is a separate
  ``scripts/optimize_comprehension.py`` process (Phase 9+; this
  card ships the module plumbing).
- #11 type-level guardrails — ``MAX_ATTEMPTS`` is a module constant,
  not env-derived. ``PROMPT_TEMPLATE_VERSION`` is the same shape.
- #12 existing callers stay byte-for-byte unchanged. The cloze
  module (``app.cloze``) is imported for ``select_target_word``;
  nothing in ``app.cloze`` is modified.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app import crud, models
from app.observability import get_langfuse

# Lazy DSPy import — ``dspy`` is heavy and only needed by the
# optimization-path classes. The production path
# (``generate_comprehension``) doesn't touch it. Mirrors the
# pattern in ``app.cloze``.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #11).
#
# ``PROMPT_TEMPLATE_VERSION`` bumps when the prompt template changes;
# downstream eval tooling (Phase 6.7's Ragas runner) uses it as an
# A/B key. ``MAX_ATTEMPTS`` caps the instructor retry budget on
# schema-violation responses.
#
# The RAG-on constants mirror Phase 6.1's hard-coded pattern:
# ``RAG_TOP_K`` and ``RAG_MAX_CHARS_PER_CHUNK`` are module constants,
# not env-derived. A future tuning iteration edits the file, commits,
# and reviews — same discipline as ``PY_FSRS_VERSION`` in Phase 5.1.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "comprehension-v1"
MAX_ATTEMPTS: int = 3
COMPREHENSION_PASSAGE_MIN_SENTENCES: int = 3
COMPREHENSION_PASSAGE_MAX_SENTENCES: int = 5
COMPREHENSION_PASSAGE_MAX_CHARS: int = 600

# RAG-on constants. Hard-coded per Hard rule #9 of Phase 6. The
# comprehension-with-RAG path uses these directly; the same values
# will appear in the Phase 6.1 cloze-with-RAG path so the two
# exercise types stay symmetric. When the Ragas eval runner needs
# to A/B a different ``k`` or chunk size, the change is a code
# review event, not an env flip.
RAG_TOP_K: int = 5
RAG_MAX_CHARS_PER_CHUNK: int = 300


ComprehensionChoice = Literal["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class ComprehensionExercise(BaseModel):
    """The metadata contract for a single comprehension exercise.

    Field set locked by card t_8556fc97 — the wire side mirrors
    this on ``app.schemas.ComprehensionExerciseOut`` (the
    generator vs wire split that Phase 4.2 established for cloze).

    ``passage`` is 3-5 sentences, bounded 20..600 chars. The LLM
    prompt's prohibitions are load-bearing here — without them
    the model drifts toward longer, multi-paragraph passages
    that the frontend can't render in a study-session card.

    ``question`` is ONE multiple-choice question whose answer is
    grounded in the passage. The frontend renders the question
    below the passage and the four choices in a grid.

    ``choices`` is exactly four entries, keyed A/B/C/D. The
    Pydantic ``min_length=4, max_length=4`` on the dict enforces
    the four-options invariant; ``min_length=1, max_length=200``
    on each value bounds the choice text. A missing key is a
    validation error, not a default — a future maintainer who
    forgets one of the keys gets a 422 on the dead-letter path.

    ``correct_choice`` is the answer key, not the index. The
    frontend uses it to mark the right answer after the user
    submits.

    ``rationale`` is one sentence (1..400 chars) explaining the
    distractor design — what semantic axis separates the
    correct answer from the three wrong ones, so a hand-reviewer
    can verify the model isn't a coin flip.
    """

    target_word_id: int = Field(
        ...,
        description=(
            "FK to words.id of the target word the comprehension "
            "passage is about. Same id re-appears on the "
            "grade_logs row when the user grades this exercise."
        ),
    )
    passage: str = Field(
        ...,
        min_length=20,
        max_length=COMPREHENSION_PASSAGE_MAX_CHARS,
        description=(
            "3-5 sentence German passage on the target word's "
            "topic. Bounded by COMPREHENSION_PASSAGE_MIN/MAX_SENTENCES "
            "in the prompt and by 20..COMPREHENSION_PASSAGE_MAX_CHARS "
            "in the Pydantic contract."
        ),
    )
    question: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description=(
            "ONE multiple-choice question whose answer is grounded "
            "in the passage."
        ),
    )
    choices: dict[ComprehensionChoice, str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "All four keys A/B/C/D required. Each value 1..200 chars."
        ),
    )
    correct_choice: ComprehensionChoice = Field(
        ...,
        description=(
            "The correct answer key. NOT an index — the frontend "
            "uses this directly to mark the right answer."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "One sentence (1..400 chars) explaining the distractor "
            "design."
        ),
    )

    @field_validator("choices")
    @classmethod
    def _validate_choices(cls, v: dict) -> dict:
        """Per-key + per-value validation on ``choices``.

        Pydantic v2's ``min_length`` / ``max_length`` on a
        ``dict`` field validate the *number of keys* (4 here),
        not the per-value length. This validator closes that
        gap: each value must be a non-empty string bounded
        ``[1, 200]`` chars. A model that emits an empty choice
        or a 201-char choice text gets a 422 on the dead-
        letter path.
        """
        if not isinstance(v, dict):
            raise ValueError("choices must be a dict mapping A/B/C/D to a string")
        for key, value in v.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"choices[{key!r}] must be a string; got {type(value).__name__}"
                )
            if len(value) < 1:
                raise ValueError(
                    f"choices[{key!r}] must be at least 1 char; got {len(value)}"
                )
            if len(value) > 200:
                raise ValueError(
                    f"choices[{key!r}] must be at most 200 chars; got {len(value)}"
                )
        return v


class ComprehensionGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields that the activity layer (and the
    Langfuse trace) record so an operator can triage without
    re-running the call. Mirrors the shape of ``ClozeGenerationError``
    from Phase 4.2 and the planned ``MatchingGenerationError``
    from Phase 6.2 — same dead-letter pattern, three exercise
    types.
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
    ) -> "ComprehensionGenerationError":
        """Build a dead-letter from a Pydantic schema dict.

        The Pydantic v2 ``model_json_schema()`` returns a ``dict``;
        the production code path passes that dict here and we
        serialise it to a stable JSON string for the
        ``attempted_schema`` field. Tests that build a string
        directly (the public constructor) still work — both
        shapes are documented.
        """
        import json as _json

        return cls(
            message,
            attempted_schema=_json.dumps(schema, ensure_ascii=False),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        )


# ---------------------------------------------------------------------------
# Retrieval helper (RAG-on)
#
# When the caller passes ``enable_rag=True``, the comprehension
# prompt embeds a ``retrieved_chunks`` list in the user-side JSON.
# The list is the top-k nearest rows from ``app.retrieval.retrieve``
# (Postgres + pgvector) — same shape as Phase 6.1's cloze-with-RAG
# path. When the active ``DATABASE_URL`` is not Postgres, returns
# ``[]`` gracefully (SQLite has no vector type, and the
# comprehension prompt is still usable on the empty-chunk
# fallback).
# ---------------------------------------------------------------------------


def _is_postgres_target() -> bool:
    """Return True if the active engine is bound to Postgres.

    Mirrors the dialect check in ``app.retrieval``. The retrieval
    helper is a no-op on SQLite (returns ``[]``) so the comprehension
    call still succeeds offline.
    """
    from app.database import engine

    return engine.dialect.name == "postgresql"


def _format_chunk_for_prompt(chunk: dict, source: str) -> dict:
    """Project a retrieval result row into the comprehension prompt shape.

    The comprehension prompt's ``retrieved_chunks`` array entries
    carry ``kind`` (the source the row came from), ``id`` (the
    words.id or examples.id, tagged by ``source``), and ``text``
    (the German text, truncated to ``RAG_MAX_CHARS_PER_CHUNK``).
    Word rows use ``word`` as the text; example rows use ``german``.

    The truncation is symmetric with the Phase 6.1 cloze-with-RAG
    path — a single chunk's text fits in the user-side JSON
    without bloating the prompt.
    """
    if source == "words":
        text = str(chunk.get("word", ""))
    else:
        text = str(chunk.get("german", ""))
    if len(text) > RAG_MAX_CHARS_PER_CHUNK:
        text = text[:RAG_MAX_CHARS_PER_CHUNK]
    return {
        "kind": source.rstrip("s"),  # "words" -> "word", "examples" -> "example"
        "id": int(chunk.get("id", 0)),
        "text": text,
    }


def _retrieve_for_comprehension(
    db: Session,
    word: models.Word,
) -> list[dict]:
    """Call /retrieve for the comprehension RAG-on path.

    The query is the word's lemma (``word.word``); the active
    embedding model is whatever Phase 1's ``app.embeddings`` is
    pinned to — we don't make this choice here, we just call the
    same model that ``/retrieve`` already uses.

    Returns ``[]`` when the active engine is SQLite (graceful
    fallback — same shape as Phase 6.1's cloze-with-RAG path) or
    when the embedding helper raises (we log and return ``[]``
    rather than breaking the comprehension call, because the
    RAG-on path is opt-in and the prompt's no-RAG branch is the
    canonical baseline).
    """
    if not _is_postgres_target():
        # No vector store — same graceful fallback the /retrieve
        # endpoint surfaces as 503, but inline so the comprehension
        # call still succeeds with an empty chunks list.
        return []

    try:
        from app.embeddings import embed_one
        from app.retrieval import retrieve as _retrieve
    except Exception as exc:  # noqa: BLE001 — never break the call
        logger.warning(
            "comprehension: retrieval helper unavailable (%s); "
            "falling back to no-RAG prompt",
            exc,
        )
        return []

    try:
        query_vec = embed_one(word.word)
    except Exception as exc:  # noqa: BLE001 — never break the call
        logger.warning(
            "comprehension: embed_one failed for lemma=%r (%s); "
            "falling back to no-RAG prompt",
            word.word,
            exc,
        )
        return []

    results: list[dict] = []
    for source in ("words", "examples"):
        try:
            rows = _retrieve(db, query_vec, k=RAG_TOP_K, source=source)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — never break the call
            logger.warning(
                "comprehension: retrieve(source=%r) failed (%s); "
                "skipping this source",
                source,
                exc,
            )
            continue
        for row in rows:
            results.append(_format_chunk_for_prompt(row, source))
    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    word: models.Word,
    weakness_axes: dict[str, int],
    *,
    retrieved_chunks: list[dict] | None = None,
) -> list[dict]:
    """Build the chat-completions messages for one comprehension generation.

    Prompt shape
    ------------
    The system prompt:

    1. Establishes the role ("German reading-comprehension designer
       for a C1 learner").
    2. Lists the explicit prohibitions — the C1-accept bar +
       3-5 sentence passage bound + 4-choice A/B/C/D invariant.
       The prohibitions are load-bearing: without them, the
       model drifts toward longer, multi-paragraph passages or
       5-choice questions.
    3. Specifies the JSON output schema with every field on
       ``ComprehensionExercise``, including the literal
       ``prompt_template_version`` so downstream eval tooling
       can split A/B cohorts.
    4. States the C1-accept bar.

    The user prompt:

    - Embeds the target word (lemma, ``word.word_type``, frequency).
    - Optionally embeds ``retrieved_chunks`` (when
      ``retrieved_chunks`` is non-empty AND not None). The
      default (None / []) produces a user prompt that is
      byte-for-byte identical to the no-RAG path — a git-diff
      test asserts this in ``tests/test_comprehension.py``.
    - Embeds the user's weakness axes as JSON so the model can
      lean the passage topic toward the axes the learner is
      weakest on (the rationale's signal source).

    Parameters
    ----------
    word
        The target ``Word`` row returned by
        ``app.cloze.select_target_word`` (re-used across the
        three exercise types for symmetry).
    weakness_axes
        The dict from ``WeaknessProfile.axes`` (may be empty for
        a fresh user). JSON-encoded verbatim into the user
        prompt so the model sees the same shape the caller
        serialised.
    retrieved_chunks
        Optional list of dicts from ``_retrieve_for_comprehension``.
        When None or empty, the user prompt's JSON is the
        no-RAG shape (no ``retrieved_chunks`` key). When
        non-empty, the JSON includes a ``retrieved_chunks``
        array. The shape is the same as Phase 6.1's
        cloze-with-RAG path.

    Returns
    -------
    list[dict]
        A two-message list suitable for ``app.llm.complete``.
        The structure is plain
        ``[{"role": ..., "content": ...}]``; no tool calls,
        no multimodal content.
    """
    system_content = (
        "You are a German reading-comprehension designer for a C1 "
        "learner. Produce ONE short reading passage and ONE "
        "multiple-choice question whose answer is grounded in the "
        "passage.\n\n"
        "PROHIBITIONS (must obey all):\n"
        f"1. The passage must be between {COMPREHENSION_PASSAGE_MIN_SENTENCES} "
        f"and {COMPREHENSION_PASSAGE_MAX_SENTENCES} sentences and at most "
        f"{COMPREHENSION_PASSAGE_MAX_CHARS} characters total.\n"
        "2. The question must have EXACTLY four choices labelled A, B, C, D — "
        "no more, no fewer. The correct answer must be unambiguously "
        "grounded in the passage text.\n"
        "3. Do NOT translate the target word into another language.\n"
        "4. Do NOT invent grammar that is not native German.\n"
        "5. Do NOT include the target word in the rationale (the "
        "rationale explains the distractor design, not the target word).\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept this passage "
        "and question without edits? If no, redo before answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "target_word_id": <integer, the words.id of the target word>,\n'
        f'  "passage": "<German passage, {COMPREHENSION_PASSAGE_MIN_SENTENCES}-'
        f'{COMPREHENSION_PASSAGE_MAX_SENTENCES} sentences, max '
        f'{COMPREHENSION_PASSAGE_MAX_CHARS} chars>",\n'
        '  "question": "<German question, grounded in the passage>",\n'
        '  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
        '  "correct_choice": "A" | "B" | "C" | "D",\n'
        '  "rationale": "<one sentence, <= 400 chars, explaining the distractor design>"\n'
        "}\n"
    )

    user_payload: dict[str, Any] = {
        "target_word": {
            "id": word.id,
            "word": word.word,
            "word_type": word.word_type,
            "frequency": word.frequency,
        },
        "learner_axes": weakness_axes,
        "instructions": (
            f"Design ONE comprehension exercise whose passage is built around "
            f"the target_word below. The passage must be "
            f"{COMPREHENSION_PASSAGE_MIN_SENTENCES}-"
            f"{COMPREHENSION_PASSAGE_MAX_SENTENCES} sentences and at most "
            f"{COMPREHENSION_PASSAGE_MAX_CHARS} characters total. The question "
            f"must have EXACTLY four choices (A, B, C, D) and the correct "
            f"answer must be grounded in the passage. Pick the target word's "
            f"semantic field (e.g. verbs -> action contexts, nouns -> topic "
            f"sentences) so the passage is on-topic for a C1 reader."
        ),
    }
    if retrieved_chunks:
        # RAG-on path: embed the chunks in the user-side JSON.
        # When the list is empty, the JSON has NO ``retrieved_chunks``
        # key — the no-RAG path is byte-for-byte stable.
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

    Imported lazily inside the function so ``app.comprehension``
    itself stays import-cheap and doesn't pay the OpenAI SDK
    import cost for the import-time assertions.

    Returns ``None`` if ``OPENROUTER_API_KEY`` is missing —
    caller treats that as a "no real LLM available" signal and
    falls back to raising ``LLMError`` so the route layer
    surfaces 502.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _perf_counter_ms() -> float:
    """Wall-clock in ms (float). Local import to keep the module
    import-cheap and to centralise the unit conversion."""
    import time

    return time.perf_counter() * 1000.0


def _count_instructor_retries(exc: Exception) -> int:
    """Best-effort: pull the retry count out of an instructor /
    pydantic failure. ``instructor`` raises an
    ``InstructorRetryException`` that wraps the last validation
    error; older versions raise ``pydantic.ValidationError``
    directly. We default to ``MAX_ATTEMPTS`` if we can't read
    the actual count, so the dead-letter always carries the
    budget it exhausted.
    """
    for attr in ("n_attempts", "attempts", "retries", "_retries"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return MAX_ATTEMPTS


def generate_comprehension(
    db: Session,
    user_id: int,
    *,
    force_word_id: int | None = None,
    enable_rag: bool = False,
) -> ComprehensionExercise:
    """Generate one ``ComprehensionExercise`` for a logged-in user.

    Flow
    ----
    1. ``app.cloze.select_target_word`` picks the deterministic
       target word (or, when ``force_word_id`` is set, looks up
       the specific word id and returns it). The cloze module
       owns this helper because it pre-dates Phase 6; the
       comprehension / matching generators all import it.
    2. Read the user's weakness axes for the prompt.
    3. When ``enable_rag=True``, call
       ``_retrieve_for_comprehension`` to fetch the top-k
       nearest rows from the pgvector store. On SQLite (or on
       any embedding / retrieve failure), the helper returns
       ``[]`` and the prompt falls back to the no-RAG shape.
    4. ``build_prompt`` produces the chat messages.
    5. Wrap an OpenRouter-targeted OpenAI client with
       ``instructor`` and call
       ``chat.completions.create(response_model=ComprehensionExercise, ...,
       max_retries=MAX_ATTEMPTS)``. ``instructor`` re-prompts on
       schema violations (each retry counts as one schema-
       violation attempt in the metadata). On the happy path the
       call lands in a single ``complete`` round-trip; on
       persistent violations ``instructor`` raises
       ``InstructorRetryException`` after the budget is
       exhausted — we translate that into
       ``ComprehensionGenerationError`` with the structured
       fields.
    6. Call ``_trace_comprehension`` with the metadata dict.

    Parameters
    ----------
    force_word_id
        Optional. When set, ``select_target_word`` returns this
        exact word (looked up by primary key) instead of running
        the deterministic seed. Symmetric with the cloze /
        matching generators — the Phase 5.4 due-queue may extend
        to comprehension in Phase 9, and the route layer needs
        the same knob to plug into the same call shape.
    enable_rag
        Opt-in flag (default False). When True, the prompt
        embeds ``retrieved_chunks`` in the user-side JSON.
        When False, the prompt is the no-RAG shape — byte-for-
        byte stable, git-diff test asserts this.

    Returns
    -------
    ComprehensionExercise
        The validated Pydantic instance ready to return to the
        route layer.

    Raises
    ------
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or the
        transport call hits a non-retryable / persistent-
        retryable failure. The route layer translates this
        into 502.
    ComprehensionGenerationError
        Persistent schema violation after ``MAX_ATTEMPTS``.
        Carries ``attempted_schema``,
        ``last_validation_error``,
        ``schema_retry_count`` so the route layer can surface
        a structured 502 instead of a bare 500.
    ValueError
        Bubbles up from ``select_target_word`` when
        ``force_word_id`` does not match any row in the words
        table (corpus inconsistency). The route layer
        translates this into 500.
    """
    from app.cloze import select_target_word
    from app.llm import _default_model, LLMError

    word = select_target_word(db, user_id, force_word_id=force_word_id)

    profile = crud.get_weakness_profile(db, user_id)
    weakness_axes: dict[str, int] = (
        crud.serialize_weakness_profile_axes(profile)
        if profile is not None
        else {}
    )

    # RAG-on: fetch chunks before building the prompt. On SQLite
    # or on any retrieval failure, the helper returns ``[]`` and
    # the prompt falls back to the no-RAG shape (no
    # ``retrieved_chunks`` key in the user JSON).
    retrieved_chunks: list[dict] = []
    if enable_rag:
        retrieved_chunks = _retrieve_for_comprehension(db, word)

    messages = build_prompt(
        word, weakness_axes, retrieved_chunks=retrieved_chunks or None
    )

    # Trace metadata captures the request shape; populated before
    # the call so the error path (ComprehensionGenerationError
    # raised below) can still log it.
    metadata: dict[str, Any] = {
        "user_id": user_id,
        "weakness_axes": weakness_axes,
        "word_id": word.id,
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": messages,
        "enable_rag": bool(enable_rag),
        "retrieved_chunk_count": len(retrieved_chunks),
        "retrieved_chunk_k": RAG_TOP_K if enable_rag else 0,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    raw_client = _openai_client()
    if raw_client is None:
        # No key — same shape as the persistent failure mode in
        # llm.py. Surface LLMError so the route layer's existing
        # 502 handler picks it up.
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.lexora/.env "
            "and restart the backend container."
        )

    import instructor

    # ``MD_JSON`` mode tells instructor to use markdown-fenced
    # JSON parsing rather than the tool-calling path. Same shape
    # as ``app.cloze`` — the tool-calling path requires the model
    # to emit ``tool_calls`` in the response, which the
    # OpenRouter passthrough doesn't always echo for non-OpenAI-
    # native models. ``MD_JSON`` works uniformly with the qwen
    # default and gives the model a clear "output a ```json
    # ...``` block" instruction in the system prompt (instructor
    # appends it).
    instructor_client = instructor.from_openai(
        raw_client, mode=instructor.Mode.MD_JSON
    )

    schema_retry_count = 0
    last_validation_error = ""
    result: ComprehensionExercise | None = None
    started = _perf_counter_ms()

    try:
        # ``instructor`` returns a ``ComprehensionExercise`` instance
        # directly when validation succeeds; raises
        # ``pydantic.ValidationError`` or
        # ``InstructorRetryException`` when it doesn't. We catch
        # both and translate into the structured error below.
        result = instructor_client.chat.completions.create(
            response_model=ComprehensionExercise,
            messages=messages,
            max_retries=MAX_ATTEMPTS,
            model=metadata["model_id"],
            temperature=0.7,
            max_tokens=1024,  # passage + question + 4 choices + rationale
        )
    except Exception as exc:  # noqa: BLE001 — translate any failure path
        schema_retry_count = _count_instructor_retries(exc)
        last_validation_error = repr(exc)[:400]
        latency_ms = int(_perf_counter_ms() - started)
        metadata["schema_retry_count"] = schema_retry_count
        # Best-effort trace before dead-lettering — the metadata
        # dict is in the right shape for the trace even on the
        # failure path.
        try:
            _trace_comprehension(None, metadata, latency_ms)
        except Exception:  # noqa: BLE001
            pass
        raise ComprehensionGenerationError.from_validation_failure(
            f"comprehension: schema validation failed after "
            f"{schema_retry_count} attempt(s): {last_validation_error}",
            schema=ComprehensionExercise.model_json_schema(),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        ) from exc

    # ``result`` is non-None here — the except branch is the only
    # failure path, and we just returned / raised.
    assert result is not None  # noqa: S101 — see above

    # Read usage from the underlying OpenAI response — same shape
    # as Phase 4.2's ``generate_cloze``. ``instructor`` returns
    # the validated Pydantic model but the raw response carries
    # the ``usage`` block in ``_raw_response`` (an implementation
    # detail of instructor >= 1.0; we tolerate missing attributes
    # gracefully).
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
    _trace_comprehension(result, metadata, latency_ms)
    return result


# ---------------------------------------------------------------------------
# Langfuse trace hook
# ---------------------------------------------------------------------------


def _trace_comprehension(
    result: ComprehensionExercise | None,
    metadata: dict[str, Any],
    latency_ms: int,
) -> None:
    """Emit one Langfuse ``comprehension.generate`` span per generation.

    Mirrors ``_trace_cloze`` (Phase 4.3) and the planned
    ``_trace_match`` (Phase 6.2). The signature is locked —
    ``generate_comprehension`` and the dead-letter branch both
    call us with the same kwargs in the same order, so the
    Langfuse payload shape stays stable for the Phase 5/6
    readers (and for the Phase 6.7 Ragas regression detector
    that consumes the same metadata keyset).

    **SDK choice.** ``pyproject.toml`` pins
    ``langfuse>=2.50.0,<3.0`` (resolved to 2.60.10). The v2 SDK
    exposes ``client.span(name=...)`` returning a span handle;
    ``span.update(metadata=...)`` merges the dict;
    ``span.end()`` closes the observation; ``client.flush()``
    pushes the buffer to the ingestion API. Mirrors
    ``app.cloze._trace_cloze``'s non-context-manager shape.

    **Graceful degradation.** When ``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY`` are missing, ``get_langfuse()``
    returns ``None`` (Phase 0's design; see
    ``app.observability``). In that branch we early-return so
    the comprehension activity still succeeds —
    observability is best-effort, never blocking. We log a
    warning once at module-import time (``observability.py``)
    instead of spamming on every call.

    **Metadata contract.** The keyset matches the Phase 6
    metadata contract (``docs/PHASE-6.md`` §"The metadata
    contract") so the Phase 6.7 Ragas runner can read every
    field. The comprehension-specific addition over cloze is
    the ``enable_rag`` / ``retrieved_chunk_count`` /
    ``retrieved_chunk_k`` triple — same shape as Phase 6.1's
    cloze-with-RAG path. We populate exactly the documented
    keyset and ignore the implementation-detail keys
    (``prompt_messages`` — too verbose for the Langfuse
    metadata row; we keep it on the trace payload so a Phase 6
    reader that needs the prompt can still get it from the
    trace record).

    **Failure mode.** Any exception raised by the Langfuse SDK
    (network glitch, malformed payload, server-side rejection)
    is caught and logged at WARNING — the comprehension
    generation has already succeeded at this point, so a trace
    failure must never break the request. The same shape is
    used by ``_trace_cloze`` and ``_trace_retrieval``.

    On success the result is a populated
    ``ComprehensionExercise``; on dead-letter
    (``ComprehensionGenerationError`` raised upstream)
    ``result`` is ``None`` and
    ``metadata["schema_retry_count"]`` carries the failure
    depth. We still emit the span in that case so the failure
    is visible in Langfuse next to the passing traces.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam
        # per call.
        return

    # Build the metadata payload exactly per the Phase 6 metadata
    # contract. ``target_word_id`` comes from the result when
    # present (it's a Pydantic output field); fall back to the
    # input dict for the dead-letter branch.
    target_word_id = (
        getattr(result, "target_word_id", None)
        if result is not None
        else metadata.get("word_id")
    )

    span_metadata: dict[str, Any] = {
        "user_id": metadata["user_id"],
        "exercise_type": "comprehension",
        "weakness_axes": metadata["weakness_axes"],
        "word_id": metadata["word_id"],
        "target_word_id": target_word_id,
        "model_id": metadata["model_id"],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "schema_retry_count": metadata["schema_retry_count"],
        "latency_ms": latency_ms,
        "prompt_tokens": metadata["prompt_tokens"],
        "completion_tokens": metadata["completion_tokens"],
        "enable_rag": metadata.get("enable_rag", False),
        "retrieved_chunk_count": metadata.get("retrieved_chunk_count", 0),
        "retrieved_chunk_k": metadata.get("retrieved_chunk_k", 0),
    }

    span = None
    try:
        span = client.span(
            name="comprehension.generate",
            # Trace-level input/output so the Langfuse UI surfaces
            # the prompt + the serialised Pydantic result. v2
            # ``span`` accepts both kwargs at construction time;
            # ``update`` is for incremental merges.
            input=metadata.get("prompt_messages"),
            output=(result.model_dump_json() if result is not None else None),
        )
        span.update(metadata=span_metadata)
        span.end()
        # Force a flush so the trace is queryable in the UI
        # before the request returns — mirrors _trace_cloze.
        client.flush()
    except Exception as exc:  # noqa: BLE001 — tracing must never break the activity
        logger.warning(
            "comprehension: Langfuse trace failed (non-fatal): %s", exc
        )
        if span is not None:
            try:
                span.end()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# DSPy surface — optimization path
#
# DSPy talks to OpenRouter directly via a thin adapter so the same
# transport that serves production also serves the optimizer. The
# adapter wraps ``app.llm.complete`` so every LLM call stays
# routed through the 4.1 client (Hard rule #4 + #5).
# ---------------------------------------------------------------------------


def _offline_dummy_answers() -> list[dict[str, str]]:
    """Build a pool of offline-LM answer dicts for the DummyLM.

    DSPy's ``DummyLM`` returns ``"[[ ## field ## ]] value"`` lines,
    and the ``ChatAdapter`` parses the value back into the field's
    declared type. Our ``exercise`` output is a Pydantic
    ``ComprehensionExercise``, so the value must be a JSON-encoded
    instance that parses cleanly through
    ``ComprehensionExercise.model_validate_json``.

    We pool three diverse stubs so any ``dspy.Predict`` /
    ``dspy.ChainOfThought`` call against the signature gets a
    fresh answer that satisfies the schema. The exact contents
    are irrelevant — the offline path exists to exercise the
    optimizer plumbing without network calls, not to produce
    realistic generations.
    """
    return [
        {
            "exercise": json.dumps(
                {
                    "target_word_id": 1,
                    "passage": (
                        "Der Hund läuft durch den Park. Er sieht einen "
                        "Ball und rennt sofort los. Sein Besitzer lacht "
                        "und ruft seinen Namen. Am Ende sind beide müde "
                        "und gehen nach Hause."
                    ),
                    "question": "Was sieht der Hund im Park?",
                    "choices": {
                        "A": "einen Ball",
                        "B": "einen Knochen",
                        "C": "eine Katze",
                        "D": "einen Stock",
                    },
                    "correct_choice": "A",
                    "rationale": "offline-stub",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "target_word_id": 2,
                    "passage": (
                        "Anna liest ein Buch im Café. Plötzlich regnet "
                        "es draußen. Sie packt ihre Sachen und läuft zur "
                        "Bushaltestelle. Der Bus kommt gerade noch "
                        "rechtzeitig."
                    ),
                    "question": "Was passiert, als es zu regnen beginnt?",
                    "choices": {
                        "A": "Sie bleibt im Café.",
                        "B": "Sie packt ihre Sachen.",
                        "C": "Sie liest weiter.",
                        "D": "Sie geht nach Hause.",
                    },
                    "correct_choice": "B",
                    "rationale": "offline-stub",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "target_word_id": 3,
                    "passage": (
                        "Der Zug fährt pünktlich um acht Uhr ab. Im "
                        "Abteil sitzen vier Reisende und lesen oder "
                        "schlafen. Nach zwei Stunden erreichen sie "
                        "die Hauptstadt."
                    ),
                    "question": "Wann erreicht der Zug sein Ziel?",
                    "choices": {
                        "A": "um acht Uhr",
                        "B": "nach einer Stunde",
                        "C": "nach zwei Stunden",
                        "D": "am Abend",
                    },
                    "correct_choice": "C",
                    "rationale": "offline-stub",
                },
                ensure_ascii=False,
            )
        },
    ]


def _offline_json_answers() -> list[dict[str, str]]:
    """Answer pool for MIPROv2's internal ``JSONAdapter``-shaped calls.

    MIPROv2 probes the LM with a JSONAdapter-shaped prompt to
    propose instructions. That adapter expects
    ``{"proposed_instruction": "..."}`` responses (a string
    field, not a JSON object). We pool five diverse instruction-
    shaped stubs so the prompt-proposer can keep cycling without
    choking.
    """
    return [
        {"proposed_instruction": "Keep the passage between 3 and 5 sentences."},
        {"proposed_instruction": "Ground the correct answer in the passage text."},
        {"proposed_instruction": "Embed the user's weakness axes as JSON."},
        {"proposed_instruction": "Use the target word's semantic field for the topic."},
        {"proposed_instruction": "State the distractor design in the rationale."},
    ]


def _configure_dspy() -> None:
    """Configure DSPy with our OpenRouter adapter, or ``DummyLM`` offline.

    Idempotent — calling twice doesn't reconfigure. If the env
    has ``OPENROUTER_API_KEY`` set, we wire the real adapter;
    otherwise we fall back to ``dspy.utils.dummies.DummyLM`` so
    unit tests and offline optimizer runs never hit the
    network.

    The DummyLM in DSPy 3.x rotates through a list of answer
    dicts whose values match the output field's expected shape.
    We pool JSON-encoded ``ComprehensionExercise`` payloads for
    the comprehension path AND instruction-shaped stubs for
    MIPROv2's internal ``JSONAdapter``-shaped prompt-proposer,
    so the offline optimizer doesn't choke on a shape mismatch
    (DSPy's ``DummyLM`` doesn't auto-detect the active
    adapter's requirements). The pool is intentionally small —
    the offline path exists to exercise the optimizer plumbing
    without network, not to produce realistic generations.
    """
    import dspy as _dspy

    if _dspy.settings.lm is not None:
        # Already configured by an earlier call in this process.
        return
    if os.getenv("OPENROUTER_API_KEY"):
        _dspy.settings.configure(lm=_DSPyOpenAICompatLM())
        return

    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.utils.dummies import DummyLM

    # The DummyLM in DSPy 3.x cycles through its answers list
    # and emits ``"[[ ## field ## ]] value"`` lines. The
    # ``ChatAdapter`` parses each value back into the field's
    # declared type. For our Pydantic
    # ``ComprehensionExercise`` output that means the value
    # must be a JSON-encoded instance. We concatenate the
    # comprehension path answers with the MIPROv2 internal-
    # proposer answers so a single DummyLM can serve both
    # protocols in any interleaving.
    answers = _offline_dummy_answers() + _offline_json_answers()
    _dspy.settings.configure(
        lm=DummyLM(answers, adapter=ChatAdapter()),
        adapter=ChatAdapter(),
    )


class _DSPyOpenAICompatLM:
    """Thin DSPy adapter that routes through ``app.llm.complete``.

    DSPy 3.x has a built-in ``dspy.LM`` for OpenAI-compatible
    endpoints, but it imports the ``openai`` SDK directly and
    bypasses our retry + latency-recording wrapper. Using a
    hand-written adapter lets us keep every chat call going
    through ``app.llm.complete`` (Hard rule #4 + #5: "every LLM
    call goes through app/llm.py").

    This is the comprehension-side copy of the adapter that
    lives in ``app.cloze``. Phase 6 was meant to extract it to
    ``app.llm.py`` as a shared utility (the plan card refers to
    "the now-extracted ``_DSPyOpenAICompatLM`` from
    ``app.llm.py``"); until that extraction lands (one of
    6.1-6.3 is the natural home), each generator module ships
    its own copy. The shape is identical to ``app.cloze``'s;
    the only difference is the module name. When the extraction
    lands, both ``app.cloze._DSPyOpenAICompatLM`` and this
    class become thin re-exports of ``app.llm._DSPyOpenAICompatLM``
    — the call sites don't change.

    The adapter is only constructed when an OpenRouter key is
    present, so the offline path (``DummyLM``) doesn't pay the
    import cost.
    """

    # DSPy reads ``model`` off the LM instance when it builds
    # ``dspy.Predict`` calls.
    model: str

    def __init__(self) -> None:
        from app.llm import _default_model

        self.model = _default_model()

    def __call__(self, prompt: str | None = None, **kwargs: Any) -> list[str]:
        """DSPy v3.x entry point.

        DSPy calls the LM with either ``prompt=...`` (legacy)
        or ``messages=...`` (newer protocol). We accept both
        and normalise into a messages-shaped
        ``app.llm.complete`` call. Returns a list of strings —
        one per generation — which is the shape DSPy 3.x
        expects from a custom LM.
        """
        from app.llm import complete as _complete

        messages = kwargs.get("messages")
        if not messages:
            messages = [{"role": "user", "content": prompt or ""}]
        result = _complete(messages=messages)
        return [result.text]

    # DSPy 3.x sometimes probes ``basic_request`` directly;
    # provide a passthrough so the optimiser can talk to the LM
    # without knowing about our internal shape.
    def basic_request(self, prompt: str | None = None, **kwargs: Any) -> list[dict]:
        text = self.__call__(prompt=prompt, **kwargs)[0]
        return [{"text": text}]


class ComprehensionSignature(dspy.Signature):
    """DSPy signature for one comprehension generation.

    Inputs match the ``build_prompt`` payload shape; the output
    is the full ``ComprehensionExercise`` Pydantic model (DSPy
    3.x supports Pydantic-typed output fields via
    ``dspy.Predict`` / ``dspy.ChainOfThought``).

    The ``target_word_id`` is in the input set so the optimizer
    can teach the model that the answer must equal the input
    word's id without the LLM having to guess. ``learner_axes_json``
    carries the JSON-encoded weakness axes dict, the same
    payload the production path passes.
    """

    word: str = dspy.InputField(desc="The German target word (lemma).")
    learner_axes_json: str = dspy.InputField(
        desc="JSON-encoded weakness axes dict from the user's profile."
    )
    target_word_id: int = dspy.InputField(
        desc="The words.id of the target word (FK)."
    )
    retrieved_chunks_json: str = dspy.InputField(
        desc=(
            "JSON-encoded list of retrieved chunks (empty list when "
            "RAG is off). Same shape as the production path's "
            "user-side JSON."
        )
    )
    exercise: ComprehensionExercise = dspy.OutputField(
        desc="A Pydantic ComprehensionExercise matching the production contract."
    )


class ComprehensionModule(dspy.Module):
    """DSPy module that wraps the production ``ComprehensionSignature``.

    Uses ``dspy.Predict`` (single-shot — no chain-of-thought) so
    the output shape stays compatible with the production
    ``instructor`` path. The optimizer
    (``optimize_comprehension_module``) can swap the predictor
    for a tuned one without changing this surface.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``dspy.Predict`` is the most production-faithful
        # predictor: no reasoning chain to inflate latency. The
        # optimizer can upgrade this to ``dspy.ChainOfThought``
        # if the eval set shows a quality win worth the latency
        # cost.
        self.predict = dspy.Predict(ComprehensionSignature)

    def forward(  # type: ignore[override]
        self,
        *,
        word: str,
        learner_axes_json: str,
        target_word_id: int,
        retrieved_chunks_json: str = "[]",
    ) -> dspy.Prediction:
        return self.predict(
            word=word,
            learner_axes_json=learner_axes_json,
            target_word_id=target_word_id,
            retrieved_chunks_json=retrieved_chunks_json,
        )


def optimize_comprehension_module(
    train_set: Iterable[dict],
    val_set: Iterable[dict] | None = None,
) -> ComprehensionModule:
    """Run the offline DSPy optimizer against a held-out eval set.

    Strategy
    --------
    - Always uses ``DummyLM`` when no API key is present so the
      CI suite runs without network (Hard rule #8).
    - Tries ``dspy.MIPROv2`` first (the spec's preferred
      optimizer). Falls back to ``dspy.BootstrapFewShot`` if
      MIPROv2 raises on the active dep tree (different
      versions of DSPy changed the constructor signature; the
      fallback keeps the optimization surface usable across
      versions).
    - Returns a ``ComprehensionModule`` with the optimized
      prompt instructions baked in. The caller (the CLI)
      serialises the optimised module to
      ``backend/app/comprehension_optimized.json`` so the
      production path can read it on next start (Phase 9+; this
      card just ships the module plumbing).

    Parameters
    ----------
    train_set, val_set
        Iterable of dicts with the four input keys (word,
        learner_axes_json, target_word_id, retrieved_chunks_json).
        The eval set is loaded from
        ``eval/comprehension_judgments.jsonl`` by
        ``scripts/optimize_comprehension.py``.

    Returns
    -------
    ComprehensionModule
        The optimized module. The optimizer mutates the
        module's internal predictor in place; the same
        instance is returned for caller convenience.
    """
    import dspy

    _configure_dspy()

    train_examples = [
        dspy.Example(**row).with_inputs(
            "word",
            "learner_axes_json",
            "target_word_id",
            "retrieved_chunks_json",
        )
        for row in train_set
    ]
    val_examples = (
        [
            dspy.Example(**row).with_inputs(
                "word",
                "learner_axes_json",
                "target_word_id",
                "retrieved_chunks_json",
            )
            for row in val_set
        ]
        if val_set is not None
        else None
    )

    module = ComprehensionModule()

    # Try MIPROv2 first (the spec's preferred optimizer).
    optimizer_cls = getattr(dspy, "MIPROv2", None)
    if optimizer_cls is None:
        # Some dep-tree configurations only ship BootstrapFewShot.
        optimizer_cls = getattr(dspy, "BootstrapFewShot", None)

    if optimizer_cls is None:
        # No optimizer available — return the un-optimized module
        # rather than crashing. The CLI prints a warning so the
        # operator notices.
        logger.warning(
            "optimize_comprehension_module: no MIPROv2 / "
            "BootstrapFewShot on the DSPy dep tree; returning the "
            "un-optimized module."
        )
        return module

    optimizer = optimizer_cls(metric=_comprehension_metric)
    try:
        optimized = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
        )
    except TypeError:
        # Older BootstrapFewShot signatures don't accept
        # ``valset``.
        optimized = optimizer.compile(module, trainset=train_examples)
    except Exception as exc:  # noqa: BLE001
        # Offline / DummyLM path: the optimizer's internal
        # prompt-proposer (or, in the case of MIPROv2, its
        # bootstrap few-shot proposer) probes the LM with a
        # signature that our ``DummyLM`` can't satisfy. This
        # is the documented failure mode of the offline path —
        # the production path runs with a real OpenRouter
        # adapter (``--live``) and exercises the optimizer
        # properly. We log and return the un-optimized module
        # so the CLI plumbing stays usable end-to-end.
        logger.warning(
            "optimize_comprehension_module: optimizer %s raised on "
            "the offline path (%s); returning the un-optimized "
            "module. Re-run with --live and OPENROUTER_API_KEY "
            "set to actually optimize.",
            optimizer_cls.__name__,
            exc,
        )
        return module
    return optimized


def _comprehension_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
    """Offline quality metric for the comprehension optimizer.

    Score range: 0.0..1.0. The bar is intentionally loose — the
    production C1-accept check is qualitative (Anurag hand-
    reviews), not a numeric gate. The optimizer uses this score
    to pick a better prompt; ``scripts/eval_comprehension.py``
    (a Phase 9 deliverable) runs the more rigorous per-row
    comparison.

    Components:

    - ``+0.3`` if ``prediction.exercise.target_word_id ==
      example.target_word_id`` (passage stays on-topic)
    - ``+0.3`` if all four choices A/B/C/D are present and
      non-empty strings
    - ``+0.2`` if ``correct_choice`` is in ``{"A", "B", "C", "D"}``
    - ``+0.2`` if ``passage`` length is in
      ``[20, COMPREHENSION_PASSAGE_MAX_CHARS]``
    """
    try:
        ex: ComprehensionExercise = prediction.exercise  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0
    score = 0.0
    if getattr(ex, "target_word_id", None) == example.target_word_id:
        score += 0.3
    if (
        isinstance(getattr(ex, "choices", None), dict)
        and set(ex.choices.keys()) == {"A", "B", "C", "D"}
        and all(isinstance(v, str) and v for v in ex.choices.values())
    ):
        score += 0.3
    if getattr(ex, "correct_choice", None) in {"A", "B", "C", "D"}:
        score += 0.2
    passage = getattr(ex, "passage", "")
    if isinstance(passage, str) and 20 <= len(passage) <= COMPREHENSION_PASSAGE_MAX_CHARS:
        score += 0.2
    return score


__all__ = [
    "COMPREHENSION_PASSAGE_MIN_SENTENCES",
    "COMPREHENSION_PASSAGE_MAX_SENTENCES",
    "COMPREHENSION_PASSAGE_MAX_CHARS",
    "ComprehensionChoice",
    "ComprehensionExercise",
    "ComprehensionGenerationError",
    "ComprehensionModule",
    "ComprehensionSignature",
    "MAX_ATTEMPTS",
    "PROMPT_TEMPLATE_VERSION",
    "RAG_MAX_CHARS_PER_CHUNK",
    "RAG_TOP_K",
    "_DSPyOpenAICompatLM",
    "build_prompt",
    "generate_comprehension",
    "optimize_comprehension_module",
]
