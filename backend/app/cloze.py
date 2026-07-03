"""Phase 4.2 — cloze exercise generator + DSPy module (card t_bdd9ffbe).

This module is the heart of Phase 4: the cloze activity that the lexora
backend generates on demand for a logged-in learner. It ships three
distinct surfaces that share one Pydantic contract (``ClozeExercise``):

1. **Production path** — ``generate_cloze(db, user_id)``. Picks a target
   word deterministically from the user's weakness profile, builds a
   constrained prompt, calls ``app.llm.complete`` wrapped with
   ``instructor`` so the response validates against the
   ``ClozeExercise`` Pydantic model. Bounded retry (Hard rule #6).
   Traces through ``_trace_cloze`` (4.3 fills this in; this card
   leaves it as a no-op).

2. **Optimization path** — ``ClozeSignature`` + ``ClozeModule`` +
   ``optimize_cloze_module``. DSPy is wired to ``app.llm.complete`` via
   a thin adapter (``_DSPyOpenAICompatLM``) so the same transport that
   serves production also serves the optimizer. The optimizer is
   offline-capable: with no API key, ``dspy.utils.dummies.DummyLM``
   swaps in automatically so the CI suite runs without network.

3. **Selector** — ``select_target_word``. Deterministic seed of
   ``(user_id, axis, date.today())``; same triple → same word across
   calls. The user re-clicking the exercise on the same day gets the
   same word; the next day → new word. Documented in the function
   docstring + in the module ``__all__`` as a stability commitment.

Hard rules enforced here:

- #1 cloze only (no matching / comprehension — single exercise type).
- #2 no ``fsrs_cards`` writes — this module does not import
  ``models.FsrsCard`` and never INSERTs into that table.
- #3 no retrieval on the cloze path. The word's first example sentence
  comes from ``Word.examples`` (a relationship on the row), NOT from
  ``app.retrieval``. Verified by the test that asserts
  ``app.retrieval`` is not in ``sys.modules`` after ``from app import
  cloze``.
- #4 every LLM call is wrapped — ``_trace_cloze`` is the hook (4.3
  fills in the real Langfuse SDK; this card's no-op keeps tests
  hermetic).
- #5 OpenRouter only — the production path goes through
  ``app.llm.complete`` (4.1's OpenAI-compatible client); the DSPy
  adapter targets the same wire format.
- #6 Pydantic v2 validated output via ``instructor``. Schema-violation
  retries are bounded by ``MAX_ATTEMPTS``.
- #8 offline-capable eval — DSPy runs on ``DummyLM`` when no key is
  present; the optimizer CLI is a separate ``scripts/optimize_cloze.py``
  process, NOT an import-time side effect.
- #11 type-level guardrails — ``MAX_ATTEMPTS`` is a module constant,
  not env-derived. ``PROMPT_TEMPLATE_VERSION`` is the same shape.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import date
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import crud, models
from app.observability import get_langfuse
from app.diagnostic.questions import ALL_AXES

# Lazy DSPy import — ``dspy`` is heavy and only needed by the
# optimization-path classes (``ClozeSignature``, ``ClozeModule``,
# ``optimize_cloze_module``). The production path (``generate_cloze``)
# doesn't touch it. Importing here is safe: dspy doesn't pull in any
# network deps at module-load time. Hard rule #3 still holds — dspy
# is not ``app.retrieval`` and the "no retrieval on cloze path"
# assertion checks for the retrieval module specifically.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #11).
#
# ``PROMPT_TEMPLATE_VERSION`` bumps when the prompt template changes;
# downstream eval tooling uses it as an A/B key. ``MAX_ATTEMPTS`` caps
# the instructor retry budget on schema-violation responses.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "cloze-v1"
MAX_ATTEMPTS: int = 3


# Mapping from weakness-profile axis name to ``Word.word_type``. The
# profile axis keys are singular concepts ("verbs", "prepositions",
# ...); the corpus ``word_type`` column uses the grammatical class
# ("Verb", "Preposition", ...). The mapping is intentionally narrow —
# we map each axis to one word_type, not a list, so the seed scheme
# and the corpus query are deterministic.
AXIS_TO_WORD_TYPE: dict[str, str] = {
    "verbs": "Verb",
    "prepositional_combos": "Preposition",
    "collocations": "Noun",
    "idioms": "Noun",
    "abstract_nouns": "Noun",
    "adjectives": "Adjective",
    "adverbs": "Adverb",
    "prepositions": "Preposition",
    "pronouns": "Pronoun",
    "conjunctions": "Conjunction",
}


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class ClozeExercise(BaseModel):
    """The metadata contract for a single cloze exercise.

    Field set locked by ``docs/PHASE-4.md`` §"The metadata contract" —
    Phase 5's FSRS grading loop reads these columns verbatim, so the
    shape is forward-frozen here.

    ``sentence_with_blank`` carries a single ``___`` (three
    underscores) marker; the UI replaces it with the input field.
    The LLM is explicitly instructed not to mutate the answer word's
    case, article, or surrounding word forms (the prompt's
    prohibition block, see ``build_prompt``).

    ``distractors`` is exactly three ``words.id`` FKs of the same
    ``word_type`` as ``answer_word_id``; the frontend renders them as
    the four multiple-choice buttons (correct answer + 3 distractors)
    in randomised order.
    """

    sentence_with_blank: str = Field(
        ...,
        description=(
            "German sentence with '___' marking the cloze position. "
            "The LLM must not mutate the answer word's case, article, "
            "or surrounding word forms."
        ),
    )
    answer_word_id: int = Field(..., description="FK to words.id of the correct answer.")
    distractors: list[int] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Exactly 3 FKs to words.id of plausible wrong answers. "
            "Same word_type as answer_word_id."
        ),
    )
    difficulty: Literal["easy", "medium", "hard"]
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="One sentence explaining the cloze design.",
    )
    prompt_template_version: str = Field(
        ...,
        description=(
            "Bump when prompt changes; enables A/B eval. Should "
            "always equal PROMPT_TEMPLATE_VERSION for production "
            "generations."
        ),
    )


class ClozeGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields that the activity layer (and the
    Langfuse trace in 4.3) record so an operator can triage without
    re-running the call. Mirrors the shape of an ``LLMError`` from
    ``app.llm`` but adds the schema-validation context that 4.1's
    transport layer doesn't know about.
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
    ) -> "ClozeGenerationError":
        """Build a dead-letter from a Pydantic schema dict.

        The Pydantic v2 ``model_json_schema()`` returns a ``dict``;
        the production code path passes that dict here and we
        serialise it to a stable JSON string for the
        ``attempted_schema`` field. Tests that build a string
        directly (the public constructor) still work — both shapes
        are documented.
        """
        import json as _json

        return cls(
            message,
            attempted_schema=_json.dumps(schema, ensure_ascii=False),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        )


# ---------------------------------------------------------------------------
# Word selection
# ---------------------------------------------------------------------------


def select_target_word(
    db: Session,
    user_id: int,
    *,
    force_word_id: int | None = None,
) -> models.Word:
    """Pick a ``Word`` for the cloze exercise.

    Two modes:

    1. **Default (``force_word_id=None``)** — deterministic seed of
       ``(user_id, axis, date.today())``; same triple → same word
       across calls. Used by the Phase 4.5 ``POST /exercises/cloze``
       surface where the server picks based on the user's weakness
       profile. Documented below; the algorithm and stability
       commitment are unchanged from Phase 4.2.

    2. **Forced (``force_word_id=<int>``)** — look up the ``Word``
       row by primary key and return it directly. Used by Phase 5.4's
       ``GET /exercises/due`` route, which has already picked a
       specific word from the ``fsrs_cards`` due queue and wants
       ``generate_cloze`` to build the exercise for THAT word (not
       re-select from the user's weakness profile). If the id does
       not exist, raise ``ValueError`` — the route layer maps that
       to 500 (corpus inconsistency; the due-queue picked a word
       that's no longer in the words table).

    Algorithm (default mode)
    -------------------------
    1. Read the user's ``WeaknessProfile.axes`` (default empty).
    2. Pick the axis with the **highest score** (the score scale is
       0=unknown / 1=shaky / 2=developing / 3=critical — the spec
       says "highest score = most critical"). Ties are broken by the
       axis order in ``ALL_AXES`` so the choice is stable across
       equal-score axes.
    3. If the profile has no axes set yet, fall back to the first axis
       in ``ALL_AXES`` (a deterministic default rather than a random
       choice, so the exercise surface still has a stable behaviour
       for fresh users — Phase 3.3's first-login gate routes them
       through the diagnostic first, but defensive default is good).
    4. Map the axis to a ``Word.word_type`` via ``AXIS_TO_WORD_TYPE``.
       If no mapping exists (e.g. a future axis), fall back to
       ``"Noun"`` so the call still returns a usable word rather than
       raising.
    5. Pull the row count of ``Word`` rows of that ``word_type`` and
       seed ``random.Random(user_id, axis, date)`` — same triple
       returns the same offset; a new day returns a different offset.
    6. Offset into the rows ordered by ``Word.id`` (stable order, no
       random shuffle needed) and return that row.

    Stability commitment
    --------------------
    The seed scheme is ``hash(user_id) XOR hash(axis) XOR date.today()``
    (where ``hash(x)`` is Python's built-in ``hash`` for a stable
    string seed). Same triple → same word; same day, re-click → same
    word; new day → new word. Documented here so a future maintainer
    doesn't break the stability promise with a naive ``random.choice``.

    Raises
    ------
    ValueError
        If ``force_word_id`` is set and the word id does not exist
        (corpus inconsistency — the caller picked an id that isn't
        in the words table). Also raised if the mapped ``word_type``
        in default mode has zero rows (caller error: the axis /
        word_type mapping is wrong, or the corpus is empty).
        Routes translate both to 500.
    """
    # Phase 5.4 — forced mode. Bypass the seed scheme entirely; the
    # caller (the /exercises/due route) has already picked the word
    # from the fsrs_cards due queue and wants THIS word, not a
    # re-selection from the user's weakness profile. The default mode
    # below stays byte-for-byte identical to Phase 4.2's behaviour —
    # the ``force_word_id`` keyword is the only addition.
    if force_word_id is not None:
        word = crud.get_word(db, word_id=force_word_id)
        if word is None:
            raise ValueError(
                f"select_target_word: force_word_id={force_word_id} "
                "does not match any Word row (corpus inconsistency)"
            )
        return word

    profile = crud.get_weakness_profile(db, user_id)
    axes: dict[str, int] = (
        crud.serialize_weakness_profile_axes(profile)
        if profile is not None
        else {}
    )
    if axes:
        # Highest score wins; tie-break by canonical axis order.
        top_axis: str | None = None
        top_score: int = -1
        for canonical_axis in ALL_AXES:
            score = axes.get(canonical_axis)
            if score is None:
                continue
            if score > top_score:
                top_score = score
                top_axis = canonical_axis
        if top_axis is None:
            # All axes present but every score is below 0 (impossible
            # under Pydantic 0-3 bounds — defensive only). Fall through
            # to the empty-axes default.
            top_axis = ALL_AXES[0]
    else:
        top_axis = ALL_AXES[0]

    word_type: str = AXIS_TO_WORD_TYPE.get(top_axis, "Noun")

    # Pull every id of that word_type in stable id order; the seed
    # picks one offset.
    candidates: list[int] = [
        row.id
        for row in db.query(models.Word)
        .filter(models.Word.word_type == word_type)
        .order_by(models.Word.id)
        .all()
    ]
    if not candidates:
        raise ValueError(
            f"select_target_word: no Word rows with word_type={word_type!r} "
            f"for axis={top_axis!r} (user_id={user_id})"
        )

    today = date.today()
    seed_str = f"{user_id}|{top_axis}|{today.isoformat()}"
    rng = random.Random(seed_str)
    chosen_id = rng.choice(candidates)
    word = crud.get_word(db, word_id=chosen_id)
    # ``crud.get_word`` returns ``None`` only on a race (someone
    # deleted the row between the id list and the lookup). Treat as
    # a hard error so the caller knows the corpus is in an inconsistent
    # state rather than silently substituting another row.
    if word is None:
        raise ValueError(
            f"select_target_word: word id {chosen_id} disappeared between "
            f"candidate list and lookup (user_id={user_id}, axis={top_axis!r})"
        )
    return word


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _first_example_sentence(word: models.Word) -> str:
    """Return the first example sentence's German text, or a fallback.

    The cloze activity does NOT call ``app.retrieval`` (Hard rule #3)
    — the context sentence comes from the ``Word.examples`` relationship
    that is already on the row. If the word has no examples, we fall
    back to a deterministic placeholder rather than raise, so a sparse
    row doesn't break the endpoint (the LLM still has the target word
    itself to work from).
    """
    if word.examples:
        return word.examples[0].german or ""
    return f"{word.word} (Beispiel nicht verfügbar)."


def build_prompt(
    word: models.Word,
    weakness_axes: dict[str, int],
) -> list[dict]:
    """Build the chat-completions messages for one cloze generation.

    Prompt shape
    ------------
    The system prompt:

    1. Establishes the role ("German cloze-exercise designer for a C1
       learner").
    2. Lists the explicit prohibitions (the C1-accept bar +
       immutable-form constraints). This is load-bearing — without it,
       the LLM drifts toward translating or paraphrasing.
    3. Specifies the JSON output schema with every field on
       ``ClozeExercise``, including the ``prompt_template_version`` so
       downstream eval tooling can split A/B cohorts.
    4. States the C1-accept bar ("would a C1 German speaker accept
       this cloze without edits?").

    The user prompt:

    - Embeds the target word (lemma, ``word.word_type``, frequency).
    - Embeds the word's first example sentence from ``Word.examples``
      (NOT a retrieval call — Hard rule #3).
    - Embeds the user's weakness axes as JSON so the model can lean
      the cloze design toward the axes the learner is weakest on
      (the rationale's signal source).

    Parameters
    ----------
    word
        The target ``Word`` row returned by ``select_target_word``.
    weakness_axes
        The dict from ``WeaknessProfile.axes`` (may be empty for a
        fresh user). JSON-encoded verbatim into the user prompt so
        the model sees the same shape the caller serialised.

    Returns
    -------
    list[dict]
        A two-message list suitable for ``app.llm.complete``. The
        structure is plain ``[{"role": ..., "content": ...}]``; no
        tool calls, no multimodal content.
    """
    example_sentence = _first_example_sentence(word)
    system_content = (
        "You are a German cloze-exercise designer for a C1 learner. "
        "Produce ONE fill-in-the-blank sentence whose missing word is "
        "the German target word below.\n\n"
        "PROHIBITIONS (must obey all):\n"
        "1. Do NOT change word forms, articles, or case endings around the blank.\n"
        "2. Do NOT invent grammar that is not native German.\n"
        "3. Do NOT translate the target word into another language.\n"
        "4. The correct answer is ALWAYS one of the German words shown "
        "(the target word itself). The three distractors are German words "
        "of the same word_type that plausibly fit but are semantically wrong.\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept this cloze "
        "without edits? If no, redo before answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "sentence_with_blank": "<German sentence with ___ as the blank marker>",\n'
        '  "answer_word_id": <integer, the words.id of the correct answer>,\n'
        '  "distractors": [<int>, <int>, <int>],\n'
        '  "difficulty": "easy" | "medium" | "hard",\n'
        '  "rationale": "<one sentence, <= 400 chars, explaining the cloze design>",\n'
        '  "prompt_template_version": "cloze-v1"\n'
        "}\n"
    )
    user_content = json.dumps(
        {
            "target_word": {
                "id": word.id,
                "word": word.word,
                "word_type": word.word_type,
                "frequency": word.frequency,
            },
            "context_sentence": example_sentence,
            "learner_axes": weakness_axes,
            "instructions": (
                "Design ONE cloze where the answer is target_word.word "
                "in context_sentence. The sentence_with_blank must "
                "embed the target word's form exactly as a C1 learner "
                "would expect. Pick three distractors of the same "
                "word_type from the corpus whose semantic field is "
                "nearby but not identical."
            ),
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Production path — instructor-wrapped chat completion
# ---------------------------------------------------------------------------


def _openai_client():
    """Build an ``OpenAI`` client pointed at OpenRouter.

    Imported lazily inside the function so ``app.cloze`` itself stays
    import-cheap and doesn't pay the OpenAI SDK import cost for the
    import-time assertions (the test that checks ``app.retrieval`` is
    not in ``sys.modules`` runs ``from app import cloze`` first). The
    same pattern as ``app.llm._get_api_key``.

    Returns ``None`` if ``OPENROUTER_API_KEY`` is missing — caller
    treats that as a "no real LLM available" signal and falls back to
    raising ``LLMError`` so the route layer surfaces 502.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def generate_cloze(
    db: Session,
    user_id: int,
    *,
    force_word_id: int | None = None,
) -> ClozeExercise:
    """Generate one ``ClozeExercise`` for a logged-in user.

    Flow
    ----
    1. ``select_target_word`` picks the deterministic target word
       (Phase 4.5 default mode), or — when ``force_word_id`` is
       set — looks up the specific word id and returns it (Phase 5.4
       ``GET /exercises/due`` mode).
    2. Read the user's weakness axes for the prompt.
    3. ``build_prompt`` produces the chat messages.
    4. Wrap an OpenRouter-targeted OpenAI client with ``instructor``
       and call ``chat.completions.create(response_model=ClozeExercise,
       ..., max_retries=MAX_ATTEMPTS)``. ``instructor`` re-prompts on
       schema violations (each retry counts as one schema-violation
       attempt in the metadata). On the happy path the call lands
       in a single ``complete`` round-trip; on persistent violations
       ``instructor`` raises ``InstructorRetryException`` after the
       budget is exhausted — we translate that into
       ``ClozeGenerationError`` with the structured fields.
    5. Stamp ``prompt_template_version`` and the metadata contract
       fields onto the result; call ``_trace_cloze`` (4.3's no-op
       stub here) with the metadata dict.

    Parameters
    ----------
    force_word_id
        Optional. When set, ``select_target_word`` returns this
        exact word (looked up by primary key) instead of running the
        deterministic seed. Used by ``GET /exercises/due`` (Phase
        5.4) which has already picked a specific word from the
        ``fsrs_cards`` due queue. Defaults to ``None`` — the
        Phase 4.5 ``POST /exercises/cloze`` path stays byte-for-byte
        identical.

    Returns
    -------
    ClozeExercise
        The validated Pydantic instance ready to return to the route
        layer. ``prompt_template_version`` is forced to
        ``PROMPT_TEMPLATE_VERSION`` so a future maintainer who
        hand-edits the prompt template doesn't silently desync the
        value.

    Raises
    ------
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or the
        transport call hits a non-retryable / persistent-retryable
        failure. The route layer translates this into 502.
    ClozeGenerationError
        Persistent schema violation after ``MAX_ATTEMPTS``. Carries
        ``attempted_schema``, ``last_validation_error``,
        ``schema_retry_count`` so the route layer can surface a
        structured 502 instead of a bare 500.
    ValueError
        Bubbles up from ``select_target_word`` when
        ``force_word_id`` does not match any row in the words table
        (corpus inconsistency). The route layer translates this into
        500.
    """
    from app.llm import _default_model, LLMError

    word = select_target_word(db, user_id, force_word_id=force_word_id)

    profile = crud.get_weakness_profile(db, user_id)
    weakness_axes: dict[str, int] = (
        crud.serialize_weakness_profile_axes(profile)
        if profile is not None
        else {}
    )

    messages = build_prompt(word, weakness_axes)

    # Trace metadata captures the request shape; populated before the
    # call so the error path (ClozeGenerationError raised below) can
    # still log it.
    prompt_messages = messages
    metadata: dict[str, Any] = {
        "user_id": user_id,
        "weakness_axes": weakness_axes,
        "word_id": word.id,
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": prompt_messages,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    raw_client = _openai_client()
    if raw_client is None:
        # No key — same shape as the persistent failure mode in llm.py.
        # Surface LLMError so the route layer's existing 502 handler
        # picks it up.
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.lexora/.env "
            "and restart the backend container."
        )

    import instructor

    # ``MD_JSON`` mode tells instructor to use markdown-fenced JSON
    # parsing rather than the tool-calling path. The tool-calling
    # path requires the model to emit ``tool_calls`` in the
    # response, which the OpenRouter passthrough doesn't always
    # echo for non-OpenAI-native models. ``MD_JSON`` works
    # uniformly with the qwen default and gives the model a
    # clear "output a ```json ...``` block" instruction in the
    # system prompt (instructor appends it).
    instructor_client = instructor.from_openai(
        raw_client, mode=instructor.Mode.MD_JSON
    )

    schema_retry_count = 0
    last_validation_error = ""
    result: ClozeExercise | None = None
    started = _perf_counter_ms()

    try:
        # ``instructor`` returns a ``ClozeExercise`` instance directly
        # when validation succeeds; raises ``pydantic.ValidationError``
        # or ``InstructorRetryException`` when it doesn't. We catch
        # both and translate into the structured error below.
        result = instructor_client.chat.completions.create(
            response_model=ClozeExercise,
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
        # Best-effort trace before dead-lettering — 4.3's no-op hook
        # ignores it today, but the metadata dict is in the right
        # shape for when 4.3 lands.
        try:
            _trace_cloze(None, metadata, latency_ms)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        raise ClozeGenerationError.from_validation_failure(
            f"cloze: schema validation failed after {schema_retry_count} "
            f"attempt(s): {last_validation_error}",
            schema=ClozeExercise.model_json_schema(),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        ) from exc

    # ``result`` is non-None here — the except branch is the only
    # failure path, and we just returned / raised.
    assert result is not None  # noqa: S101 — see above
    result = result.model_copy(update={"prompt_template_version": PROMPT_TEMPLATE_VERSION})

    # Read usage from the underlying OpenAI response — ``instructor``
    # returns the validated Pydantic model but the raw response
    # carries the ``usage`` block in ``_raw_response`` (an
    # implementation detail of instructor >= 1.0; we tolerate missing
    # attributes gracefully).
    raw_response = getattr(result, "_raw_response", None)
    if raw_response is not None:
        usage = getattr(raw_response, "usage", None)
        if usage is not None:
            metadata["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            metadata["completion_tokens"] = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
        # ``instructor`` 1.x exposes the actual retry count it
        # took via ``raw_response._instructor_retry_count`` on some
        # versions; fall back to 0 if it's not there.
        metadata["schema_retry_count"] = int(
            getattr(raw_response, "_instructor_retry_count", 0) or 0
        )

    latency_ms = int(_perf_counter_ms() - started)
    _trace_cloze(result, metadata, latency_ms)
    return result


def _perf_counter_ms() -> float:
    """Wall-clock in ms (float). Local import to keep the module
    import-cheap and to centralise the unit conversion."""
    import time

    return time.perf_counter() * 1000.0


def _count_instructor_retries(exc: Exception) -> int:
    """Best-effort: pull the retry count out of an instructor /
    pydantic failure. ``instructor`` raises a ``InstructorRetryException``
    that wraps the last validation error; older versions raise
    ``pydantic.ValidationError`` directly. We default to ``MAX_ATTEMPTS``
    if we can't read the actual count, so the dead-letter always
    carries the budget it exhausted.
    """
    # Look for the attribute instructor sets on its exception types.
    for attr in ("n_attempts", "attempts", "retries", "_retries"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    # Fall back: assume the budget was exhausted.
    return MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Langfuse trace hook — Phase 4.3 fills this in.
# ---------------------------------------------------------------------------


def _trace_cloze(
    result: ClozeExercise | None,
    metadata: dict[str, Any],
    latency_ms: int,
) -> None:
    """Emit one Langfuse ``cloze.generate`` span per generation.

    Phase 4.2 ships this as a no-op; this card is the real
    implementation. The signature is locked — ``generate_cloze`` and
    the dead-letter branch both call us with the same kwargs in the
    same order, so the Langfuse payload shape stays stable for the
    Phase 5 readers.

    **SDK choice.** ``pyproject.toml`` pins ``langfuse>=2.50.0,<3.0``
    (resolved to 2.60.10). The v2 SDK exposes ``client.span(name=...)``
    returning a span handle; ``span.update(metadata=...)`` merges the
    dict; ``span.end()`` closes the observation; ``client.flush()``
    pushes the buffer to the ingestion API. The ``start_as_current_span``
    context-manager API referenced in the Phase 4 spec doc §4.3 is a
    v3-only method and is not callable against the v2.60.10 SDK we
    pinned in the Phase 1 fix card (``t_2e386ba9``) — we mirror
    ``app.main._trace_retrieval``'s non-context-manager shape here
    for the same reason.

    **Graceful degradation.** When ``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY`` are missing, ``get_langfuse()`` returns
    ``None`` (Phase 0's design; see ``app.observability``). In that
    branch we early-return so the cloze activity still succeeds —
    observability is best-effort, never blocking. We log a warning
    once at module-import time (``observability.py``) instead of
    spamming on every call.

    **Metadata contract.** Phase 5 reads these fields; we populate
    exactly the keyset documented in ``docs/PHASE-4.md`` §"The
    metadata contract". Phase 4.2's ``generate_cloze`` already
    assembles every required key in the ``metadata`` dict; this
    function copies the right ones onto the span and ignores the
    implementation-detail keys (``prompt_messages`` — too verbose
    for the Langfuse metadata row; we keep it on the trace payload
    instead so a phase-5 reader that needs the prompt can still
    get it from the trace record).

    **Failure mode.** Any exception raised by the Langfuse SDK
    (network glitch, malformed payload, server-side rejection) is
    caught and logged at WARNING — the cloze generation has already
    succeeded at this point, so a trace failure must never break the
    request. The same shape is used by ``_trace_retrieval``.

    On success the result is a populated ``ClozeExercise``; on
    dead-letter (``ClozeGenerationError`` raised upstream) ``result``
    is ``None`` and ``metadata["schema_retry_count"]`` carries the
    failure depth. We still emit the span in that case so the
    failure is visible in Langfuse next to the passing traces.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam per call.
        return

    # Build the metadata payload exactly per docs/PHASE-4.md §"The
    # metadata contract". ``prompt_template_version`` comes from
    # the result when present (it's a Pydantic output field); fall
    # back to the input dict for the dead-letter branch.
    ptemplate_version = (
        getattr(result, "prompt_template_version", None)
        if result is not None
        else metadata.get("prompt_template_version")
    )
    span_metadata: dict[str, Any] = {
        "user_id": metadata["user_id"],
        "weakness_axes": metadata["weakness_axes"],
        "word_id": metadata["word_id"],
        "difficulty": getattr(result, "difficulty", None) if result is not None else None,
        "model_id": metadata["model_id"],
        "prompt_template_version": ptemplate_version,
        "schema_retry_count": metadata["schema_retry_count"],
        "latency_ms": latency_ms,
        "prompt_tokens": metadata["prompt_tokens"],
        "completion_tokens": metadata["completion_tokens"],
    }

    span = None
    try:
        span = client.span(
            name="cloze.generate",
            # Trace-level input/output so the Langfuse UI surfaces
            # the prompt + the serialised Pydantic result. v2
            # ``span`` accepts both kwargs at construction time;
            # ``update`` is for incremental merges.
            input=metadata.get("prompt_messages"),
            output=(result.model_dump_json() if result is not None else None),
        )
        span.update(metadata=span_metadata)
        span.end()
        # Force a flush so the trace is queryable in the UI before
        # the request returns — mirrors _trace_retrieval.
        client.flush()
    except Exception as exc:  # noqa: BLE001 — tracing must never break the activity
        logger.warning(
            "cloze: Langfuse trace failed (non-fatal): %s", exc
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
# adapter wraps ``app.llm.complete`` so every LLM call stays routed
# through the 4.1 client (Hard rule #4 + #5).
# ---------------------------------------------------------------------------


def _offline_dummy_answers() -> list[dict[str, str]]:
    """Build a pool of offline-LM answer dicts for the DummyLM.

    DSPy's ``DummyLM`` returns ``"[[ ## field ## ]] value"`` lines,
    and the ``ChatAdapter`` parses the value back into the field's
    declared type. Our ``exercise`` output is a Pydantic
    ``ClozeExercise``, so the value must be a JSON-encoded instance
    that parses cleanly through ``ClozeExercise.model_validate_json``.

    We pool three diverse stubs so any ``dspy.Predict`` /
    ``dspy.ChainOfThought`` call against the signature gets a fresh
    answer that satisfies the schema. The exact contents are
    irrelevant — the offline path exists to exercise the optimizer
    plumbing without network calls, not to produce realistic
    generations.
    """
    return [
        {
            "exercise": json.dumps(
                {
                    "sentence_with_blank": "Der ___ schläft.",
                    "answer_word_id": 1,
                    "distractors": [2, 3, 4],
                    "difficulty": "easy",
                    "rationale": "offline-stub",
                    "prompt_template_version": "cloze-v1",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "sentence_with_blank": "Die Katze ___ auf dem Sofa.",
                    "answer_word_id": 5,
                    "distractors": [6, 7, 8],
                    "difficulty": "medium",
                    "rationale": "offline-stub",
                    "prompt_template_version": "cloze-v1",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "sentence_with_blank": "Wir ___ nach Hause.",
                    "answer_word_id": 9,
                    "distractors": [10, 11, 12],
                    "difficulty": "hard",
                    "rationale": "offline-stub",
                    "prompt_template_version": "cloze-v1",
                },
                ensure_ascii=False,
            )
        },
    ]


def _offline_json_answers() -> list[dict[str, str]]:
    """Answer pool for MIPROv2's internal ``JSONAdapter``-shaped calls.

    MIPROv2 probes the LM with a JSONAdapter-shaped prompt to
    propose instructions. That adapter expects
    ``{"proposed_instruction": "..."}`` responses (a string field,
    not a JSON object). We pool five diverse instruction-shaped
    stubs so the prompt-proposer can keep cycling without choking.
    """
    return [
        {"proposed_instruction": "Always include the target word verbatim."},
        {"proposed_instruction": "Keep the distractor word_type aligned with the answer."},
        {"proposed_instruction": "Embed the user's weakness axes as JSON."},
        {"proposed_instruction": "State the C1-accept bar in the rationale."},
        {"proposed_instruction": "Use the first example sentence from the corpus row."},
    ]


def _configure_dspy() -> None:
    """Configure DSPy with our OpenRouter adapter, or ``DummyLM`` offline.

    Idempotent — calling twice doesn't reconfigure. If the env has
    ``OPENROUTER_API_KEY`` set, we wire the real adapter; otherwise
    we fall back to ``dspy.utils.dummies.DummyLM`` so unit tests
    and offline optimizer runs never hit the network.

    The DummyLM in DSPy 3.x rotates through a list of answer dicts
    whose values match the output field's expected shape. We pool
    JSON-encoded ``ClozeExercise`` payloads for the cloze path AND
    instruction-shaped stubs for MIPROv2's internal
    ``JSONAdapter``-shaped prompt-proposer, so the offline
    optimizer doesn't choke on a shape mismatch (DSPy's
    ``DummyLM`` doesn't auto-detect the active adapter's
    requirements). The pool is intentionally small — the offline
    path exists to exercise the optimizer plumbing without
    network, not to produce realistic generations.
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

    # The DummyLM in DSPy 3.x cycles through its answers list and
    # emits ``"[[ ## field ## ]] value"`` lines. The ``ChatAdapter``
    # parses each value back into the field's declared type. For our
    # Pydantic ``ClozeExercise`` output that means the value must
    # be a JSON-encoded instance. We concatenate the cloze path
    # answers with the MIPROv2 internal-proposer answers so a single
    # DummyLM can serve both protocols in any interleaving.
    answers = _offline_dummy_answers() + _offline_json_answers()
    _dspy.settings.configure(
        lm=DummyLM(answers, adapter=ChatAdapter()),
        adapter=ChatAdapter(),
    )


class _DSPyOpenAICompatLM:
    """Thin DSPy adapter that routes through ``app.llm.complete``.

    DSPy 3.x has a built-in ``dspy.LM`` for OpenAI-compatible
    endpoints, but it imports the ``openai`` SDK directly and bypasses
    our retry + latency-recording wrapper. Using a hand-written
    adapter lets us keep every chat call going through
    ``app.llm.complete`` (Hard rule #4 + #5: "every LLM call goes
    through app/llm.py").

    We don't subclass ``dspy.BaseLM`` because DSPy 3.x's ``BaseLM``
    enforces a constructor signature and `__call__` shape that
    depends on the active DSPy version; a duck-typed adapter that
    ``dspy.Predict`` accepts via ``settings.configure(lm=...)`` is
    more portable across DSPy releases.

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

        DSPy calls the LM with either ``prompt=...`` (legacy) or
        ``messages=...`` (newer protocol). We accept both and
        normalise into a messages-shaped ``app.llm.complete`` call.
        Returns a list of strings — one per generation — which is the
        shape DSPy 3.x expects from a custom LM.
        """
        from app.llm import complete as _complete

        messages = kwargs.get("messages")
        if not messages:
            messages = [{"role": "user", "content": prompt or ""}]
        result = _complete(messages=messages)
        # DSPy expects a list of strings (one per sample).
        return [result.text]

    # DSPy 3.x sometimes probes ``basic_request`` directly; provide
    # a passthrough so the optimiser can talk to the LM without
    # knowing about our internal shape.
    def basic_request(self, prompt: str | None = None, **kwargs: Any) -> list[dict]:
        text = self.__call__(prompt=prompt, **kwargs)[0]
        return [{"text": text}]


class ClozeSignature(dspy.Signature):
    """DSPy signature for one cloze generation.

    Inputs match the ``build_prompt`` payload shape; the output is
    the full ``ClozeExercise`` Pydantic model (DSPy 3.x supports
    Pydantic-typed output fields via ``dspy.Predict`` /
    ``dspy.ChainOfThought``).

    The ``target_word_id`` is in the input set so the optimizer can
    teach the model that the answer must equal the input word's id
    without the LLM having to guess.
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
    exercise: ClozeExercise = dspy.OutputField(
        desc="A Pydantic ClozeExercise matching the production contract."
    )


class ClozeModule(dspy.Module):
    """DSPy module that wraps the production ``ClozeSignature``.

    Uses ``dspy.Predict`` (single-shot — no chain-of-thought) so the
    output shape stays compatible with the production ``instructor``
    path. The optimizer (``optimize_cloze_module``) can swap the
    predictor for a tuned one without changing this surface.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``dspy.Predict`` is the most production-faithful predictor:
        # no reasoning chain to inflate latency. The optimizer can
        # upgrade this to ``dspy.ChainOfThought`` if the eval set
        # shows a quality win worth the latency cost.
        self.predict = dspy.Predict(ClozeSignature)

    def forward(  # type: ignore[override]
        self,
        *,
        word: str,
        context_sentence: str,
        learner_axes_json: str,
        target_word_id: int,
    ) -> dspy.Prediction:
        return self.predict(
            word=word,
            context_sentence=context_sentence,
            learner_axes_json=learner_axes_json,
            target_word_id=target_word_id,
        )


def optimize_cloze_module(
    train_set: Iterable[dict],
    val_set: Iterable[dict] | None = None,
) -> ClozeModule:
    """Run the offline DSPy optimizer against a held-out eval set.

    Strategy
    --------
    - Always uses ``DummyLM`` when no API key is present so the CI
      suite runs without network (Hard rule #8).
    - Tries ``dspy.MIPROv2`` first (the spec's preferred optimizer).
      Falls back to ``dspy.BootstrapFewShot`` if MIPROv2 raises on
      the active dep tree (different versions of DSPy changed the
      constructor signature; the fallback keeps the optimization
      surface usable across versions).
    - Returns a ``ClozeModule`` with the optimized prompt
      instructions baked in. The caller (the CLI) serialises the
      optimised module to ``backend/app/cloze_optimized.json`` so
      the production path can read it on next start (Phase 5+; this
      card just ships the CLI plumbing).

    Parameters
    ----------
    train_set, val_set
        Iterable of dicts with the four input keys (word, context,
        learner_axes_json, target_word_id). The eval set is loaded
        from ``eval/cloze_judgments.jsonl`` by ``scripts/optimize_cloze.py``.

    Returns
    -------
    ClozeModule
        The optimized module. The optimizer mutates the module's
        internal predictor in place; the same instance is returned
        for caller convenience.
    """
    import dspy

    _configure_dspy()

    train_examples = [
        dspy.Example(**row).with_inputs("word", "context_sentence", "learner_axes_json", "target_word_id")
        for row in train_set
    ]
    val_examples = (
        [
            dspy.Example(**row).with_inputs(
                "word", "context_sentence", "learner_axes_json", "target_word_id"
            )
            for row in val_set
        ]
        if val_set is not None
        else None
    )

    module = ClozeModule()

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
            "optimize_cloze_module: no MIPROv2 / BootstrapFewShot on the "
            "DSPy dep tree; returning the un-optimized module."
        )
        return module

    optimizer = optimizer_cls(metric=_cloze_metric)
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
        # prompt-proposer (or, in the case of MIPROv2, its bootstrap
        # few-shot proposer) probes the LM with a signature that our
        # ``DummyLM`` can't satisfy. This is the documented failure
        # mode of the offline path — the production path runs with
        # a real OpenRouter adapter (``--live``) and exercises the
        # optimizer properly. We log and return the un-optimized
        # module so the CLI plumbing stays usable end-to-end.
        logger.warning(
            "optimize_cloze_module: optimizer %s raised on the offline "
            "path (%s); returning the un-optimized module. Re-run with "
            "--live and OPENROUTER_API_KEY set to actually optimize.",
            optimizer_cls.__name__,
            exc,
        )
        return module
    return optimized


def _cloze_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
    """Offline quality metric for the cloze optimizer.

    Score range: 0.0..1.0. The bar is intentionally loose — the
    production C1-accept check is qualitative (Anurag hand-reviews),
    not a numeric gate. The optimizer uses this score to pick a
    better prompt; ``scripts/eval_cloze.py`` (4.4's deliverable)
    runs the more rigorous per-row comparison.

    Components:
    - ``+0.5`` if ``prediction.exercise.answer_word_id == example.target_word_id``
    - ``+0.3`` if ``"___"`` is present in ``prediction.exercise.sentence_with_blank``
    - ``+0.2`` if all three distractors are present and integer-typed
    """
    try:
        ex: ClozeExercise = prediction.exercise  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0
    score = 0.0
    if ex.answer_word_id == example.target_word_id:
        score += 0.5
    if "___" in ex.sentence_with_blank:
        score += 0.3
    if len(ex.distractors) == 3 and all(isinstance(d, int) for d in ex.distractors):
        score += 0.2
    return score


__all__ = [
    "AXIS_TO_WORD_TYPE",
    "ClozeExercise",
    "ClozeGenerationError",
    "ClozeModule",
    "ClozeSignature",
    "MAX_ATTEMPTS",
    "PROMPT_TEMPLATE_VERSION",
    "build_prompt",
    "generate_cloze",
    "optimize_cloze_module",
    "select_target_word",
]