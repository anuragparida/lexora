"""Phase 7.2 — collocation-cloze exercise generator + DSPy module (card t_ab77bc2b).

This module is the collocation-cloze counterpart of Phase 4.2's
``app.cloze``. It ships three distinct surfaces that share one Pydantic
contract (``CollocationExercise``):

1. **Production path** — ``generate_collocation(db, user_id, target_word_id)``.
   Picks a collocation row deterministically from the ``collocations``
   table for the target word, builds a constrained prompt, calls
   ``app.llm.complete`` wrapped with ``instructor`` so the response
   validates against the ``CollocationExercise`` Pydantic model.
   Bounded retry (Hard rule #6 — ``MAX_ATTEMPTS = 1`` per card body).
   Traces through ``_trace_collocation`` (returns ``None`` when
   Langfuse keys are missing; full span emission when present).

2. **Optimization path** — ``CollocationSignature`` + ``CollocationModule`` +
   ``optimize_collocation_module``. DSPy is wired to ``app.llm.complete``
   via the shared ``_DSPyOpenAICompatLM`` adapter (lives in ``app.llm``
   since Phase 6.2's extraction). The optimizer is offline-capable:
   with no API key, ``dspy.utils.dummies.DummyLM`` swaps in automatically
   so the CI suite runs without network.

3. **Selector** — ``select_collocation_row``. Picks one ``Collocation``
   row for the target word. Mirrors the deterministic-seed pattern of
   ``app.cloze.select_target_word``: same ``(user_id, target_word_id)``
   → same collocation row across calls. No state, no random.

Hard rules enforced here:

- **#1 (no 4th exercise type)** — ``exercise_type="cloze"`` is the
  discriminator on the wire; collocation-cloze is a cloze *variant*,
  not a new type literal. Phase 7 plan §"Hard rules" #1 + §"What is
  NOT in Phase 7".
- **#2 (READ-ONLY)** — ``generate_collocation`` only reads
  ``collocations``; no INSERT, UPDATE, DELETE from runtime. The seed
  scripts (7.1) are the only path that writes. The runtime module
  exposes a SELECT helper and nothing else; the SQLAlchemy model is
  declared here as a read-only reader rather than added to ``models.py``
  to keep 7.2's diff scoped (the 7.1 card adds the canonical model).
- **#3 (offline tests)** — DSPy module tests run with ``DummyLM``.
- **#5 (single LLM provider)** — production path goes through
  ``app.llm.complete``; the DSPy adapter targets the same wire format.
- **#6 (bounded retry)** — ``MAX_ATTEMPTS = 1`` per the card body
  ("Bounded retry (≤1) on Pydantic validation failure"). Different from
  ``app.cloze``'s ``MAX_ATTEMPTS = 3`` — collocation-cloze is a simpler
  exercise shape (single blank + partner lemma + register), so a
  tighter retry budget is appropriate.
- **#9 (Langfuse trace_id is the join key from Phase 5.3)** — the
  span carries the same metadata keyset ``_trace_cloze`` /
  ``_trace_match`` / ``_trace_comprehension`` use, plus a
  ``collocation: true`` discriminator so downstream A/B tooling can
  split cohorts.

Phase 7.0 spec gotchas addressed:

- **#7 (import-time no LLM)** — ``app.collocation`` does NOT call
  OpenRouter at import. The OpenRouter client (``_openai_client``)
  is constructed lazily inside ``generate_collocation``; the
  ``_configure_dspy`` mirror only runs when ``optimize_collocation_module``
  is called. Importing ``app.collocation`` is network-free.
- **#8 (DSPy adapter import path)** — the adapter is re-exported
  from ``app.llm`` (where it lives since Phase 6.2's extraction).
  7.2 does NOT touch ``app.llm``; the existing import path
  (``from app.llm import _DSPyOpenAICompatLM``) is the contract.
- **#12 (no schema migration)** — the ``Collocation`` SQLAlchemy
  model is defined here for read-only access. 7.1 owns the
  authoritative model in ``app.models``; 7.2's local mirror exists
  so the module compiles before 7.1 lands. When 7.1 merges, the
  canonical model is the one in ``app.models``; 7.2's read-only
  reader continues to work because it queries by table name
  (``__tablename__ = "collocations"``), not by Python identity.
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session

from app import crud, models
from app.database import Base
from app.llm import _DSPyOpenAICompatLM  # the shared adapter; see app.llm docstring
from app.observability import get_langfuse

# Lazy DSPy import — the optimization-path classes touch dspy at
# import time; the production path doesn't. Mirrors ``app.cloze``.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #11 / Phase 7 plan #9).
#
# ``PROMPT_TEMPLATE_VERSION`` is the A/B key downstream eval tooling
# splits on. ``MAX_ATTEMPTS = 1`` is the card body spec — collocation-cloze
# is a simpler shape than ``ClozeExercise`` (single blank + partner lemma +
# register), so a tighter retry budget is appropriate. ``PARTNER_REGISTER``
# is the closed literal the LLM is asked to choose from — mirrors the
# ``collocations.register`` column constraint 7.1 ships.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "collocation-cloze-v1"
MAX_ATTEMPTS: int = 1
PARTNER_REGISTER: tuple[str, ...] = ("formal", "neutral", "colloquial")


# ---------------------------------------------------------------------------
# Read-only SQLAlchemy mirror for the ``collocations`` table.
#
# The canonical model ships in 7.1's ``app/models.py`` schema migration.
# This local copy is a READ-ONLY reader (no insert/update/delete helpers
# exported, no relationship to other tables) so 7.2 compiles and the
# generator can query before 7.1 lands on main.
#
# When 7.1 merges, both models map to the same ``collocations`` table
# by name; SQLAlchemy will raise on a duplicate ``Base`` registration
# only if both are loaded into the same ``Base.metadata`` *and* one
# tries to issue DDL. Because this module imports ``Base`` from
# ``app.database`` (the same metadata registry), the 7.1 migration is
# the single source of truth on disk — this local model is a fallback
# for the pre-7.1 dev environment.
#
# Column set mirrors the spec: ``id``, ``target_word_id`` (FK to
# ``words.id``), ``partner_lemma``, ``partner_register``, ``source_corpus``
# (Literal["dwds", "wiktionary", "manual"], locked by PHASE-7 gotcha #12),
# ``created_at`` (nullable; 7.1's hard rule #1). No nullable columns
# are exposed as Pydantic response fields.
# ---------------------------------------------------------------------------


class Collocation(Base):
    """READ-ONLY mirror of the ``collocations`` table (Phase 7.1 schema).

    Used by ``generate_collocation`` and ``select_collocation_row`` to
    issue SELECTs against the seeded corpus. No helper functions for
    INSERT / UPDATE / DELETE are exported — the runtime read-only
    invariant (Hard rule #2) is enforced by omission.

    The column set mirrors 7.1's authoritative migration so the
    mirror continues to work after 7.1 lands: same ``__tablename__``,
    same column names, same nullability. If 7.1 widens the schema,
    the canonical model in ``app.models`` is the source of truth;
    this mirror is the pre-7.1 dev fallback.
    """

    __tablename__ = "collocations"

    id = Column(Integer, primary_key=True, index=True)
    target_word_id = Column(Integer, nullable=False, index=True)
    partner_lemma = Column(String, nullable=False)
    partner_register = Column(String, nullable=False)  # Literal["formal","neutral","colloquial"]
    source_corpus = Column(String, nullable=False)  # Literal["dwds","wiktionary","manual"]


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class CollocationExercise(BaseModel):
    """The metadata contract for a single collocation-cloze exercise.

    Field set locked by ``docs/PHASE-7.md`` §"Concrete cards" item 2
    and the card body "Pydantic models" section.

    ``prompt`` is the German sentence with a single ``___`` marker
    (mirrors ``ClozeExercise.sentence_with_blank``). ``partner_lemma``
    is the collocation partner — the LLM is asked to keep it verbatim
    in the prompt so the learner can recognise the fixed phrase.

    ``partner_register`` is the closed
    ``Literal["formal","neutral","colloquial"]`` tuple. The card
    body says: "Pydantic literal validation: register outside
    Literal → validation error" — the Literal is the gate. The
    field name is ``partner_register`` (matching
    ``collocations.partner_register`` on the storage side) rather
    than a bare ``register`` to avoid shadowing
    ``BaseModel.model_fields``.
    The literal on the generator side matches ``collocations.register``
    on the storage side so a seed-time drift surfaces as a Pydantic
    ValidationError at generation, not a runtime schema mismatch.

    ``source_corpus`` is the closed ``Literal["dwds","wiktionary",
    "manual"]`` tuple (PHASE-7 gotcha #12). It travels through the
    wire so the SPA can show "DWDS / Wiktionary / manual" provenance
    next to the exercise; it's also a fingerprint for offline eval.

    ``prompt_template_version`` is the A/B key, identical to the
    ``PROMPT_TEMPLATE_VERSION`` module constant for production
    generations. Hand-edits to the prompt without bumping the
    constant desync the wire field — caught at runtime when the
    activity stamps the constant onto the result.
    """

    prompt: str = Field(
        ...,
        description=(
            "German sentence with '___' marking the cloze position. "
            "The LLM must embed partner_lemma verbatim — no case / "
            "article mutation around the blank."
        ),
    )
    target_word_id: int = Field(
        ..., description="FK to words.id of the collocation's target."
    )
    partner_lemma: str = Field(
        ...,
        description=(
            "The collocation partner word the user has to fill in. "
            "Mirrors collocations.partner_lemma verbatim."
        ),
    )
    # ``partner_register`` on the wire (matching the card body's
    # ``CollocationExerciseOut(..., partner_register, source_corpus,
    # retrieval_chunks)`` shape and ``collocations.partner_register``
    # on the storage side). Avoids Pydantic's ``register``-shadows-
    # ``BaseModel`` warning — the bare name ``register`` would
    # collide with ``BaseModel.model_fields`` machinery.
    partner_register: Literal["formal", "neutral", "colloquial"] = Field(
        ...,
        description=(
            "Register label of the partner_lemma. Mirrors "
            "collocations.partner_register. Pydantic rejects any "
            "value outside the closed Literal."
        ),
    )
    source_corpus: Literal["dwds", "wiktionary", "manual"] = Field(
        ...,
        description=(
            "Provenance of the underlying collocation row. Mirrors "
            "collocations.source_corpus. PHASE-7 gotcha #12 — locked "
            "to the 3-value enum so a typo'd source never silently "
            "passes."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "One sentence explaining the collocation-cloze design "
            "(what semantic axis separates the partner from "
            "near-synonyms, why this register, etc.)."
        ),
    )
    prompt_template_version: str = Field(
        ...,
        description=(
            "Bump when prompt changes; enables A/B eval. Should "
            "always equal PROMPT_TEMPLATE_VERSION for production "
            "generations."
        ),
    )


class CollocationGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields the activity layer (and the Langfuse
    trace) record so an operator can triage without re-running the call.
    Mirrors ``ClozeGenerationError`` / ``MatchingGenerationError`` so
    the three exercise types share the same dead-letter shape.
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
    ) -> "CollocationGenerationError":
        """Build a dead-letter from a Pydantic schema dict.

        Same shape as ``ClozeGenerationError.from_validation_failure``:
        serialise the Pydantic v2 ``model_json_schema()`` dict to a
        stable JSON string so the operator can grep the dead-letter
        body without an extra step.
        """
        return cls(
            message,
            attempted_schema=json.dumps(schema, ensure_ascii=False),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        )


# ---------------------------------------------------------------------------
# Collocation row selection
# ---------------------------------------------------------------------------


def select_collocation_row(
    db: Session,
    user_id: int,
    target_word_id: int,
) -> Collocation:
    """Pick one ``Collocation`` row for the target word.

    Deterministic seed of ``(user_id, target_word_id)`` — same pair
    → same collocation row across calls. Mirrors the stability
    commitment of ``app.cloze.select_target_word``: same input →
    same output, no shuffling.

    Algorithm
    ---------
    1. Query all collocation rows for ``target_word_id`` in stable
       id order.
    2. If zero rows match, raise ``ValueError`` (the caller —
       ``generate_collocation`` — translates to 500: corpus
       inconsistency, the target word has no collocations seeded).
    3. Seed ``random.Random(f"{user_id}|{target_word_id}")`` and
       pick one row id from the candidate list.

    Stability commitment
    --------------------
    ``hash(user_id) XOR hash(target_word_id)`` (via Python's seeded
    RNG): same pair → same row across calls; different user →
    different row for the same word (so two users studying the
    same word get different collocation practice).

    Raises
    ------
    ValueError
        No ``Collocation`` row exists for ``target_word_id`` (corpus
        inconsistency — the seed scripts in 7.1 haven't covered
        this word, or the word id is wrong). Routes translate to
        500.
    """
    candidates: list[int] = [
        row.id
        for row in db.query(Collocation)
        .filter(Collocation.target_word_id == target_word_id)
        .order_by(Collocation.id)
        .all()
    ]
    if not candidates:
        raise ValueError(
            f"select_collocation_row: no Collocation rows for "
            f"target_word_id={target_word_id} (user_id={user_id})"
        )

    seed_str = f"{user_id}|{target_word_id}"
    rng = random.Random(seed_str)
    chosen_id = rng.choice(candidates)
    row = (
        db.query(Collocation)
        .filter(Collocation.id == chosen_id)
        .one_or_none()
    )
    if row is None:
        # Race — a row disappeared between id list and lookup. Treat
        # as a hard error so the operator notices corpus drift
        # rather than silently substituting another row.
        raise ValueError(
            f"select_collocation_row: collocation id {chosen_id} "
            f"disappeared between candidate list and lookup "
            f"(target_word_id={target_word_id}, user_id={user_id})"
        )
    return row


def _fetch_target_word(db: Session, target_word_id: int) -> models.Word:
    """Look up the ``Word`` row for the target.

    Used to embed the lemma / word_type / translation_en in the
    prompt. Returns ``None`` as a sentinel — caller treats that
    as "target_word_id does not match any row" and raises.
    """
    return crud.get_word(db, word_id=target_word_id)


def _target_translation_en(word: models.Word) -> str:
    """Best-effort English translation for the target word.

    The corpus's ``Word.translations`` column is a free-form string
    (a CSV of English glosses in most rows). We return the first
    one trimmed, or an empty string when the column is null —
    the prompt uses it as a learner-axes hint, not a wire field,
    so an empty fallback is acceptable.
    """
    raw = (word.translations or "").strip()
    if not raw:
        return ""
    # The translations column is a CSV of glosses; the first one is
    # the conventional primary translation. Split on the first
    # comma and trim whitespace.
    first = raw.split(",", 1)[0].strip()
    return first


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    word: models.Word,
    collocation: Collocation,
    weakness_axes: dict[str, int],
) -> list[dict]:
    """Build the chat-completions messages for one collocation-cloze generation.

    Prompt shape
    ------------
    The system prompt:

    1. Establishes the role ("German collocation-cloze designer for a
       C1 learner").
    2. Lists the explicit prohibitions — keep ``partner_lemma``
       verbatim (no case / article mutation), don't translate, C1-accept
       bar.
    3. Specifies the JSON output schema with every field on
       ``CollocationExercise``.

    The user prompt embeds:

    - The target word (lemma, word_type, frequency, first English
      translation).
    - The selected collocation row (``partner_lemma``,
      ``partner_register``, ``source_corpus``).
    - The user's weakness axes as JSON.
    - Instructions: design ONE cloze where the answer is
      ``partner_lemma`` in a natural sentence that uses it as a
      collocation of the target word.

    Parameters
    ----------
    word
        The target ``Word`` row (looked up by id).
    collocation
        The selected ``Collocation`` row (looked up by
        ``select_collocation_row``).
    weakness_axes
        Dict from ``WeaknessProfile.axes`` (may be empty for a fresh
        user).

    Returns
    -------
    list[dict]
        Two-message list suitable for ``app.llm.complete``: system
        + user content. Plain ``[{"role": ..., "content": ...}]``.
    """
    target_translation = _target_translation_en(word)

    system_content = (
        "You are a German collocation-cloze designer for a C1 learner. "
        "Produce ONE fill-in-the-blank sentence whose missing word is "
        "the partner_lemma of the target word below, used as a "
        "collocation (not a literal translation or free composition).\n\n"
        "PROHIBITIONS (must obey all):\n"
        "1. Do NOT change word forms, articles, or case endings around the blank.\n"
        "2. Do NOT translate the partner_lemma into another language.\n"
        "3. Do NOT invent a collocation that the partner_lemma does not form "
        "with the target word — use it in a natural, attested usage.\n"
        "4. The register label (formal/neutral/colloquial) must match the "
        "collocation row's register; do not switch to a different register.\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept this cloze "
        "without edits? If no, redo before answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "prompt": "<German sentence with ___ as the blank marker, '
        'embedding partner_lemma verbatim>",\n'
        '  "target_word_id": <integer, the words.id of the target>,\n'
        '  "partner_lemma": "<verbatim collocation partner>",\n'
        '  "partner_register": "formal" | "neutral" | "colloquial",\n'
        '  "source_corpus": "dwds" | "wiktionary" | "manual",\n'
        '  "rationale": "<one sentence, <= 400 chars, explaining the cloze design>",\n'
        '  "prompt_template_version": "collocation-cloze-v1"\n'
        "}\n"
    )

    user_payload: dict[str, Any] = {
        "target_word": {
            "id": word.id,
            "word": word.word,
            "word_type": word.word_type,
            "frequency": word.frequency,
            "translation_en": target_translation,
        },
        "collocation": {
            "partner_lemma": collocation.partner_lemma,
            "partner_register": collocation.partner_register,
            "source_corpus": collocation.source_corpus,
        },
        "learner_axes": weakness_axes,
        "instructions": (
            "Design ONE cloze where the answer is partner_lemma used as a "
            "collocation of target_word.word. The sentence_with_blank "
            "(prompt field) must embed partner_lemma's form exactly as a "
            "C1 learner would expect — no case / article mutation. "
            "Source the sentence from a natural German usage that "
            "exemplifies the collocation; do not invent an unattested one."
        ),
    }

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

    Lazy-imported so ``app.collocation`` stays import-cheap. Mirrors
    ``app.cloze._openai_client`` exactly: same env-var conventions,
    same ``None``-on-missing-key fallback so the production path
    raises ``LLMError`` consistently with the rest of the project.

    Returns ``None`` if ``OPENROUTER_API_KEY`` is missing — caller
    treats that as a "no real LLM available" signal and surfaces
    ``LLMError`` so the route layer can translate to 502.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def generate_collocation(
    db: Session,
    user_id: int,
    target_word_id: int,
) -> CollocationExercise:
    """Generate one ``CollocationExercise`` for a logged-in user.

    Flow
    ----
    1. ``_fetch_target_word`` looks up the target ``Word`` row by id.
       Raises ``ValueError`` on a missing row (corpus inconsistency
       — the route layer translates to 500).
    2. ``select_collocation_row`` picks a deterministic collocation
       for the (user, target) pair. Raises ``ValueError`` when no
       collocation row exists for the target (corpus inconsistency
       — 7.1's seed scripts haven't covered this word).
    3. ``build_prompt`` produces the chat messages with the
       collocation row + weakness axes embedded.
    4. Wrap an OpenRouter-targeted OpenAI client with ``instructor``
       and call ``chat.completions.create(response_model=
       CollocationExercise, ..., max_retries=MAX_ATTEMPTS)``.
       ``MAX_ATTEMPTS = 1`` per the card body — a tighter retry
       budget than ``ClozeExercise`` because the schema is simpler
       (single blank + partner lemma + register) and a misbehaving
       model is more likely to fail in obvious ways (wrong register,
       mutated lemma) than in subtle ones.
    5. Stamp ``prompt_template_version`` onto the result; call
       ``_trace_collocation`` with the metadata dict, including
       ``collocation: true`` discriminator + the trace_id carrier.

    Parameters
    ----------
    db
        Active SQLAlchemy session.
    user_id
        The logged-in learner's id (used for deterministic seed).
    target_word_id
        FK to ``words.id`` of the target word whose collocation we
        practice.

    Returns
    -------
    CollocationExercise
        The validated Pydantic instance ready to return to the
        route layer. ``prompt_template_version`` is forced to
        ``PROMPT_TEMPLATE_VERSION`` so a future maintainer who
        hand-edits the prompt template doesn't silently desync the
        value.

    Raises
    ------
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or the
        transport call hits a non-retryable / persistent-retryable
        failure. The route layer translates this into 502.
    CollocationGenerationError
        Persistent schema violation after ``MAX_ATTEMPTS`` (= 1).
        Carries ``attempted_schema``, ``last_validation_error``,
        ``schema_retry_count`` so the route layer can surface a
        structured 502 instead of a bare 500.
    ValueError
        Bubbles up from ``_fetch_target_word`` /
        ``select_collocation_row`` when the target word doesn't
        exist or has no collocation rows. The route layer
        translates this into 500.
    """
    from app.llm import _default_model, LLMError

    word = _fetch_target_word(db, target_word_id)
    if word is None:
        raise ValueError(
            f"generate_collocation: target_word_id={target_word_id} "
            f"does not match any Word row (corpus inconsistency)"
        )

    collocation = select_collocation_row(db, user_id, target_word_id)

    profile = crud.get_weakness_profile(db, user_id)
    weakness_axes: dict[str, int] = (
        crud.serialize_weakness_profile_axes(profile)
        if profile is not None
        else {}
    )

    messages = build_prompt(word, collocation, weakness_axes)

    # Trace metadata captures the request shape; populated before
    # the call so the error path (CollocationGenerationError raised
    # below) can still log it.
    metadata: dict[str, Any] = {
        "user_id": user_id,
        "weakness_axes": weakness_axes,
        "word_id": word.id,
        "collocation_id": collocation.id,
        "partner_lemma": collocation.partner_lemma,
        "partner_register": collocation.partner_register,
        "source_corpus": collocation.source_corpus,
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": messages,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        # Phase 7.2 — discriminator on the trace metadata so
        # downstream A/B tooling can split collocation-cloze
        # cohorts from regular cloze cohorts.
        "collocation": True,
    }

    raw_client = _openai_client()
    if raw_client is None:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.lexora/.env "
            "and restart the backend container."
        )

    import instructor

    instructor_client = instructor.from_openai(
        raw_client, mode=instructor.Mode.MD_JSON
    )

    schema_retry_count = 0
    last_validation_error = ""
    result: CollocationExercise | None = None
    started = _perf_counter_ms()

    try:
        result = instructor_client.chat.completions.create(
            response_model=CollocationExercise,
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
            _trace_collocation(None, metadata, latency_ms)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        raise CollocationGenerationError.from_validation_failure(
            f"collocation: schema validation failed after "
            f"{schema_retry_count} attempt(s): {last_validation_error}",
            schema=CollocationExercise.model_json_schema(),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        ) from exc

    assert result is not None  # noqa: S101 — see above
    result = result.model_copy(
        update={"prompt_template_version": PROMPT_TEMPLATE_VERSION}
    )

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
    _trace_collocation(result, metadata, latency_ms)
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
    for attr in ("n_attempts", "attempts", "retries", "_retries"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Langfuse trace hook
# ---------------------------------------------------------------------------


def _trace_collocation(
    result: CollocationExercise | None,
    metadata: dict[str, Any],
    latency_ms: int,
) -> None:
    """Emit one Langfuse ``exercise.generate`` span per collocation-cloze.

    Mirrors ``app.cloze._trace_cloze`` and ``app.match._trace_match``:
    v2 SDK shape (``client.span → span.update → span.end →
    client.flush``), graceful no-op when Langfuse keys are missing,
    SDK exceptions swallowed so the activity still succeeds.

    Span name is ``exercise.generate`` (not ``collocation.generate``)
    because the discriminator ``collocation: true`` in the metadata
    is the join key the dashboard / Phase 5.3 grading-log reader
    uses to filter — keeping a single canonical span name across
    exercise types makes the cohort-splitting query uniform. The
    ``exercise_type="cloze"`` field mirrors the cloze span's
    discriminator (collocation-cloze is a cloze *variant*, not a
    new type — Phase 7 plan Hard rule #1).

    Returns ``None`` when keys are missing (graceful degradation;
    no per-call warning spam — observability.py logs once at
    module-import time). On a populated Langfuse client, the
    function is a side-effect emitter; the return value is
    implicitly ``None``.

    Failure mode
    ------------
    Any exception raised by the Langfuse SDK is caught and logged
    at WARNING — the collocation generation has already succeeded
    at this point, so a trace failure must never break the request.
    Same shape as ``_trace_retrieval`` / ``_trace_cloze``.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam per call.
        return

    ptemplate_version = (
        getattr(result, "prompt_template_version", None)
        if result is not None
        else metadata.get("prompt_template_version")
    )
    span_metadata: dict[str, Any] = {
        "user_id": metadata["user_id"],
        "weakness_axes": metadata["weakness_axes"],
        "word_id": metadata["word_id"],
        # Phase 7.2 discriminator — splits collocation-cloze cohorts
        # from regular cloze / matching / comprehension cohorts.
        "collocation": True,
        "exercise_type": "cloze",  # cloze *variant*, not a 4th literal
        "model_id": metadata["model_id"],
        "prompt_template_version": ptemplate_version,
        "schema_retry_count": metadata["schema_retry_count"],
        "latency_ms": latency_ms,
        "prompt_tokens": metadata["prompt_tokens"],
        "completion_tokens": metadata["completion_tokens"],
        # Collocation-specific fields — preserved on the span so the
        # offline A/B reader can group by partner_register /
        # source_corpus.
        "collocation_id": metadata.get("collocation_id"),
        "partner_lemma": metadata.get("partner_lemma"),
        "partner_register": metadata.get("partner_register"),
        "source_corpus": metadata.get("source_corpus"),
    }

    span = None
    try:
        span = client.span(
            name="exercise.generate",
            input=metadata.get("prompt_messages"),
            output=(result.model_dump_json() if result is not None else None),
        )
        span.update(metadata=span_metadata)
        span.end()
        client.flush()
    except Exception as exc:  # noqa: BLE001 — tracing must never break the activity
        logger.warning(
            "collocation: Langfuse trace failed (non-fatal): %s", exc
        )
        if span is not None:
            try:
                span.end()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# DSPy surface — optimization path
# ---------------------------------------------------------------------------


def _offline_dummy_answers() -> list[dict[str, str]]:
    """Build a pool of offline-LM answer dicts for the DummyLM.

    DSPy's ``DummyLM`` returns ``"[[ ## field ## ]] value"`` lines,
    and the ``ChatAdapter`` parses the value back into the field's
    declared type. Our ``exercise`` output is a Pydantic
    ``CollocationExercise``, so the value must be a JSON-encoded
    instance that parses cleanly through
    ``CollocationExercise.model_validate_json``.
    """
    return [
        {
            "exercise": json.dumps(
                {
                    "prompt": "Er ___ eine wichtige Entscheidung.",
                    "target_word_id": 1,
                    "partner_lemma": "treffen",
                    "partner_register": "neutral",
                    "source_corpus": "dwds",
                    "rationale": "offline-stub",
                    "prompt_template_version": "collocation-cloze-v1",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "prompt": "Sie ___ großen Wert auf Pünktlichkeit.",
                    "target_word_id": 2,
                    "partner_lemma": "legen",
                    "partner_register": "formal",
                    "source_corpus": "wiktionary",
                    "rationale": "offline-stub",
                    "prompt_template_version": "collocation-cloze-v1",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "prompt": "Wir ___ uns auf den Sommer.",
                    "target_word_id": 3,
                    "partner_lemma": "freuen",
                    "partner_register": "colloquial",
                    "source_corpus": "manual",
                    "rationale": "offline-stub",
                    "prompt_template_version": "collocation-cloze-v1",
                },
                ensure_ascii=False,
            )
        },
    ]


def _offline_json_answers() -> list[dict[str, str]]:
    """Answer pool for MIPROv2's internal ``JSONAdapter``-shaped calls.

    Same pattern as ``app.cloze._offline_json_answers`` — the
    internal prompt-proposer expects
    ``{"proposed_instruction": "..."}`` responses (a string field,
    not a JSON object). Five diverse stubs keeps the proposer
    cycling without choking.
    """
    return [
        {"proposed_instruction": "Keep partner_lemma verbatim in the prompt."},
        {"proposed_instruction": "Match register to the collocation row's label."},
        {"proposed_instruction": "Embed the target word's first English translation."},
        {"proposed_instruction": "State the C1-accept bar in the rationale."},
        {"proposed_instruction": "Source the sentence from attested German usage."},
    ]


def _configure_dspy() -> None:
    """Configure DSPy with our OpenRouter adapter, or ``DummyLM`` offline.

    Mirrors ``app.cloze._configure_dspy`` exactly: idempotent,
    switches to ``DummyLM`` when no API key is present so the CI
    suite stays network-free. We pool both the cloze-path
    answers (JSON-encoded ``CollocationExercise`` payloads) and
    the MIPROv2 internal-proposer instruction-shaped stubs so a
    single ``DummyLM`` serves both protocols in any interleaving.
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


class CollocationSignature(dspy.Signature):
    """DSPy signature for one collocation-cloze generation.

    Inputs match the ``build_prompt`` payload shape; the output is
    the full ``CollocationExercise`` Pydantic model. DSPy 3.x
    supports Pydantic-typed output fields via ``dspy.Predict`` /
    ``dspy.ChainOfThought``.

    ``target_word_id`` is in the input set so the optimizer can
    teach the model that ``exercise.target_word_id`` must echo the
    input without the LLM having to guess.

    ``partner_lemma`` + ``partner_register`` + ``source_corpus``
    carry the collocation row's columns verbatim so the signature
    is faithful to the storage shape — the optimizer can teach the
    model that ``exercise.partner_lemma == input.partner_lemma``
    and ``exercise.register == input.partner_register`` invariants.
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
    partner_lemma: str = dspy.InputField(
        desc="The collocation partner word (verbatim from collocations.partner_lemma)."
    )
    partner_register: str = dspy.InputField(
        desc=(
            "The register label of the partner (verbatim from "
            "collocations.partner_register; one of formal / neutral "
            "/ colloquial)."
        )
    )
    source_corpus: str = dspy.InputField(
        desc=(
            "The provenance of the collocation row (verbatim from "
            "collocations.source_corpus; one of dwds / wiktionary / manual)."
        )
    )
    exercise: CollocationExercise = dspy.OutputField(
        desc="A Pydantic CollocationExercise matching the production contract."
    )


class CollocationModule(dspy.Module):
    """DSPy module that wraps the production ``CollocationSignature``.

    Uses ``dspy.Predict`` (single-shot — no chain-of-thought) so
    the output shape stays compatible with the production
    ``instructor`` path. The optimizer
    (``optimize_collocation_module``) can swap the predictor for a
    tuned one without changing this surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(CollocationSignature)

    def forward(  # type: ignore[override]
        self,
        *,
        word: str,
        context_sentence: str,
        learner_axes_json: str,
        target_word_id: int,
        partner_lemma: str,
        partner_register: str,
        source_corpus: str,
    ) -> dspy.Prediction:
        return self.predict(
            word=word,
            context_sentence=context_sentence,
            learner_axes_json=learner_axes_json,
            target_word_id=target_word_id,
            partner_lemma=partner_lemma,
            partner_register=partner_register,
            source_corpus=source_corpus,
        )


def optimize_collocation_module(
    train_set: Iterable[dict],
    val_set: Iterable[dict] | None = None,
) -> CollocationModule:
    """Run the offline DSPy optimizer against a held-out eval set.

    Strategy mirrors ``app.cloze.optimize_cloze_module``:

    - Always uses ``DummyLM`` when no API key is present so the CI
      suite runs without network (Hard rule #6).
    - Tries ``dspy.MIPROv2`` first (the spec's preferred optimizer).
      Falls back to ``dspy.BootstrapFewShot`` if MIPROv2 raises on
      the active dep tree.
    - Returns a ``CollocationModule`` with the optimised prompt
      instructions baked in. The caller (a future CLI script —
      7.2 doesn't ship one; Phase 9 owns it) would serialise the
      optimised module to ``backend/app/collocation_optimized.json``
      so the production path could read it on next start.

    Parameters
    ----------
    train_set, val_set
        Iterable of dicts with the seven input keys (word, context,
        learner_axes_json, target_word_id, partner_lemma,
        partner_register, source_corpus).

    Returns
    -------
    CollocationModule
        The optimised module. The optimizer mutates the module's
        internal predictor in place; the same instance is returned
        for caller convenience.
    """
    _configure_dspy()

    train_examples = [
        dspy.Example(**row).with_inputs(
            "word",
            "context_sentence",
            "learner_axes_json",
            "target_word_id",
            "partner_lemma",
            "partner_register",
            "source_corpus",
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
                "partner_lemma",
                "partner_register",
                "source_corpus",
            )
            for row in val_set
        ]
        if val_set is not None
        else None
    )

    module = CollocationModule()

    optimizer_cls = getattr(dspy, "MIPROv2", None)
    if optimizer_cls is None:
        optimizer_cls = getattr(dspy, "BootstrapFewShot", None)

    if optimizer_cls is None:
        logger.warning(
            "optimize_collocation_module: no MIPROv2 / BootstrapFewShot "
            "on the DSPy dep tree; returning the un-optimized module."
        )
        return module

    optimizer = optimizer_cls(metric=_collocation_metric)
    try:
        optimized = optimizer.compile(
            module,
            trainset=train_examples,
            valset=val_examples,
        )
    except TypeError:
        optimized = optimizer.compile(module, trainset=train_examples)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "optimize_collocation_module: optimizer %s raised on the "
            "offline path (%s); returning the un-optimized module. "
            "Re-run with --live and OPENROUTER_API_KEY set to actually "
            "optimize.",
            optimizer_cls.__name__,
            exc,
        )
        return module
    return optimized


def _collocation_metric(
    example: Any, prediction: Any, trace: Any | None = None
) -> float:
    """Offline quality metric for the collocation optimizer.

    Score range: 0.0..1.0. The bar is intentionally loose — the
    production C1-accept check is qualitative (Anurag hand-reviews),
    not a numeric gate. The optimizer uses this score to pick a
    better prompt; ``scripts/eval_collocation.py`` (a future card)
    would run the more rigorous per-row comparison.

    Components:

    - ``+0.4`` if ``prediction.exercise.partner_lemma`` equals
      ``example.partner_lemma`` verbatim (the collocation
      contract).
    - ``+0.2`` if ``prediction.exercise.register`` equals
      ``example.partner_register``.
    - ``+0.2`` if ``"___"`` is present in
      ``prediction.exercise.prompt``.
    - ``+0.2`` if ``prediction.exercise.target_word_id`` equals
      ``example.target_word_id``.
    """
    try:
        ex: CollocationExercise = prediction.exercise  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0
    score = 0.0
    if ex.partner_lemma == example.partner_lemma:
        score += 0.4
    if ex.partner_register == example.partner_register:
        score += 0.2
    if "___" in ex.prompt:
        score += 0.2
    if ex.target_word_id == example.target_word_id:
        score += 0.2
    return score


__all__ = [
    "Collocation",
    "CollocationExercise",
    "CollocationGenerationError",
    "CollocationModule",
    "CollocationSignature",
    "MAX_ATTEMPTS",
    "PARTNER_REGISTER",
    "PROMPT_TEMPLATE_VERSION",
    "_trace_collocation",
    "build_prompt",
    "generate_collocation",
    "optimize_collocation_module",
    "select_collocation_row",
]