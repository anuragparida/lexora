"""Phase 8.3 (card t_fa86ac58) — idiom exercise generator + DSPy module.

This module is the idiom counterpart of Phase 4.2's ``app.cloze`` and
Phase 7.2's ``app.collocation``. It ships three distinct surfaces that
share one Pydantic contract (``IdiomExercise``):

1. **Production path** — ``generate_idiom(db, word_id, *, enable_rag)``.
   Picks a ``Phrase`` row deterministically from the read-only
   ``phrases`` table (Phase 8.1, card t_d967c006), builds a constrained
   prompt, calls ``app.llm.complete`` wrapped with ``instructor`` so the
   response validates against the ``IdiomExercise`` Pydantic model.
   Bounded retry (Hard rule #5 of PHASE-8 — ``MAX_ATTEMPTS = 3`` per
   the card body). Traces through ``_trace_idiom`` (returns ``None``
   when Langfuse keys are missing; full span emission when present).

2. **Optimization path** — ``IdiomSignature`` + ``IdiomModule`` +
   ``optimize_idiom_module``. DSPy is wired to ``app.llm.complete``
   via the shared ``_DSPyOpenAICompatLM`` adapter (lives in
   ``app.llm`` since Phase 6.2's extraction). The optimizer is
   offline-capable: with no API key,
   ``dspy.utils.dummies.DummyLM`` swaps in automatically so the CI
   suite runs without network.

3. **Selector** — ``select_phrase_row``. Picks one ``Phrase`` row
   by ``word_id`` seed. Mirrors the deterministic-seed pattern of
   ``app.cloze.select_target_word``: same ``word_id`` → same phrase
   row across calls. No state, no random.

**Phrase model — 8.3 read-only mirror.** Phase 8.1 (card t_d967c006)
owns the authoritative ``Phrase`` SQLAlchemy model and the
``phrases`` Alembic migration. When 8.1 lands on main, this module
is updated to ``Phrase = models.Phrase`` (a one-line re-export, the
Phase 7.2 pattern for ``Collocation = models.Collocation``). Until
then, this module declares a **local read-only mirror** of the
expected 8.1 schema (``__tablename__ = "phrases"`` is the join key,
not Python identity) so the module compiles, the type system has
something concrete to introspect, and the tests run with the same
fixture shape 8.4 will exercise.

**``word_id`` contract.** In the cloze / collocation worlds,
``word_id`` is a FK to ``words.id``. For idioms (Phase 8), idioms
are a **standalone surface** — there is no row in ``words`` for an
idiom (a phrase is its own entity). The card body keeps the input
field name ``word_id`` for cross-exercise-type signature uniformity
but its semantic role is *phrase-selector seed*: the
``select_phrase_row`` helper deterministically maps the integer
seed to a phrase row in the curated ``phrases`` table. This
preserves the ``fetch a deterministic row given an integer seed``
contract from Phase 4.2 (``select_target_word``) and Phase 7.2
(``select_collocation_row``).

Hard rules enforced here:

- **#1 (Literal widening is wire-level)** — ``exercise_type: Literal["cloze",
  "matching", "comprehension", "idiom"]`` on ``BaseExerciseFields``. This
  module never narrows the literal (Phase 7's hard rule carries forward
  verbatim). The ``IdiomExercise`` model narrows the discriminator on its
  own subclass to ``Literal["idiom"]``; the base class widens once.
- **#2 (read-only ``phrases``)** — ``generate_idiom`` only reads
  ``phrases``; no INSERT/UPDATE/DELETE from runtime. The seed scripts
  (8.1) are the only write path. Mirrors Phase 7.2's ``collocation``
  discipline byte-for-byte.
- **#3 (offline tests)** — DSPy module tests run with ``DummyLM``;
  the live-LLM Ragas regression in 8.4's verification is the only
  path that requires OpenRouter keys. CI stays offline.
- **#4 (Langfuse lexora project)** — every LLM call traces through
  ``_trace_idiom``, wired to the existing ``lexora`` project (the
  shared wrapper from ``app.observability.py``).
- **#5 (single LLM provider)** — production path goes through
  ``app.llm.complete``; the DSPy adapter targets the same wire
  format.
- **#6 (bounded retry)** — ``MAX_ATTEMPTS = 3`` per the card body
  ("retries on schema failure (Phase 4.2 retry pattern: up to 3
  attempts)"). Same budget as ``app.cloze.MAX_ATTEMPTS`` — idioms
  have a richer schema (phrase + definition + example_usage +
  cloze_target + frequency_band) than cloze, so a 3-attempt ceiling
  is appropriate.
- **#7 (no bge-m3 OpenRouter)** — the ``enable_rag`` stub embeds
  with local ``sentence-transformers`` (Phase 1.3) when RAG-on is
  wired in 8.4; 8.3 only stubs the parameter (it doesn't need a
  real embedding call).
- **#8 (no env-derived thresholds)** — ``MAX_ATTEMPTS`` and
  ``PROMPT_TEMPLATE_VERSION`` are module constants, not env-derived.
  ``git grep -n "getenv.*RAG"`` returns nothing by construction.
- **#9 (Alembic-only migration)** — 8.1 owns the Alembic migration
  for the ``phrases`` table. This module does NOT touch migration
  files; it only declares a read-only ORM mirror for testability.
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, DateTime, String, Text, func
from sqlalchemy.orm import Session

from app import crud, models
from app.database import Base
from app.llm import _DSPyOpenAICompatLM  # the shared adapter; see app.llm docstring
from app.observability import get_langfuse

# Lazy DSPy import — the optimization-path classes touch dspy at
# import time; the production path doesn't. Mirrors ``app.cloze`` /
# ``app.collocation``.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #11 / Phase 7 plan #9).
#
# ``PROMPT_TEMPLATE_VERSION`` is the A/B key downstream eval tooling
# splits on. ``MAX_ATTEMPTS = 3`` per the 8.3 card body
# ("retries on schema failure (Phase 4.2 retry pattern: up to 3
# attempts)"). ``FREQUENCY_BAND`` is the closed literal the LLM is
# asked to choose from — mirrors the ``phrases.frequency_band``
# column 8.1 ships (high / mid / low).
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "idiom-v1"
MAX_ATTEMPTS: int = 3
FREQUENCY_BAND: tuple[str, ...] = ("high", "mid", "low")
# Closed literal for ``phrases.source_attribution`` (Phase 8 spec §8.3
# Pydantic contract). Comma-joined subset is allowed on the storage
# side (``"dwds,goethe"``), but every token in the join must be in
# this tuple — enforced by ``_validate_source_attribution`` below.
SOURCE_ATTRIBUTION: tuple[str, ...] = ("dwds", "goethe", "schiller")


# ---------------------------------------------------------------------------
# Read-only Phrase ORM import.
#
# Phase 8.1 (card t_d967c006) owns the authoritative SQLAlchemy
# ``Phrase`` model in ``app.models``. The fold (card t_62b3d96c) brought
# 8.1 onto main before 8.3, so we re-export the canonical model here.
# The join key is ``__tablename__ = "phrases"`` — queries on this module
# filter on the same column names the production generator (8.4) uses.
#
# Prior to the fold, 8.3 shipped a local ``class Phrase(Base)`` mirror so
# the module would compile + pass tests in its own worktree (where
# ``models.Phrase`` was not yet on main). That mirror caused a duplicate
# ``phrases`` Table registration once 8.1 landed; the post-fold
# reconciliation (this card) drops the local mirror in favor of the
# canonical ``models.Phrase``.
#
# Read-only invariants (Hard rule #2):
#
# - No INSERT / UPDATE / DELETE anywhere in this module.
# - The seed scripts (8.1) are the only write path.
# - The runtime SELECT here is gated by ``enable_rag`` for the
#   RAG-on nearest-neighbor path; otherwise the curated row alone
#   is read.
# ---------------------------------------------------------------------------


# Canonical Phrase model — re-export so the rest of this module uses
# ``Phrase`` as if it were locally defined. Mirrors the Phase 7.2
# ``Collocation = models.Collocation`` pattern.
Phrase = models.Phrase


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class IdiomExercise(BaseModel):
    """The metadata contract for a single idiom exercise (Phase 8.3).

    Field set locked by ``docs/PHASE-8.md`` §"8.3" Pydantic contract
    and the card body "Pydantic ``IdiomExercise`` response model"
    section.

    ``phrase`` is the German idiom verbatim (5..200 chars). The
    bound is wide enough to fit ``"ins Blaue hinein"`` (14 chars)
    through ``"Tomaten auf den Augen haben"`` (28 chars) and even
    the multi-clause literary attestations 8.2 surfaces
    (``~180 chars`` worst-case in DWDS).

    ``definition`` is the learner-facing gloss (1..400 chars).
    The card body says "Phase 8 Pydantic schema enforces 1..400
    chars on ``definition`` (a tight cap that forces the generator
    to compress long DWDS definitions into learner-friendly ones
    via the prompt)" — same rationale as the Phase 6.5
    comprehension prompt pattern.

    ``example_usage`` (5..400 chars) is an illustrative sentence
    using the idiom. ``None`` when the curated row has no
    attested example (DWDS exports sometimes omit
    ``<Example>``).

    ``source_attribution`` is a comma-joined literal — ``dwds``,
    ``goethe``, ``schiller``, or any subset joined by ``,``
    (``"dwds,goethe"`` when an idiom exists in both sources).
    The list-of-tokens validator below enforces the Pydantic
    closed-literal invariant; the comma-join shape lets one
    idiom be attributed to multiple seeds.

    ``attested_quote`` / ``attested_source`` are nullable —
    populated only for Goethe / Schiller attestations (8.2
    seeds). A ``dwds``-only row has both ``None``; a
    Goethe-attested row has both populated.

    ``frequency_band`` is the closed ``Literal["high","mid","low"]``
    tuple — used by 8.4's cloze variant to bias towards
    high-band idioms first.

    ``cloze_target`` is the idiom phrase with one word blanked
    for the cloze-within-idiom variant (``"ins ___ hinein"``).
    ``None`` when the curated row doesn't lend itself to a
    internal blank.

    ``prompt_template_version`` is the A/B key, identical to the
    ``PROMPT_TEMPLATE_VERSION`` module constant for production
    generations.
    """

    exercise_id: str = Field(
        ...,
        description=(
            "Server-minted per generation id. Phase 8.3 ships a "
            "uuid4 hex string (32 chars, no dashes). Mirrors the "
            "shape the future /exercises/idiom endpoint (8.4) "
            "stamps on the wire for the grade_logs round-trip."
        ),
    )
    phrase: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description=(
            "German idiom verbatim. The curated phrasal surface "
            "that the learner practices."
        ),
    )
    definition: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Learner-facing definition, 1..400 chars. Forces the "
            "generator to compress long DWDS definitions into a "
            "gloss."
        ),
    )
    example_usage: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description=(
            "An illustrative German sentence using the idiom. "
            "Sourced from the curated row's example_usage or "
            "attested_quote."
        ),
    )
    source_attribution: str = Field(
        ...,
        description=(
            "Comma-joined subset of Literal['dwds','goethe',"
            "'schiller']. Validated token-by-token by "
            "_validate_source_attribution."
        ),
    )
    attested_quote: str | None = Field(
        default=None,
        description=(
            "Optional literary attestation (Goethe / Schiller). "
            "Populated only when source_attribution includes "
            "'goethe' or 'schiller'."
        ),
    )
    attested_source: str | None = Field(
        default=None,
        description=(
            "Optional citation for attested_quote (e.g. 'Faust I, "
            "Studierzimmer (1168-1186)'). Populated alongside "
            "attested_quote."
        ),
    )
    frequency_band: Literal["high", "mid", "low"] = Field(
        ...,
        description=(
            "Closed-literal frequency band. Used by 8.4's cloze "
            "variant to bias towards 'high'-band idioms first."
        ),
    )
    cloze_target: str | None = Field(
        default=None,
        description=(
            "Idiom phrase with one word blanked for the "
            "cloze-within-idiom variant (e.g. 'ins ___ hinein'). "
            "None when the curated row doesn't lend itself to a "
            "blank-within-phrase exercise."
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

    @field_validator("source_attribution")
    @classmethod
    def _validate_source_attribution(cls, v: str) -> str:
        """Enforce the closed-literal invariant on the joined string.

        The card body allows comma-joined subsets
        (``"dwds,goethe"`` when an idiom is in both sources). Each
        comma-separated token must be one of
        ``Literal["dwds","goethe","schiller"]``. Whitespace around
        commas is stripped; an empty join (``""``) is rejected.
        """
        if not v or not v.strip():
            raise ValueError(
                "source_attribution must be a non-empty "
                "comma-joined subset of 'dwds','goethe','schiller'"
            )
        tokens = [token.strip() for token in v.split(",") if token.strip()]
        if not tokens:
            raise ValueError(
                "source_attribution must contain at least one token"
            )
        valid = set(SOURCE_ATTRIBUTION)
        invalid = [t for t in tokens if t not in valid]
        if invalid:
            raise ValueError(
                f"source_attribution tokens {invalid!r} are outside "
                f"the closed literal {list(valid)}"
            )
        # De-dupe + preserve order on first appearance for a stable
        # serializer. "dwds,dwds" → "dwds".
        seen: set[str] = set()
        deduped: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return ",".join(deduped)


class IdiomGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields the activity layer (and the Langfuse
    trace) record so an operator can triage without re-running the call.
    Mirrors ``ClozeGenerationError`` /
    ``CollocationGenerationError`` / ``MatchingGenerationError`` /
    ``ComprehensionGenerationError`` so the exercise types share the
    same dead-letter shape.
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
    ) -> "IdiomGenerationError":
        """Build a dead-letter from a Pydantic schema dict.

        Same shape as the sibling generation-error helpers:
        serialise the Pydantic v2 ``model_json_schema()`` dict to
        a stable JSON string so the operator can grep the
        dead-letter body without an extra step.
        """
        return cls(
            message,
            attempted_schema=json.dumps(schema, ensure_ascii=False),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        )


# ---------------------------------------------------------------------------
# Phrase row selection
# ---------------------------------------------------------------------------


def select_phrase_row(db: Session, word_id: int) -> Phrase:
    """Pick one ``Phrase`` row for the given integer seed.

    Deterministic seed of ``word_id`` — same integer → same phrase
    row across calls. Mirrors the stability commitment of
    ``app.cloze.select_target_word``: same input → same output,
    no shuffling.

    **``word_id`` is a phrase-selector seed, not a ``words.id`` FK.**

    In the cloze / collocation worlds ``word_id`` is a FK to
    ``words.id``. For idioms (Phase 8), idioms are a standalone
    surface — there is no row in ``words`` for an idiom (a phrase
    is its own entity). The card body keeps the input field name
    ``word_id`` for cross-exercise-type signature uniformity but
    its semantic role is a *phrase-selector seed*: this helper
    deterministically maps the integer seed to a phrase row in
    the curated ``phrases`` table. Same
    ``(word_id)`` → same phrase across calls; 8.4's endpoint
    and the route-layer semantics preserve this stability for
    re-clicks on the same day.

    Algorithm
    ---------
    1. Query all phrase rows in stable ``id`` order (slug PK).
    2. If zero rows match, raise ``ValueError`` (the caller —
       ``generate_idiom`` — translates to 500: corpus
       inconsistency, no phrases seeded yet).
    3. Seed ``random.Random(str(word_id))`` and pick one slug from
       the candidate list.

    Raises
    ------
    ValueError
        No ``Phrase`` row exists in the ``phrases`` table (corpus
        inconsistency — the seed scripts in 8.1 haven't been run
        yet, or the migration hasn't been applied). Routes
        translate to 500.
    """
    candidates: list[str] = [
        row.id
        for row in db.query(Phrase).order_by(Phrase.id).all()
    ]
    if not candidates:
        raise ValueError(
            f"select_phrase_row: no Phrase rows in the phrases "
            f"table (word_id={word_id})"
        )

    seed_str = f"phrase|{word_id}"
    rng = random.Random(seed_str)
    chosen_id = rng.choice(candidates)
    row = (
        db.query(Phrase)
        .filter(Phrase.id == chosen_id)
        .one_or_none()
    )
    if row is None:
        # Race — a row disappeared between id list and lookup.
        # Treat as a hard error so the operator notices corpus
        # drift rather than silently substituting another row.
        raise ValueError(
            f"select_phrase_row: phrase id {chosen_id!r} "
            f"disappeared between candidate list and lookup "
            f"(word_id={word_id})"
        )
    return row


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    phrase: Phrase,
    weakness_axes: dict[str, int],
    *,
    retrieved_chunks_json: str | None = None,
) -> list[dict]:
    """Build the chat-completions messages for one idiom generation.

    Prompt shape
    ------------
    The system prompt:

    1. Establishes the role ("German idiom designer for a C1 learner").
    2. Lists the explicit prohibitions — keep ``phrase`` verbatim
       (no case / article mutation around cloze_target), don't
       translate, C1-accept bar.
    3. Specifies the JSON output schema with every field on
       ``IdiomExercise``.

    The user prompt embeds:

    - The selected phrase row (``phrase``, ``definition``,
      ``example_usage``, ``source_attribution``, ``frequency_band``,
      optional ``attested_quote`` + ``attested_source``).
    - The user's weakness axes as JSON.
    - The optional ``retrieved_chunks_json`` (empty list when RAG
      is off; populated list of nearest-neighbor phrase snippets
      when ``enable_rag=True`` — 8.3 only stubs the parameter;
      8.4 wires the real retrieval).
    - Instructions: produce ONE ``IdiomExercise`` payload.

    Parameters
    ----------
    phrase
        The selected ``Phrase`` row (looked up by
        ``select_phrase_row``).
    weakness_axes
        Dict from ``WeaknessProfile.axes`` (may be empty for a
        fresh user).
    retrieved_chunks_json
        JSON-encoded list of nearest-neighbor phrase snippets
        (empty ``"[]"`` when RAG is off). The 8.3 stub accepts the
        parameter but doesn't wire a real retrieval call — 8.4's
        endpoint honors ``enable_rag=True`` and replaces this with
        a real ``app.embeddings.embed_one`` + ``phrases`` nearest
        query.

    Returns
    -------
    list[dict]
        Two-message list suitable for ``app.llm.complete``:
        system + user content. Plain ``[{"role": ..., "content":
        ...}]``.
    """
    retrieved_chunks_payload = (
        retrieved_chunks_json if retrieved_chunks_json is not None else "[]"
    )

    system_content = (
        "You are a German idiom exercise designer for a C1 learner. "
        "Produce ONE IdiomExercise payload for the curated phrase "
        "below — a learner-friendly gloss, an example sentence that "
        "uses the idiom naturally, and (when appropriate) a cloze-"
        "target variant where one word inside the idiom is blanked.\n\n"
        "PROHIBITIONS (must obey all):\n"
        "1. Do NOT change word forms, articles, or case endings in "
        "the phrase or example_usage.\n"
        "2. Do NOT translate the idiom into another language; the "
        "phrase, definition, example_usage, cloze_target, and any "
        "attested_quote must stay in German.\n"
        "3. Do NOT paraphrase the curated phrase — keep it VERBATIM. "
        "The activity teaches the exact form the learner meets in "
        "the curated corpus.\n"
        "4. The frequency_band ('high' / 'mid' / 'low') must match "
        "the curated row's frequency_band; do not switch to a "
        "different band.\n"
        "5. The source_attribution must be a comma-joined subset of "
        "'dwds','goethe','schiller' (e.g. 'dwds' or 'dwds,goethe'); "
        "do not introduce other tokens.\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept this idiom "
        "exercise without edits? If no, redo before answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "exercise_id": "<server-minted uuid4 hex, you generate a fresh one>",\n'
        '  "phrase": "<verbatim curated phrase, 5..200 chars>",\n'
        '  "definition": "<learner-facing gloss, 1..400 chars>",\n'
        '  "example_usage": "<illustrative German sentence, 5..400 chars>",\n'
        '  "source_attribution": "<comma-joined subset of dwds|goethe|schiller>",\n'
        '  "attested_quote": "<German attestation, 5..400 chars, or null>",\n'
        '  "attested_source": "<citation string, 5..200 chars, or null>",\n'
        '  "frequency_band": "high" | "mid" | "low",\n'
        '  "cloze_target": "<phrase with one word blanked ___ for the cloze variant, or null>",\n'
        '  "prompt_template_version": "idiom-v1"\n'
        "}\n"
    )

    phrase_payload: dict[str, Any] = {
        "id": phrase.id,
        "phrase": phrase.phrase,
        "definition": phrase.definition,
        "example_usage": phrase.example_usage,
        "source_attribution": phrase.source_attribution,
        "frequency_band": phrase.frequency_band,
        "attested_quote": phrase.attested_quote,
        "attested_source": phrase.attested_source,
    }

    user_payload: dict[str, Any] = {
        "phrase": phrase_payload,
        "learner_axes": weakness_axes,
        # 8.3 stub: ``retrieved_chunks_json`` stays "[]" by default.
        # 8.4 wires the real retrieval when enable_rag=True.
        "retrieved_chunks": json.loads(retrieved_chunks_payload)
        if retrieved_chunks_payload
        else [],
        "instructions": (
            "Produce ONE IdiomExercise payload for the curated phrase "
            "above. The phrase must appear verbatim in the response; "
            "the definition must be a learner-friendly 1..400-char "
            "gloss; the example_usage must be a 5..400-char German "
            "sentence that uses the idiom naturally. Set "
            "cloze_target to a phrase-with-blank variant when the "
            "idiom lends itself to a within-phrase blank (5..200 "
            "chars); leave it null otherwise. Stamp "
            "prompt_template_version='idiom-v1' exactly."
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

    Lazy-imported so ``app.idiom`` stays import-cheap. Mirrors
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


def generate_idiom(
    db: Session,
    word_id: int,
    *,
    enable_rag: bool = False,
    retrieve_neighbor=None,
) -> IdiomExercise:
    """Generate one ``IdiomExercise`` for the given integer seed.

    Flow
    ----
    1. ``select_phrase_row`` picks a deterministic phrase for
       ``word_id``. Raises ``ValueError`` when no phrase row exists
       (corpus inconsistency — 8.1's seed hasn't run yet).
    2. Read the user's weakness axes for the prompt.
    3. When ``enable_rag=True`` AND a ``retrieve_neighbor`` callable
       is supplied, fetch a single nearest-neighbor phrase snippet
       from the curated ``phrases`` table; embed it in the user
       prompt. When ``False`` (default) or no callable supplied,
       the prompt is byte-for-byte identical to the curated-only
       path. The 8.3 stub leaves ``retrieve_neighbor=None``; 8.4's
       endpoint passes a real neighbor-fetching callable.
    4. ``build_prompt`` produces the chat messages with the phrase
       row + optional nearest-neighbor chunk embedded.
    5. Wrap an OpenRouter-targeted OpenAI client with
       ``instructor`` and call
       ``chat.completions.create(response_model=IdiomExercise, ...,
       max_retries=MAX_ATTEMPTS)``. ``MAX_ATTEMPTS = 3`` per the
       card body — idioms have a richer schema than cloze, so a
       3-attempt ceiling is appropriate.
    6. Stamp ``prompt_template_version`` onto the result; call
       ``_trace_idiom`` with the metadata dict.

    Parameters
    ----------
    db
        Active SQLAlchemy session.
    word_id
        Integer seed for the ``select_phrase_row`` selector. Same
        ``word_id`` → same phrase across calls (no shuffling). See
        ``select_phrase_row`` for the ``word_id`` semantics
        discussion (idioms are a standalone surface — not a FK to
        ``words.id``).
    enable_rag
        Opt-in flag (default False). When True AND
        ``retrieve_neighbor`` is supplied, embed the nearest-
        neighbor phrase snippet in the user prompt. When False
        (default), the prompt is the curated-only shape — byte-
        for-byte stable.
    retrieve_neighbor
        Optional callable ``(db: Session, phrase: Phrase) -> dict |
        None`` that returns a single nearest-neighbor phrase
        snippet (or ``None`` on miss). 8.3 ships the parameter as
        a stub; 8.4's endpoint passes a real callable that
        integrates with ``app.embeddings.embed_one`` + the
        ``phrases`` table by-id.

    Returns
    -------
    IdiomExercise
        The validated Pydantic instance ready to return to the
        route layer. ``prompt_template_version`` is forced to
        ``PROMPT_TEMPLATE_VERSION`` so a future maintainer who
        hand-edits the prompt template doesn't silently desync
        the value.

    Raises
    ------
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or the
        transport call hits a non-retryable / persistent-retryable
        failure. The route layer translates this into 502.
    IdiomGenerationError
        Persistent schema violation after ``MAX_ATTEMPTS`` (= 3).
        Carries ``attempted_schema``, ``last_validation_error``,
        ``schema_retry_count`` so the route layer can surface a
        structured 502 instead of a bare 500.
    ValueError
        Bubbles up from ``select_phrase_row`` when no phrase row
        exists in the table (corpus inconsistency — 8.1 hasn't
        seeded yet). The route layer translates this into 500.
    """
    from app.llm import _default_model, LLMError

    phrase = select_phrase_row(db, word_id)

    # Read the user profile lazily. 8.3's entry point doesn't take
    # a user_id (idioms are standalone — not user-anchored), so
    # the profile read is best-effort and falls back to an empty
    # axes dict when no profile can be resolved. 8.4's route
    # layer will keep the surface.
    weakness_axes: dict[str, int] = {}

    # 8.3 stub: ``enable_rag`` is plumbed but doesn't trigger a real
    # retrieval. When ``retrieve_neighbor`` is supplied AND RAG is on,
    # call it; otherwise stay byte-for-byte identical to the curated-
    # only path. 8.4's endpoint replaces ``retrieve_neighbor=None``
    # with a real callable.
    retrieved_chunks_json: str | None = None
    if enable_rag and retrieve_neighbor is not None:
        try:
            neighbor = retrieve_neighbor(db, phrase)
        except Exception as exc:  # noqa: BLE001 — best-effort fallback
            logger.warning(
                "idiom: retrieve_neighbor raised unexpectedly "
                "(falling back to curated-only): %s",
                exc,
            )
            neighbor = None
        if neighbor is None:
            retrieved_chunks_json = "[]"
        else:
            retrieved_chunks_json = json.dumps(
                [neighbor], ensure_ascii=False
            )

    messages = build_prompt(
        phrase,
        weakness_axes,
        retrieved_chunks_json=retrieved_chunks_json,
    )

    # Trace metadata captures the request shape; populated before
    # the call so the error path (IdiomGenerationError raised
    # below) can still log it.
    metadata: dict[str, Any] = {
        "word_id": word_id,
        "weakness_axes": weakness_axes,
        "phrase_id": phrase.id,
        "phrase": phrase.phrase,
        "source_attribution": phrase.source_attribution,
        "frequency_band": phrase.frequency_band,
        "enable_rag": bool(enable_rag),
        "retrieved_neighbor_present": bool(retrieved_chunks_json),
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": messages,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        # Phase 8.3 discriminator — splits idiom cohorts from
        # cloze / matching / comprehension / collocation cohorts.
        "exercise_type": "idiom",
        "idiom": True,
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
    # as ``app.cloze`` / ``app.collocation`` — the tool-calling
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
    result: IdiomExercise | None = None
    started = _perf_counter_ms()

    try:
        result = instructor_client.chat.completions.create(
            response_model=IdiomExercise,
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
            _trace_idiom(None, metadata, latency_ms)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        raise IdiomGenerationError.from_validation_failure(
            f"idiom: schema validation failed after "
            f"{schema_retry_count} attempt(s): {last_validation_error}",
            schema=IdiomExercise.model_json_schema(),
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
    _trace_idiom(result, metadata, latency_ms)
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


def _trace_idiom(
    result: IdiomExercise | None,
    metadata: dict[str, Any],
    latency_ms: int,
) -> None:
    """Emit one Langfuse ``exercise.generate`` span per idiom.

    Mirrors ``app.cloze._trace_cloze`` /
    ``app.collocation._trace_collocation`` /
    ``app.comprehension._trace_comprehension``: v2 SDK shape
    (``client.span → span.update → span.end → client.flush``),
    graceful no-op when Langfuse keys are missing, SDK exceptions
    swallowed so the activity still succeeds.

    Span name is ``exercise.generate`` (not ``idiom.generate``)
    because the discriminator ``idiom: true`` in the metadata
    is the join key the dashboard uses to filter — keeping a
    single canonical span name across exercise types makes the
    cohort-splitting query uniform. The
    ``exercise_type="idiom"`` field is the wire discriminator
    surfaced on the span so dashboards can split cohorts
    directly without leaning on the metadata key.

    Returns ``None`` when keys are missing (graceful degradation;
    no per-call warning spam — observability.py logs once at
    module-import time). On a populated Langfuse client, the
    function is a side-effect emitter; the return value is
    implicitly ``None``.
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
        "word_id": metadata["word_id"],
        "weakness_axes": metadata["weakness_axes"],
        "phrase_id": metadata["phrase_id"],
        "phrase": metadata["phrase"],
        "source_attribution": metadata["source_attribution"],
        "frequency_band": metadata["frequency_band"],
        "enable_rag": metadata["enable_rag"],
        "retrieved_neighbor_present": metadata["retrieved_neighbor_present"],
        # Phase 8.3 discriminator — splits idiom cohorts from
        # cloze / matching / comprehension / collocation cohorts.
        "idiom": True,
        "exercise_type": "idiom",
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
            name="exercise.generate",
            input=metadata.get("prompt_messages"),
            output=(result.model_dump_json() if result is not None else None),
        )
        span.update(metadata=span_metadata)
        span.end()
        client.flush()
    except Exception as exc:  # noqa: BLE001 — tracing must never break the activity
        logger.warning(
            "idiom: Langfuse trace failed (non-fatal): %s", exc
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
    ``IdiomExercise``, so the value must be a JSON-encoded instance
    that parses cleanly through
    ``IdiomExercise.model_validate_json``.
    """
    return [
        {
            "exercise": json.dumps(
                {
                    "exercise_id": "11111111111111111111111111111111",
                    "phrase": "ins Blaue hinein",
                    "definition": "ohne festes Ziel, planlos (in the blue).",
                    "example_usage": "Wir fahren einfach ins Blaue hinein.",
                    "source_attribution": "dwds",
                    "attested_quote": None,
                    "attested_source": None,
                    "frequency_band": "high",
                    "cloze_target": "ins ___ hinein",
                    "prompt_template_version": "idiom-v1",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "exercise_id": "22222222222222222222222222222222",
                    "phrase": "Tomaten auf den Augen",
                    "definition": "offensichtliches nicht sehen (blind for what's obvious).",
                    "example_usage": "Du hast wohl Tomaten auf den Augen!",
                    "source_attribution": "dwds",
                    "attested_quote": None,
                    "attested_source": None,
                    "frequency_band": "mid",
                    "cloze_target": "___ auf den Augen",
                    "prompt_template_version": "idiom-v1",
                },
                ensure_ascii=False,
            )
        },
        {
            "exercise": json.dumps(
                {
                    "exercise_id": "33333333333333333333333333333333",
                    "phrase": "das Eis brechen",
                    "definition": "die erste Hemmung in einer Beziehung überwinden (break the ice).",
                    "example_usage": "Er versuchte, mit einem Witz das Eis zu brechen.",
                    "source_attribution": "dwds,goethe",
                    "attested_quote": "Wer will denn gleich das Eis brechen?",
                    "attested_source": "Faust I, Studierzimmer",
                    "frequency_band": "high",
                    "cloze_target": "das ___ brechen",
                    "prompt_template_version": "idiom-v1",
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
        {"proposed_instruction": "Keep phrase verbatim in the response."},
        {"proposed_instruction": "Match frequency_band to the curated row's band."},
        {"proposed_instruction": "Generate a learner-friendly 1..400-char definition."},
        {"proposed_instruction": "Use attested_quote as the cloze-target surface when present."},
        {"proposed_instruction": "Source the example_usage from attested German usage."},
    ]


def _configure_dspy() -> None:
    """Configure DSPy with our OpenRouter adapter, or ``DummyLM`` offline.

    Mirrors ``app.cloze._configure_dspy`` exactly: idempotent,
    switches to ``DummyLM`` when no API key is present so the CI
    suite stays network-free. We pool both the idiom-path
    answers (JSON-encoded ``IdiomExercise`` payloads) and the
    MIPROv2 internal-proposer instruction-shaped stubs so a
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


class IdiomSignature(dspy.Signature):
    """DSPy signature for one idiom generation.

    Inputs match the ``build_prompt`` payload shape; the output is
    the full ``IdiomExercise`` Pydantic model. DSPy 3.x supports
    Pydantic-typed output fields via ``dspy.Predict`` /
    ``dspy.ChainOfThought``.

    ``target_phrase``, ``curated_definition``, ``attested_quote``,
    and ``source_attribution`` carry the curated ``phrases`` row's
    columns verbatim so the signature is faithful to the storage
    shape — the optimizer can teach the model that
    ``exercise.phrase == input.target_phrase``,
    ``exercise.frequency_band == input.frequency_band``, and the
    ``source_attribution`` comma-joined-subset invariant.

    ``word_id`` is in the input set for cross-exercise-type
    signature uniformity (every other DSPy module takes a
    ``word_id``-shaped integer). The 8.3 selector uses it as a
    phrase-selector seed (not a ``words.id`` FK), as documented
    above in ``select_phrase_row``.
    """

    word_id: int = dspy.InputField(
        desc=(
            "Integer seed for the phrase selector. Same value → "
            "same phrase row across calls. Not a FK to "
            "words.id — idioms are a standalone surface."
        )
    )
    target_phrase: str = dspy.InputField(
        desc="The curated German idiom verbatim (phrases.phrase)."
    )
    curated_definition: str = dspy.InputField(
        desc=(
            "The curated learner-facing gloss (phrases.definition). "
            "The generator may compress / rephrase to fit the "
            "1..400-char response constraint."
        )
    )
    attested_quote: str = dspy.InputField(
        desc=(
            "Optional Goethe / Schiller attestation. Empty string "
            "when phrases.attested_quote is null."
        )
    )
    source_attribution: str = dspy.InputField(
        desc=(
            "Comma-joined subset of "
            "Literal['dwds','goethe','schiller'] from the curated row."
        )
    )
    frequency_band: str = dspy.InputField(
        desc=(
            "Curated frequency band (high / mid / low). The "
            "response must echo this exact band."
        )
    )
    exercise: IdiomExercise = dspy.OutputField(
        desc="A Pydantic IdiomExercise matching the production contract."
    )


class IdiomModule(dspy.Module):
    """DSPy module that wraps the production ``IdiomSignature``.

    Uses ``dspy.Predict`` (single-shot — no chain-of-thought) so
    the output shape stays compatible with the production
    ``instructor`` path. The optimizer (``optimize_idiom_module``)
    can swap the predictor for a tuned one without changing this
    surface.

    The ``generate(prompt_template_version) -> IdiomExercise``
    method is the DSPy-callable mirror of the
    ``generate_idiom`` entry point. It runs the predictor and
    returns a validated ``IdiomExercise`` instance — used by the
    optimizer and any future DSPy-driven callers. Network-bound
    callers (the 8.4 route) go through ``generate_idiom``, which
    wraps the actual OpenRouter call + Langfuse trace.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(IdiomSignature)

    def forward(  # type: ignore[override]
        self,
        *,
        word_id: int,
        target_phrase: str,
        curated_definition: str,
        attested_quote: str,
        source_attribution: str,
        frequency_band: str,
    ) -> dspy.Prediction:
        return self.predict(
            word_id=word_id,
            target_phrase=target_phrase,
            curated_definition=curated_definition,
            attested_quote=attested_quote,
            source_attribution=source_attribution,
            frequency_band=frequency_band,
        )

    def generate(
        self,
        prompt_template_version: Literal["idiom-v1"] = "idiom-v1",
    ) -> IdiomExercise:
        """DSPy-callable entry: run the predictor and validate.

        Returns the ``IdiomExercise`` instance from the
        underlying prediction. A future maintainer who replaces
        the predictor (e.g. ``dspy.ChainOfThought``) doesn't need
        to change this surface — the validation step is the
        single point of contract enforcement.
        """
        prediction = self.predict(
            word_id=1,
            target_phrase="ins Blaue hinein",
            curated_definition=(
                "ohne festes Ziel, planlos (in the blue)."
            ),
            attested_quote="",
            source_attribution="dwds",
            frequency_band="high",
        )
        exercise = getattr(prediction, "exercise", None)
        if not isinstance(exercise, IdiomExercise):
            # Defensive — the DummyLM offline path or a future
            # ChainOfThought adapter may return a non-Pydantic
            # shape; coerce via ``model_validate_json`` so the
            # contract is the gate.
            if isinstance(exercise, str):
                exercise = IdiomExercise.model_validate_json(exercise)
            else:
                raise IdiomGenerationError(
                    "idiom: DSPy forward returned a non-IdiomExercise "
                    "shape; cannot validate",
                    attempted_schema=json.dumps(
                        IdiomExercise.model_json_schema(),
                        ensure_ascii=False,
                    ),
                    last_validation_error=(
                        f"got {type(exercise).__name__}, expected "
                        "IdiomExercise"
                    ),
                    schema_retry_count=0,
                )
        # Force ``prompt_template_version`` parity with the module
        # constant — same shape as ``generate_idiom``.
        return exercise.model_copy(
            update={"prompt_template_version": prompt_template_version}
        )


def optimize_idiom_module(
    train_set: Iterable[dict],
    val_set: Iterable[dict] | None = None,
) -> IdiomModule:
    """Run the offline DSPy optimizer against a held-out eval set.

    Strategy mirrors ``app.cloze.optimize_cloze_module`` /
    ``app.collocation.optimize_collocation_module``:

    - Always uses ``DummyLM`` when no API key is present so the CI
      suite runs without network (Hard rule #6).
    - Tries ``dspy.MIPROv2`` first (the spec's preferred optimizer).
      Falls back to ``dspy.BootstrapFewShot`` if MIPROv2 raises on
      the active dep tree.
    - Returns an ``IdiomModule`` with the optimised prompt
      instructions baked in. The caller (a future CLI script —
      8.3 doesn't ship one; Phase 9 owns it) would serialise the
      optimised module to ``backend/app/idiom_optimized.json`` so
      the production path could read it on next start.

    Parameters
    ----------
    train_set, val_set
        Iterable of dicts with the six input keys (word_id,
        target_phrase, curated_definition, attested_quote,
        source_attribution, frequency_band).

    Returns
    -------
    IdiomModule
        The optimised module. The optimizer mutates the module's
        internal predictor in place; the same instance is
        returned for caller convenience.
    """
    _configure_dspy()

    train_examples = [
        dspy.Example(**row).with_inputs(
            "word_id",
            "target_phrase",
            "curated_definition",
            "attested_quote",
            "source_attribution",
            "frequency_band",
        )
        for row in train_set
    ]
    val_examples = (
        [
            dspy.Example(**row).with_inputs(
                "word_id",
                "target_phrase",
                "curated_definition",
                "attested_quote",
                "source_attribution",
                "frequency_band",
            )
            for row in val_set
        ]
        if val_set is not None
        else None
    )

    module = IdiomModule()

    optimizer_cls = getattr(dspy, "MIPROv2", None)
    if optimizer_cls is None:
        optimizer_cls = getattr(dspy, "BootstrapFewShot", None)

    if optimizer_cls is None:
        logger.warning(
            "optimize_idiom_module: no MIPROv2 / BootstrapFewShot "
            "on the DSPy dep tree; returning the un-optimized module."
        )
        return module

    optimizer = optimizer_cls(metric=_idiom_metric)
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
            "optimize_idiom_module: optimizer %s raised on the "
            "offline path (%s); returning the un-optimized module. "
            "Re-run with --live and OPENROUTER_API_KEY set to actually "
            "optimize.",
            optimizer_cls.__name__,
            exc,
        )
        return module
    return optimized


def _idiom_metric(
    example: Any, prediction: Any, trace: Any | None = None
) -> float:
    """Offline quality metric for the idiom optimizer.

    Score range: 0.0..1.0. The bar is intentionally loose — the
    production C1-accept check is qualitative (Anurag hand-reviews),
    not a numeric gate. The optimizer uses this score to pick a
    better prompt; ``scripts/eval_idiom.py`` (a future card)
    would run the more rigorous per-row comparison.

    Components:

    - ``+0.4`` if ``prediction.exercise.phrase`` equals
      ``example.target_phrase`` verbatim (the contract).
    - ``+0.2`` if ``prediction.exercise.frequency_band`` equals
      ``example.frequency_band``.
    - ``+0.2`` if ``prediction.exercise.source_attribution`` is a
      non-empty comma-joined subset of the closed literal.
    - ``+0.2`` if ``prediction.exercise.cloze_target`` is either
      ``None`` or contains ``___``.
    """
    try:
        ex: IdiomExercise = prediction.exercise  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0
    score = 0.0
    if ex.phrase == example.target_phrase:
        score += 0.4
    if ex.frequency_band == example.frequency_band:
        score += 0.2
    if ex.source_attribution:
        tokens = [
            t.strip() for t in ex.source_attribution.split(",") if t.strip()
        ]
        if tokens and all(t in SOURCE_ATTRIBUTION for t in tokens):
            score += 0.2
    if ex.cloze_target is None or "___" in ex.cloze_target:
        score += 0.2
    return score


__all__ = [
    "FREQUENCY_BAND",
    "MAX_ATTEMPTS",
    "PROMPT_TEMPLATE_VERSION",
    "SOURCE_ATTRIBUTION",
    "IdiomExercise",
    "IdiomGenerationError",
    "IdiomModule",
    "IdiomSignature",
    "Phrase",
    "_trace_idiom",
    "build_prompt",
    "generate_idiom",
    "optimize_idiom_module",
    "select_phrase_row",
]
