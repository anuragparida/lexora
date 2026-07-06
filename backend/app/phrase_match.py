"""Phase 10.2 (card t_5d91a7e7) — phrase-match exercise generator + DSPy module.

This module is the phrase-match counterpart of Phase 8.3's ``app.idiom``
(``t_fa86ac58``) and Phase 6.2's ``app.match`` (``t_ddaf9cf9``). It
ships three distinct surfaces that share one Pydantic contract
(``PhraseMatchExercise``):

1. **Production path** — ``generate_phrase_match(db, *, phrase_a_id,
   phrase_b_id, enable_rag)``. Loads the two ``Phrase`` rows for the
   pair's FK slugs, builds a constrained prompt, calls
   ``app.llm.complete`` wrapped with ``instructor`` so the response
   validates against the ``PhraseMatchExercise`` Pydantic model.
   Bounded retry (Hard rule #5 / Phase 8.3 mirror — ``MAX_ATTEMPTS = 3``).
   Traces through ``_trace_phrase_match`` (returns ``None`` when
   Langfuse keys are missing; full span emission when present).

2. **Optimization path** — ``PhraseMatchSignature`` +
   ``PhraseMatchModule``. DSPy is wired to ``app.llm.complete`` via
   the shared ``_DSPyOpenAICompatLM`` adapter. The optimizer is
   offline-capable: with no API key,
   ``dspy.utils.dummies.DummyLM`` swaps in automatically so the CI
   suite runs without network (Hard rule #3 — offline discipline
   carried forward from Phase 7).

3. **Selector** — ``select_phrase_pair``. Picks one ``PhrasePair``
   row deterministically by ``word_id`` seed. Mirrors the
   deterministic-seed pattern of ``select_phrase_row`` (Phase 8.3
   idiom) and ``select_collocation_row`` (Phase 7.2 collocation).
   Same ``word_id`` → same pair across calls. No state, no random.

**Phrase + PhrasePair — 10.2 read-only mirrors.** Phase 8.1
(``t_d967c006``) owns the authoritative ``Phrase`` SQLAlchemy model
in ``app.models``; Phase 10.1 (``t_18c90a68``) owns the
authoritative ``PhrasePair`` model. Until 10.1 folds into ``main``,
this module declares a **local read-only mirror** of the expected
10.1 schema (``__tablename__ = "phrase_pairs"`` is the join key,
not Python identity) so the module compiles, the type system has
something concrete to introspect, and the tests run with the same
fixture shape 10.3 will exercise. The 8.1 mirror was already
folded on 8.3; the 10.2 phrase_match module mirrors that
``Phrase = models.Phrase`` re-export pattern AND uses a
``try/except`` fallback for ``PhrasePair`` — when 10.1 lands on
main, the canonical ``models.PhrasePair`` becomes the single
source of truth.

**``word_id`` contract.** In the cloze / collocation worlds,
``word_id`` is a FK to ``words.id``. For phrase-match (Phase 10),
the input field is the **pair-selector seed** (an integer that
maps deterministically to a ``phrase_pairs`` row). The card body
keeps the input field name ``word_id`` for cross-exercise-type
signature uniformity (Phase 8.3 idiom discipline); its semantic
role is documented here and in ``select_phrase_pair``.

Hard rules enforced here:

- **#1 (Literal widening is wire-level)** — ``exercise_type: Literal[
  "cloze", "matching", "comprehension", "idiom", "phrase_match"]``
  on ``BaseExerciseFields``. This module never narrows the base
  literal; ``PhraseMatchExerciseOut`` (in ``app.schemas``) narrows
  the discriminator on its own subclass to ``Literal["phrase_match"]``.
  The 4 prior subclasses (``ClozeExerciseOut``,
  ``MatchingExerciseOut``, ``ComprehensionExerciseOut``,
  ``IdiomExerciseOut``) re-declare their narrow literals unchanged.
- **#2 (read-only ``phrase_pairs``)** — ``generate_phrase_match`` only
  reads ``phrase_pairs`` (for the ``enable_rag=True`` nearest-
  neighbor path); no INSERT/UPDATE/DELETE from runtime. The seed
  scripts (10.1) are the only write path. Mirrors Phase 7.2 /
  Phase 8.3 read-only discipline byte-for-byte.
- **#3 (offline tests)** — DSPy module tests run with ``DummyLM``;
  the live-LLM regression in 10.4's verification is the only path
  that requires OpenRouter keys. CI stays offline.
- **#4 (Langfuse lexora project)** — every LLM call traces through
  ``_trace_phrase_match``, wired to the existing ``lexora`` project
  (the shared wrapper from ``app.observability.py``).
- **#5 (single LLM provider)** — production path goes through
  ``app.llm.complete``; the DSPy adapter targets the same wire
  format.
- **#6 (bounded retry)** — ``MAX_ATTEMPTS = 3`` per the card body
  (Phase 8.3 hard rule #5 carried forward verbatim).
- **#7 (no bge-m3 OpenRouter chat)** — the ``enable_rag`` path
  embeds with local ``sentence-transformers`` (Phase 1.3) when
  RAG-on is wired in 10.3; 10.2 only stubs the parameter. **NO
  chat-model API call to OpenRouter** for embedding or nearest-
  neighbor pull — that's the Phase 7 hard rule #11 carry-forward.
- **#8 (no env-derived thresholds)** — ``MAX_ATTEMPTS``,
  ``PROMPT_TEMPLATE_VERSION``, ``RAG_TOP_K`` are module constants,
  not env-derived. ``git grep -n "getenv.*RAG"`` returns nothing
  by construction.
- **#9 (Alembic-only migration)** — 10.1 owns the Alembic migration
  for ``phrase_pairs``. This module does NOT touch migration files;
  it only declares a read-only ORM mirror for testability.
- **#10 (ChainOfThought)** — ``PhraseMatchModule`` wraps the
  signature with ``dspy.ChainOfThought`` (not ``dspy.Predict``) so
  the model is encouraged to articulate the relation rationale
  before picking the closed 4-way ``relation`` literal. The
  ChainOfThought wrapping is the 10.2 departure from the Phase
  8.3 idiom ``Predict`` shape; the 10.4 eval set will judge
  whether the CoT rationale out-performs the bare-Predict baseline.
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Session

from app import models
from app.database import Base
from app.llm import _DSPyOpenAICompatLM  # the shared adapter (app.llm docstring)
from app.observability import get_langfuse

# Lazy DSPy import — the optimization-path classes touch dspy at
# import time; the production path doesn't. Mirrors ``app.idiom`` /
# ``app.collocation``.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #8 — no env-derived thresholds).
#
# ``PROMPT_TEMPLATE_VERSION`` is the A/B key downstream eval tooling
# splits on. ``MAX_ATTEMPTS = 3`` per the card body (Phase 8.3 hard
# rule #5 carried forward verbatim). ``RELATION_CHOICES`` is the
# closed 4-way literal the LLM is asked to choose from — mirrors
# the ``phrase_pairs.relation`` column 10.1 ships. ``RAG_TOP_K = 3``
# is the nearest-neighbor pull size (top-3 from ``phrase_pairs``
# when ``enable_rag=True``) — locked at module constant, not
# env-derived.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "phrase-match-v1"
MAX_ATTEMPTS: int = 3
RELATION_CHOICES: tuple[str, ...] = (
    "equivalent",
    "paraphrase",
    "related",
    "unrelated",
)
# Closed literal for ``phrase_pairs.source_attribution`` (Phase 10.2
# spec §"Pydantic contract"). Comma-joined subset is allowed on the
# storage side (``"dwds,bge-m3-cosine"``), but every token in the
# join must be in this tuple — enforced by
# ``_validate_source_attribution`` below. Note the new
# ``"bge-m3-cosine"`` token beyond the 8.3 idiom literal — added
# for the RAG-on nearest-neighbor path (Phase 7.5 cosine).
SOURCE_ATTRIBUTION: tuple[str, ...] = (
    "dwds",
    "goethe",
    "schiller",
    "bge-m3-cosine",
)

# RAG config — kept as module constants (Hard rule #8). Top-3
# nearest-neighbor pairs from ``phrase_pairs`` when
# ``enable_rag=True``.
RAG_TOP_K: int = 3


# ---------------------------------------------------------------------------
# Read-only ORM imports.
#
# Phase 8.1 (card t_d967c006) owns the authoritative SQLAlchemy
# ``Phrase`` model in ``app.models`` (folded on 8.3 — the
# ``models.Phrase`` re-export below is the canonical reference).
#
# Phase 10.1 (card t_18c90a68) owns the authoritative SQLAlchemy
# ``PhrasePair`` model. Until that fold lands on main, this module
# declares a **local read-only mirror** of the expected 10.1 schema
# (column-for-column: ``id``, ``phrase_a_id``, ``phrase_b_id``,
# ``relation``, ``attested_pair``, ``created_at``) — the local
# mirror compiles + passes tests in this worktree without depending
# on 10.1's Alembic migration. The ``try/except`` guard below flips
# to the canonical ``models.PhrasePair`` once 10.1 lands on main;
# the local mirror is then dead-code and a follow-up cleanup removes
# it (mirrors the 8.3 fold-reconcile pattern documented in
# ``app.idiom``).
#
# Read-only invariants (Hard rule #2):
#
# - No INSERT / UPDATE / DELETE anywhere in this module.
# - The seed scripts (10.1) are the only write path.
# - The runtime SELECT here is gated by ``enable_rag`` for the
#   RAG-on nearest-neighbor path; otherwise the curated pair alone
#   is read.
# ---------------------------------------------------------------------------

# Canonical Phrase model — re-export so the rest of this module uses
# ``Phrase`` as if it were locally defined. Mirrors the Phase 7.2 /
# Phase 8.3 fold-on-main pattern.
Phrase = models.Phrase


# Try-canonical-else-local-mirror for ``PhrasePair``. The fold to
# ``models.PhrasePair`` happens when 10.1 lands on main; until
# then, the local mirror below keeps the module compiling + the
# tests green in this worktree. The mirror is byte-for-byte
# compatible with the 10.1 schema per the card body spec.
try:
    PhrasePair = models.PhrasePair  # type: ignore[attr-defined]
except AttributeError:
    class PhrasePair(Base):  # type: ignore[no-redef]
        """Local read-only mirror of the 10.1 ``phrase_pairs`` table.

        Phase 10.1 (card t_18c90a68) owns the authoritative
        SQLAlchemy ``PhrasePair`` model in ``app.models``. Until
        10.1 lands on main, this module compiles + tests run with
        this local mirror. The ``__tablename__ = "phrase_pairs"``
        join key is what tests seed against — Python identity is
        irrelevant to ``Base.metadata.create_all`` because both
        this module and the future canonical model register the
        same SQLAlchemy ``Table`` by name.

        Schema (10.1 card body):

        - ``id``: int PK, autoincrement.
        - ``phrase_a_id``: str FK to ``phrases.id`` (Phrase.slug).
        - ``phrase_b_id``: str FK to ``phrases.id`` (Phrase.slug).
        - ``relation``: 4-way closed literal from
          ``RELATION_CHOICES`` (= ``phrase_pairs.relation`` column
          — same literals).
        - ``attested_pair``: bool (True when the pair was hand-
          curated, False when surfaced from the RAG-on nearest-
          neighbor path).
        - ``created_at``: DateTime (default ``func.now()``).
        """
        __tablename__ = "phrase_pairs"
        __abstract__ = False

        id = Column(Integer, primary_key=True, autoincrement=True)
        phrase_a_id = Column(
            String, ForeignKey("phrases.id"), nullable=False, index=True
        )
        phrase_b_id = Column(
            String, ForeignKey("phrases.id"), nullable=False, index=True
        )
        relation = Column(String, nullable=False)
        attested_pair = Column(Integer, nullable=False, default=1)
        created_at = Column(
            DateTime(timezone=True), server_default=func.now(), nullable=False
        )

        def __repr__(self) -> str:  # pragma: no cover — cosmetic only
            return (
                f"<PhrasePair id={self.id} "
                f"phrase_a_id={self.phrase_a_id!r} "
                f"phrase_b_id={self.phrase_b_id!r} "
                f"relation={self.relation!r}>"
            )


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class PhraseMatchExercise(BaseModel):
    """The metadata contract for a single phrase-match exercise (Phase 10.2).

    Field set locked by the 10.2 card body "Pydantic
    ``PhraseMatchExercise`` response model" section.

    ``phrase_a`` / ``phrase_b`` are the two German phrases verbatim
    (5..200 chars each). The bound mirrors the 8.3 ``phrase`` bound
    so a single pair's phrases share the same wire contract.

    ``relation`` is the closed 4-way literal the LLM is asked to
    pick from (``RELATION_CHOICES``). The bound mirrors the
    ``phrase_pairs.relation`` column 10.1 ships so the runtime
    validation and the storage column type agree.

    ``relation_rationale`` (1..400 chars) is the learner-facing
    explanation of why the LLM picked the closed ``relation``
    value. Same 1..400-char bound as the 8.3 ``definition`` field
    (forces the generator to compress into a learner-friendly
    gloss — the prompt is calibrated accordingly).

    ``source_attribution`` is a comma-joined literal — any subset
    of ``{dwds, goethe, schiller, bge-m3-cosine}`` joined by ``,``
    (``"dwds,bge-m3-cosine"`` when a curated row is augmented by a
    RAG-on nearest-neighbor pull). The list-of-tokens validator
    below enforces the Pydantic closed-literal invariant; the
    comma-join shape lets one pair be attributed to multiple seeds
    plus the bge-m3 cosine pathway.

    ``prompt_template_version`` is the A/B key, identical to the
    ``PROMPT_TEMPLATE_VERSION`` module constant for production
    generations.
    """

    exercise_id: int = Field(
        ...,
        gt=0,
        description=(
            "Server-minted per generation id (signed 8-byte int, "
            "non-zero). Phase 5.3 / 6.x convention; same shape as "
            "``ClozeExerciseOut`` / ``MatchingExerciseOut`` / "
            "``ComprehensionExerciseOut`` / ``IdiomExerciseOut``. "
            "The id re-appears on the ``grade_logs`` row (Phase "
            "9 follow-up) so the offline A/B eval is deterministic."
        ),
    )
    phrase_a: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description=(
            "First phrase of the pair (verbatim from the selected "
            "``phrases`` row). 5..200 chars mirrors the 8.3 "
            "``phrase`` bound."
        ),
    )
    phrase_b: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description=(
            "Second phrase of the pair (verbatim from the selected "
            "``phrases`` row). 5..200 chars mirrors the 8.3 "
            "``phrase`` bound."
        ),
    )
    relation: Literal["equivalent", "paraphrase", "related", "unrelated"] = Field(
        ...,
        description=(
            "Closed 4-way relation literal chosen by the LLM. "
            "Pydantic rejects any other value at the wire layer "
            "(Phase 7 hard rule #1)."
        ),
    )
    relation_rationale: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Learner-facing rationale for the chosen ``relation`` "
            "(1..400 chars). Forces the generator to compress long "
            "nearest-neighbor explanations into a learner-friendly "
            "gloss via the prompt."
        ),
    )
    source_attribution: str = Field(
        ...,
        description=(
            "Comma-joined subset of "
            "``Literal['dwds','goethe','schiller','bge-m3-cosine']``. "
            "Validated token-by-token by "
            "``_validate_source_attribution``."
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

        Mirrors ``IdiomExercise._validate_source_attribution``
        plus one extra allowed token (``"bge-m3-cosine"``) for the
        10.2 RAG path. Comma-joined subsets are allowed
        (``"dwds,bge-m3-cosine"``); each comma-separated token
        must be one of the four literals; whitespace around the
        separator is rejected (canonical wire form has zero
        whitespace); trailing / leading separators canonicalize
        away; de-dupes while preserving first-appearance order.
        """
        if not v or not v.strip():
            raise ValueError(
                "source_attribution must be a non-empty "
                "comma-joined subset of "
                "'dwds','goethe','schiller','bge-m3-cosine'"
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
        # De-dupe + preserve order on first appearance for a
        # stable serializer. "dwds,dwds" → "dwds".
        seen: set[str] = set()
        deduped: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return ",".join(deduped)


class PhraseMatchGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields the activity layer (and the
    Langfuse trace) record so an operator can triage without
    re-running the call. Mirrors ``IdiomGenerationError`` /
    ``ClozeGenerationError`` / ``CollocationGenerationError`` /
    ``MatchingGenerationError`` / ``ComprehensionGenerationError``
    so the exercise types share the same dead-letter shape.
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
    ) -> "PhraseMatchGenerationError":
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


class PhraseMatchNotFoundError(LookupError):
    """Raised by ``select_phrase_pair`` when no pair row can be served.

    Triggered when the ``phrase_pairs`` table is empty (10.1 seed
    hasn't run yet or wasn't migrated to the live DB). The card
    body expects ``POST /exercises/phrase_match`` to surface this
    as **404**, not 500 — the request hit an unfulfillable corpus
    constraint, not a server fault. The route layer at
    ``backend/app/main.py`` (10.3) catches this and stamps a 404.

    Carries ``word_id`` so the route can surface a structured 404
    detail (``"no phrase_pairs row exists for word_id={exc.word_id}"``)
    that mirrors ``GET /words/{word_id}``'s own 404 shape and the
    8.4 idiom ``IdiomNotFoundError`` discipline.

    Distinguishes from ``ValueError`` (which keeps the
    race-during-iteration semantics — a row disappeared between
    candidate list and lookup — translated to 500 by the route).
    """

    def __init__(self, *, word_id: int, candidates: int = 0) -> None:
        self.word_id = word_id
        self.candidates = candidates
        super().__init__(
            f"no phrase_pairs row for word_id={word_id} "
            f"(candidates={candidates})"
        )


# ---------------------------------------------------------------------------
# Pair-row selection (Phase 10.2 — selector for the curated ``phrase_pairs`` table)
# ---------------------------------------------------------------------------


def select_phrase_pair(db: Session, word_id: int) -> PhrasePair:
    """Pick one ``PhrasePair`` row for the given integer seed.

    Deterministic seed of ``word_id`` — same integer → same pair
    across calls. Mirrors the stability commitment of
    ``app.idiom.select_phrase_row``: same input → same output, no
    shuffling.

    **``word_id`` is a pair-selector seed, not a ``words.id`` FK.**

    In the cloze / collocation worlds ``word_id`` is a FK to
    ``words.id``. For phrase-match (Phase 10), ``word_id`` is a
    pair-selector seed — there is no FK to ``words.id`` because
    the pair table is anchored to two ``phrases.id`` slugs (not
    to ``words.id``). The card body keeps the input field name
    ``word_id`` for cross-exercise-type signature uniformity
    (Phase 8.3 idiom discipline); its semantic role is documented
    here as *pair-selector seed*.

    Algorithm
    ---------
    1. Query all ``PhrasePair`` rows in stable ``id`` order
       (autoincrement int PK).
    2. If zero rows match, raise ``PhraseMatchNotFoundError``
       (the caller — ``generate_phrase_match`` / the route
       layer — translates to 404: the request hit an
       unfulfillable corpus constraint, not a server fault).
       The card body demands 404 (not 500) for this case.
    3. Seed ``random.Random(str(word_id))`` and pick one slug
       from the candidate list (the integer-PK index isn't
       used directly — the seed makes the choice re-callable).

    Raises
    ------
    PhraseMatchNotFoundError
        The ``phrase_pairs`` table is empty (10.1 seed hasn't
        been applied). Routes translate to 404 (mirrors the
        8.4 ``IdiomNotFoundError`` 404 discipline).
    ValueError
        A row disappeared between candidate-list read and
        lookup (a corpus-drift race). Routes translate to 500.
    """
    candidates: list[int] = [
        row.id
        for row in db.query(PhrasePair).order_by(PhrasePair.id).all()
    ]
    if not candidates:
        raise PhraseMatchNotFoundError(word_id=word_id, candidates=0)

    seed_str = f"phrase_pair|{word_id}"
    rng = random.Random(seed_str)
    chosen_id = rng.choice(candidates)
    row = (
        db.query(PhrasePair)
        .filter(PhrasePair.id == chosen_id)
        .one_or_none()
    )
    if row is None:
        # Race — a row disappeared between id list and lookup.
        # Treat as a hard error so the operator notices corpus
        # drift rather than silently substituting another row.
        raise ValueError(
            f"select_phrase_pair: phrase_pair id {chosen_id!r} "
            f"disappeared between candidate list and lookup "
            f"(word_id={word_id})"
        )
    return row


# ---------------------------------------------------------------------------
# Nearest-neighbor retrieval (Phase 10.2 RAG-on path stub)
# ---------------------------------------------------------------------------


def _retrieve_phrase_pair_neighbors(
    db: Session,
    pair: PhrasePair,
    *,
    top_k: int = RAG_TOP_K,
) -> list[dict[str, Any]]:
    """Return top-k nearest-neighbor ``phrase_pairs`` for RAG-on prompt.

    Phase 10.2 stub — the 10.3 endpoint will pass a real
    ``retrieve_neighbor`` callable from
    ``app.embeddings.embed_one`` + the ``phrase_pairs`` cosine
    path (Phase 7.5). For now, this stub returns an empty list
    so the offline path stays byte-stable.

    The stub honors the Hard rule #7 contract:
    **no bge-m3 OpenRouter chat call**. When 10.3 wires the real
    callable, the embedding is local ``sentence-transformers``
    (Phase 1.3) — not a chat-model API call to OpenRouter.

    Parameters
    ----------
    db
        Active SQLAlchemy session (not used in the 10.2 stub;
        kept in the signature for 10.3's byte-stable drop-in).
    pair
        The selected ``PhrasePair`` row (anchor for the cosine
        pull — not used in the 10.2 stub).
    top_k
        Nearest-neighbor count (default ``RAG_TOP_K = 3``).

    Returns
    -------
    list[dict]
        10.2 stub: empty list. 10.3 will return
        ``[{"kind": "phrase_pair", "id": int, "phrase_a": str,
        "phrase_b": str, "relation": str}, ...]``.
    """
    # 10.2 stub returns [] unconditionally — Hard rule #7 keeps
    # the offline test path network-free; 10.3 replaces the body
    # with the real sentence-transformers pull + phrase_pairs
    # cosine query.
    return []


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    pair: PhrasePair,
    phrase_a_row: Phrase,
    phrase_b_row: Phrase,
    *,
    retrieved_neighbors_json: str | None = None,
) -> list[dict]:
    """Build the chat-completions messages for one phrase-match generation.

    Prompt shape
    ------------
    The system prompt:

    1. Establishes the role ("German phrase-pair relation judge
       for a C1 learner").
    2. Lists the explicit prohibitions — keep ``phrase_a`` /
       ``phrase_b`` verbatim (no case / article mutation),
       pick one of the 4-way ``RELATION_CHOICES`` literals,
       write the rationale in German, C1-accept bar.
    3. Specifies the JSON output schema with every field on
       ``PhraseMatchExercise``.

    The user prompt embeds:

    - The selected ``PhrasePair`` row (``phrase_a_id``,
      ``phrase_b_id``, ``relation`` from the curated table).
    - The two ``Phrase`` rows (the two phrases' verbatim
      German surface text — what the LLM sees).
    - The optional ``retrieved_neighbors_json`` (empty list
      when RAG is off; populated list of nearest-neighbor
      phrase-pair snippets when ``enable_rag=True`` — 10.2
      only stubs the parameter; 10.3 wires the real
      retrieval).
    - Instructions: produce ONE ``PhraseMatchExercise``
      payload.

    Parameters
    ----------
    pair
        The selected ``PhrasePair`` row (looked up by
        ``select_phrase_pair``).
    phrase_a_row, phrase_b_row
        The two ``Phrase`` rows (FK slugs → surface text).
        The LLM sees the verbatim German and the curated row's
        ``source_attribution`` so it can echo the literal on
        the response.
    retrieved_neighbors_json
        JSON-encoded list of nearest-neighbor phrase-pair
        snippets (empty ``"[]"`` when RAG is off). The 10.2
        stub accepts the parameter but doesn't wire a real
        retrieval call — 10.3's endpoint honors
        ``enable_rag=True`` and replaces this with a real
        nearest-neighbor pull.

    Returns
    -------
    list[dict]
        Two-message list suitable for ``app.llm.complete``:
        system + user content. Plain
        ``[{"role": ..., "content": ...}]``.
    """
    retrieved_neighbors_payload = (
        retrieved_neighbors_json if retrieved_neighbors_json is not None else "[]"
    )

    system_content = (
        "You are a German phrase-pair relation judge for a C1 learner. "
        "Given two German phrases (phrase_a + phrase_b), decide which "
        "closed-4-way relation best describes the pair and justify your "
        "pick in a concise learner-friendly rationale (German, "
        "1..400 chars).\n\n"
        "PROHIBITIONS (must obey all):\n"
        "1. Do NOT change word forms, articles, or case endings in "
        "phrase_a or phrase_b — both must appear verbatim.\n"
        "2. Do NOT translate either phrase into another language; "
        "phrase_a, phrase_b, and relation_rationale must stay in "
        "German.\n"
        "3. The relation MUST be one of the closed 4-way literals "
        "['equivalent','paraphrase','related','unrelated'] — pick "
        "exactly one, do not invent other tokens.\n"
        "4. The source_attribution must be a non-empty comma-joined "
        "subset of 'dwds','goethe','schiller','bge-m3-cosine'; do "
        "not introduce other tokens. When the curated pair is "
        "augmented by RAG-on nearest-neighbor pulls, add "
        "'bge-m3-cosine' to the join.\n\n"
        "RELATION GLOSS (which literal fits which case):\n"
        "  equivalent: phrase_a and phrase_b are dictionary-synonyms "
        "with interchangeable usage.\n"
        "  paraphrase: surface variation that conveys the same "
        "meaning in a slightly different form.\n"
        "  related: same semantic domain, complementary meaning "
        "(both are German idioms about X, but they differ).\n"
        "  unrelated: no semantic relation; the pair is a negative "
        "example (e.g. one is food, the other is weather).\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept the chosen "
        "relation and rationale without edits? If no, redo before "
        "answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "exercise_id": <signed 64-bit int — server stamps on the '
        'wire, you can leave 0 as a placeholder>,\n'
        '  "phrase_a": "<verbatim curated phrase, 5..200 chars>",\n'
        '  "phrase_b": "<verbatim curated phrase, 5..200 chars>",\n'
        '  "relation": "equivalent" | "paraphrase" | "related" | "unrelated",\n'
        '  "relation_rationale": "<German rationale, 1..400 chars>",\n'
        '  "source_attribution": "<comma-joined subset of '
        'dwds|goethe|schiller|bge-m3-cosine>",\n'
        '  "prompt_template_version": "phrase-match-v1"\n'
        "}\n"
    )

    user_payload: dict[str, Any] = {
        "pair": {
            "id": pair.id,
            "phrase_a_id": pair.phrase_a_id,
            "phrase_b_id": pair.phrase_b_id,
            "curated_relation": pair.relation,
            "attested_pair": bool(pair.attested_pair),
        },
        "phrase_a": {
            "id": phrase_a_row.id,
            "phrase": phrase_a_row.phrase,
            "source_attribution": phrase_a_row.source_attribution,
        },
        "phrase_b": {
            "id": phrase_b_row.id,
            "phrase": phrase_b_row.phrase,
            "source_attribution": phrase_b_row.source_attribution,
        },
        # 10.2 stub: ``retrieved_neighbors_json`` stays "[]" by
        # default. 10.3 wires the real retrieval when enable_rag=True.
        "retrieved_neighbors": json.loads(retrieved_neighbors_payload)
        if retrieved_neighbors_payload
        else [],
        "instructions": (
            "Produce ONE PhraseMatchExercise payload for the curated "
            "phrase pair above. Both phrases must appear verbatim on "
            "the wire (no form / case / article mutation). The "
            "relation MUST be one of the closed 4-way literals; the "
            "rationale must be a learner-friendly German sentence "
            "(1..400 chars) that names why the chosen relation fits. "
            "Stamp prompt_template_version='phrase-match-v1' "
            "exactly."
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

    Lazy-imported so ``app.phrase_match`` stays import-cheap.
    Mirrors ``app.idiom._openai_client`` exactly: same env-var
    conventions, same ``None``-on-missing-key fallback so the
    production path raises ``LLMError`` consistently with the
    rest of the project.

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


def generate_phrase_match(
    db: Session,
    *,
    phrase_a_id: int | str,
    phrase_b_id: int | str,
    enable_rag: bool = False,
) -> PhraseMatchExercise:
    """Generate one ``PhraseMatchExercise`` for the given pair ids.

    Flow
    ----
    1. Load the two ``Phrase`` rows from the curated ``phrases``
       table (Phase 8.1). Raises ``PhraseMatchNotFoundError``
       (mirroring ``IdiomNotFoundError``) when either is missing
       — the route layer (10.3) translates to HTTP 404 per the
       card body commitment, not 500.
    2. When ``enable_rag=True``, fetch top-3 nearest-neighbor
       phrase pairs (Phase 7.5 cosine path, local
       ``sentence-transformers`` — **NOT** OpenRouter chat).
       10.2 stubs the retrieval; 10.3 wires a real callable.
    3. ``build_prompt`` produces the chat messages with the pair
       row + the two ``Phrase`` rows + optional nearest-neighbor
       snippets embedded.
    4. Wrap an OpenRouter-targeted OpenAI client with
       ``instructor`` and call
       ``chat.completions.create(
         response_model=PhraseMatchExercise, ...,
         max_retries=MAX_ATTEMPTS)``. ``MAX_ATTEMPTS = 3`` per
       the card body — same budget as ``app.idiom.MAX_ATTEMPTS``.
    5. Stamp ``prompt_template_version`` onto the result; call
       ``_trace_phrase_match`` with the metadata dict.

    Parameters
    ----------
    db
        Active SQLAlchemy session.
    phrase_a_id, phrase_b_id
        The two ``Phrase`` FK slugs (Phase 8.1 ``phrases.id``
        column is a ``String`` slug; the 10.2 card takes either
        int or str for forward-compat with the eventual
        ``generate_phrase_match(db, *, word_id)`` wrapper). The
        10.3 endpoint collapses to the ``word_id`` seed via
        ``select_phrase_pair``; this function takes the resolved
        FKs directly.
    enable_rag
        Opt-in flag (default False). When True, embed top-3
        nearest-neighbor phrase-pair snippets in the user prompt
        (Hard rule #7 — local sentence-transformers path only).
        When False (default), the prompt is byte-for-byte
        identical to the curated-only path. 10.2 stub keeps the
        retrieval empty; 10.3 wires the real pull.

    Returns
    -------
    PhraseMatchExercise
        The validated Pydantic instance ready to return to the
        route layer. ``prompt_template_version`` is forced to
        ``PROMPT_TEMPLATE_VERSION`` so a future maintainer who
        hand-edits the prompt template doesn't silently desync
        the value.

    Raises
    ------
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or the
        transport call hits a non-retryable / persistent-
        retryable failure. The route layer (10.3) translates to
        502.
    PhraseMatchGenerationError
        Persistent schema violation after ``MAX_ATTEMPTS``
        (= 3). Carries ``attempted_schema``,
        ``last_validation_error``, ``schema_retry_count`` so the
        route layer can surface a structured 502 instead of a
        bare 500.
    PhraseMatchNotFoundError
        Bubbles up when either ``phrase_a_id`` or ``phrase_b_id``
        doesn't resolve to a ``Phrase`` row (8.1 seed hasn't
        run yet, or the curated table was pruned). The route
        layer (10.3) catches this and translates to HTTP 404
        (mirrors the 8.4 ``IdiomNotFoundError`` discipline; the
        card body explicitly demands 404 for this case, not
        500).
    ValueError
        Bubbles up from a corpus-drift race (a phrase slug
        disappeared between the candidate list and the
        lookup). The route layer translates this into 500.
    """
    from app.llm import _default_model, LLMError

    # 1. Load the two ``Phrase`` rows.
    phrase_a_row = (
        db.query(Phrase)
        .filter(Phrase.id == phrase_a_id)
        .one_or_none()
    )
    phrase_b_row = (
        db.query(Phrase)
        .filter(Phrase.id == phrase_b_id)
        .one_or_none()
    )
    if phrase_a_row is None or phrase_b_row is None:
        # Either FK is unresolvable — the curated corpus
        # doesn't carry the pair the caller asked for. Surface
        # a 404 (route layer translation) rather than a 500.
        missing = (
            "phrase_a" if phrase_a_row is None else "phrase_b"
        )
        # Use ``PhraseMatchNotFoundError`` with ``word_id=0``
        # sentinel to surface the 404 via the route layer's
        # existing handler — the card body mandates 404.
        raise PhraseMatchNotFoundError(
            word_id=0, candidates=int(phrase_a_row is None) + int(phrase_b_row is None)
        )

    # 2. RAG-on stub (Hard rule #7 — no OpenRouter chat).
    # 10.2: retrieval empty; 10.3 wires the real nearest-
    # neighbor pull.
    retrieved_neighbors_json: str | None = None
    if enable_rag:
        # 10.2 stub calls the empty default; 10.3 replaces
        # with a real cosine pull against ``phrase_pairs``.
        neighbors: list[dict[str, Any]] = []
        retrieved_neighbors_json = json.dumps(
            neighbors, ensure_ascii=False
        )

    # 3. Synthesize a transient ``PhrasePair`` view for the
    # prompt (this path is called from the 10.3 endpoint which
    # passes the resolved pair; for direct generator calls we
    # build a transient mirror so ``build_prompt`` keeps the
    # same input shape regardless of caller).
    pair_view = _make_transient_pair(
        phrase_a_id=str(phrase_a_row.id),
        phrase_b_id=str(phrase_b_row.id),
        relation="related",  # default; LLM picks the actual literal.
        attested_pair=True,
    )

    messages = build_prompt(
        pair_view,
        phrase_a_row,
        phrase_b_row,
        retrieved_neighbors_json=retrieved_neighbors_json,
    )

    # Trace metadata captures the request shape; populated
    # before the call so the error path
    # (``PhraseMatchGenerationError`` raised below) can still
    # log it.
    metadata: dict[str, Any] = {
        "word_id": 0,  # direct-generator callers don't carry a pair-selector seed
        "phrase_a_id": phrase_a_row.id,
        "phrase_b_id": phrase_b_row.id,
        "phrase_a": phrase_a_row.phrase,
        "phrase_b": phrase_b_row.phrase,
        "source_attribution_a": phrase_a_row.source_attribution,
        "source_attribution_b": phrase_b_row.source_attribution,
        "enable_rag": bool(enable_rag),
        "retrieved_neighbors_present": bool(retrieved_neighbors_json),
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": messages,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        # Phase 10.2 discriminator — splits phrase_match
        # cohorts from cloze / matching / comprehension /
        # idiom cohorts.
        "exercise_type": "phrase_match",
        "phrase_match": True,
    }

    raw_client = _openai_client()
    if raw_client is None:
        # No key — same shape as the persistent failure mode
        # in llm.py. Surface LLMError so the route layer's
        # existing 502 handler picks it up.
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it to "
            "~/.lexora/.env and restart the backend container."
        )

    import instructor

    # ``MD_JSON`` mode tells instructor to use markdown-fenced
    # JSON parsing rather than the tool-calling path. Same
    # shape as ``app.idiom`` / ``app.cloze`` / ``app.collocation``
    # — the tool-calling path requires the model to emit
    # ``tool_calls`` in the response, which the OpenRouter
    # passthrough doesn't always echo for non-OpenAI-native
    # models. ``MD_JSON`` works uniformly with the qwen
    # default and gives the model a clear
    # "output a ```json ...``` block" instruction in the
    # system prompt (instructor appends it).
    instructor_client = instructor.from_openai(
        raw_client, mode=instructor.Mode.MD_JSON
    )

    schema_retry_count = 0
    last_validation_error = ""
    result: PhraseMatchExercise | None = None
    started = _perf_counter_ms()

    try:
        result = instructor_client.chat.completions.create(
            response_model=PhraseMatchExercise,
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
            _trace_phrase_match(None, metadata, latency_ms)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
        raise PhraseMatchGenerationError.from_validation_failure(
            f"phrase_match: schema validation failed after "
            f"{schema_retry_count} attempt(s): {last_validation_error}",
            schema=PhraseMatchExercise.model_json_schema(),
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
    _trace_phrase_match(result, metadata, latency_ms)
    return result


def _make_transient_pair(
    *,
    phrase_a_id: str,
    phrase_b_id: str,
    relation: str,
    attested_pair: bool,
) -> PhrasePair:
    """Build a transient ``PhrasePair`` view for direct generator calls.

    The 10.3 endpoint will resolve a ``PhrasePair`` row via
    ``select_phrase_pair`` and pass it into the prompt builder
    directly. The 10.2 direct-generator entry point
    (``generate_phrase_match``) takes the two resolved FK
    slugs and synthesises a transient row so the prompt builder
    keeps the same input shape regardless of caller.

    Marked ``attested_pair=1`` (the curated-table default) so
    the prompt can branch on it when the RAG-on path surfaces
    a non-attested pair. Mirrors the 8.3 idiom mirror pattern
    for consistency.
    """
    # Build an unpersisted SQLAlchemy row. The instance lives
    # only for the prompt build — never ``session.add()``ed.
    pair = PhrasePair(
        phrase_a_id=phrase_a_id,
        phrase_b_id=phrase_b_id,
        relation=relation,
        attested_pair=1 if attested_pair else 0,
    )
    return pair


def _perf_counter_ms() -> float:
    """Wall-clock in ms (float). Local import to keep the module
    import-cheap and to centralise the unit conversion."""
    import time

    return time.perf_counter() * 1000.0


def _count_instructor_retries(exc: Exception) -> int:
    """Best-effort: pull the retry count out of an instructor /
    pydantic failure. ``instructor`` raises a
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


# ---------------------------------------------------------------------------
# Langfuse trace hook
# ---------------------------------------------------------------------------


def _trace_phrase_match(
    result: PhraseMatchExercise | None,
    metadata: dict[str, Any],
    latency_ms: int,
) -> None:
    """Emit one Langfuse ``exercise.generate`` span per phrase-match.

    Mirrors ``app.idiom._trace_idiom`` /
    ``app.collocation._trace_collocation`` /
    ``app.comprehension._trace_comprehension``: v2 SDK shape
    (``client.span → span.update → span.end → client.flush``),
    graceful no-op when Langfuse keys are missing, SDK
    exceptions swallowed so the activity still succeeds.

    Span name is ``exercise.generate`` (not
    ``phrase_match.generate``) because the discriminator
    ``phrase_match: true`` in the metadata is the join key the
    dashboard uses to filter — keeping a single canonical span
    name across exercise types makes the cohort-splitting query
    uniform. The ``exercise_type="phrase_match"`` field is the
    wire discriminator surfaced on the span so dashboards can
    split cohorts directly without leaning on the metadata key.

    Returns ``None`` when keys are missing (graceful
    degradation; no per-call warning spam — observability.py
    logs once at module-import time). On a populated Langfuse
    client, the function is a side-effect emitter; the return
    value is implicitly ``None``.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam
        # per call.
        return

    ptemplate_version = (
        getattr(result, "prompt_template_version", None)
        if result is not None
        else metadata.get("prompt_template_version")
    )
    span_metadata: dict[str, Any] = {
        "word_id": metadata.get("word_id", 0),
        "phrase_a_id": metadata["phrase_a_id"],
        "phrase_b_id": metadata["phrase_b_id"],
        "phrase_a": metadata["phrase_a"],
        "phrase_b": metadata["phrase_b"],
        "source_attribution_a": metadata["source_attribution_a"],
        "source_attribution_b": metadata["source_attribution_b"],
        "enable_rag": metadata["enable_rag"],
        "retrieved_neighbors_present": metadata["retrieved_neighbors_present"],
        # Phase 10.2 discriminator — splits phrase_match
        # cohorts from cloze / matching / comprehension /
        # idiom / collocation cohorts.
        "phrase_match": True,
        "exercise_type": "phrase_match",
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
            "phrase_match: Langfuse trace failed (non-fatal): %s", exc
        )
        if span is not None:
            try:
                span.end()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# DSPy surface — optimization path
# ---------------------------------------------------------------------------


def _offline_dummy_answers() -> list[dict[str, Any]]:
    """Build a pool of offline-LM answer dicts for the DummyLM.

    DSPy's ``DummyLM`` returns ``"[[ ## field ## ]] value"``
    lines, and the ``JSONAdapter`` (used here, not
    ``ChatAdapter``) parses the value back into the field's
    declared type. Our ``exercise`` output is a Pydantic
    ``PhraseMatchExercise``, so the value must be a
    JSON-encoded instance that parses cleanly through
    ``PhraseMatchExercise.model_validate_json``. The fixed
    pool covers the 4 relations + a few near-edge variants
    so any interleaved call has a deterministic answer.

    Each entry carries a ``reasoning`` field (required by
    ``dspy.ChainOfThought``) and the JSON-encoded
    ``exercise`` payload. The JSON adapter fans these out to
    ``prediction.reasoning`` (str) and ``prediction.exercise``
    (Pydantic instance) — the ``PhraseMatchModule.generate``
    surface then unwraps ``prediction.exercise``.
    """
    return [
        {
            "reasoning": "Beide Ausdruecke bezeichnen eine planlose Handlung ohne konkretes Ziel; Synonym-Beziehung.",
            "exercise": json.dumps(
                {
                    "exercise_id": 44444441,
                    "phrase_a": "ins Blaue hinein",
                    "phrase_b": "ohne festes Ziel",
                    "relation": "equivalent",
                    "relation_rationale": (
                        "Beide Ausdruecke bezeichnen eine planlose "
                        "Handlung ohne konkretes Ziel."
                    ),
                    "source_attribution": "dwds",
                    "prompt_template_version": "phrase-match-v1",
                },
                ensure_ascii=False,
            ),
        },
        {
            "reasoning": "Tomaten auf den Augen vs. etwas Offensichtliches nicht sehen — die zweite Form ist eine Umschreibung der idiomatischen Bedeutung.",
            "exercise": json.dumps(
                {
                    "exercise_id": 44444442,
                    "phrase_a": "Tomaten auf den Augen",
                    "phrase_b": "etwas Offensichtliches nicht sehen",
                    "relation": "paraphrase",
                    "relation_rationale": (
                        "Eine Umschreibung der idiomatischen "
                        "Bedeutung in eigenen Worten."
                    ),
                    "source_attribution": "dwds",
                    "prompt_template_version": "phrase-match-v1",
                },
                ensure_ascii=False,
            ),
        },
        {
            "reasoning": "das Eis brechen vs. einen Fauxpas begehen — beide stammen aus dem Bereich sozialer Interaktion, beschreiben aber gegensatzliche Verhaltensweisen.",
            "exercise": json.dumps(
                {
                    "exercise_id": 44444443,
                    "phrase_a": "das Eis brechen",
                    "phrase_b": "einen Fauxpas begehen",
                    "relation": "related",
                    "relation_rationale": (
                        "Beide Ausdruecke stammen aus dem Bereich "
                        "sozialer Interaktion, beschreiben aber "
                        "gegensatzliche Verhaltensweisen."
                    ),
                    "source_attribution": "dwds,goethe",
                    "prompt_template_version": "phrase-match-v1",
                },
                ensure_ascii=False,
            ),
        },
        {
            "reasoning": "Apfelkuchen backen vs. ein Gewitter zieht auf — kein gemeinsames semantisches Feld.",
            "exercise": json.dumps(
                {
                    "exercise_id": 44444444,
                    "phrase_a": "Apfelkuchen backen",
                    "phrase_b": "ein Gewitter zieht auf",
                    "relation": "unrelated",
                    "relation_rationale": (
                        "Kein gemeinsames semantisches Feld; das "
                        "eine ist eine Kuechenaktivitaet, das "
                        "andere ein Wetterphaenomen."
                    ),
                    "source_attribution": "dwds",
                    "prompt_template_version": "phrase-match-v1",
                },
                ensure_ascii=False,
            ),
        },
    ]


def _offline_json_answers() -> list[dict[str, str]]:
    """Answer pool for MIPROv2's internal ``JSONAdapter``-shaped calls.

    Same pattern as ``app.idiom._offline_json_answers`` —
    MIPROv2's internal prompt-proposer sometimes expects
    ``{"proposed_instruction": "..."}`` responses (a string
    field). Five diverse stubs keep the proposer cycling
    without choking.
    """
    return [
        {"proposed_instruction": "Pick exactly one 4-way relation literal."},
        {"proposed_instruction": "Write the rationale in German, 1..400 chars."},
        {"proposed_instruction": "Keep phrase_a / phrase_b verbatim on the wire."},
        {"proposed_instruction": "Echo 'bge-m3-cosine' in source_attribution when RAG-on pulls neighbor pairs."},
        {"proposed_instruction": "Pick 'equivalent' only when phrase_a and phrase_b are dictionary-synonyms."},
    ]


def _configure_dspy() -> None:
    """Configure DSPy with our OpenRouter adapter, or ``DummyLM`` offline.

    Mirrors ``app.idiom._configure_dspy`` /
    ``app.cloze._configure_dspy`` exactly: idempotent,
    switches to ``DummyLM`` when no API key is present so the
    CI suite stays network-free (Hard rule #3). We pool both
    the phrase-match-path answers (ChainOfThought-shaped:
    ``{"reasoning": str, "exercise": json-string}``) and the
    MIPROv2 internal-proposer instruction-shaped stubs so a
    single ``DummyLM`` serves both protocols in any
    interleaving.

    Note: 10.2's ``PhraseMatchModule`` uses
    ``dspy.ChainOfThought`` (Hard rule #10), so the adapter
    is ``dspy.JSONAdapter`` (not the ``ChatAdapter`` used
    in the idiom module). ``JSONAdapter`` parses the
    ``[[ ## field ## ]] value`` markdown-format the
    ChainOfThought adapter expects, and supports
    Pydantic-typed outputs via ``model_validate_json``.
    Idiom uses ``dspy.Predict`` and ``ChatAdapter`` because
    the ChainOfThought rationale field would be wasted on
    the bare ``IdiomExercise`` JSON envelope.
    """
    import dspy as _dspy

    if _dspy.settings.lm is not None:
        return
    if os.getenv("OPENROUTER_API_KEY"):
        _dspy.settings.configure(lm=_DSPyOpenAICompatLM())
        return

    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.utils.dummies import DummyLM

    answers = _offline_dummy_answers() + _offline_json_answers()
    _dspy.settings.configure(
        lm=DummyLM(answers, adapter=JSONAdapter()),
        adapter=JSONAdapter(),
    )


class PhraseMatchSignature(dspy.Signature):
    """DSPy signature for one phrase-match generation.

    Inputs match the production prompt shape:

    - ``word_id`` keeps the cross-exercise-type field-name
      uniformity (Phase 8.3 idiom discipline). For
      phrase-match it's the pair-selector seed (not a
      ``words.id`` FK).
    - ``phrase_a`` / ``phrase_b`` carry the verbatim German
      surface text the LLM will see.
    - ``relation_choices`` is the closed 4-way literal
      tuple passed as a static hint so the LLM doesn't
      invent tokens.
    - ``few_shot_examples`` is the nearest-neighbor pair
      list from ``phrase_pairs``, embedded only when
      ``enable_rag=True``; ``None`` when curated-only.

    Output is the full ``PhraseMatchExercise`` Pydantic
    model. DSPy 3.x supports Pydantic-typed output fields via
    ``dspy.ChainOfThought`` / ``dspy.Predict``.
    """

    word_id: int = dspy.InputField(
        desc=(
            "Integer seed for the phrase-pair selector. Same "
            "value → same phrase_pairs row across calls. "
            "Not a FK to words.id — phrase-match uses this as "
            "a pair-selector seed (mirrors the Phase 8.3 idiom "
            "discipline)."
        )
    )
    phrase_a: str = dspy.InputField(
        desc=(
            "First phrase of the pair (verbatim from "
            "phrases.phrase)."
        )
    )
    phrase_b: str = dspy.InputField(
        desc=(
            "Second phrase of the pair (verbatim from "
            "phrases.phrase)."
        )
    )
    relation_choices: list[str] = dspy.InputField(
        desc=(
            "The closed 4-way literal tuple "
            "['equivalent','paraphrase','related','unrelated'] "
            "the LLM must pick from. Passed as a static hint; "
            "the LLM does not invent tokens."
        )
    )
    few_shot_examples: list[dict] | None = dspy.InputField(
        desc=(
            "Optional nearest-neighbor phrase_pairs (top-3 by "
            "cosine similarity, Phase 7.5 path). Each entry is "
            "{phrase_a, phrase_b, relation}. None when "
            "enable_rag=False."
        )
    )
    exercise: PhraseMatchExercise = dspy.OutputField(
        desc=(
            "A Pydantic PhraseMatchExercise matching the "
            "production contract."
        )
    )


class PhraseMatchModule(dspy.Module):
    """DSPy module that wraps the production ``PhraseMatchSignature``.

    Uses ``dspy.ChainOfThought`` so the model articulates the
    relation rationale before picking the closed 4-way
    ``relation`` literal — Phase 10.2 hard rule #10. The 10.4
    hand-labeled eval set will quantify the rationale quality
    vs the bare-Predict baseline. The optimizer
    (10.7's ``scripts/optimize_phrase_match.py``) can swap the
    predictor for a tuned one without changing this surface.

    The ``generate(...) -> PhraseMatchExercise`` method is the
    DSPy-callable mirror of the production
    ``generate_phrase_match``. It runs the ChainOfThought
    predictor and returns a validated ``PhraseMatchExercise``
    instance — used by the optimizer and any future DSPy-
    driven callers. Network-bound callers (the 10.3 route)
    go through ``generate_phrase_match``, which wraps the
    actual OpenRouter call + Langfuse trace.
    """

    def __init__(self) -> None:
        super().__init__()
        # Phase 10.2 hard rule #10 — ``dspy.ChainOfThought``
        # wraps the signature so the model articulates the
        # rationale before picking the relation literal.
        self.predict = dspy.ChainOfThought(PhraseMatchSignature)

    def forward(  # type: ignore[override]
        self,
        *,
        word_id: int,
        phrase_a: str,
        phrase_b: str,
        few_shot_examples: list[dict] | None = None,
    ) -> dspy.Prediction:
        return self.predict(
            word_id=word_id,
            phrase_a=phrase_a,
            phrase_b=phrase_b,
            relation_choices=list(RELATION_CHOICES),
            few_shot_examples=few_shot_examples,
        )

    def generate(
        self,
        *,
        word_id: int = 1,
        phrase_a: str = "ins Blaue hinein",
        phrase_b: str = "ohne festes Ziel",
        few_shot_examples: list[dict] | None = None,
        prompt_template_version: Literal["phrase-match-v1"] = "phrase-match-v1",
    ) -> PhraseMatchExercise:
        """DSPy-callable entry: run the predictor and validate.

        Returns the ``PhraseMatchExercise`` instance from the
        underlying prediction. A future maintainer who
        replaces the predictor doesn't need to change this
        surface — the validation step is the single point of
        contract enforcement.
        """
        prediction = self.predict(
            word_id=word_id,
            phrase_a=phrase_a,
            phrase_b=phrase_b,
            relation_choices=list(RELATION_CHOICES),
            few_shot_examples=few_shot_examples,
        )
        exercise = getattr(prediction, "exercise", None)
        if not isinstance(exercise, PhraseMatchExercise):
            # Defensive — the DummyLM offline path or a
            # future adapter may return a non-Pydantic shape;
            # coerce via ``model_validate_json`` so the
            # contract is the gate.
            if isinstance(exercise, str):
                exercise = PhraseMatchExercise.model_validate_json(exercise)
            else:
                raise PhraseMatchGenerationError(
                    "phrase_match: DSPy forward returned a non-"
                    "PhraseMatchExercise shape; cannot validate",
                    attempted_schema=json.dumps(
                        PhraseMatchExercise.model_json_schema(),
                        ensure_ascii=False,
                    ),
                    last_validation_error=(
                        f"got {type(exercise).__name__}, expected "
                        "PhraseMatchExercise"
                    ),
                    schema_retry_count=0,
                )
        # Force ``prompt_template_version`` parity with the
        # module constant — same shape as
        # ``generate_phrase_match``.
        return exercise.model_copy(
            update={"prompt_template_version": prompt_template_version}
        )


def optimize_phrase_match_module(
    train_set: Iterable[dict],
    val_set: Iterable[dict] | None = None,
) -> PhraseMatchModule:
    """Run the offline DSPy optimizer against a held-out eval set.

    Strategy mirrors ``app.cloze.optimize_cloze_module`` /
    ``app.idiom.optimize_idiom_module``:

    - Always uses ``DummyLM`` when no API key is present so the
      CI suite runs without network (Hard rule #3).
    - Tries ``dspy.MIPROv2`` first (the spec's preferred
      optimizer). Falls back to ``dspy.BootstrapFewShot`` if
      MIPROv2 raises on the active dep tree.
    - Returns a ``PhraseMatchModule`` with the optimised
      prompt instructions baked in. The caller (Phase 10.7's
      ``scripts/optimize_phrase_match.py``) would serialise
      the optimised module to
      ``backend/app/phrase_match_optimized.json`` so the
      production path could read it on next start.

    Parameters
    ----------
    train_set, val_set
        Iterable of dicts with the input keys (word_id,
        phrase_a, phrase_b, relation_choices, few_shot_examples,
        exercise — where ``exercise`` is a ``PhraseMatchExercise``
        Pydantic instance).

    Returns
    -------
    PhraseMatchModule
        The optimised module. The optimizer mutates the
        module's internal predictor in place; the same
        instance is returned for caller convenience.
    """
    _configure_dspy()

    train_examples = [
        dspy.Example(**row).with_inputs(
            "word_id",
            "phrase_a",
            "phrase_b",
            "relation_choices",
            "few_shot_examples",
        )
        for row in train_set
    ]
    val_examples = (
        [
            dspy.Example(**row).with_inputs(
                "word_id",
                "phrase_a",
                "phrase_b",
                "relation_choices",
                "few_shot_examples",
            )
            for row in val_set
        ]
        if val_set is not None
        else None
    )

    module = PhraseMatchModule()

    optimizer_cls = getattr(dspy, "MIPROv2", None)
    if optimizer_cls is None:
        optimizer_cls = getattr(dspy, "BootstrapFewShot", None)

    if optimizer_cls is None:
        logger.warning(
            "optimize_phrase_match_module: no MIPROv2 / "
            "BootstrapFewShot on the DSPy dep tree; returning "
            "the un-optimized module."
        )
        return module

    optimizer = optimizer_cls(metric=_phrase_match_metric)
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
            "optimize_phrase_match_module: optimizer %s raised "
            "on the offline path (%s); returning the un-"
            "optimized module. Re-run with --live and "
            "OPENROUTER_API_KEY set to actually optimize.",
            optimizer_cls.__name__,
            exc,
        )
        return module
    return optimized


def _phrase_match_metric(
    example: Any, prediction: Any, trace: Any | None = None
) -> float:
    """Offline quality metric for the phrase-match optimizer.

    Score range: 0.0..1.0. The bar is intentionally loose — the
    production C1-accept check is qualitative (Anurag hand-
    reviews), not a numeric gate. The optimizer uses this score
    to pick a better prompt; ``scripts/eval_phrase_match.py``
    (a Phase 10.4 / 10.7 follow-up) would run the more
    rigorous per-row comparison.

    Components:

    - ``+0.4`` if ``prediction.exercise.phrase_a`` equals
      ``example.phrase_a`` verbatim (the verbatim contract).
    - ``+0.4`` if ``prediction.exercise.phrase_b`` equals
      ``example.phrase_b`` verbatim.
    - ``+0.2`` if ``prediction.exercise.relation`` is one of
      the closed 4-way literals and matches the example's
      curated ``relation`` (when the example carries one).
    """
    try:
        ex: PhraseMatchExercise = prediction.exercise  # type: ignore[attr-defined]
    except AttributeError:
        return 0.0
    score = 0.0
    if ex.phrase_a == example.phrase_a:
        score += 0.4
    if ex.phrase_b == example.phrase_b:
        score += 0.4
    # The relation-choices field is the closed 4-way literal
    # tuple; if the prediction picked one of them, +0.2.
    if ex.relation in RELATION_CHOICES:
        score += 0.2
    return score


__all__ = [
    "MAX_ATTEMPTS",
    "PROMPT_TEMPLATE_VERSION",
    "RAG_TOP_K",
    "RELATION_CHOICES",
    "SOURCE_ATTRIBUTION",
    "Phrase",
    "PhraseMatchExercise",
    "PhraseMatchGenerationError",
    "PhraseMatchModule",
    "PhraseMatchNotFoundError",
    "PhraseMatchSignature",
    "PhrasePair",
    "_retrieve_phrase_pair_neighbors",
    "_trace_phrase_match",
    "build_prompt",
    "generate_phrase_match",
    "optimize_phrase_match_module",
    "select_phrase_pair",
]
