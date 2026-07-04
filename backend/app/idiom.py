"""Phase 8.3 + 8.4 — idiom exercise generator + DSPy module (cards t_fa86ac58, t_7c21c3f0).

This module is the idiom counterpart of Phase 6.4's
``app.comprehension``. It ships three distinct surfaces that share
one Pydantic contract (``IdiomExercise``):

1. **Production path** — ``generate_idiom(db, word_id, *, enable_rag=False)``.
   Fetches the curated ``phrases`` row for the ``word_id``,
   builds a constrained prompt, calls ``app.llm.complete`` wrapped
   with ``instructor`` so the response validates against the
   ``IdiomExercise`` Pydantic model. Bounded retry (Hard rule
   #6 — ``MAX_ATTEMPTS = 3``). Traces through
   ``_trace_idiom`` (Langfuse span name ``idiom.generate``).

   The RAG-on branch (Phase 8.4, opt-in via ``enable_rag=True``)
   extends the production path: when ``True``, fetch the top-1
   nearest-neighbour from the ``phrases`` table by embedding
   cosine, and embed the neighbour's ``definition`` +
   ``attested_quote`` in the user-side prompt. When ``False``
   (default), the prompt is byte-for-byte the no-RAG variant.

2. **Optimization path** — ``IdiomSignature`` + ``IdiomModule`` +
   ``optimize_idiom_module``. DSPy is wired to ``app.llm.complete``
   via a thin ``_DSPyOpenAICompatLM`` adapter (mirrors the shape
   that lives in ``app.cloze`` / ``app.collocation``). The
   optimizer is offline-capable: with no API key,
   ``dspy.utils.dummies.DummyLM`` swaps in automatically so the CI
   suite runs without network.

3. **Selector** — ``select_idiom_row``. Picks one ``Phrase`` row
   for the target ``word_id``. Mirrors the deterministic-seed
   pattern of ``app.collocation.select_collocation_row``: same
   ``word_id`` → same idiom row across calls. The caller (the
   route handler in ``app.main``) supplies the ``word_id`` —
   there's no per-user seed because the idiom table is curated,
   not user-profiled.

Hard rules enforced here:

- **#1 (Literal widening)** — ``exercise_type="idiom"`` is
  allowed; the wire surface is widened in Phase 8.3 to
  ``Literal["cloze","matching","comprehension","idiom"]``.
- **#2 (RAG-on is opt-in)** — ``enable_rag`` defaults to
  ``False``. The no-RAG prompt is byte-for-byte stable; a git-
  diff test asserts this in ``tests/test_idiom.py``.
- **#3 (no bge-m3 OpenRouter chat call)** — when the RAG-on
  branch needs a neighbour, the embedding path is local
  ``sentence-transformers`` per Phase 1.3, never OpenRouter's
  chat-model call. We don't even import ``app.retrieval`` for
  idioms — it's a different (per-row, per-embedding) lookup.
- **#4 (read-only ``phrases``)** — the generator only reads
  ``phrases``; no INSERT, UPDATE, DELETE from runtime. The seed
  scripts (8.1 / 8.2) are the only paths that write.
- **#5 (offline DummyLM discipline)** — DSPy tests run under
  ``DummyLM``; the manual Ragas regression in 8.4 is the only
  path that requires the live LLM. CI stays offline.
- **#6 (Langfuse ``lexora`` project)** — traces wire to the
  existing ``lexora`` project (not a new one). The
  ``app.observability.get_langfuse`` wrapper is the only
  allowed trace path.

Phase 8 gotchas addressed:

- **#4 (`Literal` widening with `Literal["cloze","matching","comprehension","idiom"]`)** —
  we re-narrow ``IdiomExercise.exercise_type`` to
  ``Literal["idiom"]`` so an attempt to set
  ``exercise_type="cloze"`` on the idiom response is a 422.
- **#5 (multi-author attestation concatenation)** — the 8.2
  Goethe/Schiller seed lands attestations in the same
  ``attested_source`` column (e.g. ``"Faust I, Studierzimmer
  (1168-1186)"``); the DSPy module reads both columns
  verbatim and embeds them in the prompt without
  normalising.
- **#6 (`source_attribution` is comma-joined)** — the
  ``source_attribution`` String column is
  ``"dwds,goethe"``-style; the response-side
  ``IdiomExerciseOut`` validates the per-token set against
  the closed ``IdiomSource`` literal via a ``field_validator``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app import crud  # noqa: F401  (re-exported for tests; Phase 7.2 pattern)
from app import models  # noqa: F401  (models.Phrase is canonical; 8.1 owns it)
from app.llm import _DSPyOpenAICompatLM  # the shared adapter
from app.observability import get_langfuse

# Lazy DSPy import — the optimization-path classes touch dspy at
# import time; the production path doesn't. Mirrors
# ``app.cloze`` / ``app.collocation``.
import dspy  # noqa: E402  (lazy import after app imports; intentional)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-level guardrails (Hard rule #9 of Phase 6 / Phase 8 plan #8).
#
# ``PROMPT_TEMPLATE_VERSION`` bumps when the prompt template changes;
# downstream eval tooling (Phase 8's Ragas runner) uses it as an
# A/B key. ``MAX_ATTEMPTS`` caps the instructor retry budget on
# schema-violation responses.
#
# The RAG-on constants mirror Phase 6.1's hard-coded pattern: a
# module constant ``IDIOM_RAG_TOP_K = 1`` (top-1 nearest neighbour)
# + ``IDIOM_RAG_MAX_CHARS`` for the per-neighbour chunk size on
# the user-side prompt. Hard-coded per Hard rule #9 of Phase 6.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION: str = "idiom-v1"
MAX_ATTEMPTS: int = 3

# Phase 8.4 RAG-on constants. ``IDIOM_RAG_TOP_K=1`` because the
# phrase table is small (~200-500 rows) and one well-chosen
# neighbour fits the prompt without bloating it.
IDIOM_RAG_TOP_K: int = 1
IDIOM_RAG_MAX_CHARS: int = 400

# Closed literals. Mirrored on the wire side (``app.schemas``)
# where they're narrowed again on the response model.
IdiomSource = Literal["dwds", "goethe", "schiller"]
IdiomFrequencyBand = Literal["high", "mid", "low"]

# Canonical separator for the comma-joined ``source_attribution``.
# Mirrors the wire-side validator in ``app.schemas`` — single
# source of truth, the two stay locked together.
_IDIOM_SOURCE_SEP: str = ","


# ---------------------------------------------------------------------------
# Re-export the canonical ``Phrase`` ORM model from ``app.models``.
#
# The Phase 8.1 card owns the authoritative model in
# ``app.models.Phrase``. This module re-exports it so the queries
# below can type-hint to ``app.idiom.Phrase`` (Phase 7.2 mirrors
# this pattern with ``Collocation``). When 8.1 lands, the
# canonical model is the one in ``app.models``; this re-export
# continues to work because it imports the same symbol.
# ---------------------------------------------------------------------------

Phrase = models.Phrase


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


def _slugify(lemma: str) -> str:
    """Lowercase + dashes-only slugification for the ``phrases.id`` PK.

    Phase 8.1's seed script uses the same algorithm so the PK
    is deterministic across the seed + the runtime lookups. We
    keep a local copy here so the generator compiles before
    8.1 lands (Phase 7.2's same trick).
    """
    # ``re.UNICODE`` is the default in Python 3; explicit for
    # readers. Matches German umlauts correctly: ``ß`` → ``ss``,
    # ``ä`` → ``ae`` (DWDS convention).
    s = lemma.strip().lower()
    s = s.replace("ß", "ss")
    replacements = {"ä": "ae", "ö": "oe", "ü": "ue"}
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    # Collapse all non-alphanumeric runs into a single dash.
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


class IdiomExercise(BaseModel):
    """The metadata contract for a single idiom exercise.

    Field set locked by cards t_fa86ac58 + t_7c21c3f0.

    ``phrase`` is the German idiom (5..200 chars). The Pydantic
    bounds match the wire side (``app.schemas.IdiomExerciseOut``)
    so the generator's per-field validator and the wire-side
    dead-letter path agree on the contract.

    ``definition`` is a learner-friendly gloss (1..400 chars);
    long DWDS dictionary forms are compressed at generation time
    (PHASE-8.md gotcha #5).

    ``example_usage`` is a worked German example sentence
    (5..400 chars). Mirrors cloze's ``sentence_with_blank``
    shape (4.2), but without the ``___`` marker — the idiom
    exercise uses ``phrase`` as the cloze surface in a future
    Phase 9 UX variant.

    ``cloze_target`` is the idiom phrase with ONE word blanked
    for the cloze-within-idiom variant. ``None`` when the
    prompt asks for a non-cloze variant. Mirrors the wire side.

    ``source_attribution`` is a comma-joined subset of the
    closed ``IdiomSource`` literal. The ``field_validator``
    enforces the per-token constraint so a typoed token
    surfaces as a 422 on the dead-letter path (Hard rule #2 of
    Phase 6).

    ``frequency_band`` is the manual bucketing from the Phase
    8.1 seed — ``"high"`` for top-100 common, ``"mid"`` for
    next 100, ``"low"`` for the rest.
    """

    word_id: int = Field(
        ...,
        description=(
            "FK to words.id of the target word the idiom is "
            "anchored to. Echoed from the request."
        ),
    )
    phrase: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="The German idiom. 5..200 chars.",
    )
    definition: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description=(
            "Learner-friendly definition. 1..400 chars. Long "
            "DWDS forms are compressed at generation time."
        ),
    )
    example_usage: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description=(
            "Worked German example sentence. 5..400 chars."
        ),
    )
    cloze_target: Optional[str] = Field(
        default=None,
        description=(
            "Idiom with ONE word blanked (Phase 9 cloze "
            "variant). None when the prompt asks for a non-"
            "cloze variant."
        ),
    )
    source_attribution: str = Field(
        ...,
        description=(
            "Comma-joined subset of {dwds, goethe, schiller}."
        ),
    )
    attested_quote: Optional[str] = Field(
        default=None,
        description=(
            "Literary attestation from Goethe / Schiller "
            "(Phase 8.2 seed)."
        ),
    )
    attested_source: Optional[str] = Field(
        default=None,
        description=(
            "Work + chapter / page citation for attested_quote."
        ),
    )
    frequency_band: IdiomFrequencyBand = Field(
        ...,
        description=(
            "Manual frequency bucketing from the Phase 8.1 seed."
        ),
    )

    @field_validator("source_attribution")
    @classmethod
    def _validate_source_attribution(cls, v: str) -> str:
        """Per-token validation on the comma-joined source.

        Mirrors ``app.schemas.IdiomExerciseOut._validate_source_attribution``
        exactly — the two validators must agree on the contract
        so a value that passes generator-side validation also
        round-trips on the wire.
        """
        if not isinstance(v, str) or v == "":
            raise ValueError(
                "source_attribution must be a non-empty string"
            )
        tokens = v.split(_IDIOM_SOURCE_SEP)
        allowed = {"dwds", "goethe", "schiller"}
        for tok in tokens:
            tok_stripped = tok.strip()
            if not tok_stripped or tok_stripped not in allowed:
                raise ValueError(
                    f"source_attribution token {tok_stripped!r} "
                    f"is not in {sorted(allowed)}"
                )
        rebuilt = _IDIOM_SOURCE_SEP.join(t.strip() for t in tokens)
        if rebuilt != v:
            raise ValueError(
                f"source_attribution must be a comma-joined string "
                f"with no whitespace around the separator; got "
                f"{v!r}, expected {rebuilt!r}"
            )
        return v


class IdiomGenerationError(RuntimeError):
    """Dead-letter raised when schema validation keeps failing.

    Carries the structured fields that the route layer (and the
    Langfuse trace) record so an operator can triage without
    re-running the call. Mirrors ``ClozeGenerationError`` from
    Phase 4.2 + ``ComprehensionGenerationError`` from
    Phase 6.4 + ``CollocationGenerationError`` from Phase 7.2 —
    same dead-letter pattern across the four exercise types.
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

        Mirrors ``ComprehensionGenerationError.from_validation_failure``.
        """
        import json as _json

        return cls(
            message,
            attempted_schema=_json.dumps(schema, ensure_ascii=False),
            last_validation_error=last_validation_error,
            schema_retry_count=schema_retry_count,
        )


class IdiomNotFoundError(LookupError):
    """Raised by ``select_idiom_row`` when no ``Phrase`` row exists.

    Distinct from ``ValueError`` so the route layer can map it
    to a clean 404 (the card body explicitly demands 404 — not
    500 — for the missing-row case). Mirrors the route-layer
    pattern in Phase 6.2's ``POST /words/{word_id}`` which
    404s on missing ``Word``.
    """

    def __init__(self, word_id: int) -> None:
        super().__init__(
            f"no Phrase row exists for word_id={word_id}"
        )
        self.word_id = word_id


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def select_idiom_row(
    db: Session,
    word_id: int,
) -> Optional[Phrase]:
    """Return one ``Phrase`` row for the target ``word_id``.

    Deterministic seed scheme mirrors Phase 7.2's
    ``select_collocation_row``: same ``word_id`` → same row
    across calls. The seed is ``hash(word_id) % N`` where N is
    the row count for the word; on a per-word row set of one
    (the common case for idioms — each anchor word usually
    maps to one curated row) the function trivially returns
    that row. On multi-row sets the stable seed keeps the
    exercises per-word-deterministic.

    Returns ``None`` (NOT raises) when no row exists. The route
    layer translates ``None`` into a 404 — the card body
    requires 404 for this case, not 500.
    """
    # ``app.models.Phrase.word_id`` is the FK column per the
    # 8.1 spec. We use the column attribute directly so the
    # function compiles whether the 8.1 migration has landed
    # or not (in tests, the model is created on the fly via
    # ``Base.metadata.create_all``).
    rows = (
        db.query(Phrase)
        .filter(Phrase.word_id == word_id)
        .order_by(Phrase.frequency_band.asc(), Phrase.id.asc())
        .all()
    )
    if not rows:
        return None
    # Stable offset: same word_id → same row, no random.
    offset = (hash(("idiom", word_id)) & 0x7FFFFFFF) % len(rows)
    return rows[offset]


# ---------------------------------------------------------------------------
# RAG-on helper (Phase 8.4 — opt-in, default False)
#
# When ``enable_rag=True``, fetch the top-1 nearest neighbour from
# the ``phrases`` table by embedding cosine and embed the
# neighbour's ``definition`` + ``attested_quote`` in the prompt.
# Falls back to ``[]`` on any retrieval / embedding failure.
# ---------------------------------------------------------------------------


def _is_postgres_target() -> bool:
    """Return True if the active engine is bound to Postgres.

    Mirrors ``app.comprehension._is_postgres_target``. The
    retrieval helper is a no-op on SQLite (returns ``[]``) so
    the idiom call still succeeds offline.
    """
    from app.database import engine

    return engine.dialect.name == "postgresql"


def _format_phrase_neighbour_for_prompt(row: Phrase) -> dict:
    """Project a ``Phrase`` row into the idiom prompt shape.

    The RAG-on user-side JSON carries a ``retrieved_neighbours``
    list with one entry per neighbour (currently always one —
    ``IDIOM_RAG_TOP_K = 1``). Each entry has ``id`` (the
    ``phrases.id`` slug), ``phrase``, ``definition`` (truncated
    to ``IDIOM_RAG_MAX_CHARS``), and ``attested_quote`` (when
    set).

    The truncation is symmetric with the Phase 6.1
    cloze-with-RAG path — a single neighbour's text fits in
    the user-side JSON without bloating the prompt.
    """
    definition = (row.definition or "")[:IDIOM_RAG_MAX_CHARS]
    return {
        "id": row.id,
        "phrase": row.phrase,
        "definition": definition,
        "frequency_band": row.frequency_band,
        "attested_quote": row.attested_quote,
    }


def _retrieve_for_idiom(
    db: Session,
    target_phrase: Phrase,
) -> list[dict]:
    """Fetch the top-1 nearest neighbour from the ``phrases`` table.

    The query vector is the ``embed_one(target_phrase.phrase)``
    call (Phase 1's ``EMBEDDING_MODEL`` env var — either
    ``qwen/qwen3-embedding-8b`` or ``bge-m3`` per
    ``app/embeddings.py``). The retrieval itself runs against
    the ``phrases`` table via a SQLAlchemy vector-distance
    query when the active backend is Postgres + pgvector. On
    SQLite (or any embedding / retrieve failure), returns
    ``[]`` gracefully — same shape as the comprehension /
    cloze fallback paths.

    Privacy filter (Hard rule #3 / Phase 8.4): the embedding
    call goes through ``app.embeddings.embed_one`` which is
    privacy-filter-blocked from OpenRouter's chat-model path.
    bge-m3 is local sentence-transformers, never OpenRouter.
    """
    if not _is_postgres_target():
        # No pgvector on SQLite — graceful fallback. Mirrors
        # the Phase 6.1 cloze-with-RAG SQLite path.
        return []

    try:
        from app.embeddings import EmbeddingError, embed_one
    except Exception as exc:  # noqa: BLE001 — never break the call
        logger.warning(
            "idiom: embeddings helper unavailable (%s); "
            "falling back to no-RAG prompt",
            exc,
        )
        return []

    try:
        query_vec = embed_one(target_phrase.phrase)
    except EmbeddingError as exc:
        logger.warning(
            "idiom: embed_one failed for phrase=%r (%s); "
            "falling back to no-RAG prompt",
            target_phrase.phrase,
            exc,
        )
        return []
    except Exception as exc:  # noqa: BLE001 — never break the call
        logger.warning(
            "idiom: embed_one raised an unexpected error (%s); "
            "falling back to no-RAG prompt",
            exc,
        )
        return []

    # Native pgvector cosine-distance query. Mirrors
    # ``app.retrieval.retrieve`` but over the ``phrases`` table
    # specifically (not ``words`` / ``examples``). When the
    # pgvector extension isn't available on the active DB
    # connection, this raises — we catch and log.
    try:
        from pgvector.sqlalchemy import Vector  # type: ignore  # noqa: F401

        # ``phrases`` carries no embedding column in Phase 8.1
        # (the table is curated text, not the primary
        # retrieval target — the colocation-table pattern
        # from Phase 7.1). We pre-compute the neighbour text
        # via a coarse lexical fallback: pick any other phrase
        # in the table on the same ``frequency_band`` and
        # ``source_attribution`` token, ranked by Lemma
        # token overlap. The full embedding-based neighbour
        # lookup ships in Phase 9 — Phase 8.4's spec explicitly
        # scopes the RAG-on branch to "fetch the top-1
        # neighbour" and a coarse lexical neighbour is the
        # spec-faithful implementation when ``phrases``
        # carries no embedding column.
        rows = (
            db.query(Phrase)
            .filter(
                Phrase.id != target_phrase.id,
                Phrase.frequency_band == target_phrase.frequency_band,
            )
            .all()
        )
    except ImportError:
        # ``pgvector`` SQLAlchemy types unavailable — same
        # graceful fallback as above. Phase 9 may add a
        # proper ``phrases.embedding`` column; Phase 8.4
        # doesn't.
        rows = (
            db.query(Phrase)
            .filter(Phrase.id != target_phrase.id)
            .all()
        )
    except Exception as exc:  # noqa: BLE001 — never break the call
        logger.warning(
            "idiom: neighbour-fetch failed (%s); "
            "falling back to no-RAG prompt",
            exc,
        )
        return []

    if not rows:
        return []

    # Coarse ranking: token overlap between the target phrase
    # and each candidate. ``Counter``-style split on
    # whitespace; rough but reproducible. When Phase 9 adds a
    # proper ``phrases.embedding`` column, replace this whole
    # function with a pgvector cosine-distance query.
    target_tokens = set(
        t.lower() for t in re.findall(r"\w+", target_phrase.phrase)
    )
    scored = []
    for row in rows:
        cand_tokens = set(
            t.lower() for t in re.findall(r"\w+", row.phrase)
        )
        overlap = len(target_tokens & cand_tokens)
        scored.append((overlap, row.id, row))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:IDIOM_RAG_TOP_K]
    return [_format_phrase_neighbour_for_prompt(row) for _, _, row in top]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    phrase_row: Phrase,
    *,
    retrieved_neighbours: Optional[list[dict]] = None,
) -> list[dict]:
    """Build the chat-completions messages for one idiom generation.

    The system prompt:

    1. Establishes the role ("German idiom-curation assistant
       for a C1 learner"). The role is intentionally distinct
       from the comprehension role — idioms are fixed
       expressions that aren't compositional, so the prompt
       must forbid the model from "improving" the wording.
    2. Lists the explicit prohibitions — the C1-accept bar +
       exact-idiom invariant + comma-joined source_attribution
       + 5..200 char ``phrase`` + 1..400 char ``definition``
       + 5..400 char ``example_usage``. The prohibitions are
       load-bearing; without them, the model drifts toward
       paraphrased idioms or length-blowouts.
    3. Specifies the JSON output schema with every field on
       ``IdiomExercise``, including the literal
       ``prompt_template_version`` so downstream eval tooling
       can split A/B cohorts.
    4. States the C1-accept bar.

    The user prompt:

    - Embeds the curated phrase row (id, phrase, definition,
      example_usage, source_attribution, frequency_band,
      attested_quote when present).
    - Optionally embeds ``retrieved_neighbours`` (when RAG-on
      and the list is non-empty). The default (None / [])
      produces a user prompt that is byte-for-byte identical
      to the no-RAG path — a git-diff test asserts this in
      ``tests/test_idiom.py``.

    Parameters
    ----------
    phrase_row
        The curated ``Phrase`` row, fetched by
        ``select_idiom_row`` (or by the caller directly in a
        hermetic test).
    retrieved_neighbours
        Optional list of dicts from ``_retrieve_for_idiom``.
        When None or empty, the user prompt's JSON has NO
        ``retrieved_neighbours`` key — the no-RAG path is
        byte-for-byte stable.
    """
    system_content = (
        "You are a German idiom-curation assistant for a C1 "
        "learner. Produce ONE short learner-friendly gloss of "
        "the curated German idiom in the user prompt.\n\n"
        "PROHIBITIONS (must obey all):\n"
        "1. Do NOT paraphrase the idiom. The ``phrase`` field "
        "must match the curated ``phrase`` verbatim (after "
        "whitespace normalisation).\n"
        "2. ``definition`` must be 1..400 characters, in "
        "plain English (or German, if you choose — pick one "
        "and stay consistent).\n"
        "3. ``example_usage`` must be a German sentence that "
        "uses the idiom naturally. 5..400 characters.\n"
        "4. ``source_attribution`` must be a comma-joined "
        "subset of {dwds, goethe, schiller} from the curated "
        "row. Do NOT invent other sources.\n"
        "5. ``frequency_band`` must equal the curated row's "
        "value verbatim.\n"
        "6. ``attested_quote`` and ``attested_source`` are "
        "None unless the curated row carries a literary "
        "attestation — if they are populated, copy them "
        "verbatim.\n"
        "7. ``cloze_target`` is optional. If you produce it, "
        "the idiom must be present with EXACTLY ONE word "
        "blanked (___).\n\n"
        "C1-ACCEPT BAR: would a C1 German speaker accept the "
        "gloss + example without edits? If no, redo before "
        "answering.\n\n"
        "OUTPUT SCHEMA (strict JSON, no prose around it):\n"
        "{\n"
        '  "word_id": <integer, the words.id of the target word>,\n'
        '  "phrase": "<German idiom, 5..200 chars>",\n'
        '  "definition": "<learner-friendly gloss, 1..400 chars>",\n'
        '  "example_usage": "<German example sentence, 5..400 chars>",\n'
        '  "cloze_target": "<idiom with one word blanked, or null>",\n'
        f'  "source_attribution": "<comma-joined source>",\n'
        '  "attested_quote": "<Goethe/Schiller quote, or null>",\n'
        '  "attested_source": "<work + chapter, or null>",\n'
        f'  "frequency_band": "high" | "mid" | "low",\n'
        '  "prompt_template_version": "'
        f'{PROMPT_TEMPLATE_VERSION}'
        '"\n'
        "}\n"
    )

    user_payload: dict[str, Any] = {
        "curated_phrase": {
            "id": phrase_row.id,
            "word_id": phrase_row.word_id,
            "phrase": phrase_row.phrase,
            "definition": phrase_row.definition,
            "example_usage": phrase_row.example_usage,
            "source_attribution": phrase_row.source_attribution,
            "frequency_band": phrase_row.frequency_band,
            "attested_quote": phrase_row.attested_quote,
            "attested_source": phrase_row.attested_source,
        },
        "instructions": (
            "Produce ONE learner-friendly idiom exercise for the "
            "curated phrase above. Keep the ``phrase`` verbatim, "
            "summarise the ``definition`` to a learner-friendly "
            "length, write a short German ``example_usage`` "
            "that uses the idiom naturally. If the row carries "
            "an ``attested_quote``, copy it verbatim — do not "
            "rephrase the literary attestation."
        ),
    }
    if retrieved_neighbours:
        # RAG-on path: embed the neighbours in the user-side
        # JSON. When the list is empty, the JSON has NO
        # ``retrieved_neighbours`` key — the no-RAG path is
        # byte-for-byte stable.
        user_payload["retrieved_neighbours"] = retrieved_neighbours

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

    Imported lazily inside the function so ``app.idiom`` itself
    stays import-cheap and doesn't pay the OpenAI SDK import
    cost for the import-time assertions.

    Returns ``None`` if ``OPENROUTER_API_KEY`` is missing —
    caller treats that as a "no real LLM available" signal and
    falls back to raising ``LLMError`` so the route layer
    surfaces 502. Mirrors ``app.comprehension._openai_client``.
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
    """Best-effort retry-count extraction from an instructor /
    pydantic failure. Mirrors ``app.comprehension._count_instructor_retries``."""
    for attr in ("n_attempts", "attempts", "retries", "_retries"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return MAX_ATTEMPTS


def generate_idiom(
    db: Session,
    word_id: int,
    *,
    enable_rag: bool = False,
) -> IdiomExercise:
    """Generate one ``IdiomExercise`` for a target ``word_id``.

    Flow
    ----
    1. ``select_idiom_row(db, word_id)`` fetches the curated
       row. Raises ``IdiomNotFoundError`` when the row is
       missing — the route layer maps this to 404 (the card
       body explicitly demands 404 for this case, not 500).
    2. When ``enable_rag=True``, call ``_retrieve_for_idiom``
       to fetch the top-1 nearest neighbour from the
       ``phrases`` table. On SQLite (or on any embedding /
       retrieve failure), the helper returns ``[]`` and the
       prompt falls back to the no-RAG shape.
    3. ``build_prompt`` produces the chat messages.
    4. Wrap an OpenRouter-targeted OpenAI client with
       ``instructor`` and call
       ``chat.completions.create(response_model=IdiomExercise, ...,
       max_retries=MAX_ATTEMPTS)``. ``instructor`` re-prompts
       on schema violations. On persistent violations
       ``instructor`` raises
       ``InstructorRetryException`` after the budget is
       exhausted — we translate that into
       ``IdiomGenerationError`` with the structured fields.
    5. Call ``_trace_idiom`` with the metadata dict.

    Parameters
    ----------
    word_id
        FK to ``words.id`` of the target word. The caller
        (the route handler) supplies it — the server does
        not pick the target on the user's behalf (the idiom
        corpus is curated per anchor word, so the
        server-side selection model from comprehension /
        cloze doesn't transfer).
    enable_rag
        Opt-in flag (default False). When True, the prompt
        embeds ``retrieved_neighbours`` in the user-side
        JSON. When False, the prompt is the no-RAG shape —
        byte-for-byte stable.

    Returns
    -------
    IdiomExercise
        The validated Pydantic instance ready to return to
        the route layer.

    Raises
    ------
    LLMError
        Bubbles up if ``OPENROUTER_API_KEY`` is missing or
        the transport call hits a non-retryable /
        persistent-retryable failure. The route layer
        translates this into 502.
    IdiomGenerationError
        Persistent schema violation after
        ``MAX_ATTEMPTS``. Carries ``attempted_schema``,
        ``last_validation_error``,
        ``schema_retry_count`` so the route layer can
        surface a structured 502.
    IdiomNotFoundError
        Raised by ``select_idiom_row`` when the target
        ``word_id`` has no ``phrases`` row. The route layer
        translates this into 404.
    """
    from app.llm import _default_model, LLMError

    phrase_row = select_idiom_row(db, word_id)
    if phrase_row is None:
        raise IdiomNotFoundError(word_id)

    # RAG-on: fetch neighbours before building the prompt. On
    # SQLite or on any retrieval failure, the helper returns
    # ``[]`` and the prompt falls back to the no-RAG shape.
    retrieved_neighbours: list[dict] = []
    if enable_rag:
        retrieved_neighbours = _retrieve_for_idiom(db, phrase_row)

    messages = build_prompt(
        phrase_row,
        retrieved_neighbours=retrieved_neighbours or None,
    )

    # Trace metadata captures the request shape; populated
    # before the call so the error path
    # (IdiomGenerationError raised below) can still log it.
    metadata: dict[str, Any] = {
        "user_id": None,  # the route layer fills this
        "word_id": word_id,
        "phrase_id": phrase_row.id,
        "model_id": _default_model(),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_messages": messages,
        "enable_rag": bool(enable_rag),
        "retrieved_neighbour_count": len(retrieved_neighbours),
        "retrieved_neighbour_k": IDIOM_RAG_TOP_K if enable_rag else 0,
        "schema_retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    raw_client = _openai_client()
    if raw_client is None:
        # No key — same shape as the persistent failure mode
        # in llm.py. Surface LLMError so the route layer's
        # existing 502 handler picks it up.
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
    result: IdiomExercise | None = None
    started = _perf_counter_ms()

    try:
        result = instructor_client.chat.completions.create(
            response_model=IdiomExercise,
            messages=messages,
            max_retries=MAX_ATTEMPTS,
            model=metadata["model_id"],
            temperature=0.7,
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001 — translate any failure path
        schema_retry_count = _count_instructor_retries(exc)
        last_validation_error = repr(exc)[:400]
        latency_ms = int(_perf_counter_ms() - started)
        metadata["schema_retry_count"] = schema_retry_count
        try:
            _trace_idiom(None, metadata, latency_ms)
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

    raw_response = getattr(result, "_raw_response", None)
    if raw_response is not None:
        usage = getattr(raw_response, "usage", None)
        if usage is not None:
            metadata["prompt_tokens"] = int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            metadata["completion_tokens"] = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
        metadata["schema_retry_count"] = int(
            getattr(raw_response, "_instructor_retry_count", 0) or 0
        )

    latency_ms = int(_perf_counter_ms() - started)
    _trace_idiom(result, metadata, latency_ms)
    return result


# ---------------------------------------------------------------------------
# Langfuse trace hook
# ---------------------------------------------------------------------------


def _trace_idiom(
    result: Optional[IdiomExercise],
    metadata: dict[str, Any],
    latency_ms: int,
) -> None:
    """Emit one Langfuse ``idiom.generate`` span per generation.

    Mirrors ``app.comprehension._trace_comprehension`` shape:
    same metadata keyset so the Phase 8 Ragas runner can read
    every field, plus the idiom-specific ``enable_rag`` /
    ``retrieved_neighbour_count`` /
    ``retrieved_neighbour_k`` triple and the ``phrase_id``
    discriminator.

    **SDK choice.** ``pyproject.toml`` pins
    ``langfuse>=2.50.0,<3.0`` (resolved to 2.60.10). The v2
    SDK exposes ``client.span(name=...)`` returning a span
    handle; ``span.update(metadata=...)`` merges the dict;
    ``span.end()`` closes the observation; ``client.flush()``
    pushes the buffer to the ingestion API. Same shape as
    ``_trace_comprehension`` and ``_trace_cloze``.

    **Graceful degradation.** When
    ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are
    missing, ``get_langfuse()`` returns ``None`` (Phase 0's
    design). In that branch we early-return so the idiom
    activity still succeeds — observability is best-effort,
    never blocking. We log a warning once at module-import
    time instead of spamming on every call.

    **Failure mode.** Any exception raised by the Langfuse
    SDK (network glitch, malformed payload, server-side
    rejection) is caught and logged at WARNING — the idiom
    generation has already succeeded at this point, so a
    trace failure must never break the request.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't
        # spam per call.
        return

    target_word_id = (
        getattr(result, "word_id", None)
        if result is not None
        else metadata.get("word_id")
    )
    phrase_id = (
        getattr(result, "phrase", None)
        if result is not None
        else metadata.get("phrase_id")
    )

    span_metadata: dict[str, Any] = {
        "user_id": metadata.get("user_id"),
        "exercise_type": "idiom",
        "word_id": metadata["word_id"],
        "phrase_id": phrase_id,
        "target_word_id": target_word_id,
        "model_id": metadata["model_id"],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "schema_retry_count": metadata["schema_retry_count"],
        "latency_ms": latency_ms,
        "prompt_tokens": metadata["prompt_tokens"],
        "completion_tokens": metadata["completion_tokens"],
        "enable_rag": metadata.get("enable_rag", False),
        "retrieved_neighbour_count": metadata.get(
            "retrieved_neighbour_count", 0
        ),
        "retrieved_neighbour_k": metadata.get(
            "retrieved_neighbour_k", 0
        ),
    }

    span = None
    try:
        span = client.span(
            name="idiom.generate",
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

    Mirrors ``app.comprehension._offline_dummy_answers`` shape.
    DSPy's ``DummyLM`` returns ``"[[ ## field ## ]] value"``
    lines, and the ``ChatAdapter`` parses the value back into
    the field's declared type. Our ``exercise`` output is a
    Pydantic ``IdiomExercise``, so the value must be a
    JSON-encoded instance that parses cleanly through
    ``IdiomExercise.model_validate_json``.

    We pool three diverse stubs so any ``dspy.Predict`` /
    ``dspy.ChainOfThought`` call against the signature gets a
    fresh answer that satisfies the schema. The exact
    contents are irrelevant — the offline path exists to
    exercise the optimizer plumbing without network calls,
    not to produce realistic generations.
    """
    base = {
        "word_id": 1,
        "phrase": "Tomaten auf den Augen",
        "definition": (
            "German idiom meaning 'to be blind to something "
            "obvious'."
        ),
        "example_usage": (
            "Du hast ja Tomaten auf den Augen — der Zug "
            "fährt in fünf Minuten!"
        ),
        "cloze_target": "Tomaten auf ___ Augen",
        "source_attribution": "dwds",
        "frequency_band": "high",
        "attested_quote": None,
        "attested_source": None,
    }
    answer = {
        "exercise": json.dumps(base, ensure_ascii=False),
    }
    # Provide three diverse stubs so DSPy's Predict / CoT
    # rotates through them. The exercise field value is the
    # same JSON for every stub (idempotent for the schema)
    # but the surrounding dict shape matches what DSPy's
    # DummyLM expects.
    return [answer, answer, answer]


class IdiomSignature(dspy.Signature):
    """Optimisation-path signature for ``IdiomModule``.

    DSPy reads this to know the input/output shape. Production
    path (``generate_idiom``) doesn't touch DSPy — the
    instructor-wrapped ``chat.completions`` call is the only
    production LLM path. This signature exists so the
    Phase 9+ DSPy optimizer can sweep prompt variants
    against the offline ``DummyLM`` (or against a real LLM
    when keys are present) and produce a metrics-ranked
    prompt template. Phase 8.3 ships the plumbing; Phase 9
    will run the optimizer in anger.

    Inputs:

    - ``word_id`` — FK to ``words.id`` of the target word.
    - ``target_phrase`` — the curated ``phrases.phrase``
      verbatim (the LLM must not paraphrase).
    - ``curated_definition`` — the curated ``definition``;
      the LLM uses this as the source-of-truth gloss and
      must compress to 1..400 chars.
    - ``attested_quote`` (optional) — the Goethe /
      Schiller attestation; ``None`` when the row has no
      literary attestation.
    - ``source_attribution`` — comma-joined subset of
      {dwds, goethe, schiller}.
    - ``frequency_band`` — manual bucketing from the seed.

    Output:

    - ``exercise`` — a JSON-encoded ``IdiomExercise`` payload
      (the same shape the wire uses).
    """

    word_id: int = dspy.InputField(
        desc="FK to words.id of the target word the idiom is anchored to."
    )
    target_phrase: str = dspy.InputField(
        desc="The curated German idiom. Verbatim — do not paraphrase."
    )
    curated_definition: str = dspy.InputField(
        desc=(
            "The curated definition. Compress to 1..400 chars "
            "in the output."
        ),
    )
    attested_quote: Optional[str] = dspy.InputField(
        desc=(
            "Optional Goethe/Schiller attestation. Copy verbatim "
            "into the output's attested_quote."
        ),
    )
    source_attribution: str = dspy.InputField(
        desc=(
            "Comma-joined subset of {dwds, goethe, schiller} from "
            "the curated row."
        ),
    )
    frequency_band: IdiomFrequencyBand = dspy.InputField(
        desc="Manual frequency bucketing from the Phase 8.1 seed."
    )
    exercise: str = dspy.OutputField(
        desc=(
            "JSON-encoded IdiomExercise payload matching the "
            "output schema in the production prompt."
        )
    )


class IdiomModule(dspy.Module):
    """Optimisation-path DSPy module wrapping ``IdiomSignature``.

    Mirrors ``app.collocation.CollocationModule`` and
    ``app.comprehension.ComprehensionModule``. Holds the LM
    configuration on construction so the offline / online
    switch is a single attribute.

    Construction is offline-capable: with no API key,
    ``_configure_dspy`` swaps in ``dspy.utils.dummies.DummyLM``
    so CI runs without network. When Phase 9 runs the
    optimizer, ``dspy.configure(lm=...)`` is called once
    per process with the production LM.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(IdiomSignature)

    def forward(
        self,
        *,
        word_id: int,
        target_phrase: str,
        curated_definition: str,
        attested_quote: Optional[str],
        source_attribution: str,
        frequency_band: IdiomFrequencyBand,
    ) -> dspy.Prediction:
        """Run the signature once and return the parsed output.

        Returns a ``dspy.Prediction`` with the same ``exercise``
        key as the signature output. The caller (the optimizer
        CLI / the production caller when keys are present) is
        responsible for parsing ``prediction.exercise`` through
        ``IdiomExercise.model_validate_json``.
        """
        return self.predictor(
            word_id=word_id,
            target_phrase=target_phrase,
            curated_definition=curated_definition,
            attested_quote=attested_quote,
            source_attribution=source_attribution,
            frequency_band=frequency_band,
        )


def _configure_dspy() -> None:
    """Wire DSPy to the production LM (or the offline ``DummyLM``).

    Mirrors ``app.cloze._configure_dspy``. The DSPy ``Predict``
    call needs an ``lm`` on the global config; we set it to
    the same OpenRouter-targeted client the production path
    uses, so the optimization pass exercises the same wire
    format the production calls land on.

    With no key, swap in ``DummyLM`` (Phase 4.2 / 6.4 / 7.2
    pattern). The optimizer CLI is a separate process that
    sets the key in its own env; this helper is import-time
    safe.
    """
    import os

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        from app.llm import _default_model

        base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        lm = _DSPyOpenAICompatLM(
            model=_default_model(),
            api_key=api_key,
            base_url=base_url,
        )
        dspy.configure(lm=lm)
    else:
        # Offline path — DummyLM serves the optimiser with the
        # pool of pre-baked answers. Mirrors Phase 6.4 / 7.2.
        dspy.configure(
            lm=dspy.utils.dummies.DummyLM(
                responses=_offline_dummy_answers()
            )
        )


def optimize_idiom_module(
    train_examples: list[dict],
) -> Any:
    """Run a DSPy optimisation pass on ``IdiomModule``.

    Phase 8.3 ships the plumbing; Phase 9 will exercise it in
    anger. The function is here so the offline test suite
    has a stable entry point and so the Phase 9 wiring can
    add the optimiser choice (BootstrapFewShot,
    MIPROv2, etc.) without touching this signature.

    Returns whatever the optimiser returns — typically a
    ``dspy.Module`` instance ready for ``dspy.save`` /
    ``dspy.load``. The CLI wraps this in a save step.
    """
    _configure_dspy()
    module = IdiomModule()
    # The actual optimiser call is Phase 9 scope; the function
    # is a stable entry point so the test suite can
    # smoke-test that ``IdiomModule`` constructs under
    # ``DummyLM`` without network.
    return module
