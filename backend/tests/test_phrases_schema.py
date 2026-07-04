"""Tests for Phase 8.1 — phrases schema (Pydantic boundaries).

Card: t_d967c006.

Coverage map (mirrors the card body's "Verification" list and the
Phase 7.1 schema-test pattern):

1. ``PhraseSeedRow`` Pydantic enforcement:
   - ``phrase``: 5–200 chars.
   - ``definition``: 1–400 chars.
   - ``example_usage``: 5–400 chars when set, None when absent.
   - ``frequency_band``: ``Literal["high","mid","low"]``.
   - ``source_attribution``: per-element literal at parse time
     (matches tokens in ``{"dwds","goethe","schiller","manual"}``),
     comma-joined strings like ``"dwds,goethe"`` accepted, unknown
     tokens rejected.
   - ``id`` (slug PK): 3–120 chars.

2. ``PhraseOut`` outbound view preserves the same shape; the
   validator still catches an invalid ``source_attribution`` token.

3. The literal enums are the wire-level guardrails (gotcha #6 of
   PHASE-8.md). A typo'd value is caught at parse time, NOT later
   when the row silently propagates into the Phase 8.3 idiom
   generator.

Hermetic: no DB connection, no LLM call. Pure Pydantic v2 unit
tests on the ``app.schemas`` module.

Run from ``backend/``::

    bash /tmp/runpytest.sh tests/test_phrases_schema.py
"""
from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from app.schemas import (
    PhraseFrequencyBand,
    PhraseOut,
    PhraseSeedRow,
    PhraseSourceAttribution,
    _PHRASE_SOURCE_TOKENS,
    _split_source_attribution,
)


# ---------------------------------------------------------------------------
# Tiny fixture builders — keep the test matrix legible.
# ---------------------------------------------------------------------------


def _good_row(**overrides) -> dict:
    """Return a valid ``PhraseSeedRow`` payload (with optional overrides)."""
    base = {
        "id": "tomaten-auf-den-augen",
        "phrase": "Tomaten auf den Augen",
        "definition": "blind to what is plainly obvious",
        "example_usage": "Du hast Tomaten auf den Augen.",
        "source_attribution": "dwds",
        "frequency_band": "high",
        "dwds_url": "https://www.dwds.de/wb/Tomaten%20auf%20den%20Augen",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# PhraseSeedRow — happy paths
# ---------------------------------------------------------------------------


def test_phrase_seed_row_accepts_a_valid_payload():
    """The simplest valid row round-trips cleanly through
    ``PhraseSeedRow``. Negative coverage for the union of fields:
    if this test fails, the column list has drifted from the
    SQLAlchemy model."""
    row = PhraseSeedRow.model_validate(_good_row())
    assert row.id == "tomaten-auf-den-augen"
    assert row.phrase == "Tomaten auf den Augen"
    assert row.definition == "blind to what is plainly obvious"
    assert row.example_usage == "Du hast Tomaten auf den Augen."
    assert row.source_attribution == "dwds"
    assert row.frequency_band == "high"
    assert row.dwds_url.startswith("https://www.dwds.de/")
    # Reserved-for-8.2 attestation fields default to None.
    assert row.attested_quote is None
    assert row.attested_source is None


def test_phrase_seed_row_accepts_example_usage_none():
    """``example_usage=None`` is the documented case for DWDS
    lemmas that ship without an ``<Example>`` child. The
    validator must accept the missing value without raising."""
    row = PhraseSeedRow.model_validate(_good_row(example_usage=None))
    assert row.example_usage is None


def test_phrase_seed_row_normalises_source_attribution_whitespace():
    """The validator strips whitespace around comma-separated
    tokens and rejoins, so ``"dwds , goethe"`` becomes
    ``"dwds,goethe"`` on round-trip. Mirrors the storage-shape
    contract (the DB column carries a normalised, comma-joined
    string)."""
    row = PhraseSeedRow.model_validate(
        _good_row(source_attribution="  dwds , goethe  ")
    )
    assert row.source_attribution == "dwds,goethe"


def test_phrase_seed_row_accepts_all_three_frequency_band_values():
    """All three valid ``frequency_band`` literals round-trip
    through ``PhraseSeedRow``. Negative coverage for the Literal:
    every valid value passes."""
    for band in ("high", "mid", "low"):
        row = PhraseSeedRow.model_validate(_good_row(frequency_band=band))
        assert row.frequency_band == band


# ---------------------------------------------------------------------------
# PhraseSeedRow — bounds enforcement
# ---------------------------------------------------------------------------


def test_phrase_seed_row_rejects_short_phrase():
    """``phrase`` is bounded 5..200 chars (PHASE-8.md item 4). A
    3-char phrase like ``"ab"`` is rejected at validation time."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(phrase="ab"))
    assert "phrase" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_long_phrase():
    """``phrase`` is bounded 5..200 chars. A 250-char phrase is
    rejected at validation time."""
    long_phrase = "a" * 201
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(phrase=long_phrase))
    assert "phrase" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_short_definition():
    """``definition`` is bounded 1..400 chars. An empty string is
    rejected (the column is NOT NULL on the DB side AND the
    Pydantic floor is 1 char, not 0)."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(definition=""))
    assert "definition" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_long_definition():
    """``definition`` is bounded 1..400 chars. A 500-char gloss is
    rejected — the cap forces the seed author to compress long
    DWDS definitions into a single tight sentence (PHASE-8.md
    gotcha #5)."""
    long_def = "x" * 401
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(definition=long_def))
    assert "definition" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_short_example_usage_when_set():
    """``example_usage`` is optional, 5..400 chars when set. A
    3-char example is rejected (the wire-level 1-char-or-greater
    floor only applies when the value is NOT None — None is the
    documented DWDS-without-<Example> shape)."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(example_usage="hm"))
    assert "example_usage" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_long_example_usage():
    """``example_usage`` is bounded 5..400 chars when set."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(example_usage="x" * 401))
    assert "example_usage" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_unknown_frequency_band():
    """``frequency_band`` is a closed ``Literal["high","mid","low"]``.
    A typo'd value like ``"Hi"`` must raise ``ValidationError``
    at parse time — the type system is the gate (PHASE-8.md
    gotcha #6)."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(frequency_band="Hi"))
    assert "frequency_band" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# PhraseSeedRow — source_attribution per-token enforcement
# ---------------------------------------------------------------------------


def test_phrase_seed_row_rejects_unknown_source_token():
    """``source_attribution`` validates each comma-separated token
    against ``PhraseSourceAttribution``. A typo like
    ``"google"`` must be rejected at parse time (the DB column
    is loose String — the Pydantic layer is the only guardrail,
    so a typo'd value would otherwise silently propagate into
    the seeded corpus)."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(
            _good_row(source_attribution="google")
        )
    assert "source_attribution" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_mixed_known_unknown_tokens():
    """A mix of one valid token and one unknown token in the same
    comma-joined string is rejected — partial writes would
    mislead future readers. ``"dwds,google"`` is a clean
    422 at parse time, not a partial-accept."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(
            _good_row(source_attribution="dwds,google")
        )
    assert "source_attribution" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_empty_source_attribution():
    """An empty ``source_attribution`` is rejected — the column
    is non-null on the DB side and the corpus always has at
    least one source."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(source_attribution=""))
    assert "source_attribution" in str(exc_info.value).lower()


def test_phrase_seed_row_accepts_phase_82_multi_token_string():
    """The Phase 8.2 attestation seed shape — ``"dwds,goethe"``
    or ``"goethe,schiller"`` — is parsed and re-joined by the
    validator. This is forward-compat for 8.2 without needing a
    Pydantic model widening."""
    row = PhraseSeedRow.model_validate(
        _good_row(source_attribution="dwds,goethe")
    )
    assert row.source_attribution == "dwds,goethe"


def test_phrase_source_attribution_literal_includes_all_four_tokens():
    """The ``PhraseSourceAttribution`` Literal must include
    ``"dwds"``, ``"goethe"``, ``"schiller"``, and ``"manual"``
    (the last is reserved for future hand-curated entries — NOT
    in 8.1, but kept on the wire contract so 8.x and Phase 9
    don't need to re-widen)."""
    assert {"dwds", "goethe", "schiller", "manual"} == set(
        PhraseSourceAttribution.__args__
    )


def test_split_source_attribution_strips_empty_tokens():
    """``_split_source_attribution`` strips whitespace per token
    and drops empty entries from trailing commas / double-
    commas. Internal helper — exercised via the validator above,
    but the unit test pins the behaviour for the next maintainer.
    """
    assert _split_source_attribution(
        "dwds, goethe, , schiller"
    ) == ["dwds", "goethe", "schiller"]


def test_phrase_source_tokens_set_matches_literal():
    """The frozen set used by the validators must agree with the
    Literal — drifting the two would surface as a confusing
    "this raises on round-trip" bug, so this test pins both
    arms of the contract."""
    assert _PHRASE_SOURCE_TOKENS == frozenset(
        PhraseSourceAttribution.__args__
    )


# ---------------------------------------------------------------------------
# id (slug PK) bounds
# ---------------------------------------------------------------------------


def test_phrase_seed_row_rejects_short_id():
    """``id`` is bounded 3..120 chars (slug PK). A 1-char slug is
    rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(id="ab"))
    assert "id" in str(exc_info.value).lower()


def test_phrase_seed_row_rejects_long_id():
    """``id`` is bounded 3..120 chars. A 130-char slug is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PhraseSeedRow.model_validate(_good_row(id="a" * 130))
    assert "id" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# PhraseOut — outbound shape
# ---------------------------------------------------------------------------


def test_phrase_out_round_trips_a_valid_row():
    """``PhraseOut`` outbound model mirrors the SQLAlchemy
    ``Phrase`` columns. Constructing it from a valid dict
    succeeds and round-trips every field, including the
    ``created_at`` timestamp."""
    payload = {
        "id": "tomaten-auf-den-augen",
        "phrase": "Tomaten auf den Augen",
        "definition": "blind to what is plainly obvious",
        "example_usage": "Du hast Tomaten auf den Augen.",
        "source_attribution": "dwds",
        "frequency_band": "high",
        "dwds_url": "https://www.dwds.de/wb/Tomaten%20auf%20den%20Augen",
        "attested_quote": None,
        "attested_source": None,
        "created_at": datetime.datetime(2026, 7, 4, 12, 0, 0),
    }
    out = PhraseOut.model_validate(payload)
    assert out.id == payload["id"]
    assert out.phrase == payload["phrase"]
    assert out.definition == payload["definition"]
    assert out.example_usage == payload["example_usage"]
    assert out.source_attribution == payload["source_attribution"]
    assert out.frequency_band == payload["frequency_band"]
    assert out.dwds_url == payload["dwds_url"]
    assert out.created_at == payload["created_at"]


def test_phrase_out_rejects_bad_source_attribution():
    """``PhraseOut`` validates ``source_attribution`` per token at
    parse time. A typo'd ``"google"`` is rejected on the outbound
    too (the validator on this model is symmetric with
    ``PhraseSeedRow``)."""
    payload = {
        "id": "x",
        "phrase": "Tomaten auf den Augen",
        "definition": "blind to what is plainly obvious",
        "example_usage": None,
        "source_attribution": "google",
        "frequency_band": "high",
        "dwds_url": None,
        "attested_quote": None,
        "attested_source": None,
        "created_at": datetime.datetime(2026, 7, 4, 12, 0, 0),
    }
    with pytest.raises(ValidationError) as exc_info:
        PhraseOut.model_validate(payload)
    assert "source_attribution" in str(exc_info.value).lower()


def test_phrase_out_rejects_unknown_frequency_band():
    """``PhraseOut`` enforces the ``PhraseFrequencyBand`` Literal
    (``"high" / "mid" / "low"``). A typo'd value is rejected
    on the outbound view too."""
    payload = {
        "id": "x",
        "phrase": "Tomaten auf den Augen",
        "definition": "blind to what is plainly obvious",
        "example_usage": None,
        "source_attribution": "dwds",
        "frequency_band": "high-plus",  # NOT in the literal
        "dwds_url": None,
        "attested_quote": None,
        "attested_source": None,
        "created_at": datetime.datetime(2026, 7, 4, 12, 0, 0),
    }
    with pytest.raises(ValidationError) as exc_info:
        PhraseOut.model_validate(payload)
    assert "frequency_band" in str(exc_info.value).lower()


def test_phrase_frequency_band_literal_is_closed_three_value():
    """The ``PhraseFrequencyBand`` Literal is locked to the three
    documented values. A future widening (e.g. ``"very-high"``)
    must be a deliberate PR — this test pins the contract so
    the change is loud."""
    assert {"high", "mid", "low"} == set(PhraseFrequencyBand.__args__)
