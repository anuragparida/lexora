"""Tests for Phase 10.1 — phrase_pairs Pydantic schema.

Card: t_18c90a68.

Coverage map (mirrors the card body's "Verification" list and
the Phase 8.1 schema-test pattern):

1. ``PhrasePairSeedRow`` Pydantic enforcement:
   - ``phrase_a_id`` / ``phrase_b_id`` are slug-shaped FKs
     (lowercase-hyphenated, 3-120 chars).
   - ``relation`` is the 4-way Literal — typo'd literals
     are caught here, not at INSERT time.
   - ``attested_pair`` is a bool, default ``False``.
   - The ``phrase_a_id != phrase_b_id`` cross-field rule
     fires when the two slugs are equal (mirrors the DB
     CHECK constraint).

2. ``PhrasePairOut`` outbound view: same field set, plus
   ``id: int`` and ``created_at: datetime``. The ``relation``
   validator catches unknown relation values from a raw-SQL
   row.

3. ``PhrasePairSeedManifest`` wrapper: ``pairs`` is a list
   (may be empty), ``attested_pairs_filename`` defaults to
   ``"attested_pairs.json"`` and rejects path separators.

4. ``phrase_pairs_fixture.json`` parses cleanly via the
   ``PhrasePairSeedRow`` validator — mirrors the Phase 8.1
   ``phrases_fixture.json`` discipline.

Hermetic: no DB connection, no LLM call. Pure Pydantic v2 unit
tests on the ``app.schemas`` module.

Run from ``backend/``::

    uv run pytest -q tests/test_phrase_pairs_schema.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    PhrasePairRelation,
    PhrasePairSeedManifest,
    PhrasePairSeedRow,
    PhrasePairOut,
)


# ---------------------------------------------------------------------------
# Tiny fixture builders — keep the test matrix legible.
# ---------------------------------------------------------------------------


def _good_row(**overrides) -> dict:
    """Return a valid ``PhrasePairSeedRow`` payload (with overrides)."""
    base = {
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "da-steppt-der-baer",
        "relation": "paraphrase",
        "attested_pair": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# PhrasePairSeedRow — happy paths
# ---------------------------------------------------------------------------


def test_phrase_pair_seed_row_accepts_a_valid_payload():
    """The simplest valid row round-trips cleanly through
    ``PhrasePairSeedRow``. Negative coverage for the union of
    fields: if this test fails, the column list has drifted
    from the SQLAlchemy model."""
    row = PhrasePairSeedRow.model_validate(_good_row())
    assert row.phrase_a_id == "tomaten-auf-den-augen"
    assert row.phrase_b_id == "da-steppt-der-baer"
    assert row.relation == "paraphrase"
    assert row.attested_pair is False


def test_phrase_pair_seed_row_default_attested_pair_is_false():
    """``attested_pair`` defaults to ``False`` so the seed
    script's candidate-pool rows don't have to set it."""
    row = PhrasePairSeedRow.model_validate({
        "phrase_a_id": "ins-blaue-hinein",
        "phrase_b_id": "die-kirche-im-dorf-lassen",
        "relation": "equivalent",
    })
    assert row.attested_pair is False


def test_phrase_pair_seed_row_accepts_all_four_relation_values():
    """All four valid ``relation`` literals round-trip through
    ``PhrasePairSeedRow``. Negative coverage for the Literal:
    every valid value passes."""
    for relation in (
        "equivalent", "paraphrase", "related", "unrelated"
    ):
        row = PhrasePairSeedRow.model_validate(
            _good_row(relation=relation)
        )
        assert row.relation == relation


def test_phrase_pair_seed_row_rejects_unknown_relation():
    """A typo'd ``relation`` value (e.g. ``"equiv"``) is caught
    at parse time, not at INSERT time — this is the
    wire-level guardrail (PHASE-7 gotcha #6, restated for
    Phase 10)."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedRow.model_validate(
            _good_row(relation="equiv")
        )
    assert "relation" in str(exc_info.value)


def test_phrase_pair_seed_row_rejects_uppercase_slug():
    """Slug FKs are lowercase-hyphenated ASCII. An uppercase
    character (``Tomaten-auf-den-Augen``) is caught here, not
    at INSERT time where it would silently fail the FK
    constraint."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedRow.model_validate(_good_row(
            phrase_a_id="Tomaten-auf-den-Augen",
        ))
    assert "phrase_a_id" in str(exc_info.value)


def test_phrase_pair_seed_row_rejects_underscore_in_slug():
    """Slugs are hyphen-separated (no underscores) — mirrors the
    Phase 8.1 ``phrases.id`` discipline. An underscore-bearing
    slug is rejected here."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedRow.model_validate(_good_row(
            phrase_b_id="da_steppt_der_baer",
        ))
    assert "phrase_b_id" in str(exc_info.value)


def test_phrase_pair_seed_row_rejects_slug_too_short():
    """Slugs must be 3-120 chars (the DB column cap). A
    2-char slug is rejected here."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedRow.model_validate(_good_row(
            phrase_a_id="ab",
        ))
    assert "phrase_a_id" in str(exc_info.value)


def test_phrase_pair_seed_row_rejects_self_pair():
    """Hard rule: ``phrase_a_id != phrase_b_id``. The cross-
    field validator catches the violation at Pydantic parse
    time (mirrors the DB CHECK constraint)."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedRow.model_validate({
            "phrase_a_id": "tomaten-auf-den-augen",
            "phrase_b_id": "tomaten-auf-den-augen",
            "relation": "paraphrase",
        })
    assert "distinct" in str(exc_info.value)


def test_phrase_pair_seed_row_rejects_non_string_relation():
    """Defensive guard — ``relation`` must be a string. A
    Python ``True`` or ``42`` is rejected at Pydantic parse
    time."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedRow.model_validate({
            "phrase_a_id": "tomaten-auf-den-augen",
            "phrase_b_id": "da-steppt-der-baer",
            "relation": True,
        })
    assert "relation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# PhrasePairOut — outbound shape
# ---------------------------------------------------------------------------


def test_phrase_pair_out_accepts_a_complete_row():
    """``PhrasePairOut`` mirrors the SQLAlchemy model column-
    for-column. The ``id`` and ``created_at`` fields round-trip
    cleanly via Pydantic."""
    from datetime import datetime

    row = PhrasePairOut.model_validate({
        "id": 42,
        "phrase_a_id": "tomaten-auf-den-augen",
        "phrase_b_id": "da-steppt-der-baer",
        "relation": "equivalent",
        "attested_pair": True,
        "created_at": datetime(2026, 7, 6, 12, 0, 0),
    })
    assert row.id == 42
    assert row.relation == "equivalent"
    assert row.attested_pair is True
    assert row.created_at == datetime(2026, 7, 6, 12, 0, 0)


def test_phrase_pair_out_rejects_unknown_relation():
    """Defensive: even if the DB stored an unknown relation
    literal (a future-proofing scenario), the wire layer
    catches it here, not at serializer time."""
    from datetime import datetime

    with pytest.raises(ValidationError) as exc_info:
        PhrasePairOut.model_validate({
            "id": 1,
            "phrase_a_id": "tomaten-auf-den-augen",
            "phrase_b_id": "da-steppt-der-baer",
            "relation": "antonym",  # not in the 4-way literal
            "attested_pair": False,
            "created_at": datetime(2026, 7, 6, 12, 0, 0),
        })
    assert "relation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# PhrasePairSeedManifest — wrapper
# ---------------------------------------------------------------------------


def test_phrase_pair_seed_manifest_accepts_empty_pairs_list():
    """Phase 10.1 ships ``attested_pairs.json`` empty (Phase
    10.4 populates it). The manifest accepts an empty
    ``pairs`` list without complaint."""
    manifest = PhrasePairSeedManifest.model_validate({
        "pairs": [],
        "attested_pairs_filename": "attested_pairs.json",
    })
    assert manifest.pairs == []
    assert manifest.attested_pairs_filename == "attested_pairs.json"


def test_phrase_pair_seed_manifest_default_filename():
    """The default ``attested_pairs_filename`` is
    ``"attested_pairs.json"`` — callers don't have to repeat
    the filename."""
    manifest = PhrasePairSeedManifest.model_validate({
        "pairs": [],
    })
    assert manifest.attested_pairs_filename == "attested_pairs.json"


def test_phrase_pair_seed_manifest_rejects_path_separator():
    """The filename must be a bare filename (no path
    separators). A path like ``"data/attested.json"`` is
    rejected here so the seed script doesn't have to defend
    against it."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedManifest.model_validate({
            "pairs": [],
            "attested_pairs_filename": "data/attested_pairs.json",
        })
    assert "filename" in str(exc_info.value)


def test_phrase_pair_seed_manifest_rejects_absolute_filename():
    """An absolute path in the filename field is also rejected
    (forward-slash is the trigger)."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedManifest.model_validate({
            "pairs": [],
            "attested_pairs_filename": "/etc/attested.json",
        })
    assert "filename" in str(exc_info.value)


def test_phrase_pair_seed_manifest_validates_each_pair():
    """Every entry in ``pairs`` is validated against
    ``PhrasePairSeedRow``. A typo'd slug in one row causes
    the whole manifest to fail (no partial validation)."""
    with pytest.raises(ValidationError) as exc_info:
        PhrasePairSeedManifest.model_validate({
            "pairs": [
                {
                    "phrase_a_id": "tomaten-auf-den-augen",
                    "phrase_b_id": "da-steppt-der-baer",
                    "relation": "paraphrase",
                    "attested_pair": False,
                },
                {
                    # Row 2: invalid slug (uppercase)
                    "phrase_a_id": "BadSlug",
                    "phrase_b_id": "die-daumen-druecken",
                    "relation": "related",
                    "attested_pair": False,
                },
            ],
        })
    assert "phrase_a_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Fixture test — phrase_pairs_fixture.json parses via PhrasePairSeedRow
# ---------------------------------------------------------------------------


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "phrase_pairs_fixture.json"
)


def test_phrase_pairs_fixture_exists():
    """The fixture JSON ships with the test bundle. If it's
    missing, the schema-test coverage is incomplete."""
    assert FIXTURE_PATH.exists(), f"missing {FIXTURE_PATH}"


def test_phrase_pairs_fixture_parses_as_list():
    """The fixture is a JSON array of seed row dicts (not
    JSON-Lines, not a single object). Mirrors the seed
    script's expectation documented in its docstring."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_phrase_pairs_fixture_every_row_validates():
    """Every row in the bundled fixture validates through
    ``PhrasePairSeedRow``. A typo'd slug or invalid relation
    is caught here, not at INSERT time."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    for i, row in enumerate(payload, 1):
        PhrasePairSeedRow.model_validate(row)


def test_phrase_pairs_fixture_has_distinct_pairs():
    """Every row in the fixture has
    ``phrase_a_id != phrase_b_id`` (the
    self-pair hard rule)."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    for i, row in enumerate(payload, 1):
        assert row["phrase_a_id"] != row["phrase_b_id"], (
            f"row {i}: phrase_a_id == phrase_b_id ({row['phrase_a_id']!r})"
        )


def test_phrase_pairs_fixture_covers_all_four_relations():
    """The fixture exercises each of the four ``relation``
    literals at least once — mirrors the Phase 8.1
    ``phrases_fixture.json`` discipline (every frequency
    band has at least one row)."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    relations = {row["relation"] for row in payload}
    assert {
        "equivalent", "paraphrase", "related", "unrelated"
    }.issubset(relations), (
        f"fixture missing some relation literals; got {relations!r}"
    )
