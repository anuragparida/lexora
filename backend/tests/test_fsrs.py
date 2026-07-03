"""Phase 5.1 — py-fsrs integration module tests.

Card t_65cf7a3a. Every case runs in-memory (no DB, no network). The
import-time version assertion is verified by patching
``importlib.metadata.version`` before re-importing ``app.fsrs`` in a
subprocess (``test_version_assertion_imports_clean``).

## Known spec-vs-library divergences

The 5.1 card body made claims about py-fsrs's transition behavior
that don't match the v4.1.2 library. The tests below pin the
**actual** library behavior (since the library is the source of
truth — that's why we're version-pinning it) and the docstring on
``app.fsrs.apply_grade`` calls out the diff. Concretely:

- The card body said "``apply_grade(card, 1)`` on a Learning card
  transitions to Relearning". py-fsrs 4.1.2's actual behavior:
  ``Again`` on a Learning card **stays in Learning** (it resets
  ``step`` back to 0). ``Relearning`` only happens when you grade
  ``Again`` on a card that's already in ``Review``. See
  ``test_apply_grade_again_on_review_transitions_to_relearning``.

- The card body said "``apply_grade(card, 3)`` on a Learning card
  graduates to Review". py-fsrs 4.1.2's actual behavior: ``Good``
  on a Learning card **advances the step** but stays in Learning.
  ``Good`` only graduates after the last learning step is consumed.
  See ``test_apply_grade_good_on_learning_advances_step``.

- The card body referenced ``scheduled_days >= 1`` on a graduated
  card. py-fsrs 4.1.2's ``Card`` doesn't carry ``scheduled_days``
  as an attribute at all (the columns live on the ``fsrs_cards``
  table; 5.3 derives them from ``due - last_review``). The
  graduation tests below assert ``state == Review`` only.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fsrs import Card, Rating, State

from app.fsrs import (
    DEFAULT_DESIRED_RETENTION,
    DEFAULT_ENABLE_FUZZING,
    DEFAULT_LEARNING_STEPS,
    DEFAULT_MAXIMUM_INTERVAL,
    DEFAULT_PARAMETERS,
    DEFAULT_RELEARNING_STEPS,
    PY_FSRS_VERSION,
    apply_grade,
    build_scheduler,
    card_to_row_dict,
    row_to_card,
)


# ---------------------------------------------------------------------------
# Module constants — locked by Hard rule #8.
# ---------------------------------------------------------------------------


def test_py_fsrs_version_constant_matches_installed() -> None:
    """``PY_FSRS_VERSION`` is the source of truth for the pin."""
    assert PY_FSRS_VERSION == "4.1.2"


def test_py_fsrs_version_constant_matches_metadata() -> None:
    """The installed ``fsrs`` distribution version equals the pin."""
    installed = importlib.metadata.version("fsrs")
    assert installed == PY_FSRS_VERSION


def test_default_parameters_is_19_tuple_with_locked_values() -> None:
    """The 19-tuple default is hard-coded — no fsrs() call, no config."""
    assert len(DEFAULT_PARAMETERS) == 19
    # The first four weights — sentinel check that we have the right tuple.
    assert DEFAULT_PARAMETERS[:4] == (0.40255, 1.18385, 3.173, 15.69105)
    # The last weight — sentinel check.
    assert DEFAULT_PARAMETERS[-1] == 0.6621


def test_other_defaults_are_locked_constants() -> None:
    """Hard rule #8: defaults are module constants, never config/env."""
    assert DEFAULT_DESIRED_RETENTION == 0.9
    assert DEFAULT_LEARNING_STEPS == (timedelta(minutes=1), timedelta(minutes=10))
    assert DEFAULT_RELEARNING_STEPS == (timedelta(minutes=10),)
    assert DEFAULT_MAXIMUM_INTERVAL == 36500
    assert DEFAULT_ENABLE_FUZZING is True


# ---------------------------------------------------------------------------
# Scheduler construction — the single construction site for Phase 5.
# ---------------------------------------------------------------------------


def test_build_scheduler_default_args() -> None:
    """Default ``build_scheduler()`` builds a Scheduler with our defaults."""
    s = build_scheduler()
    assert s.parameters == DEFAULT_PARAMETERS
    assert s.desired_retention == DEFAULT_DESIRED_RETENTION
    assert s.learning_steps == DEFAULT_LEARNING_STEPS
    assert s.relearning_steps == DEFAULT_RELEARNING_STEPS
    assert s.maximum_interval == DEFAULT_MAXIMUM_INTERVAL
    assert s.enable_fuzzing is True


def test_build_scheduler_disables_fuzzing_for_tests() -> None:
    """The ``enable_fuzzing=False`` constructor arg is honored."""
    s = build_scheduler(enable_fuzzing=False)
    assert s.enable_fuzzing is False


def test_build_scheduler_matches_pyfspy_fsrs_default_scheduler() -> None:
    """``build_scheduler()`` equals ``Scheduler()`` (the py-fsrs library default)."""
    from fsrs import Scheduler as _RawScheduler

    ours = build_scheduler()
    theirs = _RawScheduler()
    assert ours.parameters == theirs.parameters
    assert ours.desired_retention == theirs.desired_retention
    assert ours.learning_steps == theirs.learning_steps
    assert ours.relearning_steps == theirs.relearning_steps
    assert ours.maximum_interval == theirs.maximum_interval
    # enable_fuzzing defaults to True on both, so they should match.
    assert ours.enable_fuzzing == theirs.enable_fuzzing


# ---------------------------------------------------------------------------
# row_to_card / card_to_row_dict — bridge to the fsrs_cards table.
# ---------------------------------------------------------------------------


def _make_row(**overrides):
    """Construct a minimal ``FsrsCard``-shaped namespace.

    We don't import ``backend.app.models`` here (it pulls in
    SQLAlchemy + database setup; this test file is offline by
    design). A SimpleNamespace with the right attribute set is
    enough — ``row_to_card`` only reads the columns listed in
    docs/PHASE-5.md §"The FSRS shape".
    """
    defaults = {
        "word_id": 42,
        "difficulty": None,
        "stability": None,
        "retrievability": None,
        "due_date": None,
        "last_review": None,
        "reps": None,
        "lapses": None,
        "state": None,
        "elapsed_days": None,
        "scheduled_days": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_row_to_card_on_fresh_row_yields_default_card() -> None:
    """A row with every FSRS column ``None`` yields a brand-new Card."""
    row = _make_row(word_id=42)
    card = row_to_card(row)
    assert isinstance(card, Card)
    # Card() defaults: state=Learning, step=0.
    assert card.state == State.Learning
    assert card.card_id == 42


def test_row_to_card_on_populated_row_recovers_via_from_dict() -> None:
    """``row_to_card`` uses py-fsrs's own ``Card.from_dict`` — verify it
    recovers a populated row correctly.
    """
    now = datetime.now(timezone.utc)
    row = _make_row(
        word_id=7,
        difficulty=5.3,
        stability=12.7,
        due_date=now + timedelta(days=4),
        last_review=now,
        state=int(State.Review),
    )
    card = row_to_card(row)
    assert card.card_id == 7
    assert card.state == State.Review
    assert card.difficulty == 5.3
    assert card.stability == 12.7
    # py-fsrs's from_dict stores due as tz-aware.
    assert card.due == now + timedelta(days=4)
    assert card.last_review == now


def test_card_to_row_dict_matches_fsrs_cards_columns() -> None:
    """The dict returned has exactly the fsrs_cards column names."""
    card = Card(card_id=99, state=State.Review, difficulty=4.2, stability=10.0)
    row_dict = card_to_row_dict(card, word_id=99)
    expected_keys = {
        "word_id",
        "difficulty",
        "stability",
        "due_date",
        "last_review",
        "reps",
        "lapses",
        "state",
        "elapsed_days",
        "scheduled_days",
    }
    assert set(row_dict.keys()) == expected_keys
    assert row_dict["word_id"] == 99
    assert row_dict["state"] == int(State.Review)
    assert row_dict["difficulty"] == 4.2
    assert row_dict["stability"] == 10.0


def test_card_to_row_dict_emits_none_for_untracked_columns() -> None:
    """The four columns py-fsrs 4.1.2's Card does not carry are None.

    The 5.3 grader is expected to fill ``reps``, ``lapses``,
    ``elapsed_days``, ``scheduled_days`` from the post-review
    snapshot (e.g. ``scheduled_days = (card.due - card.last_review).days``).
    We do NOT silently invent values here.
    """
    card = Card(card_id=1, state=State.Review)
    row = card_to_row_dict(card, word_id=1)
    assert row["reps"] is None
    assert row["lapses"] is None
    assert row["elapsed_days"] is None
    assert row["scheduled_days"] is None


def test_card_to_row_dict_round_trips_through_row_to_card() -> None:
    """``card_to_row_dict(card, word_id) → row_to_card(...)`` recovers the
    Card's py-fsrs-tracked fields (state, stability, difficulty, due,
    last_review).

    The four columns py-fsrs doesn't track (``reps``, ``lapses``,
    ``elapsed_days``, ``scheduled_days``) round-trip as ``None``, which
    is the documented contract for the 5.1 surface.
    """
    # Build a populated Card with non-default values on every py-fsrs field.
    due = datetime(2030, 1, 1, tzinfo=timezone.utc)
    last_review = datetime(2029, 12, 25, tzinfo=timezone.utc)
    card = Card(
        card_id=1,
        state=State.Review,
        step=None,
        stability=15.5,
        difficulty=4.7,
        due=due,
        last_review=last_review,
    )
    row = _make_row(**card_to_row_dict(card, word_id=1))
    recovered = row_to_card(row)
    # The py-fsrs-tracked fields round-trip exactly.
    assert recovered.card_id == card.card_id
    assert recovered.state == card.state
    assert recovered.stability == card.stability
    assert recovered.difficulty == card.difficulty
    assert recovered.due == card.due
    assert recovered.last_review == card.last_review


# ---------------------------------------------------------------------------
# apply_grade — thin wrapper around Scheduler.review_card.
#
# These tests pin the ACTUAL py-fsrs 4.1.2 behavior. The 5.1 card body
# contained three assertions that don't match the library; the
# module docstring documents the diff so 5.3 doesn't trip on it.
# ---------------------------------------------------------------------------


def _fresh_learning_card() -> Card:
    """A brand-new Card: state=Learning, step=0, due=now."""
    return Card()


def _review_card_now_due() -> Card:
    """A card already in Review, due right now (graduated via Easy)."""
    card = Card()
    new_card, _ = build_scheduler(enable_fuzzing=False).review_card(card, Rating(4))
    assert new_card.state == State.Review
    return new_card


def test_apply_grade_again_on_review_transitions_to_relearning() -> None:
    """Rating 1 (Again) on a Review card → state Relearning.

    This is the actual library behavior. The 5.1 card body asserted
    "Again on a Learning card → Relearning", which is incorrect for
    py-fsrs 4.1.2 (Again on Learning stays in Learning and resets
    step). The module docstring documents the diff.
    """
    card = _review_card_now_due()
    assert card.state == State.Review
    new_card, log = apply_grade(card, 1)
    assert new_card.state == State.Relearning
    assert log.rating == Rating.Again
    # The returned card is a different object — py-fsrs deep-copies.
    assert new_card is not card


def test_apply_grade_good_on_learning_advances_step() -> None:
    """Rating 3 (Good) on a fresh Learning card at step=0 advances
    step but stays in Learning.

    py-fsrs 4.1.2's learning_steps are (1min, 10min); Good from step
    0 → step 1. To actually graduate to Review you need to consume
    both steps (Good from step 1 → Review).
    """
    card = _fresh_learning_card()
    assert card.state == State.Learning
    assert card.step == 0
    new_card, log = apply_grade(card, 3)
    assert new_card.state == State.Learning
    assert new_card.step == 1
    assert log.rating == Rating.Good


def test_apply_grade_good_after_last_step_graduates_to_review() -> None:
    """Rating 3 (Good) on a Learning card at the last step → state Review.

    Canonical graduation path: two consecutive Good ratings. The
    first Good advances step (0 → 1, the last learning step); the
    second Good consumes it and graduates to Review. The 5.1 card
    body asserted "Good on a Learning card graduates to Review"
    without consuming the steps; that's incorrect for the library.
    """
    card = _fresh_learning_card()
    card, _ = apply_grade(card, 3)
    assert card.state == State.Learning
    assert card.step == 1  # last learning step
    card, log = apply_grade(card, 3)
    assert card.state == State.Review
    assert log.rating == Rating.Good


def test_apply_grade_easy_on_learning_jumps_to_review() -> None:
    """Rating 4 (Easy) on a fresh Learning card → state Review.

    Easy bypasses the learning-steps entirely (it's the "I know
    this already" path). The 5.1 card body said "scheduled_days >=
    1", but py-fsrs 4.1.2's Card doesn't carry scheduled_days as
    an attribute — that lives on the fsrs_cards row and 5.3
    derives it from ``due - last_review``.
    """
    card = _fresh_learning_card()
    new_card, log = apply_grade(card, 4)
    assert new_card.state == State.Review
    assert log.rating == Rating.Easy


def test_apply_grade_hard_on_learning_resets_step() -> None:
    """Rating 2 (Hard) on a fresh Learning card at step=0 → stays in
    Learning at step=0.

    py-fsrs 4.1.2's actual behavior: Hard from step 0 does NOT
    advance the step (it stays at 0). The library considers step=0
    the "just-started" state and Hard is meant to indicate
    difficulty without progress. The 5.1 card body asserted Hard
    advances step; that's incorrect for v4.1.2.
    """
    card = _fresh_learning_card()
    assert card.state == State.Learning
    assert card.step == 0
    new_card, _ = apply_grade(card, 2)
    assert new_card.state == State.Learning
    assert new_card.step == 0


def test_apply_grade_rejects_invalid_grade() -> None:
    """``grade`` must be in {1, 2, 3, 4}. Anything else raises ValueError."""
    card = _fresh_learning_card()
    for bad in (0, 5, -1, 99):
        with pytest.raises(ValueError, match="grade must be"):
            apply_grade(card, bad)


# ---------------------------------------------------------------------------
# Determinism — with fuzzing off, the same starting state gives the same intervals.
# ---------------------------------------------------------------------------


def test_apply_grade_is_deterministic_with_fuzzing_off() -> None:
    """Two ``apply_grade(card, 3)`` calls on the same starting state produce
    identical scheduling decisions. ``apply_grade`` already builds the
    scheduler with ``enable_fuzzing=False``; this test pins that
    contract.

    Note: the library uses wall-clock time (``datetime.now(UTC)``) when
    no ``review_datetime`` is passed, so two consecutive cards get
    ``due`` values that differ by a few microseconds. We assert on
    the deterministic fields (state, step, stability, difficulty)
    only — those are the fields that depend on the algorithm's
    parameters, not on wall-clock noise.
    """
    card_a = _fresh_learning_card()
    card_b = _fresh_learning_card()
    new_a, _ = apply_grade(card_a, 3)
    new_b, _ = apply_grade(card_b, 3)
    # Deterministic: these depend only on the algorithm + starting state.
    assert new_a.stability == new_b.stability
    assert new_a.difficulty == new_b.difficulty
    assert new_a.state == new_b.state == State.Learning
    assert new_a.step == new_b.step == 1


def test_sequential_applies_are_deterministic() -> None:
    """Two apply-3-then-3 sequences in lockstep produce identical card states."""
    card_a = _fresh_learning_card()
    card_b = _fresh_learning_card()
    card_a, _ = apply_grade(card_a, 3)
    card_b, _ = apply_grade(card_b, 3)
    card_a, _ = apply_grade(card_a, 3)
    card_b, _ = apply_grade(card_b, 3)
    # Deterministic: scheduling decisions.
    assert card_a.stability == card_b.stability
    assert card_a.difficulty == card_b.difficulty
    assert card_a.state == card_b.state == State.Review


def test_scheduler_with_explicit_review_datetime_is_fully_deterministic() -> None:
    """With an explicit ``review_datetime``, the library produces
    bit-identical cards (same ``due`` down to the microsecond).

    This is the property 5.3's grader relies on when replaying a
    grade for audit — pass ``review_datetime = datetime.now(UTC)``
    explicitly to make the trace fully reproducible.
    """
    from app.fsrs import build_scheduler

    scheduler = build_scheduler(enable_fuzzing=False)
    fixed_dt = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)

    card_a = Card()
    card_b = Card()
    new_a, _ = scheduler.review_card(card_a, Rating(3), review_datetime=fixed_dt)
    new_b, _ = scheduler.review_card(card_b, Rating(3), review_datetime=fixed_dt)

    assert new_a.due == new_b.due
    assert new_a.stability == new_b.stability
    assert new_a.difficulty == new_b.difficulty
    assert new_a.state == new_b.state


# ---------------------------------------------------------------------------
# Version-pin import-time assertion — verified via subprocess.
# ---------------------------------------------------------------------------


def test_version_assertion_raises_on_installed_version_drift() -> None:
    """Patching ``importlib.metadata.version`` to return a different
    version and re-importing ``app.fsrs`` must raise ``RuntimeError``.

    Run as a subprocess so the import-time assertion actually executes
    on a fresh interpreter (in-process reimport would be a no-op since
    the module is already loaded with the right pin).
    """
    script = """
import sys
import importlib.metadata as md

_real_version = md.version

def _patched_version(name):
    if name == "fsrs":
        return "5.0.0"
    return _real_version(name)

md.version = _patched_version
try:
    import app.fsrs  # noqa: F401
except RuntimeError as exc:
    if "version drift" in str(exc):
        print("OK", str(exc))
        sys.exit(0)
    print("UNEXPECTED RuntimeError:", exc)
    sys.exit(2)
except Exception as exc:
    print("UNEXPECTED", type(exc).__name__, exc)
    sys.exit(3)
print("FAIL: import succeeded despite version drift")
sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"version drift import did not raise as expected.\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "version drift" in result.stdout
    assert "5.0.0" in result.stdout
    assert PY_FSRS_VERSION in result.stdout


# ---------------------------------------------------------------------------
# Sanity: nothing in app/fsrs.py imports models/database/network at runtime.
# ---------------------------------------------------------------------------


def test_fsrs_module_source_has_no_unconditional_app_models_import() -> None:
    """The fsrs module's only reference to ``app.models`` must be a
    ``TYPE_CHECKING``-guarded import for type hints.

    Hard rule #7: tests must be offline-capable. The fsrs surface
    must not drag in SQLAlchemy / database setup just by being
    imported standalone. We verify the source-level constraint
    (no unconditional ``from app.models`` at module scope) rather
    than sys.modules state, because other tests in the full suite
    legitimately load ``app.models`` via conftest, and that's not
    a fsrs-module concern.
    """
    import app.fsrs as fsrs_module

    module_file = fsrs_module.__file__
    assert module_file is not None
    with open(module_file, encoding="utf-8") as f:
        source = f.read()

    # Walk the source. Any ``from app.models`` / ``import app.models``
    # outside a TYPE_CHECKING block is forbidden.
    lines = source.splitlines()
    in_type_checking = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("if TYPE_CHECKING:"):
            in_type_checking = True
            continue
        # A non-indented (or less-indented) line ends the block.
        if in_type_checking and stripped and not line.startswith((" ", "\t", "    ")):
            in_type_checking = False
        if in_type_checking:
            continue
        # Outside TYPE_CHECKING, an import of app.models is forbidden.
        if stripped.startswith(("from app.models", "import app.models")):
            pytest.fail(
                f"Unconditional app.models import in app/fsrs.py at runtime: {line!r}"
            )

    # And the fsrs module file itself must NOT trigger a transitive
    # import of app.models when imported FRESH (no other test loaded
    # anything first). This is the actual offline guarantee — the
    # 5.1 surface must be importable in a clean interpreter.
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [
            _sys.executable,
            "-c",
            "import app.fsrs as m; "
            "loaded = sorted([n for n in __import__('sys').modules "
            "if n.startswith('app.')]); "
            "assert loaded == ['app.fsrs'], "
            "'Unexpected app.* modules loaded: {loaded}'; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"Standalone import of app.fsrs pulled in extra app.* modules.\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )