r"""py-fsrs integration layer for the Lexora spaced-repetition loop.

Phase 5.1 (card t_65cf7a3a). The single construction site for the
``Scheduler`` used by Phase 5.3 (the grade endpoint) and Phase 5.4
(the due-queue endpoint). Bridges the SQLAlchemy ``FsrsCard`` row to
py-fsrs's ``Card`` via the dictionary serializer built into the
library (``Card.to_dict`` / ``Card.from_dict``).

## Hard rules encoded here

1. **Version pin.** ``fsrs==4.1.2`` is the last release before the
   5.x breaking change to 21-parameter weights and renamed serializer
   methods (v6.x renamed ``to_dict``/``from_dict`` to ``to_json``/
   ``from_json``). The pin lives as ``PY_FSRS_VERSION`` and is
   asserted on import — a wrong-version install fails immediately
   with a ``RuntimeError`` rather than corrupting schedules silently.
2. **Defaults only.** All FSRS hyperparameters are module constants
   below. Never config, never env. ``git grep -n "getenv.*FSRS\|getenv.*RETENTION"``
   must return nothing (Hard rule #8).
3. **Single scheduler construction site.** ``build_scheduler`` is the
   only place a ``Scheduler`` is built in Phase 5. ``git grep -n
   "Scheduler(" backend/app/`` returns only this module.

## Out of scope for 5.1

- ``grade_logs`` audit table (5.2).
- HTTP routes (5.3, 5.4).
- Langfuse wrapping on the grade path (5.3).
- Any change to ``backend/app/models.py``, ``backend/app/main.py``,
  ``backend/app/cloze.py``, ``backend/app/llm.py`` (Hard rule from
  the card body).
"""

from __future__ import annotations

import importlib.metadata
from datetime import timedelta
from typing import TYPE_CHECKING

from fsrs import Card, Rating, ReviewLog, Scheduler, State

if TYPE_CHECKING:
    from app.models import FsrsCard


# ---------------------------------------------------------------------------
# Version pin — fails fast on a wrong-version install.
# ---------------------------------------------------------------------------

PY_FSRS_VERSION = "4.1.2"

_installed_py_fsrs_version = importlib.metadata.version("fsrs")
if _installed_py_fsrs_version != PY_FSRS_VERSION:
    raise RuntimeError(
        "py-fsrs version drift: installed="
        f"{_installed_py_fsrs_version} pinned={PY_FSRS_VERSION}"
    )


# ---------------------------------------------------------------------------
# Defaults — hard-coded constants, never config, never env.
# These match py-fsrs v4.1.2's Scheduler.__init__ defaults verbatim so
# that ``Scheduler(...)`` with no args produces the same object as ours.
# ---------------------------------------------------------------------------

# The 19-tuple FSRS default weights (py-fsrs v4.1.2's default scheduler).
DEFAULT_PARAMETERS: tuple[float, ...] = (
    0.40255,
    1.18385,
    3.173,
    15.69105,
    7.1949,
    0.5345,
    1.4604,
    0.0046,
    1.54575,
    0.1192,
    1.01925,
    1.9395,
    0.11,
    0.29605,
    2.2698,
    0.2315,
    2.9898,
    0.51655,
    0.6621,
)

DEFAULT_DESIRED_RETENTION: float = 0.9
DEFAULT_LEARNING_STEPS: tuple[timedelta, ...] = (
    timedelta(minutes=1),
    timedelta(minutes=10),
)
DEFAULT_RELEARNING_STEPS: tuple[timedelta, ...] = (timedelta(minutes=10),)
DEFAULT_MAXIMUM_INTERVAL: int = 36500
DEFAULT_ENABLE_FUZZING: bool = True


# ---------------------------------------------------------------------------
# Scheduler construction — the single construction site for Phase 5.
# ---------------------------------------------------------------------------


def build_scheduler(enable_fuzzing: bool = DEFAULT_ENABLE_FUZZING) -> Scheduler:
    """Construct the Phase 5 ``Scheduler`` with the locked defaults.

    The single construction site for the whole phase. Production code
    calls this with no args (so ``enable_fuzzing=True``); tests call
    it with ``enable_fuzzing=False`` so two ``apply_grade`` calls on
    the same starting state produce identical intervals (determinism).
    """
    return Scheduler(
        parameters=DEFAULT_PARAMETERS,
        desired_retention=DEFAULT_DESIRED_RETENTION,
        learning_steps=DEFAULT_LEARNING_STEPS,
        relearning_steps=DEFAULT_RELEARNING_STEPS,
        maximum_interval=DEFAULT_MAXIMUM_INTERVAL,
        enable_fuzzing=enable_fuzzing,
    )


# ---------------------------------------------------------------------------
# Row <-> Card bridge.
#
# The fsrs_cards row is what py-fsrs reads on every grade. The schema
# is fixed by Phase 0 (backend/app/models.py:86-110); Phase 5 populates
# it. The mapping table in docs/PHASE-5.md "The FSRS shape" section
# is authoritative.
#
# | fsrs_cards column | py-fsrs Card field |
# |-------------------|--------------------|
# | word_id           | (FK only — not in Card) |
# | difficulty        | card.difficulty     |
# | stability         | card.stability      |
# | retrievability    | (derived — not stored) |
# | due_date          | card.due            |
# | last_review       | card.last_review    |
# | reps              | card.reps           |
# | lapses            | card.lapses         |
# | state             | card.state          |
# | elapsed_days      | card.elapsed_days   |
# | scheduled_days    | card.scheduled_days |
# ---------------------------------------------------------------------------


def row_to_card(row: "FsrsCard") -> Card:
    """Reconstruct a py-fsrs ``Card`` from a ``FsrsCard`` SQLAlchemy row.

    The py-fsrs ``Card`` carries its own dictionary serializer
    (``to_dict`` / ``from_dict``) that knows how to round-trip
    ``card_id``, ``state``, ``step``, ``stability``, ``difficulty``,
    ``due`` and ``last_review``. The remaining fsrs_cards columns
    (``reps``, ``lapses``, ``elapsed_days``, ``scheduled_days``) are
    derived by the scheduler during ``review_card`` and don't need to
    be reconstructed here — they're written back after the review.

    A fresh-card row (every column ``None``) yields a brand-new
    ``Card()`` with ``state=Learning`` and ``due=now``.
    """
    source: dict = {
        "card_id": row.word_id if row.word_id is not None else 0,
        "state": row.state if row.state is not None else int(State.Learning),
        "step": None,
        "stability": row.stability,
        "difficulty": row.difficulty,
        "due": row.due_date.isoformat() if row.due_date is not None else None,
        "last_review": (
            row.last_review.isoformat() if row.last_review is not None else None
        ),
    }
    # Brand-new cards: every FSRS-relevant column is None. ``Card.from_dict``
    # would crash on a None ``due``; fall back to a default-constructed Card
    # tagged with the row's word_id.
    if row.state is None and row.due_date is None:
        return Card(card_id=int(row.word_id) if row.word_id is not None else None)
    return Card.from_dict(source)


def card_to_row_dict(
    card: Card,
    word_id: int,
    prev_state: int | None = None,
) -> dict:
    """Serialize a py-fsrs ``Card`` into a dict matching ``fsrs_cards`` columns.

    ``prev_state`` is unused on the 5.1 surface but the parameter is
    reserved for 5.3's ``grade_logs`` audit row (which needs both the
    pre-review and post-review state for interval-delta observability).
    Accepting it here keeps the bridge signature stable across phases
    so 5.3 doesn't have to refactor the call site.

    Returns a dict whose keys are exactly the columns of
    ``backend.app.models.FsrsCard`` so the caller can splat it into
    ``FsrsCard(**row_dict)`` (or use it as an update mapping).

    ## Note on PHASE-5.md vs. py-fsrs 4.1.2 reality

    The PHASE-5.md "FSRS shape" mapping table lists four columns that
    py-fsrs 4.1.2's ``Card`` does not carry as attributes:
    ``reps``, ``lapses``, ``elapsed_days``, ``scheduled_days``. The
    v4.1.2 ``Card`` only has ``card_id / state / step / stability /
    difficulty / due / last_review`` (see ``fsrs/fsrs.py:65-86``).
    v5.x added those counters; v6.x renamed the serializer methods.

    For the columns the library *does* carry (``state / difficulty /
    stability / due / last_review``) this function reads them off
    the card. For the four columns py-fsrs 4.1.2 doesn't carry we
    emit ``None``; the grader in 5.3 is expected to fill them
    (e.g. ``scheduled_days = (card.due - card.last_review).days``)
    from the post-review snapshot. We do NOT silently invent values
    here — a wrong number on the audit row is worse than a ``None``
    that 5.3 owns.
    """
    _ = prev_state  # Reserved for 5.3; intentionally unused on the 5.1 surface.
    return {
        "word_id": word_id,
        "difficulty": card.difficulty,
        "stability": card.stability,
        "due_date": card.due,
        "last_review": card.last_review,
        # Columns py-fsrs 4.1.2's Card does NOT carry — set None, 5.3 fills.
        "reps": None,
        "lapses": None,
        "elapsed_days": None,
        "scheduled_days": None,
        "state": int(card.state),
    }


# ---------------------------------------------------------------------------
# Grading — thin wrapper around ``Scheduler.review_card``.
# ---------------------------------------------------------------------------


def apply_grade(card: Card, grade: int) -> tuple[Card, ReviewLog]:
    """Apply a 1-4 FSRS grade to a card and return the updated card + log.

    Thin wrapper around ``Scheduler.review_card(card, Rating(grade))``.
    Asserts ``grade ∈ {1, 2, 3, 4}`` so a bad input fails fast here
    rather than in py-fsrs's ``IntEnum`` machinery (which would raise
    a less obvious ``ValueError``). The wrapper exists so the route
    layer (5.3) doesn't have to import ``Rating`` itself — keeping the
    py-fsrs dependency surface narrow.
    """
    if grade not in (1, 2, 3, 4):
        raise ValueError(
            f"grade must be 1, 2, 3 or 4 (FSRS Rating); got {grade!r}"
        )
    scheduler = build_scheduler(enable_fuzzing=False)
    return scheduler.review_card(card, Rating(grade))


__all__ = [
    "PY_FSRS_VERSION",
    "DEFAULT_PARAMETERS",
    "DEFAULT_DESIRED_RETENTION",
    "DEFAULT_LEARNING_STEPS",
    "DEFAULT_RELEARNING_STEPS",
    "DEFAULT_MAXIMUM_INTERVAL",
    "DEFAULT_ENABLE_FUZZING",
    "build_scheduler",
    "row_to_card",
    "card_to_row_dict",
    "apply_grade",
]