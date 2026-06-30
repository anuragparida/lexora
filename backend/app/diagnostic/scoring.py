"""Phase 3.1 — deterministic rule-based scoring for the diagnostic probe.

Card: t_41d85c32.

``score(answers)`` is a pure function: the same answer set always
produces the same ``(axes, reasons)`` pair. No LLM, no Langfuse, no
randomness, no clock reads. That determinism is the whole point of
the MC-only probe — it's why the route can recompute the result on
every ``GET /diagnostic/result`` without persisting it.

``axes`` maps every one of the ten axes -> an int in [0, 3] (0 for
axes no answer touched). ``reasons`` maps only the axes with a
score > 0 -> a one-line string naming the top contributing
questions, so the frontend can show *why* an axis scored high.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from app.diagnostic.questions import ALL_AXES, QUESTION_BY_ID


@dataclass(frozen=True)
class AnswerRecord:
    """One recorded answer: which question, which choice label.

    The route layer builds these from the session's ``answers_json``
    dict (``question_id -> choice_label``). Kept as a tiny dataclass
    so ``score`` has a typed, attribute-access input rather than
    raw tuples.
    """

    question_id: str
    choice_label: str


def answers_from_dict(answers_json: Dict[str, str]) -> List[AnswerRecord]:
    """Build the ordered ``AnswerRecord`` list from a session's
    ``{question_id: choice_label}`` dict.

    Unknown question ids (e.g. a bank entry removed in a later
    version) are skipped silently — a stale recorded answer must not
    crash the read path. The route validates ids on write, so this
    only matters for historical rows.
    """
    records: List[AnswerRecord] = []
    for question_id, choice_label in answers_json.items():
        if question_id in QUESTION_BY_ID:
            records.append(
                AnswerRecord(question_id=question_id, choice_label=choice_label)
            )
    return records


def score(
    answers: Iterable[AnswerRecord],
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Pure scoring function.

    Returns ``(axes, reasons)``:

    - ``axes``: every axis in ``ALL_AXES`` -> clamped score in [0, 3].
    - ``reasons``: only axes with a final score > 0 -> a one-line
      string listing the top-2 weighted contributing questions.

    A choice label that doesn't match any of its question's choices
    is skipped (defensive — the route validates labels on write, so
    this only guards malformed historical rows).
    """
    axes: Dict[str, int] = {axis: 0 for axis in ALL_AXES}
    # axis -> list of (question_id, weight, choice_delta) contributions
    contributions: Dict[str, List[Tuple[str, int, int]]] = {
        axis: [] for axis in ALL_AXES
    }

    for ans in answers:
        question = QUESTION_BY_ID.get(ans.question_id)
        if question is None:
            continue
        choice = next(
            (c for c in question.choices if c.label == ans.choice_label),
            None,
        )
        if choice is None:
            continue
        for axis, delta in choice.delta.items():
            if axis not in axes:
                # Defensive: a choice delta referencing an axis outside
                # ALL_AXES is ignored rather than crashing. The bank
                # test asserts deltas are a subset of ALL_AXES, so this
                # never fires for the shipped bank.
                continue
            axes[axis] += question.weight * delta
            contributions[axis].append((question.id, question.weight, delta))

    # Clamp each axis to [0, 3] AFTER summing all contributions.
    axes = {axis: max(0, min(3, raw)) for axis, raw in axes.items()}

    reasons: Dict[str, str] = {}
    for axis, contribs in contributions.items():
        if axes[axis] == 0:
            continue
        # Sort by weighted contribution (weight * delta) descending so
        # the most influential questions surface first. Take the top 2.
        contribs_sorted = sorted(
            contribs, key=lambda c: c[1] * c[2], reverse=True
        )
        top = contribs_sorted[:2]
        reason_parts = [
            f"{q_id} (weight {w}, delta {d})" for q_id, w, d in top
        ]
        reasons[axis] = f"{axis}: " + " + ".join(reason_parts)

    return axes, reasons
