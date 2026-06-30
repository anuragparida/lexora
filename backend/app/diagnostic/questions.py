"""Phase 3.1 — the diagnostic probe question bank (code, not DB).

Card: t_41d85c32.

Ten fixed multiple-choice questions, one frozen module-level list.
The bank is version-controlled code, never stored in the database —
the ``diagnostic_sessions`` table only records a user's *answers*
(``question_id -> choice_label``), not the questions themselves.

Each question targets one or two of the ten weakness axes shared
with the manual profile page. ``ALL_AXES`` is the canonical axis
list; every axis appears in at least one question's ``axis_tags``.

The probe is deterministic: scoring is a pure function of the
recorded answers (see ``scoring.py``). No LLM call — the user knows
how comfortable they are with each topic better than any model
could infer. The MC-only shape is what makes deterministic scoring
possible; free-text would force either brittle keyword matching or
an LLM, both deferred to Phase 4+.

Server-side scoring fields (``delta``, ``weight``, ``axis_tags``)
are NEVER serialized to the client — the ``/diagnostic/start``
route strips them and sends only ``{id, prompt, kind, choices:
[{label}]}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal


# ---------------------------------------------------------------------------
# Axes — the canonical ten, shared with the manual WeaknessProfile page.
#
# Kept here (not imported from schemas) so the probe module is
# self-contained; the manual-profile validator accepts any axis name
# in [0, 3], so there's no single enum to import. Keep this list in
# sync with the frontend's axis labels (Phase 2.4).
# ---------------------------------------------------------------------------

ALL_AXES: tuple[str, ...] = (
    "verbs",
    "prepositional_combos",
    "collocations",
    "idioms",
    "abstract_nouns",
    "adjectives",
    "adverbs",
    "prepositions",
    "pronouns",
    "conjunctions",
)


@dataclass(frozen=True)
class QuestionChoice:
    """One answer option. ``delta`` maps axis name -> 0..3 raw points
    (multiplied by the question ``weight`` during scoring). A choice
    that signals confidence has all-zero (or omitted) deltas; a choice
    that signals a gap has higher deltas on the probed axes."""

    label: str
    delta: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Question:
    """A single multiple-choice probe question.

    ``axis_tags`` lists every axis this question can move (the union
    of the choices' ``delta`` keys). ``weight`` (1-3) scales every
    choice delta — a high-weight question moves the axis faster, so a
    single confident/uncertain answer on a load-bearing topic counts
    more. ``kind`` is always ``"multiple_choice"`` in Phase 3.
    """

    id: str
    axis_tags: List[str]
    weight: int
    prompt: str
    kind: Literal["multiple_choice"]
    choices: List[QuestionChoice]


# A reusable 4-rung comfort scale for single-axis questions. The
# rungs map confidence -> weakness: a confident learner adds 0, a
# struggling learner adds 3 (then scaled by the question weight and
# clamped to [0, 3] in scoring).
def _comfort_choices(axis: str) -> List[QuestionChoice]:
    return [
        QuestionChoice(
            label="Confident — I rarely get this wrong",
            delta={axis: 0},
        ),
        QuestionChoice(
            label="Mostly fine — the occasional slip",
            delta={axis: 1},
        ),
        QuestionChoice(
            label="Shaky — I mix these up often",
            delta={axis: 2},
        ),
        QuestionChoice(
            label="I struggle — never properly learned this",
            delta={axis: 3},
        ),
    ]


QUESTIONS: List[Question] = [
    Question(
        id="q-verb-conjugation-present",
        axis_tags=["verbs"],
        weight=3,
        prompt=(
            "How comfortable are you conjugating German verbs in the "
            "present tense (including strong/irregular stems)?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("verbs"),
    ),
    Question(
        id="q-verb-preposition-combos",
        axis_tags=["verbs", "prepositional_combos"],
        weight=3,
        prompt=(
            "How comfortable are you with verb + preposition "
            "combinations (e.g. 'warten auf', 'denken an', 'sich "
            "freuen über')?"
        ),
        kind="multiple_choice",
        choices=[
            QuestionChoice(
                label="Confident — I know which preposition each verb takes",
                delta={"verbs": 0, "prepositional_combos": 0},
            ),
            QuestionChoice(
                label="Mostly fine — I get the common ones right",
                delta={"verbs": 1, "prepositional_combos": 1},
            ),
            QuestionChoice(
                label="Shaky — I guess the preposition",
                delta={"verbs": 1, "prepositional_combos": 2},
            ),
            QuestionChoice(
                label="I struggle — I never know which preposition to use",
                delta={"verbs": 1, "prepositional_combos": 3},
            ),
        ],
    ),
    Question(
        id="q-collocations",
        axis_tags=["collocations"],
        weight=2,
        prompt=(
            "How comfortable are you with natural word pairings / "
            "collocations (e.g. 'eine Entscheidung treffen', not "
            "'eine Entscheidung machen')?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("collocations"),
    ),
    Question(
        id="q-idioms",
        axis_tags=["idioms"],
        weight=2,
        prompt=(
            "How comfortable are you understanding and using common "
            "German idioms (e.g. 'die Daumen drücken', 'ins Wasser "
            "fallen')?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("idioms"),
    ),
    Question(
        id="q-abstract-nouns",
        axis_tags=["abstract_nouns"],
        weight=2,
        prompt=(
            "How comfortable are you with abstract nouns and their "
            "genders (e.g. 'die Gerechtigkeit', 'das Verständnis', "
            "'der Zweifel')?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("abstract_nouns"),
    ),
    Question(
        id="q-adjective-endings",
        axis_tags=["adjectives"],
        weight=3,
        prompt=(
            "How comfortable are you with adjective endings across "
            "the cases (e.g. 'ein guter Mann', 'mit dem guten "
            "Wein')?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("adjectives"),
    ),
    Question(
        id="q-adverbs",
        axis_tags=["adverbs"],
        weight=2,
        prompt=(
            "How comfortable are you placing adverbs correctly "
            "(time-manner-place order, 'schon', 'noch', 'erst')?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("adverbs"),
    ),
    Question(
        id="q-prepositions-cases",
        axis_tags=["prepositions"],
        weight=3,
        prompt=(
            "How comfortable are you choosing the right case after a "
            "preposition (two-way prepositions like 'in', 'auf', "
            "'über' taking accusative vs dative)?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("prepositions"),
    ),
    Question(
        id="q-pronouns",
        axis_tags=["pronouns"],
        weight=2,
        prompt=(
            "How comfortable are you with pronouns in the right case "
            "(e.g. 'mir' vs 'mich', relative pronouns 'der/die/das' "
            "vs 'dem/den')?"
        ),
        kind="multiple_choice",
        choices=_comfort_choices("pronouns"),
    ),
    Question(
        id="q-conjunctions",
        axis_tags=["conjunctions", "adverbs"],
        weight=2,
        prompt=(
            "How comfortable are you with conjunctions and the word "
            "order they trigger (subordinating 'weil', 'dass', "
            "'obwohl' sending the verb to the end)?"
        ),
        kind="multiple_choice",
        choices=[
            QuestionChoice(
                label="Confident — I get the word order right",
                delta={"conjunctions": 0, "adverbs": 0},
            ),
            QuestionChoice(
                label="Mostly fine — the common ones are automatic",
                delta={"conjunctions": 1, "adverbs": 0},
            ),
            QuestionChoice(
                label="Shaky — I forget to move the verb",
                delta={"conjunctions": 2, "adverbs": 1},
            ),
            QuestionChoice(
                label="I struggle — subordinate clauses trip me up",
                delta={"conjunctions": 3, "adverbs": 1},
            ),
        ],
    ),
]


# ``question_id -> Question`` lookup, built once at import. Used by
# the scoring function and the answer-validation route.
QUESTION_BY_ID: Dict[str, Question] = {q.id: q for q in QUESTIONS}

# Total questions the client must answer for a complete probe.
TOTAL_QUESTIONS: int = len(QUESTIONS)
