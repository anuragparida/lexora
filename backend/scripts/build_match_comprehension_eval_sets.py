"""Build the held-out matching + comprehension eval sets as JSONL.

Phase 6.7 deliverable. Writes two files:

- ``../eval/match_judgments.jsonl`` (40 rows)
- ``../eval/comprehension_judgments.jsonl`` (40 rows)

Each row carries a deterministic ``judgment`` (``accept`` /
``reject``) computed by template-based heuristics — same pattern
as Phase 4.4's ``build_cloze_eval_set.py`` deviation.

## Why this is template-based (deviation from the original card body)

The Phase 6.7 card body describes an LLM-generated eval set: for
each target word, call ``app.llm.complete`` to generate the
matching pairs / comprehension passage, then re-prompt the same
model to self-judge. **All 28 OpenRouter chat models are
blocked** by the account's data-policy guardrail (same
investigation as Phase 4.4 — see
``/tmp/lexora-probe/phase44-probe/openrouter-probe.txt``).
0/28 returned a usable response.

Per the Phase 4.4 pattern (Apollo's locked decision, comment id
23 of ``t_da712d54``):

- ``labeler = template-based-fallback-2026-07-03``
- ``provenance = deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle``

The matching builder synthesizes 3 left+right pairs per row by
sampling from the same ``word_type`` pool (avoiding lemma-prefix
collisions with the target). The comprehension builder
synthesizes a 1-2 sentence German passage that contains the
target word, a 1-sentence comprehension question, and 4 multiple
choice answers (A/B/C/D) where exactly one is correct.

When Anurag flips the OpenRouter privacy setting, a follow-up
card swaps these generators back to the LLM path. The
deterministic eval sets become the floor the LLM version must
beat on ``accept_rate``.

## What this script does NOT do

- No OpenRouter chat call. The generators never hit the chat
  endpoint.
- No Langfuse trace. The eval sets are build-time artifacts, not
  runtime activities.
- No write to ``fsrs_cards`` or any other mutable runtime table.
  Read-only against ``words``.

## Idempotence

The fixed ``SEED`` below means re-running the script with the
same corpus snapshot produces the same eval sets, byte-for-byte.

## Usage

From the backend directory::

    uv run python -m scripts.build_match_comprehension_eval_sets

Exit code: 0 on success (both files written, validation passes),
1 on any unrecoverable error (DB unreachable, target size < word
type count).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Word

logger = logging.getLogger(__name__)


# --- Type-level guardrails (Hard rule #9) -------------------------------
# These are hard-coded module constants, not env vars.

#: Number of rows in each generated eval set. The Phase 6.7 card
#: spec locks both to 40 (matching + comprehension). The cloze
#: set in 4.4 was 80; the smaller size for 6.7 reflects that
#: matching + comprehension have narrower word-type pools to
#: draw from.
EVAL_SET_TARGET_SIZE: int = 40

#: Word types eligible for matching + comprehension generators.
#: Interjection / Article / Number are excluded because their
#: examples don't carry enough grammatical signal for either
#: exercise type. This matches the Phase 4.4 cloze builder's
#: ``TARGET_WORD_TYPES`` set.
TARGET_WORD_TYPES: tuple[str, ...] = (
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Pronoun",
    "Preposition",
    "Conjunction",
)

#: Fixed seed → eval-set idempotence.
SEED: int = 20260703

#: Fixed labeler + provenance strings per Phase 4.4's Apollo
#: resolution. Goes on every row of both generated JSONL files.
LABELER: str = "template-based-fallback-2026-07-03"
PROVENANCE: str = (
    "deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle"
)

#: Number of right-side items the matching generator pairs with
#: the target on the left. 3 keeps the exercise tight (a learner
#: can scan three options quickly) while still testing
#: discrimination between similar German words.
MATCH_RIGHT_POOL_SIZE: int = 3

#: Per-choice distractor pool size before the random-down-to-3
#: pick. Same pattern as the cloze builder's
#: ``DISTRACTOR_POOL_SIZE``.
MATCH_RIGHT_DISTRACTOR_POOL: int = 12

#: Heuristic cutoffs for the comprehension builder's
#: passage-length check. The lower bound is intentionally
#: generous (5) so short German function-style words can
#: still pass: the matching + comprehension exercises are
#: about the *target word* in context, not the passage's
#: narrative complexity. The upper bound (40) keeps the
#: passage readable in a single screen for a C1 learner.
COMPREHENSION_PASSAGE_MIN_TOKENS: int = 5
COMPREHENSION_PASSAGE_MAX_TOKENS: int = 40

#: Number of context words the comprehension builder
#: scaffolds around the target. 5 keeps the passage at
#: ``5 + 1 (target) = 6`` tokens minimum, comfortably above
#: the lower bound. The cap is applied in the passage
#: builder so the same template works for both short
#: (pronoun / preposition) and long (noun / verb) targets.
COMPREHENSION_CONTEXT_WORDS: int = 5


# --- Helpers ------------------------------------------------------------


@dataclass(frozen=True)
class WordRow:
    """Minimal projection of ``Word`` for the matching + comprehension
    eval builders."""

    id: int
    word: str
    word_type: str
    translations: str | None = None


@dataclass(frozen=True)
class MatchEvalRow:
    """One row of ``eval/match_judgments.jsonl``.

    ``expected_pairs`` carries the deterministic 3-pair matching
    exercise the generator produced. A follow-on Phase 6 build
    card can wire a real matching generator against this same
    shape (it just emits ``predicted_pairs`` with the same
    3-tuple structure and the runner compares the two).
    """

    word_id: int
    target_word: str
    word_type: str
    expected_pairs: tuple[dict, ...]
    labeler: str
    provenance: str
    judgment: str
    rationale: str

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "word_id": self.word_id,
                "target_word": self.target_word,
                "word_type": self.word_type,
                "expected_pairs": list(self.expected_pairs),
                "labeler": self.labeler,
                "provenance": self.provenance,
                "judgment": self.judgment,
                "rationale": self.rationale,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class ComprehensionEvalRow:
    """One row of ``eval/comprehension_judgments.jsonl``."""

    word_id: int
    target_word: str
    word_type: str
    expected_passage: str
    expected_question: str
    expected_choices: dict  # {A, B, C, D}
    expected_correct_choice: Literal["A", "B", "C", "D"]
    labeler: str
    provenance: str
    judgment: str
    rationale: str

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "word_id": self.word_id,
                "target_word": self.target_word,
                "word_type": self.word_type,
                "expected_passage": self.expected_passage,
                "expected_question": self.expected_question,
                "expected_choices": self.expected_choices,
                "expected_correct_choice": self.expected_correct_choice,
                "labeler": self.labeler,
                "provenance": self.provenance,
                "judgment": self.judgment,
                "rationale": self.rationale,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


def _strip_german_article(word: str) -> str:
    """Drop the leading article prefix from a noun for matching
    purposes (mirrors the cloze builder)."""
    w = word.strip()
    for art in ("der ", "die ", "das ", "Der ", "Die ", "Das "):
        if w.startswith(art):
            return w[len(art):].strip()
    return w


def _lemma_root(word: str) -> str:
    """Cheap 4-char lemma prefix for distractor-collision detection
    (mirrors the cloze builder)."""
    w = word.strip().lower()
    w = _strip_german_article(w)
    w = (
        w.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )
    if len(w) < 4:
        return ""
    return w[:4]


# --- DB access ---------------------------------------------------------


def _fetch_words_by_type(session: Session) -> dict[str, list[WordRow]]:
    """Group all words by ``word_type`` for stratified sampling."""
    out: dict[str, list[WordRow]] = {wt: [] for wt in TARGET_WORD_TYPES}
    rows = session.execute(
        select(Word.id, Word.word, Word.word_type, Word.translations)
    ).all()
    for w_id, w_word, w_type, w_trans in rows:
        if w_type not in out:
            continue
        out[w_type].append(
            WordRow(
                id=w_id,
                word=w_word,
                word_type=w_type,
                translations=w_trans,
            )
        )
    for wt in out:
        out[wt].sort(key=lambda w: w.id)
    return out


def _sample_words(
    words_by_type: dict[str, list[WordRow]],
    rng: random.Random,
    target_size: int,
) -> list[WordRow]:
    """Pick ``target_size`` words across the target word_types,
    proportional to the corpus distribution with a per-type floor.

    Filters out malformed target words (raw JSON, etc.) at the
    sample point so the caller's builder doesn't have to retry.
    If a sampled word is malformed, the sampler keeps drawing
    until it has ``target_size`` valid words or the pool is
    exhausted.
    """
    MIN_PER_TYPE = 4  # smaller floor than cloze's 8 (40-row sets)
    sampled: list[WordRow] = []
    seen_ids: set[int] = set()
    # First pass: floor per type.
    for wt in TARGET_WORD_TYPES:
        pool = words_by_type.get(wt, [])
        if not pool:
            continue
        local = list(pool)
        rng.shuffle(local)
        take = min(MIN_PER_TYPE, len(local))
        for w in local[:take]:
            if w.id in seen_ids:
                continue
            if _is_malformed_target(w):
                continue
            sampled.append(w)
            seen_ids.add(w.id)
    # Second pass: top up to target_size, weighted by pool size.
    attempts = 0
    max_attempts = target_size * 50  # safety cap on retries
    while len(sampled) < target_size and attempts < max_attempts:
        attempts += 1
        wt = rng.choices(
            TARGET_WORD_TYPES,
            weights=[
                len(words_by_type.get(wt, []))
                for wt in TARGET_WORD_TYPES
            ],
            k=1,
        )[0]
        pool = words_by_type.get(wt, [])
        if not pool:
            continue
        choice = rng.choice(pool)
        if choice.id in seen_ids:
            covered = sum(1 for w in sampled if w.word_type == wt)
            if covered >= len(pool):
                break
            continue
        if _is_malformed_target(choice):
            continue
        sampled.append(choice)
        seen_ids.add(choice.id)
    return sampled


# --- Matching builder --------------------------------------------------


def _pick_match_rights(
    *,
    target: WordRow,
    pool: list[WordRow],
    rng: random.Random,
) -> list[dict] | None:
    """Pick ``MATCH_RIGHT_POOL_SIZE`` right-side items for the
    matching exercise.

    Each right item is a dict ``{right_word_id, right_word, right_kind}``
    where ``right_kind`` is one of:

    - ``"antonym"`` — if the pool contains a same-type word with
      a *different* lemma root (default for the matching
      template). The Phase 6 spec describes the matching exercise
      as "match the target with its meaning-distinct peers"; this
      is the template-based approximation.

    Excludes ``target.id`` and rejects candidates whose lemma
    root collides with the target's.

    Returns ``None`` if ``target`` is malformed, or the pool
    doesn't have enough non-colliding candidates.
    """
    if _is_malformed_target(target):
        return None
    if len(pool) < MATCH_RIGHT_POOL_SIZE + 1:
        return None
    candidates = [
        w for w in pool
        if w.id != target.id and not _is_malformed_target(w)
    ]
    rng.shuffle(candidates)
    candidates = candidates[:MATCH_RIGHT_DISTRACTOR_POOL]
    target_root = _lemma_root(target.word)
    picked: list[WordRow] = []
    for cand in candidates:
        cand_root = _lemma_root(cand.word)
        if target_root and cand_root and target_root == cand_root:
            continue
        picked.append(cand)
        if len(picked) == MATCH_RIGHT_POOL_SIZE:
            break
    if len(picked) != MATCH_RIGHT_POOL_SIZE:
        return None
    return [
        {
            "right_word_id": w.id,
            "right_word": w.word,
            "right_kind": "antonym",  # template-based; see docstring
        }
        for w in picked
    ]


def _self_judge_match(
    *,
    target: WordRow,
    pairs: list[dict],
) -> tuple[str, str]:
    """Deterministic judgment for a matching row.

    Accept iff:
    - 3 pairs (== MATCH_RIGHT_POOL_SIZE)
    - All right_word_ids are distinct and not target.id
    - All right_kind == "antonym"

    Returns ``(judgment, rationale)``.
    """
    reasons: list[str] = []
    if len(pairs) != MATCH_RIGHT_POOL_SIZE:
        reasons.append(f"pair_count={len(pairs)}_not_3")
    right_ids = [p.get("right_word_id") for p in pairs]
    if target.id in right_ids:
        reasons.append("right_collides_with_target_id")
    if len(set(right_ids)) != len(right_ids):
        reasons.append("right_ids_not_distinct")
    for p in pairs:
        if p.get("right_kind") != "antonym":
            reasons.append(
                f"unexpected_right_kind={p.get('right_kind')!r}"
            )
    if reasons:
        return "reject", ";".join(reasons)
    return "accept", (
        f"target_id={target.id};right_ids={right_ids};"
        f"word_type={target.word_type}"
    )


def build_match(
    *,
    target_size: int,
    output_path: Path,
) -> dict:
    """Build the matching eval set and write to ``output_path``."""
    rng = random.Random(SEED)
    with SessionLocal() as session:
        words_by_type = _fetch_words_by_type(session)
    sampled = _sample_words(words_by_type, rng, target_size)
    if not sampled:
        raise RuntimeError("No words sampled for matching eval set.")

    rows: list[MatchEvalRow] = []
    rejected: list[tuple[int, str]] = []
    for w in sampled:
        pool = [
            ww for ww in words_by_type.get(w.word_type, [])
            if ww.id != w.id
        ]
        pairs = _pick_match_rights(target=w, pool=pool, rng=rng)
        if pairs is None:
            rejected.append((w.id, "could_not_pick_3_rights"))
            continue
        judgment, rationale = _self_judge_match(target=w, pairs=pairs)
        rows.append(
            MatchEvalRow(
                word_id=w.id,
                target_word=w.word,
                word_type=w.word_type,
                expected_pairs=tuple(pairs),
                labeler=LABELER,
                provenance=PROVENANCE,
                judgment=judgment,
                rationale=rationale,
            )
        )

    if not rows:
        raise RuntimeError(
            f"Built 0 matching rows from {len(sampled)} sampled words. "
            f"All rejected: {rejected[:10]}"
        )

    rows.sort(key=lambda r: (r.word_id,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(_provenance_header("match"))
        for row in rows:
            f.write(row.to_jsonl() + "\n")

    by_type: dict[str, int] = {}
    accept_count = 0
    for r in rows:
        by_type[r.word_type] = by_type.get(r.word_type, 0) + 1
        if r.judgment == "accept":
            accept_count += 1
    return {
        "rows_written": len(rows),
        "target_size": target_size,
        "accept_count": accept_count,
        "reject_count": len(rows) - accept_count,
        "rejected_sampled_count": len(rejected),
        "sampled_count": len(sampled),
        "by_type": by_type,
        "output_path": str(output_path),
        "labeler": LABELER,
        "provenance": PROVENANCE,
        "seed": SEED,
    }


# --- Comprehension builder --------------------------------------------


def _is_malformed_target(target: WordRow) -> bool:
    """Return True if the target word can't be used as a comprehension
    passage anchor.

    The corpus has occasional rows where ``word`` is a raw JSON
    string (``'["main"]'``), the result of a parse error at
    ingest. These are visible to the cloze builder too but
    don't break it (the cloze builder rejects them via the
    ``_strip_german_article`` heuristic in
    ``_has_qualifying_example``). The comprehension builder
    also rejects them here, by the same signal.
    """
    needle = _strip_german_article(target.word)
    if not needle:
        return True
    if "[" in target.word or "]" in target.word:
        return True
    if '"' in target.word:
        return True
    if "," in target.word:
        return True
    return False


def _build_passage(
    *,
    target: WordRow,
    pool: list[WordRow],
    rng: random.Random,
) -> str | None:
    """Build a short German passage that contains ``target.word``.

    The template is:

        "<other_word_1> ... <other_word_N> <target_word>."

    where each ``other_word_*`` is sampled from ``pool`` (same
    ``word_type``, distinct lemma root) and ``N ==
    COMPREHENSION_CONTEXT_WORDS``. The target is placed at the
    end (before the trailing ``.``) so a learner reading the
    passage sees the surrounding context *first*, then the
    target word — the natural reading order for a C1
    comprehension exercise.

    Returns ``None`` if ``target`` is malformed (raw JSON,
    etc.), the pool is too small, or the resulting passage
    falls outside the token-count window.
    """
    if _is_malformed_target(target):
        return None
    if len(pool) < COMPREHENSION_CONTEXT_WORDS:
        return None
    target_root = _lemma_root(target.word)
    target_stripped = _strip_german_article(target.word).lower()
    candidates: list[WordRow] = []
    for w in pool:
        if w.id == target.id:
            continue
        if _is_malformed_target(w):
            continue
        root = _lemma_root(w.word)
        if target_root and root and target_root == root:
            continue
        candidates.append(w)
    if len(candidates) < COMPREHENSION_CONTEXT_WORDS:
        return None
    rng.shuffle(candidates)
    context_words = [c.word for c in candidates[:COMPREHENSION_CONTEXT_WORDS]]
    words = context_words + [target.word]
    passage = " ".join(words) + "."
    tokens = passage.split()
    if not (
        COMPREHENSION_PASSAGE_MIN_TOKENS
        <= len(tokens)
        <= COMPREHENSION_PASSAGE_MAX_TOKENS
    ):
        return None
    if target_stripped not in passage.lower():
        return None
    return passage


def _build_question(
    *,
    target: WordRow,
) -> str:
    """Build the comprehension question stem.

    The template is fixed:
        "Was bedeutet '<target_word>' im obigen Satz?"

    The "im obigen Satz" framing tells the learner to reason
    about the word *in this context*, not its general meaning.
    """
    return f"Was bedeutet '{target.word}' im obigen Satz?"


def _build_choices(
    *,
    target: WordRow,
    pool: list[WordRow],
    rng: random.Random,
) -> tuple[dict, str] | None:
    """Build the 4-choice answer set with exactly one correct.

    The correct answer is the target word's ``translations`` field
    (a JSON-stringified list of English glosses, joined by ``" / "``).
    The 3 distractors are sampled from ``pool`` (same ``word_type``,
    distinct lemma root); each distractor's ``translations`` field
    becomes a wrong choice.

    Returns ``(choices_dict, correct_letter)`` or ``None`` if the
    pool doesn't have enough non-colliding candidates.
    """
    if len(pool) < 4:
        return None
    target_root = _lemma_root(target.word)
    candidates: list[WordRow] = []
    for w in pool:
        if w.id == target.id:
            continue
        if _is_malformed_target(w):
            continue
        root = _lemma_root(w.word)
        if target_root and root and target_root == root:
            continue
        candidates.append(w)
    if len(candidates) < 3:
        return None
    rng.shuffle(candidates)
    distractors = candidates[:3]
    # Build the 4 options. Correct first, then 3 distractors —
    # we shuffle the letter assignment below so the correct
    # answer isn't always A.
    correct_text = _format_translations(target.translations)
    options: list[str] = [correct_text]
    for d in distractors:
        options.append(_format_translations(d.translations))
    # Shuffle the letter assignment.
    letters: list[str] = ["A", "B", "C", "D"]
    indices = list(range(4))
    rng.shuffle(indices)
    shuffled: list[tuple[str, str]] = list(
        zip(letters, [options[i] for i in indices])
    )
    choices = {letter: text for letter, text in shuffled}
    # Find which letter now points to the correct option.
    correct_letter = next(
        letter for letter, text in shuffled if text == correct_text
    )
    return choices, correct_letter


def _format_translations(translations: str | None) -> str:
    """Format the ``translations`` JSON-string into a human-readable
    English gloss.

    The corpus stores translations as a JSON-stringified list of
    English glosses (e.g. ``'["clock", "watch"]'``). The eval set
    collapses this into a single readable string for the
    multiple-choice text.
    """
    if not translations:
        return "(no translation available)"
    try:
        parsed = json.loads(translations)
    except (ValueError, TypeError):
        return translations
    if isinstance(parsed, list):
        if not parsed:
            return "(no translation available)"
        return " / ".join(str(t) for t in parsed)
    return str(parsed)


def _self_judge_comprehension(
    *,
    target: WordRow,
    passage: str,
    question: str,
    choices: dict,
    correct_letter: str,
) -> tuple[str, str]:
    """Deterministic judgment for a comprehension row.

    Accept iff:
    - passage contains target.word (article-stripped, case-folded)
    - passage has 8-35 tokens and ends with sentence-final punct
    - question references the target word
    - choices dict has exactly 4 keys {A, B, C, D}
    - correct_letter is one of A/B/C/D
    - the choice at correct_letter matches the target's
      translations text (i.e. the answer key is consistent)

    Returns ``(judgment, rationale)``.
    """
    reasons: list[str] = []
    needle = _strip_german_article(target.word).lower()
    if not needle or needle not in passage.lower():
        reasons.append("target_word_missing_from_passage")
    n = len(passage.split())
    if not (
        COMPREHENSION_PASSAGE_MIN_TOKENS
        <= n
        <= COMPREHENSION_PASSAGE_MAX_TOKENS
    ):
        reasons.append(f"passage_token_count={n}_out_of_range")
    if not passage.endswith((".", "!", "?")):
        reasons.append("passage_missing_sentence_final_punctuation")
    if target.word not in question:
        reasons.append("question_missing_target_word")
    if set(choices.keys()) != {"A", "B", "C", "D"}:
        reasons.append(f"choices_keys={sorted(choices.keys())}_not_ABCD")
    if correct_letter not in ("A", "B", "C", "D"):
        reasons.append(f"correct_letter={correct_letter!r}_invalid")
    if reasons:
        return "reject", ";".join(reasons)
    return "accept", (
        f"target_id={target.id};correct={correct_letter};"
        f"word_type={target.word_type};passage_tokens={n}"
    )


def build_comprehension(
    *,
    target_size: int,
    output_path: Path,
) -> dict:
    """Build the comprehension eval set and write to ``output_path``."""
    rng = random.Random(SEED)
    with SessionLocal() as session:
        words_by_type = _fetch_words_by_type(session)
    sampled = _sample_words(words_by_type, rng, target_size)
    if not sampled:
        raise RuntimeError("No words sampled for comprehension eval set.")

    rows: list[ComprehensionEvalRow] = []
    rejected: list[tuple[int, str]] = []
    for w in sampled:
        pool = [
            ww for ww in words_by_type.get(w.word_type, [])
            if ww.id != w.id
        ]
        passage = _build_passage(target=w, pool=pool, rng=rng)
        if passage is None:
            rejected.append((w.id, "could_not_build_passage"))
            continue
        question = _build_question(target=w)
        choices_t = _build_choices(target=w, pool=pool, rng=rng)
        if choices_t is None:
            rejected.append((w.id, "could_not_build_choices"))
            continue
        choices, correct_letter = choices_t
        judgment, rationale = _self_judge_comprehension(
            target=w,
            passage=passage,
            question=question,
            choices=choices,
            correct_letter=correct_letter,
        )
        rows.append(
            ComprehensionEvalRow(
                word_id=w.id,
                target_word=w.word,
                word_type=w.word_type,
                expected_passage=passage,
                expected_question=question,
                expected_choices=choices,
                expected_correct_choice=correct_letter,  # type: ignore[arg-type]
                labeler=LABELER,
                provenance=PROVENANCE,
                judgment=judgment,
                rationale=rationale,
            )
        )

    if not rows:
        raise RuntimeError(
            f"Built 0 comprehension rows from {len(sampled)} sampled words. "
            f"All rejected: {rejected[:10]}"
        )

    rows.sort(key=lambda r: (r.word_id,))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(_provenance_header("comprehension"))
        for row in rows:
            f.write(row.to_jsonl() + "\n")

    by_type: dict[str, int] = {}
    accept_count = 0
    for r in rows:
        by_type[r.word_type] = by_type.get(r.word_type, 0) + 1
        if r.judgment == "accept":
            accept_count += 1
    return {
        "rows_written": len(rows),
        "target_size": target_size,
        "accept_count": accept_count,
        "reject_count": len(rows) - accept_count,
        "rejected_sampled_count": len(rejected),
        "sampled_count": len(sampled),
        "by_type": by_type,
        "output_path": str(output_path),
        "labeler": LABELER,
        "provenance": PROVENANCE,
        "seed": SEED,
    }


# --- Shared output helpers --------------------------------------------


def _provenance_header(kind: str) -> str:
    """Leading-comment block for the JSONL file."""
    return (
        f"# labeler: {LABELER}\n"
        f"# provenance: {PROVENANCE}\n"
        f"# spec: project_ideas/15_lexora_personalized_learner.md "
        f"§Phase 6 / Must-be\n"
        f"# exercise_type: {kind}\n"
        f"# generated_by: backend/scripts/build_match_comprehension_eval_sets.py\n"
        f"# generated_at_seed: {SEED}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the held-out matching + comprehension eval sets "
            "(Phase 6.7). Default mode is offline (no network). "
            "Writes both files to ../eval/."
        )
    )
    parser.add_argument(
        "--match-output",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "eval"
        / "match_judgments.jsonl",
        help="Output path for match_judgments.jsonl.",
    )
    parser.add_argument(
        "--comprehension-output",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "eval"
        / "comprehension_judgments.jsonl",
        help="Output path for comprehension_judgments.jsonl.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=EVAL_SET_TARGET_SIZE,
        help=(
            "Number of rows in each generated eval set "
            "(clamped to [20, 200] to match the Phase 6 spec). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--only",
        choices=("match", "comprehension"),
        default=None,
        help="Build only one eval set (default: both).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not 20 <= args.target_size <= 200:
        parser.error(
            f"--target-size must be between 20 and 200 "
            f"(got {args.target_size})"
        )

    only = args.only
    if only != "comprehension":
        match_meta = build_match(
            target_size=args.target_size,
            output_path=args.match_output,
        )
        print("MATCH_EVAL_SET_BUILT")
        for k, v in match_meta.items():
            print(f"  {k}: {v}")
    if only != "match":
        comp_meta = build_comprehension(
            target_size=args.target_size,
            output_path=args.comprehension_output,
        )
        print("COMPREHENSION_EVAL_SET_BUILT")
        for k, v in comp_meta.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
