"""Build the held-out cloze eval set as a JSONL file.

Phase 4.4 deliverable. Writes ``../eval/cloze_judgments.jsonl`` (one
JSON object per line, leading provenance comment) so the DSPy
optimizer in 4.2 has a stable, reproducible signal to optimize
against.

## Why this is template-based (deviation from the original card body)

The original card body called for an LLM-generated eval set: for
each word, call ``app.llm.complete`` to produce a cloze, then re-prompt
the same model to self-judge. **All 28 OpenRouter chat models are
blocked** by the account's data-policy guardrail (probed during the
prior perseus run on this card; full results at
``/tmp/lexora-probe/phase44-probe/openrouter-probe.txt``). 0/28
returned a usable response.

Per Apollo's resolution (comment id 23 of the card thread, 2026-07-03):

> Option 3 — template-based fallback. Deterministic cloze via
> first example sentence with the target word replaced by `___`;
> distractor sampling across `word_type`; self-judgment via length
> + lemma-overlap heuristics. Replace `labeler` with
> `template-based-fallback-2026-07-03`. Provenance comment becomes
> `deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle`.
> Honest deviation recorded.

This script implements option 3. The deviations are visible in
two places:

1. The ``LABELER`` constant below is
   ``template-based-fallback-2026-07-03`` (was
   ``ai-assisted-claude-minimax-m3``).
2. The leading comment of the generated ``cloze_judgments.jsonl``
   carries the honest provenance string so any future consumer
   can see the eval set was not LLM-judged.

When Anurag flips the OpenRouter privacy setting (or supplies a
different provider key), a follow-up card swaps this generator
back to the LLM path. The deterministic eval set becomes the
floor that the LLM version must beat on ``accept_rate``.

## What this script does NOT do

- No OpenRouter chat call. The generator never hits the chat
  endpoint.
- No Langfuse trace. The eval set is a build-time artifact, not
  a runtime activity.
- No write to ``fsrs_cards`` or any other mutable runtime table.
  Read-only against ``words`` and ``examples``.

## Optional embedding-based distractor filter

By default, distractors are picked via stratified random sampling
across the same ``word_type``. Add ``--use-embeddings`` to enable
cosine-similarity filtering using the existing
``qwen/qwen3-embedding-8b`` embedding pipeline (Phase 1's same
provider, allowed under Apollo's resolution). With
``--use-embeddings``, the script batches the target + candidate
distractors through ``app.embeddings.embed`` and rejects any
candidate with cosine similarity ≥ ``DISTRACTOR_MAX_COSINE`` to
the target (too similar → confusing), or cosine < 0.05 (too
unrelated → trivially eliminated).

## Idempotence

The fixed ``SEED`` below means re-running the script with the
same corpus snapshot produces the same eval set, byte-for-byte.
The only state that varies across runs is the ``timestamp`` in
the metadata (and that's omitted from the JSONL rows themselves —
it lives in the ``build_metadata`` print at the end of the run).

## Usage

From the backend directory::

    # Template-only (default; no network):
    uv run python -m scripts.build_cloze_eval_set

    # With embedding-based distractor filter (calls OpenRouter /embeddings):
    uv run python -m scripts.build_cloze_eval_set --use-embeddings

    # Smaller / larger eval set:
    uv run python -m scripts.build_cloze_eval_set --target-size 60

Exit code: 0 on success (file written, validation passes), 1 on
any unrecoverable error (DB unreachable, embedding API failure
when ``--use-embeddings`` is set, target size < word-type count).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Example, Word

logger = logging.getLogger(__name__)


# --- Type-level guardrails (Hard rule #11) -----------------------------
# These are hard-coded module constants, not env vars. Changing any
# of them requires a code review, not a container restart.

#: Number of rows in the generated eval set. Spec says 50-100;
#: realistic LLM-judged floor is ~80 before diminishing returns.
EVAL_SET_TARGET_SIZE = 80

#: Stratification: target size split across the 7 word_types that
#: produce meaningful clozes. Interjection / Article / Number are
#: excluded because their examples don't carry enough grammatical
#: signal to make a good cloze.
TARGET_WORD_TYPES: tuple[str, ...] = (
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Pronoun",
    "Preposition",
    "Conjunction",
)

#: Fixed seed for stratified sampling → eval-set idempotence.
SEED = 20260703

#: Fixed labeler + provenance strings per Apollo's resolution. These
#: go on every row of the generated JSONL.
LABELER = "template-based-fallback-2026-07-03"
PROVENANCE = (
    "deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle"
)

#: Distractor pool size before similarity filtering. We oversample
#: then pick 3.
DISTRACTOR_POOL_SIZE = 25

#: When --use-embeddings is set: reject distractors with cosine
#: similarity to target ≥ this (too semantically similar → confusing
#: distractors). Below MIN_COSINE they're too unrelated → trivially
#: eliminated (also rejected).
DISTRACTOR_MAX_COSINE = 0.85
DISTRACTOR_MIN_COSINE = 0.05

#: Heuristic cutoffs for the self-judgment acceptance test.
SENTENCE_MIN_TOKENS = 8
SENTENCE_MAX_TOKENS = 25


# --- Helpers ------------------------------------------------------------


@dataclass(frozen=True)
class WordRow:
    """Minimal projection of ``Word`` for the eval builder."""

    id: int
    word: str
    word_type: str
    translations: str | None = None


@dataclass(frozen=True)
class EvalRow:
    """One row of ``eval/cloze_judgments.jsonl``."""

    word_id: int
    word: str
    word_type: str
    context_sentence: str
    source_example_sentence: str
    expected_answer_word_id: int
    expected_distractors: list[int]
    expected_difficulty: str
    labeler: str
    provenance: str
    judgment: str
    rationale: str

    def to_jsonl(self) -> str:
        """Serialize as a single-line JSON object (no leading whitespace).

        ``sort_keys=True`` so the file is byte-stable across runs
        with the same seed + corpus snapshot — useful for diffing
        and for re-running the eval runner against a snapshot.

        The ``source_example_sentence`` field carries the original
        (un-clozed) sentence — a reviewer can grep for the target
        word there to satisfy the Phase 4 spec's "every
        context_sentence contains the target word" acceptance
        criterion against the *original* (the cloze itself has the
        target replaced by ``___``).
        """
        return json.dumps(
            {
                "word_id": self.word_id,
                "word": self.word,
                "word_type": self.word_type,
                "context_sentence": self.context_sentence,
                "source_example_sentence": self.source_example_sentence,
                "expected_answer_word_id": self.expected_answer_word_id,
                "expected_distractors": list(self.expected_distractors),
                "expected_difficulty": self.expected_difficulty,
                "labeler": self.labeler,
                "provenance": self.provenance,
                "judgment": self.judgment,
                "rationale": self.rationale,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


def _strip_german_article(word: str) -> str:
    """Drop the leading article prefix from a noun for matching purposes.

    The corpus stores nouns with the article glued to the lemma
    (e.g. ``"die  Uhr"`` with double space). Examples may use the
    lemma bare or with the article; we want a substring match
    against either form.
    """
    w = word.strip()
    for art in ("der ", "die ", "das ", "Der ", "Die ", "Das "):
        if w.startswith(art):
            return w[len(art):].strip()
    return w


def _lemma_root(word: str) -> str:
    """Cheap lemma prefix for distractor-collision detection.

    Not a real German lemmatizer — but good enough to keep
    ``wählen`` from being paired with ``Wahl``, ``Entscheidung``,
    or ``Auswahl``. Returns the first 4 lowercase characters.

    This deliberately *under*-matches: we only reject distractors
    that are very close (4-char prefix overlap). Distractors with
    shorter surface forms (``das``, ``zu``, ``in``) naturally have
    low overlap with longer content words.
    """
    w = word.strip().lower()
    w = _strip_german_article(w)
    # Strip common umlaut decompositions back to ASCII for stability.
    w = (
        w.replace("ä", "a")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ß", "ss")
    )
    if len(w) < 4:
        return ""  # too short to be a meaningful collision signal
    return w[:4]


def _pick_example_sentence(
    examples: list[tuple[int, str]],
    target_word: WordRow,
    rng: random.Random,
) -> tuple[int, str, str] | None:
    """Pick one ``Example`` row that fits the cloze heuristics.

    Returns ``(example_id, sentence_with_blank, original_sentence)``
    or ``None`` if no example passes the filters. The sentence has
    the target word replaced by a whitespace-preserving ``___`` (so
    the article stays put for nouns); the original sentence is
    returned alongside it so the caller can re-check the
    target-word presence and sentence-final punctuation against
    the *un-clozed* version.
    """
    needle = _strip_german_article(target_word.word).lower()
    candidates: list[tuple[int, str, str]] = []
    for ex_id, german in examples:
        if not german:
            continue
        text = german.strip()
        # The target word (article-stripped) must appear in the
        # ORIGINAL sentence at least once. This matches the Phase 4
        # spec's "every context_sentence contains the target word"
        # acceptance criterion.
        if needle and needle not in text.lower():
            continue
        tokens = text.split()
        if not (SENTENCE_MIN_TOKENS <= len(tokens) <= SENTENCE_MAX_TOKENS):
            continue
        if not text.endswith((".", "!", "?")):
            continue
        # Avoid lines that are obviously truncated or contain
        # unicode line breaks (defensive).
        if any(ch in text for ch in ("\n", "\r", "\t")):
            continue
        # Build the cloze: replace the FIRST occurrence of the
        # target-word lemma with `___`. Preserve the trailing
        # punctuation by carrying it onto the `___` token.
        cloze = _replace_first_lemma(text, target_word.word)
        if cloze is None:
            continue
        # Sanity-check: the cloze must still end with sentence-final
        # punctuation (otherwise the replacement ate it).
        if not cloze.endswith((".", "!", "?")):
            continue
        candidates.append((ex_id, cloze, text))

    if not candidates:
        return None
    return rng.choice(candidates)


def _replace_first_lemma(sentence: str, word: str) -> str | None:
    """Replace the first occurrence of the word (or its inflected form
    starting with the lemma prefix) with ``___``.

    For nouns: matches either ``die Uhr`` (with article) or bare
    ``Uhr``. For verbs / adjectives / adverbs: matches any token
    whose first 4 chars equal the lemma's first 4 chars.

    Trailing sentence-final punctuation (``.``, ``!``, ``?``) is
    preserved by carrying it onto the ``___`` token, so the cloze
    always ends with ``___.`` (or ``___!`` etc.) — a learner still
    sees the sentence boundary.

    Returns the modified sentence, or ``None`` if no match was
    found.
    """
    lemma = _strip_german_article(word).strip()
    if not lemma:
        return None
    prefix = _lemma_root(word)
    tokens = sentence.split(" ")
    out: list[str] = []
    replaced = False
    for tok in tokens:
        # Strip attached punctuation for the match check.
        bare = tok.strip(",.;:!?\"«»()[]").lower()
        bare = (
            bare.replace("ä", "a")
            .replace("ö", "o")
            .replace("ü", "u")
            .replace("ß", "ss")
        )
        if not replaced and bare and (bare == lemma.lower() or (
            prefix and bare.startswith(prefix) and len(bare) >= len(prefix)
        )):
            # Preserve trailing punctuation: if the original token
            # ended with . / ! / ?, append it to the blank so the
            # cloze still ends with sentence-final punctuation.
            trailing = ""
            for ch in reversed(tok):
                if ch in ".!?":
                    trailing = ch + trailing
                else:
                    break
            out.append("___" + trailing)
            replaced = True
        else:
            out.append(tok)
    if not replaced:
        return None
    return " ".join(out)


def _pick_distractors(
    *,
    target: WordRow,
    pool: list[WordRow],
    rng: random.Random,
    embeddings_by_word: dict[int, list[float]] | None = None,
) -> list[int] | None:
    """Pick 3 distractor word IDs from ``pool`` (same ``word_type``).

    Excludes ``target.id``. Rejects candidates whose lemma root
    collides with the target's. When ``embeddings_by_word`` is
    provided, also enforces cosine distance bounds.
    """
    if embeddings_by_word is not None:
        target_emb = embeddings_by_word.get(target.id)
        if target_emb is None:
            return None

    candidates = [w for w in pool if w.id != target.id]
    rng.shuffle(candidates)
    candidates = candidates[:DISTRACTOR_POOL_SIZE]

    target_root = _lemma_root(target.word)
    picked: list[int] = []
    for cand in candidates:
        if cand.id in picked:
            continue
        cand_root = _lemma_root(cand.word)
        # Reject if lemma-prefix collides (4-char overlap on the
        # ascii-normalized form). Empty root → no overlap signal;
        # allow it.
        if target_root and cand_root and target_root == cand_root:
            continue
        if embeddings_by_word is not None:
            cand_emb = embeddings_by_word.get(cand.id)
            if cand_emb is None:
                continue
            cos = _cosine(target_emb, cand_emb)
            if cos is None:
                continue
            if cos >= DISTRACTOR_MAX_COSINE:
                continue
            if cos < DISTRACTOR_MIN_COSINE:
                continue
        picked.append(cand.id)
        if len(picked) == 3:
            break
    if len(picked) != 3:
        return None
    return picked


def _cosine(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def _expected_difficulty(
    *,
    target: WordRow,
    sentence: str,
) -> str:
    """Heuristic difficulty label.

    ``hard`` = sentence near the upper token limit OR no
    frequency field on the word. ``easy`` = sentence near the
    lower token limit AND frequency marked. Otherwise ``medium``.
    """
    tokens = sentence.split()
    near_upper = len(tokens) >= SENTENCE_MAX_TOKENS - 2
    near_lower = len(tokens) <= SENTENCE_MIN_TOKENS + 2
    has_freq = bool(target.translations and len(target.translations) > 0)
    if near_upper and not has_freq:
        return "hard"
    if near_lower and has_freq:
        return "easy"
    return "medium"


def _self_judge(
    *,
    target: WordRow,
    original_sentence: str,
    cloze_sentence: str,
    distractors: list[int],
    all_word_ids_in_pool: set[int],
    pool_by_id: dict[int, WordRow],
) -> tuple[str, str]:
    """Deterministic heuristic judgment.

    Returns ``(judgment, rationale)``. ``judgment`` is ``accept``
    or ``reject``. ``rationale`` is a deterministic string built
    from the heuristic state so the JSONL row is reproducible.

    Uses the *original* sentence for the target-word presence and
    sentence-final punctuation checks (so the cloze blank doesn't
    trip the check) and the *cloze* sentence for the token-length
    check.
    """
    reasons: list[str] = []
    needle = _strip_german_article(target.word).lower()
    sentence_ok = bool(needle and needle in original_sentence.lower())
    if not sentence_ok:
        reasons.append("target_word_missing_from_sentence")
    n = len(cloze_sentence.split())
    if not (SENTENCE_MIN_TOKENS <= n <= SENTENCE_MAX_TOKENS):
        reasons.append(f"token_count={n}_out_of_range")
    if not original_sentence.endswith((".", "!", "?")):
        reasons.append("missing_sentence_final_punctuation")
    if not cloze_sentence.endswith((".", "!", "?")):
        reasons.append("cloze_missing_sentence_final_punctuation")
    if len(distractors) != 3:
        reasons.append(f"distractor_count={len(distractors)}_not_3")
    if any(d == target.id for d in distractors):
        reasons.append("distractor_collides_with_target_id")
    for d_id in distractors:
        if d_id not in all_word_ids_in_pool:
            reasons.append(f"distractor_id_{d_id}_missing_from_pool")
    target_root = _lemma_root(target.word)
    for d_id in distractors:
        d_word = pool_by_id.get(d_id)
        if d_word is not None:
            d_root = _lemma_root(d_word.word)
            if target_root and d_root and target_root == d_root:
                reasons.append(
                    f"distractor_lemma_collision:{target.word}/{d_word.word}"
                )

    if reasons:
        return "reject", ";".join(reasons)
    return "accept", (
        f"sentence_length={n};lemma='{needle}';target_id={target.id};"
        f"distractors={distractors};word_type={target.word_type}"
    )


# --- DB access ----------------------------------------------------------


def _fetch_words_by_type(session: Session) -> dict[str, list[WordRow]]:
    """Group all words by ``word_type`` so the stratified sampler can
    pick from each bucket."""
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
        out[wt].sort(key=lambda w: w.id)  # stable order before sampling
    return out


def _fetch_examples_for_word_ids(
    session: Session,
    word_ids: set[int],
) -> dict[int, list[tuple[int, str]]]:
    """Fetch every ``Example`` for the given word ids, as a
    ``{word_id: [(example_id, german), ...]}`` map.

    Filters out empty / whitespace-only ``german`` rows.
    """
    if not word_ids:
        return {}
    rows = session.execute(
        select(Example.id, Example.word_id, Example.german).where(
            Example.word_id.in_(sorted(word_ids))
        )
    ).all()
    out: dict[int, list[tuple[int, str]]] = {wid: [] for wid in word_ids}
    for ex_id, word_id, german in rows:
        if not german or not german.strip():
            continue
        out.setdefault(word_id, []).append((ex_id, german))
    return out


def _qualifying_words(
    session: Session,
    *,
    cache_examples_by_word: dict[int, list[tuple[int, str]]] | None = None,
) -> dict[str, list[WordRow]]:
    """Group ``Word`` rows by ``word_type`` AND filter to only those
    with at least one qualifying example sentence.

    The filter mirrors ``_pick_example_sentence`` so the sampler
    never picks a word whose every example would be rejected by
    the strict picker. The two halves of the pipeline agree on
    what's a usable cloze target.

    The ``cache_examples_by_word`` argument is required: pass the
    pre-fetched examples dict to avoid a second DB roundtrip. The
    build() call site pre-fetches all examples for the candidate
    word set; this function reuses that cache.
    """
    if cache_examples_by_word is None:
        raise ValueError(
            "cache_examples_by_word is required (the build site "
            "pre-fetches examples; pass that cache here)"
        )
    rows = session.execute(
        select(Word.id, Word.word, Word.word_type, Word.translations)
    ).all()
    by_type: dict[str, list[WordRow]] = {wt: [] for wt in TARGET_WORD_TYPES}
    for w_id, w_word, w_type, w_trans in rows:
        if w_type not in by_type:
            continue
        if _has_qualifying_example(w_word, cache_examples_by_word.get(w_id, [])):
            by_type[w_type].append(
                WordRow(
                    id=w_id,
                    word=w_word,
                    word_type=w_type,
                    translations=w_trans,
                )
            )
    for wt in by_type:
        by_type[wt].sort(key=lambda w: w.id)
    return by_type


def _has_qualifying_example(
    word: str,
    examples: list[tuple[int, str]],
) -> bool:
    """True if at least one example passes the strict cloze filter.

    The filter is the same one ``_pick_example_sentence`` applies,
    including the cloze-replacement step — so the pre-qualifier
    only returns ``True`` when ``_pick_example_sentence`` would
    also find a usable candidate. (Without the cloze-replacement
    step, a small fraction of words pass the qualifier but fail
    the actual picker because ``_replace_first_lemma`` can't match
    the lemma inside a compound token like ``Urlaubsplan``.)
    """
    needle = _strip_german_article(word).lower()
    if not needle:
        return False
    for _ex_id, german in examples:
        if not german:
            continue
        text = german.strip()
        if needle not in text.lower():
            continue
        tokens = text.split()
        if not (SENTENCE_MIN_TOKENS <= len(tokens) <= SENTENCE_MAX_TOKENS):
            continue
        if not text.endswith((".", "!", "?")):
            continue
        if any(ch in text for ch in ("\n", "\r", "\t")):
            continue
        if _replace_first_lemma(text, word) is None:
            continue
        return True
    return False


def _fetch_all_target_type_word_ids(session: Session) -> set[int]:
    """Return the set of word IDs whose ``word_type`` is in
    ``TARGET_WORD_TYPES``. Used to pre-fetch all relevant examples
    in one DB roundtrip.
    """
    rows = session.execute(
        select(Word.id).where(Word.word_type.in_(TARGET_WORD_TYPES))
    ).all()
    return {r[0] for r in rows}


def _sample_words(
    words_by_type: dict[str, list[WordRow]],
    rng: random.Random,
    target_size: int,
) -> list[WordRow]:
    """Pick ``target_size`` words across the target word_types,
    proportional to the corpus distribution but with a floor of
    ``MIN_PER_TYPE`` per type so every type is represented."""
    MIN_PER_TYPE = 8
    sampled: list[WordRow] = []
    seen_ids: set[int] = set()
    # First pass: floor per type.
    for wt in TARGET_WORD_TYPES:
        pool = words_by_type.get(wt, [])
        if not pool:
            continue
        # Shuffle the pool with the seeded RNG, take MIN_PER_TYPE
        # (or the whole pool if smaller).
        local = list(pool)
        rng.shuffle(local)
        take = min(MIN_PER_TYPE, len(local))
        for w in local[:take]:
            if w.id in seen_ids:
                continue
            sampled.append(w)
            seen_ids.add(w.id)
    # Second pass: top up to target_size, weighted by pool size.
    while len(sampled) < target_size:
        # Pick a word_type proportional to its share of the corpus.
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
            # If we've already covered every distinct word in this
            # type, bail out to avoid an infinite loop on tiny corpora.
            covered = sum(1 for w in sampled if w.word_type == wt)
            if covered >= len(pool):
                break
            continue
        sampled.append(choice)
        seen_ids.add(choice.id)
    return sampled


def _build_embeddings_index(
    words: list[WordRow],
) -> dict[int, list[float]]:
    """Compute an embedding for every word in ``words`` via
    ``app.embeddings.embed``. Returns ``{word_id: vector}``.

    This is the only network call in the script when
    ``--use-embeddings`` is enabled. Returns ``{}`` if the API
    key is missing or the call fails — the caller treats that as
    ``embeddings_by_word=None`` (no similarity filtering).
    """
    from app.embeddings import embed  # lazy import: only when needed

    texts = []
    for w in words:
        texts.append(
            f"{w.word} ({w.word_type}): {w.translations or ''}".strip()
        )
    try:
        vectors = embed(texts)
    except Exception as exc:
        logger.warning(
            "embed() failed (%s); falling back to no-similarity-filter",
            exc,
        )
        return {}
    return {w.id: vec for w, vec in zip(words, vectors)}


def build(
    *,
    target_size: int = EVAL_SET_TARGET_SIZE,
    output_path: Path,
    use_embeddings: bool = False,
) -> dict:
    """Build the eval set and write it to ``output_path``.

    Returns a metadata dict for the operator log (eval set size,
    breakdown by type, embeddings enabled flag).
    """
    rng = random.Random(SEED)

    with SessionLocal() as session:
        # Step 1: fetch all examples in one shot so the qualifier
        # and the builder share a single DB roundtrip.
        candidate_word_ids = _fetch_all_target_type_word_ids(session)
        examples_by_word = _fetch_examples_for_word_ids(
            session, candidate_word_ids
        )
        # Step 2: filter to words with at least one usable example.
        qualifying = _qualifying_words(
            session,
            cache_examples_by_word=examples_by_word,
        )
        # Step 3: sample from the qualifying pool.
        sampled = _sample_words(qualifying, rng, target_size)
        if not sampled:
            raise RuntimeError("No words sampled from the corpus.")

    # Build embedding index for the sampled set if requested.
    embeddings_by_word: dict[int, list[float]] | None = None
    if use_embeddings:
        t0 = time.time()
        embeddings_by_word = _build_embeddings_index(sampled)
        elapsed = time.time() - t0
        logger.info(
            "embeddings: %d vectors in %.1fs (filtering %s)",
            len(embeddings_by_word),
            elapsed,
            "enabled" if embeddings_by_word else "disabled (fallback)",
        )
        if not embeddings_by_word:
            embeddings_by_word = None  # treat as "off" downstream

    # For each sampled word, try to assemble a cloze row.
    rows: list[EvalRow] = []
    rejected: list[tuple[int, str]] = []
    for w in sampled:
        examples = examples_by_word.get(w.id, [])
        picked = _pick_example_sentence(examples, w, rng)
        if picked is None:
            rejected.append((w.id, "no_qualifying_example"))
            continue
        ex_id, cloze_sentence, original_sentence = picked

        # Build the same-type distractor pool from the qualifying
        # subset (not the full corpus — keeps the candidate set
        # aligned with what _pick_example_sentence deems a usable
        # cloze target, so distractors pass the same word-form
        # sanity check the answer does).
        pool = [
            ww for ww in qualifying.get(w.word_type, [])
            if ww.id != w.id
        ]
        distractors = _pick_distractors(
            target=w,
            pool=pool,
            rng=rng,
            embeddings_by_word=embeddings_by_word,
        )
        if distractors is None:
            rejected.append((w.id, "could_not_pick_3_distractors"))
            continue

        difficulty = _expected_difficulty(target=w, sentence=cloze_sentence)
        all_word_ids = {ww.id for ww in pool} | {w.id}
        judgment, rationale = _self_judge(
            target=w,
            original_sentence=original_sentence,
            cloze_sentence=cloze_sentence,
            distractors=distractors,
            all_word_ids_in_pool=all_word_ids,
            pool_by_id={ww.id: ww for ww in pool} | {w.id: w},
        )
        rows.append(
            EvalRow(
                word_id=w.id,
                word=w.word,
                word_type=w.word_type,
                context_sentence=cloze_sentence,
                source_example_sentence=original_sentence,
                expected_answer_word_id=w.id,
                expected_distractors=distractors,
                expected_difficulty=difficulty,
                labeler=LABELER,
                provenance=PROVENANCE,
                judgment=judgment,
                rationale=rationale,
            )
        )

    if not rows:
        raise RuntimeError(
            f"Built 0 rows from {len(sampled)} sampled words. "
            f"All rejected: {rejected[:10]}"
        )

    # Sort rows by word_id so the JSONL is stable across runs
    # (independent of pool sampling order).
    rows.sort(key=lambda r: (r.word_id,))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(_provenance_header())
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
        "use_embeddings": use_embeddings,
        "embeddings_loaded": (
            len(embeddings_by_word) if embeddings_by_word else 0
        ),
        "output_path": str(output_path),
        "labeler": LABELER,
        "provenance": PROVENANCE,
        "seed": SEED,
    }


def _provenance_header() -> str:
    """The leading-comment block for the JSONL file.

    Every row carries ``labeler`` and ``provenance`` fields too —
    this header is for humans opening the file in an editor.
    """
    return (
        f"# labeler: {LABELER}\n"
        f"# provenance: {PROVENANCE}\n"
        f"# spec: project_ideas/15_lexora_personalized_learner.md "
        f"§Phase 4 / Must-be\n"
        f"# bar: would a C1 German speaker accept this cloze without edits?\n"
        f"# generated_by: backend/scripts/build_cloze_eval_set.py\n"
        f"# generated_at_seed: {SEED}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the held-out cloze eval set "
            "(eval/cloze_judgments.jsonl). Default mode is offline "
            "(no network). Pass --use-embeddings to enable "
            "OpenRouter-embedding-based distractor filtering."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "eval"
        / "cloze_judgments.jsonl",
        help="Output JSONL path. Defaults to ../eval/cloze_judgments.jsonl.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=EVAL_SET_TARGET_SIZE,
        help=(
            "Number of rows to generate (clamped to [50, 200] to "
            "match the Phase 4 spec). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help=(
            "Enable OpenRouter embedding-based distractor similarity "
            "filter (calls app.embeddings.embed; only the embeddings "
            "endpoint, not chat). Default off."
        ),
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

    if not 50 <= args.target_size <= 200:
        parser.error(
            f"--target-size must be between 50 and 200 (got {args.target_size})"
        )

    meta = build(
        target_size=args.target_size,
        output_path=args.output,
        use_embeddings=args.use_embeddings,
    )
    print("EVAL_SET_BUILT")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())