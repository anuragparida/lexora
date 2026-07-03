# Lexora — Eval set + offline runner

> Card `t_da712d54` (Phase 4.4, `perseus`).
> Parent: `t_706a4ffa` (Phase 4 plan).

This document explains how to re-generate the held-out cloze eval
set (`eval/cloze_judgments.jsonl`) and how to run a model's
predictions against it. Mirrors the README's "Embeddings &
retrieval" section shape from Phase 1 — quickstart, then the
deep reference.

## What the eval set is

`eval/cloze_judgments.jsonl` is a held-out, labeled triple set
for the cloze-exercise generator. Each row carries:

- `word_id` (FK to `words.id`) and `word` (the surface form).
- `word_type` (Noun / Verb / Adjective / Adverb / Pronoun /
  Preposition / Conjunction — Phase 4's seven clozable types).
- `context_sentence` — the German sentence with the target word
  replaced by `___`.
- `source_example_sentence` — the original un-clozed sentence
  (so a reviewer can grep for the target word there).
- `expected_answer_word_id` — FK to the target word (always equal
  to `word_id`).
- `expected_distractors` — exactly 3 FKs to words of the same
  `word_type`, chosen from the corpus with no lemma-prefix
  collision with the target.
- `expected_difficulty` — heuristic label (`easy` / `medium` /
  `hard`) derived from sentence length and word frequency.
- `labeler` — provenance string.
- `provenance` — long-form provenance.
- `judgment` — `accept` or `reject` from the build-time self-check.
- `rationale` — deterministic rationale string describing the
  heuristic state.

The leading comment block on the file declares the provenance
so the file is self-describing when opened in an editor:

```
# labeler: template-based-fallback-2026-07-03
# provenance: deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle
# spec: project_ideas/15_lexora_personalized_learner.md §Phase 4 / Must-be
# bar: would a C1 German speaker accept this cloze without edits?
```

## How the eval set is generated (template-based fallback)

The original Phase 4 plan called for an LLM-generated eval set
(`labeler: ai-assisted-claude-minimax-m3`). All 28 OpenRouter
chat models are blocked by the account's data-policy guardrail
(probed during card 4.4's first perseus run; results at
`/tmp/lexora-probe/phase44-probe/openrouter-probe.txt`). Apollo
resolved the card with **option 3 — template-based fallback**:

- For each target word, the script picks a `Word.examples` row
  whose sentence contains the lemma in surface form, is 8-25
  tokens, ends with sentence-final punctuation, and where the
  cloze-replacement step succeeds.
- Distractors are picked from the same `word_type` pool with a
  4-character-prefix collision filter (so `wählen` doesn't pair
  with `Wahl`).
- Self-judgment is deterministic: sentence length, target-word
  presence in the original sentence, distractor count, lemma
  collision, sentence-final punctuation.

The locked deviations from the original card body are recorded
in the card thread (comment id 23 of `t_da712d54`):

- `labeler` = `template-based-fallback-2026-07-03` (was
  `ai-assisted-claude-minimax-m3`)
- `provenance` =
  `deterministic-template-openrouter-chat-blocked-pending-anurag-privacy-toggle`
  (was `llm-generated-v1-pending-anurag-hand-review`)
- The generator does not call `app.llm.complete`. No OpenRouter
  chat call.

When Anurag flips the OpenRouter privacy setting (or supplies a
different provider key), a follow-up card will swap the generator
back to the LLM path. The deterministic eval set becomes the
floor that the LLM version must beat on `accept_rate`.

## How to re-generate the eval set

From `backend/`:

```bash
# Default: 80 rows, offline (no OpenRouter contact).
uv run python -m scripts.build_cloze_eval_set

# Smaller / larger eval set (clamped to [50, 200] per the spec).
uv run python -m scripts.build_cloze_eval_set --target-size 60

# With OpenRouter embedding-based distractor filtering (only the
# /embeddings endpoint, never chat). Recommended for production
# eval sets where you want high-quality distractors.
uv run python -m scripts.build_cloze_eval_set --use-embeddings

# Custom output path (defaults to ../eval/cloze_judgments.jsonl).
uv run python -m scripts.build_cloze_eval_set --output /tmp/eval.jsonl
```

The script is idempotent — re-running with the same seed
(`SEED = 20260703`, locked at module level) reproduces the same
file byte-for-byte against the same corpus snapshot. This makes
the eval set safe to commit and diff.

Exit code: 0 on success (file written, validation passes), 1 on
any unrecoverable error (DB unreachable, embedding API failure
when `--use-embeddings` is set, target size out of range).

## How to run a model against the eval set

The offline runner takes a JSONL of predictions and computes
three metrics:

```bash
# CI smoke path (no model, no OpenRouter contact). Uses the
# held-out set's own ``judgment`` column as the accept proxy so
# the metric is well-defined even without a live model.
uv run python -m scripts.eval_cloze --dry-run

# Live path: feed a JSONL of ClozeExercise predictions (one per
# line, same order as the held-out rows). Computes the three
# metrics and writes eval/results_<timestamp>.json.
uv run python -m scripts.eval_cloze \
    --predictions /path/to/preds.jsonl \
    --output-dir /path/to/results
```

The runner writes `eval/results_<timestamp>.json` with the
metrics plus a per-row diff (`schema_valid`, `semantic_equivalent`,
`predicted_answer_word_id`). The timestamp keeps concurrent
runs from clobbering each other.

## How to interpret the metrics

| Metric | What it measures | Healthy value |
|---|---|---|
| `accept_rate` | Fraction of held-out rows whose predicted `answer_word_id` matches and whose sentence shares ≥60% token overlap with the held-out sentence. | ≥ 0.95 in `--dry-run` (the held-out rows are self-accepting by construction). |
| `schema_validity_rate` | Fraction of predictions that pass Pydantic `ClozeExercise` validation. | 1.0 in live mode (anything below means a structural regression). |
| `rationale_quality_proxy` | Average character length of the `rationale` field across schema-passing predictions. | Heuristic; report the number so reviewers can spot collapses to empty strings. |

The spec's hard floor for cloze quality is qualitative — "would
a C1 German speaker accept this cloze without edits?" — not a
numeric threshold. The metrics above are the deterministic
proxies that fit in CI; the qualitative bar is Anurag's
hand-review pass when he has time (marked in the eval set's
`provenance` field).

## How to re-run the DSPy optimizer with the eval set

The DSPy optimizer CLI ships with Phase 4.2
(`scripts/optimize_cloze.py`). When 4.2 lands, the standard
loop is:

```bash
# 1. Make sure the eval set is fresh.
uv run python -m scripts.build_cloze_eval_set

# 2. Run the optimizer (uses DummyLM in CI / offline mode; real
#    LM in --live mode).
uv run python -m scripts.optimize_cloze

# 3. Measure the optimized module against the eval set.
uv run python -m scripts.eval_cloze \
    --predictions /tmp/cloze_predictions.jsonl
```

`scripts/optimize_cloze.py` is out of scope for 4.4 — this
section exists so reviewers can see the eval set's role in the
Phase 4 pipeline at a glance.

## Where this lives

```
lexora/
├── docs/
│   ├── PHASE-4.md       (Phase 4 plan; 4.4 section)
│   └── EVAL.md          (this file)
├── eval/
│   ├── cloze_judgments.jsonl      (the eval set; 80 rows)
│   └── results_<timestamp>.json   (one per runner invocation)
└── backend/
    └── scripts/
        ├── build_cloze_eval_set.py  (the generator)
        └── eval_cloze.py            (the runner)
```

## Phase 6 follow-up (Ragas + retrieval eval)

Phase 6 will add:

- Ragas eval (LLM-judged retrieval quality)
- Retrieval eval (Recall@k, MRR)
- A combined report that cites both `eval/cloze_judgments.jsonl`
  (this card) and the Phase 6 retrieval metrics.

Card 4.4 doesn't ship any of that. This card's eval set is the
floor that Phase 6's retrieval-aware eval set must beat.