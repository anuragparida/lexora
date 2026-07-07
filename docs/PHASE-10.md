# lexora — Phase 10: phrase-to-phrase matching (`phrase_pairs` table) + 5th exercise type

> **Outcome-led spec.** This doc is the source of truth for the Phase 10
> rollout on the lexora board. The kanban plan card defers to this file
> for scope; the build cards (10.1–10.7, review card 10.9, doc card 10.10)
> read it as their first hand-off document.
>
> **Authoritative references** for Phase 10 scope:
> - `lexora/docs/PHASE-8.md` §"What is NOT in Phase 8" (the explicit
>   "LLM-curated phrase generation is its own multi-card phase" deferral
>   that Phase 10 honors as a content deferral, not a bug).
> - `lexora/NOTES.md` §"Future direction — three credible paths" (the
>   planted `phrases` table from Phase 8.1 is the surface Phase 10 widens
>   via `phrase_pairs`).
> - `lexora/docs/PHASE-7.md` §"Retrieval-quality A/B verdict" (the
>   Phase 7.5 verdict carries forward; Phase 10 does NOT re-run the A/B).

## What Phase 10 ships (outcome-led)

Phase 8 planted a `phrases` table (idioms + Goethe/Schiller attestations)
and an `idiom` exercise type. Phase 9 mixed the four exercise types into
the `/exercises/session` study-session UI. Phase 10 closes the
phrase-to-phrase deferred item: a curated `phrase_pairs` table that
pairs phrases by relation (equivalent / paraphrase / related / unrelated),
a `phrase_match` exercise type that asks the learner to judge the
relation between two phrases, and a fourth DSPy optimizer script.

The closed-loop outcome: `POST /exercises/phrase_match` returns a
generated phrase-match exercise on the same auth-gated, Ragas-traced,
RAG-on-optional shape as the Phase 6 exercise types — but with a
**4-button relation picker** as the answer surface (instead of a cloze
text input or matching grid) and a **3-button FSRS grade** as the
self-assessment. The two are stored in distinct fields (`relation` vs.
`grade`). Wire-level exercise surface widens from four types to five.

### Why this is `weekend-scale` not `phase-scale`

- **No new infra.** No new service, no new dependency beyond the
  Phase 6 `sentence-transformers` + `instructor` + DSPy stack, plus
  the existing `phrases` table from Phase 8.1.
- **No schema-curated vs LLM-learned debate.** `phrase_pairs` is a
  deterministic-seeding curated surface, NOT LLM-written. The pairing
  rule lives in code (`backend/scripts/seed_phrase_pairs.py --seed 42`),
  not in a DSPy signature output. This is the Phase 8 explicit deferral
  ("LLM-curated phrase generation is its own multi-card phase"),
  honored as a content deferral, not a bug.
- **No retrieval-quality A/B reset.** The Phase 7.5 A/B verdict
  (`no_significant_lift`) carries forward. We do not re-run the A/B.
  The `phrase_pairs` table is a parallel surface to `phrases` /
  `collocations` / `prepositional_objects`, not a deeper one; it
  widens the exercise surface, not the retrieval space. The
  noun-phrase retrieval space is unchanged.
- **Mirrors Phase 8's frontend discipline.** Wire-level endpoint +
  curated table only. Phase 10 widens the existing
  `/exercises/session` mixer (`SessionPage.tsx`) to route
  `phrase_match` cards to a new `PhraseMatchPage.tsx`. Phase 10
  does not introduce a new study-session flow.

## Concrete cards

1. **10.1 — `phrase_pairs` schema + Alembic migration + seed
   script (perseus, 2h).** One SQLAlchemy model, one Pydantic
   schema, one Alembic migration, one deterministic-seeding script.
   `phrase_pairs` rows are `(phrase_a_id, phrase_b_id, relation,
   attested_pair, created_at)` with a CHECK constraint
   `phrase_a_id != phrase_b_id` and a UNIQUE
   `(phrase_a_id, phrase_b_id)` constraint (the seed script sorts
   lexicographically before insert so the (a, b) pair never
   collides with its (b, a) mirror). All four columns indexed
   (`ix_phrase_pairs_phrase_a_id`, `ix_phrase_pairs_phrase_b_id`,
   `ix_phrase_pairs_relation`, `ix_phrase_pairs_attested_pair`).
   100% offline test against a fixture subset. Mirror the Phase 7.1
   `collocations` discipline: hand-curated, no LLM writes,
   `inspect()`-guarded migration so re-running `alembic upgrade
   head` is a clean no-op.
2. **10.2 — Goethe/Schiller attested-pair extension (perseus, 1.5h).**
   A second seed script adding `attested_pair=True` rows from
   `backend/data/attested_pairs.json` (Goethe/Schiller concordances
   that explicitly pair two phrases). The generator's planner uses
   the attested subset when available (so the learner sees
   real-corpus pairs before generic bge-m3-bucketed ones). Mirror
   Phase 8.2's attestation discipline: `attested_pair` is the
   hard-coded `True/False` column, indexed for a fast subset pull.
3. **10.3 — `app.phrase_match.py` DSPy module + `Literal` widening
   (perseus, 2h).** `PhraseMatchSignature`, `PhraseMatchModule`,
   `generate_phrase_match`. The
   `Literal["cloze","matching","comprehension","idiom"]` on
   `exercise_type` widens to
   `Literal["cloze","matching","comprehension","idiom","phrase_match"]`
   (Hard rule: widening is wire-level; existing callers parse as
   before). The `PhraseMatchExerciseOut` Pydantic model enforces
   bounds on `phrase_a` (5–200 chars), `phrase_b` (5–200 chars),
   `relation` (`Literal["equivalent","paraphrase","related","unrelated"]`),
   `relation_rationale` (1–400 chars), and `source_attribution`
   (`Literal["curated","attested","nearest_neighbor"]` or
   comma-joined). DSPy offline-capable via `DummyLM` swap (Phase
   4.2 + 7.2 + 8.3 pattern).
4. **10.4 — `eval/phrase_match_judgments.jsonl` + manifest
   (perseus, 1.5h).** 50 hand-labeled phrase-match pairs, each
   row carrying the pair, the relation, the rationale, and the
   source (curated / attested). Manifest
   `eval/phrase_match_judgments.manifest.json` declares the set
   `HUMAN-LABELED` (Phase 1.5a eval-set honesty discipline). No
   LLM-vs-human agreement claim is made on this set; the
   optimization signal is the hand-labeled grader, full stop.
   100% offline test against the fixture subset; no OpenRouter
   chat call; no `bge-m3` OpenRouter probe.
5. **10.5 — `frontend/src/pages/PhraseMatchPage.tsx` (perseus, 1.5h).**
   New per-type page mirroring `ComprehensionPage.tsx` (Phase 6.5):
   renders `phrase_a` + `phrase_b` side-by-side, exposes a
   4-button relation picker (equivalent / paraphrase / related /
   unrelated), and a 3-button FSRS grade as the self-assessment.
   Stores the relation as the answer field; stores the grade as the
   self-assessment field. The two are persisted separately.
6. **10.6 — `SessionPage.tsx` mixer widening + `postAuthRoute.ts`
   gate (perseus, 1h).** Wire `phrase_match` into the existing
   `/exercises/session` mixer; update the per-row `exercise_type`
   union so the union-of-`fsrs_cards` pull includes the fifth
   type. Gate widening (Phase 9 → Phase 10) is additive; no
   narrowing. The Phase 6 hard rule #11 exception ("the
   first-login gate stays cloze-only") still applies and is
   named in the README Limitations section.
7. **10.7 — `scripts/optimize_phrase_match.py` DSPy optimizer
   (perseus, 1.5h).** Mirror of `scripts/optimize_match.py`
   (Phase 9.3): same `DummyLM` discipline, same offline dry-run
   default, same Ragas-floor gate before any live run. Same
   `inspect()`-guarded manifest write, same `--live` opt-in, same
   hard-coded floor constants from `backend/app/eval/`. The
   cloze optimizer template is mirrored 1x; Phase 10 does not
   introduce a new DSPy paradigm.
8. **10.9 — Helena code review PASS/FAIL (helena, 1.5h).** Severity-
   tagged findings on the 7 build cards (10.1–10.7). Child of all
   7 builds. Same shape as Phase 6.9 / 7.7 / 8.5.
9. **10.10 — README Limitations + NOTES.md Path A flip (athena, 1h).**
   Doc-only followup card; runs after Helena's PASS. Flips
   `## Limitations (Phase 9)` to `## Limitations (Phase 10)`,
   flips `Path A` in NOTES.md from "Future direction" to
   `DONE (Phase 10)`, adds `POST /exercises/phrase_match` to the
   API table, updates the grade row's `Literal` widening
   annotation.

Phase 10 budget: ~11 build hours + 1.5 review hours + 1 doc hour.
Roughly 1.5x Phase 8 — five build cards (vs. four in Phase 8), the
optimizer card, and the new `PhraseMatchPage.tsx`.

## What is NOT in Phase 10 (deferred — keep the discipline)

- **Sixth exercise type.** The `Literal` stays at five entries.
  Phrase-to-context matching (`"fill in the missing phrase in this
  sentence"`) is a 6th type and is deferred indefinitely.
- **LLM-writes-back to `phrase_pairs`.** The Phase 8 deferral
  ("LLM-curated phrase generation is its own multi-card phase")
  holds. Phase 10 *reads* from `phrase_pairs`, never writes. The
  pairing rule is in code (`seed_phrase_pairs.py --seed 42`),
  not in a DSPy signature output. This is a content deferral,
  not a bug.
- **Cross-language phrase-to-phrase matching.** Deferred
  indefinitely per plan body. The shipped surface is German-only.
- **In-app audio / IPA / native-speaker sentences.** That's Path B
  in NOTES.md, still deferred. The `words` table doesn't have
  audio columns; Path B is its own multi-card phase.
- **FSRS in-app review / personalization.** That's Path C in
  NOTES.md, still deferred. The `fsrs_cards` table exists (with
  `exercise_type` from Phase 9.1) but is still empty. Phase 10
  ships no new personalization surface.
- **Re-running the Phase 7.5 retrieval-quality A/B.** The verdict
  holds (`no_significant_lift`). `phrase_pairs` is a parallel
  exercise surface, not a deeper retrieval space. Revisit only
  if Phase 11 introduces a retrieval path that materially changes
  the noun-phrase retrieval space.
- **`bge-m3` OpenRouter chat call claims.** `phrase_pairs` uses
  local `sentence-transformers` for the nearest-neighbor pull at
  request time, NOT OpenRouter chat. The OpenRouter account's
  privacy filter blocks `baai/bge-m3` as a chat-model call
  (Phase 1 deviation, carried forward); the local
  `sentence-transformers` cache is the only path.

## Hard rules (apply to every 10.x build card)

1. **Alembic owns migrations, not `Base.metadata.create_all`.** 10.1
   ships an Alembic migration; the Phase 6 lifespan logger's
   `create_all` call stays unchanged (it never touches the new
   table — same pattern as Phase 7.1 / 8.1). If the lifespan ever
   tries to `create_all` the `phrase_pairs` table, that's a
   regression to flag.
2. **`phrase_pairs` is read-only at runtime.** No `INSERT` /
   `UPDATE` / `ON CONFLICT` from the generator. The deterministic
   seed script is the only write path. Mirrors Phase 7.1
   `collocations` discipline + Phase 8.1 `phrases` discipline.
3. **Pydantic `Literal` widening is wire-level.** 10.3 widens the
   `exercise_type` literal to include `"phrase_match"`. Existing
   callers passing `"cloze"` / `"matching"` / `"comprehension"` /
   `"idiom"` parse as before. The opposite direction — narrowing —
   would silently break callers. Phase 10 never narrows.
4. **Langfuse traces on every LLM call.** 10.3's `PhraseMatchModule`
   wires to the `lexora` project (the existing one), not a new
   project. The Langfuse wrapper from `app/observability.py` is
   the only allowed trace path.
5. **Offline / `DummyLM` discipline.** 10.3 tests run under
   `DummyLM`; the manual Ragas regression in 10.7's
   `make eval-optimize-phrase-match` is the only path that
   requires the live LLM. CI stays offline. The hand-labeled
   eval set (10.4) is the primary optimization signal — not the
   Ragas lift number.
6. **Bge-m3 / OpenRouter privacy filter.** If 10.3 ever needs an
   embedding call (it does — for the `enable_rag=True` nearest-
   neighbor pull from `phrases`), the path is local
   `sentence-transformers` per Phase 1.3, never OpenRouter's
   chat-model call (which is privacy-filter-blocked). The pull is
   cosine-distance, not chat-completions.
7. **No env-derived thresholds.** 10.3's `enable_rag` is a
   boolean flag, not a confidence floor. The Phase 6.7 Ragas
   floor constants apply to the live `make eval` run, not the
   wire surface.
8. **Schema-curated, not LLM-learned.** The `phrase_pairs` table
   is deterministically seeded. The generator reads, never writes.
   No DSPy optimizer path touches the table. The pairing rule
   lives in `seed_phrase_pairs.py`, not in any prompt.
9. **Phase 1.5a eval-set honesty discipline.** The hand-labeled
   phrase-match pairs are tagged `HUMAN-LABELED` in the eval
   manifest. No LLM-vs-human agreement claim is made. LLM-
   generated eval sets are tagged as such; hand-labeled is the
   rare exception. Phase 10 honors the precedent.
10. **Phase 6 hard rule #11 still applies.** The first-login gate
    stays cloze-only. The Phase 9 plan card (`t_6e784cf1`) is
    the documented exception (gate widens to any due type after
    first login). Phase 10 is the surface widening on top of it.

## Files affected (anticipated)

- `backend/app/models.py` — new `PhrasePair` SQLAlchemy model
  (mirrors `Phrase` from Phase 8.1).
- `backend/app/schemas.py` — new `PhrasePair` Pydantic model +
  `PhraseMatchExerciseOut` response model + widened `Literal` on
  `BaseExerciseFields.exercise_type`.
- `backend/alembic/versions/` — new migration
  (`10a1_phrase_pairs_table.py`) adding `phrase_pairs` table with
  columns `id` (autoincrement PK, NOT slug), `phrase_a_id` (FK
  to `phrases.id`, `ondelete="RESTRICT"`), `phrase_b_id` (FK to
  `phrases.id`, `ondelete="RESTRICT"`), `relation` (TEXT),
  `attested_pair` (BOOLEAN, default `False`), `created_at`
  (TIMESTAMPTZ). Plus four indexes + UNIQUE constraint + CHECK
  constraint.
- `backend/app/phrase_match.py` — new DSPy module
  (`PhraseMatchSignature`, `PhraseMatchModule`,
  `generate_phrase_match`).
- `backend/app/main.py` — new `/exercises/phrase_match` route
  mirroring `/exercises/comprehension` + `/exercises/idiom`.
- `backend/scripts/seed_phrase_pairs.py` — deterministic-seeding
  script (paired-pair pull from `phrases` + `attested_pairs.json`
  + bge-m3 nearest-neighbor bucketing under
  `enable_rag=False`).
- `backend/scripts/optimize_phrase_match.py` — DSPy optimizer
  (Phase 10.7, mirrors Phase 4.4 + Phase 9.3 templates).
- `backend/tests/test_phrase_match.py` — 100% offline tests
  under `DummyLM`.
- `backend/tests/fixtures/phrase_pairs_fixture.json` — small
  fixture for the seed-script tests.
- `eval/phrase_match_judgments.jsonl` — 50 hand-labeled
  phrase-match pairs (Phase 10.4).
- `eval/phrase_match_judgments.manifest.json` — manifest
  declaring the set `HUMAN-LABELED`.
- `frontend/src/pages/PhraseMatchPage.tsx` — new per-type page
  (4-button relation picker + 3-button FSRS grade).
- `frontend/src/pages/SessionPage.tsx` — mixer widening (route
  `phrase_match` cards to `PhraseMatchPage.tsx`).
- `frontend/src/routing/postAuthRoute.ts` — gate widening to
  include the fifth type (additive on Phase 9 widening).
- `README.md` — Limitations honesty update for Phase 10 (Athena's
  10.10 card) + API table addition for `POST /exercises/phrase_match`
  + grade row's `Literal` widening annotation.
- `NOTES.md` — Path A flipped to `DONE (Phase 10)`; Path B + C
  deferral status preserved.

## Gotchas anticipated (the lessons learned)

1. **`phrase_pairs` PK is autoincrement, not slug.** Unlike
   `phrases.id` (a slug like `ins-blaue-hinein`), `phrase_pairs.id`
   is a synthetic autoincrement integer. The pair identity is the
   FK pair `(phrase_a_id, phrase_b_id)`, not the PK. The UNIQUE
   constraint on `(phrase_a_id, phrase_b_id)` is what enforces
   uniqueness; the PK is just a row handle.
2. **Lexicographic sort before insert.** The seed script must
   sort `(a, b)` lexicographically before insert so the `(a, b)`
   pair never collides with its `(b, a)` mirror in the UNIQUE
   constraint. The CHECK constraint
   `phrase_a_id != phrase_b_id` is the second guard against
   self-pairs.
3. **`ondelete="RESTRICT"` on the FKs.** A paired-pair row
   outlives even the removal of its parent phrase (an audit trail
   is more useful than a silent cascade). Same discipline as Phase
   7.1 `collocations.verb_lemma` FK. Phase 10.1's migration
   declares `ondelete="RESTRICT"`; the Phase 10.3 wire endpoint
   does not catch the FK violation silently.
4. **Relation picker is 4 buttons, not 3.** The answer surface is
   `Literal["equivalent","paraphrase","related","unrelated"]` —
   strictly four values, not a graded 1–5 scale. The 3-button
   FSRS grade (`again` / `hard` / `good`) is the self-assessment,
   not the answer. The two are persisted as distinct fields.
   `frontend/src/pages/PhraseMatchPage.tsx` must wire both; the
   `/exercises/phrase_match` POST stores the relation in
   `relation`, the `/exercises/grade` POST stores the FSRS grade
   in `grade`.
5. **`source_attribution` literal shape.** Phase 10.3 declares it
   as a `Literal["curated","attested","nearest_neighbor"]` (or
   comma-joined). `curated` rows come from the deterministic
   seed script; `attested` rows come from
   `backend/data/attested_pairs.json` (Goethe/Schiller); the
   comma-joined `"curated,attested"` shape is allowed when a row
   has both origins.
6. **Local `sentence-transformers` for the nearest-neighbor pull.**
   `enable_rag=True` pulls the top-1 nearest-neighbor from
   `phrases` via cosine distance on local
   `sentence-transformers`. This is NOT a chat-completions call
   to OpenRouter — OpenRouter's privacy filter blocks
   `baai/bge-m3` as a chat-model call. The pull is embedding
   math, not prompt engineering.
7. **Phase 7.5 verdict carry-forward.** Phase 10 does NOT re-run
   the retrieval-quality A/B. The `phrase_pairs` table is a
   parallel exercise surface, not a deeper retrieval space. The
   noun-phrase retrieval space is unchanged. If a future phase
   materially changes the retrieval space (e.g. adds a phrase-
   to-phrase retrieval path), revisit the A/B.
8. **Eval-set honesty discipline.** The 50 hand-labeled
   phrase-match pairs are tagged `HUMAN-LABELED` in the manifest.
   No LLM-vs-human agreement claim is made on this set. The
   optimization signal is the hand-labeled grader, full stop. If
   the optimizer ever runs against this set, the report must
   surface the `HUMAN-LABELED` tag prominently; an
   LLM-generated eval set would not get the same treatment.