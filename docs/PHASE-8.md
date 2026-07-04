# lexora — Phase 8: idioms (`phrases` table) + idiom exercise type

> **Outcome-led spec.** This doc is the source of truth for the Phase 8
> rollout on the lexora board. The kanban plan card defers to this file
> for scope; the build cards (8.1–8.4, review card 8.5) read it as
> their first hand-off document.
>
> **Authoritative references** for Phase 8 scope:
> - `lexora/docs/PHASE-7.md` lines 232–246 (the explicit "What Phase 8
>   picks up" deferral).
> - `lexora/NOTES.md` §"The hidden gem" (the planted `fsrs_cards` table
>   is Phase 9+, not Phase 8 — Phase 8 stays on the wire-level
>   exercise-surface widening).

## What Phase 8 ships (outcome-led)

Phase 7 added a 4th exercise type's worth of **schema + DSPy + endpoint**
plumbing, but didn't ship idioms. Phase 8 closes the deferred idiom
work: a curated `phrases` table (multi-word fixed expressions that
aren't compositional — `ins Blaue hinein`, `Tomaten auf den Augen`),
an `app.idiom.py` DSPy module that consumes it, a new
`Literal["cloze","matching","comprehension","idiom"]` exercise type,
and 200–500 hand-curated idioms (DWDS Idiome + selected Goethe /
Schiller attested usages).

The closed-loop outcome: `/exercises/idiom` returns a generated
idiom-exercise on the same auth-gated, Ragas-traced, RAG-on-optional
shape as the Phase 6 exercise types. Wire-level; Phase 9 picks up
the in-app study-session UI that mixes exercise types.

### Why this is `weekend-scale` not `phase-scale`

- **No new infra.** No new service, no new dependency beyond the
  Phase 6 `sentence-transformers` + `instructor` + DSPy stack.
- **No schema-curated vs LLM-learned debate.** Idioms are
  hand-curated from public reference sources, the same model as
  Phase 7's `collocations` and `prepositional_objects` tables. The
  generator reads, never writes.
- **No retrieval-quality A/B reset.** The Phase 7.5 A/B verdict
  (`no_significant_lift`) carries forward. We do not re-run the A/B
  unless the new phrase table materially changes the noun-phrase
  retrieval space (it doesn't — phrases are a parallel surface, not
  a deeper one).
- **No frontend.** Wire-level endpoint only. Phase 9 is the study-
  session mixing UI.

## Concrete cards

1. **8.1 — `phrases` schema + Alembic migration + 200-row DWDS
   seed (perseus, 2h).** One SQLAlchemy model, one Pydantic schema,
   one Alembic migration, one seed script pulling from a public
   DWDS Idiome export (curated subset, target ≥200 rows with
   source URL per row). 100% offline test against a fixture subset.
   Mirror the Phase 7.1 `collocations` discipline: hand-curated,
   `Phrase.id` is a slug (`ins-blaue-hinein`), no LLM writes.
2. **8.2 — Goethe/Schiller attestation extension (200–300 rows)
   (perseus, 1.5h).** A second seed script adding 200–300 idiom
   attestations from a public Goethe/Schiller concordance subset
   (Project Gutenberg or DWDS Belegarchiv). Each row carries
   `attested_quote`, `attested_source` (work + chapter or page),
   and a `frequency_band: Literal["high","mid","low"]` column. The
   generator may use `attested_quote` as the surface form for the
   cloze-with-context variant (Phase 8.4).
3. **8.3 — `app.idiom.py` DSPy module + `Literal` widening
   (perseus, 2h).** `IdiomSignature`, `IdiomModule`,
   `generate_idiom`. The `Literal["cloze","matching","comprehension"]`
   on `exercise_type` widens to
   `Literal["cloze","matching","comprehension","idiom"]` (Hard rule:
   widening is wire-level; existing callers parse as before). DSPy
   offline-capable via `DummyLM` swap (Phase 4.2 + 7.2 pattern).
4. **8.4 — `POST /exercises/idiom` endpoint + RAG-on opt-in
   (perseus, 1.5h).** Mirror of Phase 6.5 `/exercises/comprehension`:
   same auth gate, same `enable_rag: bool = False` body, same
   `IdiomExercise` Pydantic schema enforcing bounds on `phrase`
   (5–200 chars), `definition` (1–400 chars), `example_usage` (5–400
   chars), and `source_attribution: Literal["dwds","goethe","schiller"]`
   (or comma-joined). When `enable_rag=True`, the generator pulls
   the top-1 nearest-neighbor from `phrases` table and embeds the
   `definition` + `attested_quote` in the prompt. When `False`,
   byte-for-byte the curated-table-only path. 422 on
   `source_attribution` outside the literal.
5. **8.5 — Helena code review PASS/FAIL (helena, 1.5h).** Severity-
   tagged findings on the 4 build cards (8.1–8.4). Child of all 4
   builds. Same shape as Phase 6.9 / 7.7.

Phase 8 budget: ~7 build hours + 1.5 review hours. Roughly half the
size of Phase 7 — one exercise type, no bilingual flag, no
retrieval A/B.

## What is NOT in Phase 8 (deferred — keep the discipline)

- **In-app study-session UI** that mixes cloze + matching +
  comprehension + idiom in a single session flow. That's Phase 9.
  Phase 8 is wire-level only.
- **FSRS in-app review** (the planted `fsrs_cards` table in
  `backend/data/vocabeo_words.db`). That's a multi-card phase
  (FSRS schema wire-up + review endpoint + FSRS-graded recall
  surface + due-queue UX). Phase 8 doesn't touch it.
- **LLM-curated phrase generation** (a DSPy optimizer loop that
  proposes idioms and writes them back to the table). The
  `phrases` table is hand-curated, like the `collocations` and
  `prepositional_objects` tables. No `INSERT` from runtime.
- **Re-running the retrieval-quality A/B.** The Phase 7.5 verdict
  holds; idioms don't change the noun-phrase retrieval space.
  Revisit only if Phase 9 introduces FSRS-graded recall that needs
  a different retrieval quality floor.
- **Phrase-to-phrase matching exercise** (`"match the idiom to its
  meaning"`). The `Literal` widening in 8.3 adds the `idiom`
  *type*; a multi-idiom matching exercise would be a 5th type and
  is Phase 9 scope.

## Hard rules (apply to every 8.x build card)

1. **Alembic owns migrations, not `Base.metadata.create_all`.** 8.1
   ships an Alembic migration; the Phase 6 lifespan logger's
   `create_all` call stays unchanged (it never touches the new
   table — same pattern as Phase 7.1). If the lifespan ever tries
   to `create_all` the `phrases` table, that's a regression to
   flag.
2. **`phrases` is read-only at runtime.** No `INSERT` / `UPDATE` /
   `ON CONFLICT` from the generator. Hand-curated seed scripts are
   the only write path. Mirrors Phase 7.1 collocations discipline.
3. **Pydantic `Literal` widening is wire-level.** 8.3 widens the
   `exercise_type` literal to include `"idiom"`. Existing callers
   passing `"cloze"` / `"matching"` / `"comprehension"` parse as
   before. The opposite direction — narrowing — would silently
   break callers. Phase 8 never narrows.
4. **Langfuse traces on every LLM call.** 8.3's `IdiomModule` wires
   to the `lexora` project (the existing one), not a new project.
   The Langfuse wrapper from `app/observability.py` is the only
   allowed trace path.
5. **Offline / `DummyLM` discipline.** 8.3 tests run under
   `DummyLM`; the manual Ragas regression in 8.4's `make eval` is
   the only path that requires the live LLM. CI stays offline.
6. **Bge-m3 / OpenRouter privacy filter.** If 8.3 ever needs an
   embedding call (it shouldn't — the `phrases` table is the
   retrieval target, not bge-m3), the path is local
   `sentence-transformers` per Phase 1.3, never OpenRouter's
   chat-model call (which is privacy-filter-blocked).
7. **No env-derived thresholds.** 8.4's `enable_rag` is a boolean
   flag, not a confidence floor. The Phase 6.7 Ragas floor
   constants apply to the live `make eval` run, not the wire
   surface.
8. **Schema-curated, not LLM-learned.** The `phrases` table is
   hand-curated from public reference sources. The generator
   reads, never writes. No DSPy optimizer path touches the table.
9. **No frontend work.** 8.1–8.5 are all backend / data / review.
   The `frontend/src/App.tsx` is unchanged. Wire-level endpoints
   only. Phase 9 picks up the study-session mixing UI.

## Files affected (anticipated)

- `backend/app/models.py` — new `Phrase` SQLAlchemy model.
- `backend/app/schemas.py` — new `Phrase` Pydantic model +
  `IdiomExercise` response model + widened `Literal` on
  `BaseExerciseFields.exercise_type`.
- `backend/alembic/versions/` — new migration adding `phrases`
  table with columns `id` (slug PK), `phrase` (TEXT, UNIQUE),
  `definition` (TEXT), `example_usage` (TEXT), `source_attribution`
  (TEXT — comma-joined literal), `frequency_band` (TEXT literal),
  `dwds_url` (TEXT, nullable), `attested_quote` (TEXT, nullable),
  `attested_source` (TEXT, nullable), `created_at` (TIMESTAMPTZ).
- `backend/app/idiom.py` — new DSPy module (`IdiomSignature`,
  `IdiomModule`, `generate_idiom`).
- `backend/app/main.py` — new `/exercises/idiom` route mirroring
  `/exercises/comprehension`.
- `backend/scripts/seed_phrases_dwds.py` — DWDS Idiome seed
  (≥200 rows).
- `backend/scripts/seed_phrases_attestations.py` — Goethe/Schiller
  attestation extension (200–300 rows).
- `backend/tests/test_idiom.py` — 100% offline tests under
  `DummyLM`.
- `backend/tests/fixtures/phrases_fixture.json` — small fixture
  for the seed-script tests.
- `README.md` — Limitations honesty update for Phase 8 (Athena's
  card, separate from 8.1–8.4 — added as 8.6 if budget allows, or
  folded into Phase 9's README pass).

## Gotchas anticipated (the lessons learned)

1. **DWDS Idiome export format.** DWDS exports idioms as XML with
   `<Lemma>` + `<Definition>` + optional `<Example>` blocks. The
   8.1 seed script must handle the optional `<Example>` (some
   idioms have no attested example in the export) and must slugify
   the `Lemma` for the `id` PK. Test fixture covers both shapes.
2. **Multi-author attestation concatenation.** Goethe's *Faust* and
   Schiller's *Wilhelm Tell* attestations land in the same
   `attested_source` column (e.g. `"Faust I, Studierzimmer (1168-1186)"`).
   The 8.2 seed script preserves the original punctuation and
   doesn't try to normalize to a single citation format. Phase 9
   may add a citation-formatter; Phase 8 doesn't.
3. **`Literal` widening with `Literal["cloze","matching","comprehension","idiom"]`.**
   Existing `ClozeGenerateRequest` and `MatchingGenerateRequest`
   and `ComprehensionGenerateRequest` Pydantic models don't carry
   the `exercise_type` literal — they have a `prompt_template_version`
   field instead. The widening happens on
   `BaseExerciseFields.exercise_type`. 8.3's tests assert that
   none of the 3 existing endpoints regress to a 422 on the default
   call shape (the `exercise_type` widening is additive, not
   breaking).
4. **`Base.metadata.create_all` short-circuit.** Same risk as
   Phase 7.1 — if the lifespan tries to `create_all` the `phrases`
   table before Alembic, the migration becomes a no-op. 8.1's
   body explicitly verifies the lifespan doesn't list `phrases` in
   its `create_all` call. (It shouldn't — the lifespan uses
   `models.Base.metadata.create_all` for ALL tables, so any new
   Alembic-added table gets created inline; the fix from Phase 7.1
   was to drop the `create_all` entirely. Phase 8.1 verifies the
   drop is still in place.)
5. **Idiom definition length variance.** Some idioms have
   dictionary definitions under 50 chars (`Tomaten auf den Augen`
   = "blind for what's obvious"); others have multi-sentence
   explanations. The 8.4 Pydantic schema enforces 1–400 chars on
   `definition` (a tight cap that forces the generator to compress
   long DWDS definitions into learner-friendly ones via the
   prompt, mirroring the Phase 6.5 comprehension rationale rule).
6. **`source_attribution` column shape.** Phase 8.1 declares it as
   `TEXT` (comma-joined literal values), not as a separate
   junction table. The literal is `Literal["dwds","goethe","schiller"]`
   and a row can have `"dwds,goethe"` if the same idiom is in both
   sources. 8.4's endpoint validates the comma-joined string
   against the literal. Phase 9 may refactor to a junction table
   if querying by source becomes common.
