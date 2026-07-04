# lexora — Phase 7: collocations + prepositional-objects schema + corpus extension + retrieval-quality A/B

> **Outcome-led spec.** This doc is the source of truth for the Phase 7
> rollout on the lexora board. The kanban plan card `t_474acaff`
> defers to this file for scope; the build cards (7.1–7.5, the
> review card 7.7) read it as their first hand-off document.
>
> **Authoritative references** for Phase 7 scope:
> - `lexora/docs/PHASE-6.md` lines 22–27 (explicit Phase 7 deferral).
> - `lexora/NOTES.md` §"Translation" (the retrieval-quality comparison
>   one-env-var change).
> - `project_ideas/15_lexora_personalized_learner.md` line 36
>   (the verbatim "Weekend 8: Phase 7 — collocations +
>   prepositional-objects schema + corpus extension").

## What Phase 7 ships (outcome-led)

Phase 6 left the exercise generator surface stable: three exercise
types (`cloze`, `matching`, `comprehension`), each as a DSPy module
plus a FastAPI endpoint, each wired to Optuna-traced cloze
generation + Ragas regression. Phase 7 widens the surface in two
directions that the Phase 6 review's "Out of scope" line (PHASE-6.md
line 24) explicitly defers:

1. **Collocations + prepositional-objects corpus.** Two new
   read-only tables — `collocations` and `prepositional_objects` —
   seeded with hand-curated German pairs (DWDS Kollokationen +
   a curated subset of Wiktionary Kollokationen). The exercise
   generator *consumes* these tables on opt-in; it never writes
   back. The schema is a curated corpus, not a learned one.
2. **Bilingual optionality.** A `partner_lang: Literal["de","en"]`
   flag on `/exercises/match` and `/exercises/cloze` enables a
   small bilingual-exercise variant (DE word + EN collocation
   match) for the 200 most-frequent words. The default stays
   German-only — bilingual is opt-in.
3. **Retrieval-quality A/B lift.** A reproducible offline runner
   that compares the current OpenRouter embedding
   (`qwen/qwen3-embedding-8b`, Phase 1 default) against the
   locally-cached `baai/bge-m3` on the Phase 4.4 held-out cloze
   set under the Phase 6.7 Ragas metrics. The deliverable is the
   comparison report (CSV + markdown), not the lift decision.

The closed-loop outcome is the bilingual corpus + comparison
report landing on disk — every other piece (schemas, DSPy module,
endpoint flags, A/B runner) is plumbing for that outcome.

### Concrete cards

1. **7.1 — `collocations` + `prepositional_objects` schema
   (perseus, 2h).** Two SQLAlchemy models, two Pydantic schemas,
   one Alembic migration, two seed scripts (DWDS +
   Wiktionary, hand-curated subset, target ≥200 rows per table),
   100% offline test.
2. **7.2 — `app/collocation.py` DSPy module + collocation-cloze
   generator (perseus, 2h).** Mirror of Phase 4.2's
   `app/cloze.py` shape: `CollocationSignature`,
   `CollocationModule`, `generate_collocation`. Langfuse trace
   wires to the `lexora` project (Hard rule). DSPy offline-capable
   via `DummyLM` swap (per Phase 4.2 pattern).
3. **7.3 — `/exercises/cloze?collocation=true` opt-in flag
   (perseus, 1.5h).** Pydantic widens `ClozeGenerateRequest` with
   `collocation: bool = False`. When `True`, the cloze prompt
   consumes the `collocations` table for the target word. Default
   `False` keeps Phase 4.2 + Phase 6.1 callers byte-for-byte
   stable (Hard rule #11).
4. **7.4 — Bilingual matching (`partner_lang` opt-in)
   (perseus, 1.5h).** `MatchingGenerateRequest` and
   `ClozeGenerateRequest` widen with
   `partner_lang: Literal["de", "en"] = "de"`. The English
   counterpart for the top 200 German collocations is a curated
   subset, seeded alongside `collocations` rows in 7.1. Tests
   cover 422 on `partner_lang` outside the literal.
5. **7.5 — Retrieval-quality A/B runner (perseus, 2.5h).**
   `backend/app/eval/retrieval_compare.py` + matching CLI
   `backend/scripts/eval_retrieval_compare.py`. Two CSV outputs
   (current / bge-m3), one markdown comparison table,
   `RAGAS_MIN_CONTEXT_PRECISION` floor encoded as a module
   constant (Hard rule #7). Test SKIPs if bge-m3 cache is cold.
6. **7.6 — README + Limitations honesty update (athena, 1h).**
   Phase 6's "RAG-on wired + Ragas regression" line becomes
   "Collocations + prepositional-objects schema + retrieval-quality
   A/B landed; bilingual exercise opt-in; bge-m3 alt is one env
   var". Includes A/B lift numbers in Limitations when known.
7. **7.7 — Helena code review PASS/FAIL (helena, 1.5h).**
   Severity-tagged findings on the 5 build cards (7.1–7.5) +
   the doc card (7.6). Child of all 5 builds + 7.6.

Phase 7 budget: ~12 build hours + 1.5 review hours. Smaller than
Phase 6 (16h) — fewer endpoints, more schema/eval work.

## What is NOT in Phase 7 (deferred — keep the discipline)

- **No idiom entries.** Phase 6 deferred them to Phase 8; that
  deferral holds. The schema's enum never reaches "idiom"; the
  collocation table's `register` column accommodates formal /
  colloquial labels, not a 4th register class.
- **No LLM-generated collocations.** All rows in `collocations`
  and `prepositional_objects` are hand-curated (DWDS +
  Wiktionary). The exercise generator reads, never writes. No
  DSPy optimizer path touches these tables in Phase 7.
- **No `app.collocation.py` prompt-optimizer run.** The DSPy
  module ships with the default prompt; offline optimization
  against a held-out collocation-cloze set is Phase 9 work.
- **No new exercise type.** The litany stays `Literal["cloze",
  "matching", "comprehension"]` plus the existing opt-in flags
  (`enable_rag`, `enable_rerank`, `collocation`, `partner_lang`).
  No `exercise_type="collocation"` 4th literal — collocations are
  a *cloze variant*, not a 4th type.
- **No new frontend pages.** The cloze, match, comprehension,
  due-cards pages from Phase 4.5 + 5.5 stay; `collocation=true`
  renders inside the existing cloze surface. Bilingual exercises
  render inside the existing match surface. Wire-level ≠ UI.
- **No new LLM provider.** OpenRouter only. Same `app.llm`
  client, same DSPy adapter, same `instructor` wrapper.
- **No new retrieval vector store.** The Phase 1 `pgvector`
  store is the only one consumed. The retrieval-quality
  comparison in 7.5 swaps *embeddings*, not *stores*.
- **No GPT-4 / Claude / paid LLM swap.** The Phase 4.1 OpenRouter
  + DSPy pipeline is the single LLM path. bge-m3 is local
  (sentence-transformers cache), no hosted API call.
- **No `fsrs_cards` row shape change.** The Phase 5.1 schema is
  the final shape; Phase 7 adds nothing to it. The
  `grade_logs.exercise_type` column accommodates the new
  collocation-cloze as a `cloze` variant, no widening.
- **No first-login gate change.** Phase 5.6's gate stays
  cloze-only — same shape as Phase 6. Collocation-cloze and
  bilingual match are reachable via direct navigation, not via
  the gate (same as Phase 6's match + comprehension pattern).
- **No `/retrieve` change.** Phase 1's `/retrieve` endpoint is
  the contract; if 7.5 needs a per-source filter, that's a
  separate card, not 7.5.

## Hard rules (apply to every 7.x build card)

These rules are enforced by each card body and Helena's review.
A build that violates any one of them is `FAIL`.

1. **One env var, one offline eval.** The retrieval-quality A/B
   (7.5) flips a single env var (`EMBEDDING_MODEL` →
   `"baai/bge-m3"` vs the current `qwen/qwen3-embedding-8b`),
   run against the Phase 4.4 held-out cloze set + the Phase 6.7
   Ragas runner. No code change to the embedding client —
   `app/embeddings.py` consumes `EMBEDDING_MODEL` already (Hard
   rule #6 of Phase 1, restated in NOTES.md §"Gotchas hit").
2. **Collocations + prepositional-objects are READ-ONLY inputs.**
   The exercise generator (7.2 / 7.3 / 7.4) reads from both
   tables; it never writes back. No `INSERT`, no `UPDATE`, no
   `ON CONFLICT` writes from runtime. The seed scripts in 7.1
   are the only path that touches these tables outside Alembic
   migrations. **Type-level guardrail**: the SQLAlchemy models
   define the columns; the generator's Pydantic response shapes
   never include a row-construction model.
3. **Bilingual exercise is opt-in.** Default `partner_lang="de"`.
   Existing `/exercises/match` + `/exercises/cloze` callers see
   no schema change (the flag has a default). When
   `partner_lang="en"`, the response's `partner_translation`
   field is populated from `collocations.partner_lemma`.
4. **Pydantic v2 validated input/output** on every new endpoint
   and every new schema migration. Alembic migration for both
   tables — no PRAGMA-band-aid `Base.metadata.create_all` (Hard
   rule #4 in PHASE-6.md, restated).
5. **No committed secrets.** Same pattern as Phases 4–6. The
   `qwen/qwen3-embedding-8b` key stays in `~/.lexora/.env`;
   `baai/bge-m3` is local (sentence-transformers cache), no API
   call. The `EMBEDDING_MODEL` env var name follows Phase 1's
   non-triggering pattern (no `KEY` / `SECRET` / `TOKEN`).
6. **Offline-capable tests.** `app/collocation.py`'s DSPy module
   tests run with `dspy.utils.dummies.DummyLM` (Phase 4.2
   pattern). The retrieval-quality runner test SKIPs (not fails)
   when bge-m3 cache is cold. The A/B is two CSV outputs + one
   markdown comparison table — the report is the deliverable,
   not the lift decision.
7. **Type-level guardrails on thresholds.** The
   `RETRIEVAL_MIN_QUALITY_FLOOR` (the lift threshold below
   which bge-m3 is rejected as a regression) is a hard-coded
   module constant in `backend/app/eval/retrieval_compare.py`.
   **Not** env, **not** config. Same pattern as Phase 6.7's
   `RAGAS_MIN_CONTEXT_PRECISION = 0.6`.
8. **All Phase 7 work on the lexora board.** Not `default`.
   Includes the schema migration (7.1), the collocations seed
   (7.1), the bge-m3 A/B runner (7.5), the report (7.5), the
   doc update (7.6).
9. **No `notify-subscribe` to Anurag's Discord/Telegram.** Per
   the standing framework rule (the `completed` builder caps
   summaries at ~200 chars; broken path). Workers self-send at
   the end of their turn via `hermes send`.
10. **Existing callers stay byte-for-byte unchanged.** The
    `/exercises/cloze` route without `collocation=true` and
    `/exercises/match` route without `partner_lang` produce
    output byte-for-byte identical to Phase 6.1 + 6.2. A
    `git diff main -- backend/app/main.py` for the no-flag
    branch shows only the conditional plumbing (the
    `if collocation:` / `if partner_lang != "de":` arms).

## Files affected (anticipated)

```
lexora/
├── backend/
│   ├── alembic/versions/
│   │   ├── 7a1_collocations_table.py        NEW (7.1)
│   │   └── 7a2_prepositional_objects_table.py NEW (7.1)
│   ├── app/
│   │   ├── collocation.py                   NEW (7.2)
│   │   ├── main.py                          MODIFIED (7.3, 7.4 — opt-in flags)
│   │   ├── models.py                        MODIFIED (7.1 — two new SQLAlchemy models)
│   │   ├── schemas.py                       MODIFIED (7.3, 7.4 — request flag widening)
│   │   ├── seeds/
│   │   │   ├── collocations_seed.json       NEW (7.1)
│   │   │   └── prepositional_objects_seed.json NEW (7.1)
│   │   ├── eval/
│   │   │   ├── __init__.py                  MODIFIED (export retrieval_compare)
│   │   │   └── retrieval_compare.py         NEW (7.5)
│   │   └── embeddings.py                    MODIFIED (7.5 — env var fallback hook only)
│   ├── data/
│   │   └── collocations.db                  NEW (7.1, optional — for offline seed)
│   ├── scripts/
│   │   ├── eval_retrieval_compare.py        NEW (7.5)
│   │   ├── seed_collocations.py             NEW (7.1)
│   │   └── seed_prepositional_objects.py    NEW (7.1)
│   └── tests/
│       ├── test_collocation.py              NEW (7.2)
│       ├── test_collocations_schema.py      NEW (7.1)
│       ├── test_cloze_collocation_flag.py   NEW (7.3)
│       ├── test_match_partner_lang.py       NEW (7.4)
│       └── test_retrieval_compare.py        NEW (7.5)
├── docs/
│   └── PHASE-7.md                           NEW (this file)
└── README.md                                MODIFIED (7.6)
```

## What Phase 8 picks up

Phase 8 (deferred) will:

- Add a `phrases` table for fixed idioms (i.e. multi-word
  expressions that are not compositional — `ins Blaue hinein`,
  `Tomaten auf den Augen`). The current schema has no slot for
  them; collocations are pairwise, idioms are n-ary.
- Extend the corpus with 200–500 idioms (curated, hand-written
  examples from DWDS Idiome + selected Goethe / Schiller texts).
- Add an `app.idiom.py` DSPy module + a third exercise type
  (`Literal["cloze","matching","comprehension","idiom"]`).
- Possibly re-run the retrieval-quality A/B (the lifted
  thresholds from Phase 7.5's report may shift once idioms
  enlarge the noun-phrase retrieval space).

## Gotchas anticipated (the lessons learned)

These are the patterns the team has hit in earlier phases that
this plan encodes around:

1. **Pydantic `Literal` widening is wire-level.** 7.3 + 7.4 add
   `collocation: bool = False` and `partner_lang: Literal["de",
   "en"] = "de"` to *request* Pydantic models. Existing cloze
   callers (no flag) parse as the default value; Pydantic honours
   the literal widening without breaking. The opposite
   direction — narrowing, e.g. dropping the `partner_lang="en"`
   option — would silently break callers passing `"en"`. Phase 7
   never narrows.
2. **`bge-m3` OpenRouter privacy filter.** Per NOTES.md
   §"Gotchas hit", OpenRouter's privacy filter blocks
   `baai/bge-m3` as a *chat* model (it's an embedding model;
   the filter is over-broad). The Phase 1 workaround — running
   bge-m3 locally via `sentence-transformers` — is the only
   path the project has used. 7.5's offline runner reuses that
   path; no OpenRouter call to bge-m3.
3. **Bge-m3 cold cache.** The first run of 7.5 will download
   ~2.3GB from HuggingFace. Subsequent runs load from
   `~/.cache/huggingface/`. The 7.5 test SKIPs (not fails) on
   cold cache — the manual `make eval-retrieval-compare` is the
   only path that requires the warm cache. This mirrors Phase
   4.4's deviation pattern.
4. **`Base.metadata.create_all` short-circuits Alembic.** The
   Phase 6 lifespan logger calls `models.Base.metadata.
   create_all` on startup; if 7.1 adds new tables via Alembic
   but the lifespan creates them inline first, the Alembic
   migration becomes a no-op. The 7.1 body explicitly removes
   the `create_all` call for the new tables (or, better, drops
   the line entirely — Alembic owns migrations, same as Phase
   5.1's pattern).
5. **`exercise_type` does NOT widen.** `collocation=true` is an
   opt-in flag on `ClozeGenerateRequest`, not a 4th
   `exercise_type` literal. The `grade_logs.exercise_type` row
   is still `"cloze"`. This mirrors Phase 5.6's `enable_rag`
   pattern: opt-in flags vs type widening are deliberately
   different mechanisms.
6. **`partner_lang="en"` is a curated subset, not a translation
   pipeline.** The 200 English counterparts in
   `collocations.partner_lemma` are hand-picked. Phase 7 does
   *not* call any LLM to translate German → English at
   seed-time. The DSPy module in 7.2 still uses OpenRouter for
   the cloze *prompt*, not for translation.
7. **`app/llm.py` import order.** Phase 6.1 + 6.2 + 6.4 all
   import from `app/llm` for the shared `_DSPyOpenAICompatLM`
   adapter (PHASE-6.md gotcha #6). 7.2 follows the same import
   path; no duplicate adapter copy.
8. **DSPy adapter for collocation module.** `CollocationSignature`
   in `app/collocation.py` re-exports the same `_DSPyOpenAICompatLM`
   adapter via `from app.llm import _openai_client, _trace_collocation`
   (mirroring Phase 4.2's pattern). 7.2's body explicitly
   instructs the coder to extract the adapter if it hasn't
   been extracted yet by 6.2 (it has — confirmed by Phase 6
   landing).
9. **Langfuse trace_id is the join key (Phase 5.3 contract).**
   The `collocation.generate` trace span emits `trace_id` on
   the same `grade_logs` row as the cloze/match/comprehension
   spans. The wire guardrail from Phase 5.3 is unchanged.
10. **The retrieval-quality comparison swaps embeddings, not
    stores.** Phase 7.5's A/B reuses the Phase 1 `pgvector` store
    unchanged. The two CSVs are *embedding-model-keyed*, not
    *store-keyed*. If a future card wants to A/B stores
    (qdrant vs pgvector), that's a Phase 9 card, not Phase 7.5.
11. **`make eval-retrieval-compare` is a Makefile alias.** The
    7.5 CLI script is the source of truth; the Makefile alias
    keeps the operational entry point consistent with
    `make eval-ragas` from Phase 6.7. The Makefile regex is the
    same single-line `eval-${target}:` shell-out pattern.
12. **`collocations.source_corpus` enum is locked.** DWDS,
    Wiktionary, and a 3rd "manual" value are the only allowed
    sources for now. A Pydantic `Literal["dwds",
    "wiktionary", "manual"]` on the seed rows means Phase 7
    never silently accepts a typo'd source. Same pattern as
    Phase 6's `exercise_type` literals.
