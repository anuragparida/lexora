# Phase 6 — RAG-on + matching + comprehension exercise types + Ragas eval

> Spec card: `t_bd35b840` on the lexora board (apollo, plan).
> Parent: `t_2cd86c93` (Phase 5.7 review verdict PASS on 2026-07-03).
> Scope source: `project_ideas/15_lexora_personalized_learner.md` line 38 (Phase 6) + `lexora/README.md` §"Embeddings & retrieval (Phase 1)" + §"Observability (Langfuse)" + `lexora/docs/PHASE-5.md` §"What is NOT in Phase 5 (deferred — keep the discipline)" + `lexora/NOTES.md` §"Translation".
> Standing permission: Anurag 2026-06-28 — "Do it with AI, it's fine. I'm not gonna touch it... just do it and get the tickets moving." No fresh sign-off needed for this plan card.

## Outcome of this phase

By the time Phase 6 closes, lexora has the **full three-exercise-type study surface with retrieval-augmented prompts and a regression-detector eval layer**:

1. **RAG-on for cloze.** The cloze generator's prompt template (Phase 4.2's `backend/app/cloze.py`) actually consumes the existing `/retrieve` endpoint (Phase 1 infrastructure, returns top-k cosine-similar Word + Example rows). The LLM call now sees `word + retrieved_chunks` instead of just `word + context`. **Opt-in:** a `enable_rag: bool = False` flag on the request schema (mirroring Phase 5.4's `due_from_fsrs` and Phase 5.6's priorart-style `enable_rerank` pattern) keeps the offline eval runnable as non-RAG for A/B comparison. Backwards-compatible: existing clients see no schema change.
2. **Matching exercise type.** A new `POST /exercises/match` endpoint. DSPy module generates N word-pair match items (DE word ↔ EN translation, or DE word ↔ synonym set). Same Pydantic-validated structured-response shape as cloze, Langfuse-traced through the existing `lexora` project. `exercise_type` discriminator is `"matching"`.
3. **Comprehension exercise type.** A new `POST /exercises/comprehension` endpoint. DSPy module generates a short reading-comprehension passage (3–5 sentences, drawn from the corpus chunk that `/retrieve` returns for the target word, or LLM-generated on the target word when no chunk exists) + a multiple-choice question. Multiple-choice answer keys are `A | B | C | D` as a Pydantic `Literal`. `exercise_type` discriminator is `"comprehension"`.
4. **Ragas eval runner.** A new offline runner (`backend/scripts/eval_ragas.py`) that scores retrieval + generation against the held-out set. Metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevance`. Used as a **regression detector** on a config change touching `/retrieve` or the cloze/match/comprehension prompt templates — **not** the primary optimization target. The Phase 4.4 hand-labeled cloze judgments remain the primary eval signal; Ragas is layered on top as a complementary metric, per the lexora spec §"Eval".
5. **`/exercises/grade` accepts all three exercise types.** The `exercise_type` Literal expands from `Literal["cloze"]` (Phase 5.2's wire guardrail) to `Literal["cloze", "matching", "comprehension"]`. Schema + handler + Langfuse trace + `grade_logs` row + `fsrs_cards` write all extend. Phase 5's `Literal["cloze"]` 422 path is replaced by a 3-way literal guard — any other value still → 422 (type-level guardrail, not a soft check).
6. **All Phase 6 test cases green**, no secrets in the repo, OpenRouter remains the only LLM provider, no new LLM keys, no new retrieval vector store, no new retrieval model. The Phase 1 `/retrieve` (Postgres + pgvector) is consumed as-is. The offline Ragas runner uses the same held-out set shape as Phase 4.4 (per-row JSONL), with new match + comprehension rows that mirror the cloze eval's structure.
7. **README Limitations section is honest.** The Phase 5 "no retrieval-augmented prompts, no Ragas eval" line becomes "RAG-on wired + Ragas regression in place; A/B lift numbers reported against the v80 cloze set". The Phase 1 retrieval-quality disclaimer stays — Ragas makes it measurable, not zero.

The closed-loop outcome is the deliverable. Each piece (a new endpoint, a new exercise type, a Ragas runner, a 3-way grading guardrail) is plumbing for that outcome.

## What is NOT in Phase 6 (deferred — keep the discipline)

- **No collocations, no prepositional-objects schema, no corpus extension.** Phase 7.
- **No idiom entries, no LLM-generated idiom exercises.** Phase 8.
- **No new retrieval vector store** (Qdrant / Weaviate / pgvector alternatives). The Phase 1 pgvector store is the only one consumed. NOTES.md's "If retrieval quality is poor, the swap back to bge-m3 is a one-env-var change" is exactly that — a one-line config flip, not a Phase 6 deliverable.
- **No new embedding model.** The Phase 1 OpenRouter `qwen/qwen3-embedding-8b` (or whatever is currently pinned) is the only one. The retrieval-quality comparison that NOTES.md mentions runs in Phase 7+.
- **No new LLM provider.** OpenRouter only. Same `app.llm` client (Phase 4.1), same `instructor` wrapper, same DSPy adapter.
- **No multi-agent orchestration.** Each exercise type is a single DSPy module + a single endpoint. The spec explicitly forbids multi-agent (project_ideas/15 §"Pitfalls": "Do not start with multi-agent").
- **No new auth model.** Reuse the Phase 2 JWT cookie. No refresh tokens, no session expiry changes.
- **No diagnostic probe changes.** The first-login gate update logic from Phase 5.6 stays. If a user has due matching/comprehension cards, the gate routes to `/exercises/due`; the existing cloze-only behavior is unchanged (the existing branch is "due cards exist" — `fsrs_cards` is the single source of truth, so matching/comprehension rows just fall into the same check).
- **No study session mixing logic** (mixing cloze + matching + comprehension cards in one study session). Phase 9 or later. Phase 6 ships each exercise type as its own endpoint; the first-login gate sees them as additional `exercise_type` variants in the `grade_logs` table — the `/exercises/due` route stays cloze-only (Phase 5.4's contract; expanding it to pick across exercise types is Phase 9).
- **No deck export changes.** Phase 0's `anki_builder.py` stays as-is.
- **No multi-language expansion.** German-only, same as Phase 1–5.
- **No prompt optimization via DSPy on the matching/comprehension paths.** The DSPy modules exist for each exercise type, but the offline optimizer runs only on cloze (Phase 4.4's `scripts/optimize_cloze.py`). Phase 6 ships the three generator surfaces; tuning prompts via BootstrapFewShot / MIPROv2 on match / comprehension eval sets is Phase 9+ work.
- **No new frontend pages** (Phase 6 endpoints are wire-level — a frontend to render the match pairs + comprehension passages is Phase 9, alongside the study-session mixing UI). The cloze page from Phase 4.5/5.5 stays; `app.match` and `app.comprehension` endpoints are exercisable via curl.

## Hard rules (apply to every 6.x build card)

These rules are enforced by the card body and Helena's review. A build that violates any one of them is `FAIL`.

1. **RAG-on is opt-in.** Default behavior stays non-RAG. The `/exercises/cloze` request schema gains `enable_rag: bool = False`; existing clients see no schema change. New `enable_rag=True` callers get the retrieval-augmented verdict. When `enable_rag=False`, `retrieved_chunks=[]` and the prompt template is byte-for-byte identical to Phase 4.2's prompt (a git-diff test asserts this). This mirrors Phase 5.6's `enable_rerank` flag and is the same pattern priorart uses for opt-in retrieval configs.
2. **`/retrieve` is consumed as-is.** No new retrieval routes, no new vector stores, no new embedding model. The Phase 1 endpoint shape is the contract; if it needs to grow (e.g. per-source filters, hybrid sparse+dense), that's a separate card.
3. **Three exercise types only:** `cloze` (Phase 4), `matching` (Phase 6.2–6.3), `comprehension` (Phase 6.4–6.5). `exercise_type` Literal expands to the 3-way union. Any other value → 422.
4. **Single LLM provider (OpenRouter).** No new keys, no LangChain, no new model registry. The Phase 4.1 `app/llm.py` is reused; the new DSPy modules live in `app/match.py` and `app/comprehension.py` and import from `app/llm.py` and `app/cloze.py` for shared utilities (`select_target_word`, `_first_example_sentence`, `_openai_client`, `_trace_cloze`, etc.).
5. **Every state-mutating call is traced.** `get_langfuse()` from `backend/app/observability.py` is called before each exercise generation (`exercise.generate` span, scoped by `exercise_type` field) and each grade-persist write (Phase 5.3's `exercise.grade` span). The existing graceful-degradation path (returns `None` when keys missing) is reused — the call succeeds with `trace_id = None` rather than failing.
6. **Pydantic v2 validated input/output.** Each new endpoint has a `pydantic.BaseModel` request and response. Matching and comprehension response shapes are siblings to the cloze response shape — they share a `BaseExerciseResponse` model with an `exercise_type` discriminator (`Literal["cloze", "matching", "comprehension"]`).
7. **No committed secrets.** Same pattern as Phases 4 + 5. The literal API key never enters the repo; `OPENROUTER_API_KEY` and Langfuse keys stay in `~/.lexora/.env`. The Phase 1 harness-redaction gotcha (env-var names containing `KEY`/`SECRET`/`TOKEN`) applies if 6.1's schema adds a new env var — use non-triggering names (e.g. `LEXORA_RAG_TOP_K`, not `LEXORA_RAG_SECRET`).
8. **Offline-capable tests.** DSPy module tests run with a stubbed `llm.complete()` (the same `dspy.utils.dummies.DummyLM` pattern Phase 4.2 uses). Integration tests use the same test DB; no live Langfuse, no live OpenRouter. Ragas offline runner uses the same held-out set as Phase 4.4.
9. **Type-level guardrails on thresholds.** Ragas config constants (e.g. `RAGAS_MIN_CONTEXT_PRECISION = 0.6`, `RAGAS_MIN_FAITHFULNESS = 0.7`, `RAG_TOP_K = 5`, `RAG_MAX_CHARS = 1500`) are hard-coded module constants in `backend/app/eval/ragas.py` (6.7's deliverable). **Not** config, **not** env. If someone wants to tune, they edit the file, commit, and review — the same pattern Phase 5.1 uses for `PY_FSRS_VERSION`.
10. **All Phase 6 work goes on the lexora board.** Not `default`. This includes the build cards, the schema additions (if any), the Ragas offline runner, the doc updates, the eval set extensions — anything the team produces.
11. **No `notify-subscribe` to Anurag's Discord/Telegram.** Per the standing framework rule (the `completed` builder caps summaries at ~200 chars; broken path). Workers self-send at the end of their turn via `hermes send`.
12. **Existing callers stay byte-for-byte unchanged.** The `/exercises/cloze` route without `enable_rag` produces the prompt Phase 4.2 produces; a `git diff main -- backend/app/cloze.py` for the non-RAG branch shows only the conditional plumbing (the `if enable_rag:` branch). The Phase 5.6 first-login gate logic, the Phase 5.3 grading endpoint, the `/retrieve` endpoint — all stay untouched outside the scope of their 6.x card.

## The exercise-type wire (locked contract for Phase 6)

Each exercise type shares a `BaseExerciseResponse` with an `exercise_type` discriminator. Phase 4.2's `ClozeExerciseOut`, Phase 6.2's `MatchingExerciseOut`, and Phase 6.4's `ComprehensionExerciseOut` are concrete subclasses / variants. The shared fields:

| Field | Type | Source | Notes |
|---|---|---|---|
| `exercise_type` | `Literal["cloze", "matching", "comprehension"]` | Server-side tag | Wire discriminator — added to every response in Phase 6 |
| `exercise_id` | `int` | Server-side | Deterministic-ish: `int.from_bytes(os.urandom(8), "big", signed=True)` per generation. Same id re-appears on the `grade_logs` row for the same exercise. |
| `target_word_id` | `int` | Picked by `select_target_word` | FK to `words.id` |
| `prompt_template_version` | `str` | Module constant | `cloze-v1` / `match-v1` / `comprehension-v1` — bump on prompt change, used as A/B key by Ragas |
| `enable_rag` | `bool` | Echoed from request | True iff the caller passed `enable_rag=True`; the metadata field surfaces in Langfuse |
| `trace_id` | `str \| None` | Langfuse span id | `None` when keys are unset (graceful degradation) |
| `latency_ms` | `int` | Activity boundary | Wall-clock end-to-end |

Cloze-specific (Phase 4.2, unchanged): `sentence_with_blank`, `answer_word_id`, `distractors[3]`, `difficulty`, `rationale`.

Matching-specific (Phase 6.2): `pairs: list[MatchingPair]` where each pair is `{left_word_id: int, right_word_id: int, right_kind: Literal["translation", "synonym"]}`. The frontend shuffles the right-hand side for the user to match (a "concentration"-style memory game — drag left → right). `count: int` in `[2, 8]`; default 4.

Comprehension-specific (Phase 6.4): `passage: str` (3–5 sentences), `question: str`, `choices: dict[Literal["A", "B", "C", "D"], str]`, `correct_choice: Literal["A", "B", "C", "D"]`, `rationale: str`.

The base model + per-type variants land in `backend/app/schemas.py` (single source of truth, mirrors Phase 4.2 + Phase 5.2's pattern).

## The metadata contract (Langfuse + `grade_logs` row)

Phase 4 + 5 already lock the cloze trace contract + the `grade_logs` row shape. Phase 6 extends, never edits:

| Field | Type | Notes |
|---|---|---|
| `exercise_type` | `Literal["cloze", "matching", "comprehension"]` | Phase 5.2's `Literal["cloze"]` widens in 6.6 |
| `enable_rag` | `bool` | Echoed from request — Phase 6.1's new field; `False` for non-cloze or for cloze without the flag |
| `retrieved_chunk_count` | `int` | Number of chunks `/retrieve` returned; `0` when RAG is off |
| `retrieved_chunk_k` | `int` | The `k` value passed to `/retrieve`; defaults to `RAG_TOP_K = 5` |
| `retrieved_chunk_source` | `Literal["words", "examples", "both"]` | The `source` value passed to `/retrieve`; defaults to `"both"` |
| (existing Phase 4 + 5 fields) | … | `user_id`, `word_id`, `difficulty`, `model_id`, `prompt_template_version`, `schema_retry_count`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `grade`, `scheduled_next_due_at`, `prev_due_at`, `state`, `stability`, `difficulty`, `reps`, `lapses`, `trace_id`, `graded_at` |

The Pydantic wire models (`MatchingExerciseOut`, `ComprehensionExerciseOut`, the widened `GradeRequest`/`GradeResponse`) land in 6.2, 6.4, and 6.6.

## Build-card breakdown

Seven build cards (6.1–6.7) + one Athena doc card (6.8) + one Helena review card (6.9). All nine parent-linked to `t_bd35b840` (this card). The dependency ordering:

```
6.1 (RAG-on schema + cloze prompt change) ──────────────────┐
                                                            │
6.2 (app/match.py DSPy + schemas) ─────┐                    │
                                       ├──► 6.3 (/exercises/match) ──┐
6.4 (app/comprehension.py + schemas) ──┤                            │
                                       └──► 6.5 (/exercises/compr) ─┤
                                                                │
                                                                ├──► 6.6 (/exercises/grade 3-way Literal) ──► 6.9 (review)
6.7 (Ragas runner + eval set extensions) ───────────────────┐    │
                                                            │    │
6.8 (athena: README Limitations update) ──────────────────── ┴────┘
```

`6.1` is the cloze-only opt-in retrieval augmentation — independent of match/comprehension. `6.2` + `6.3` chain on match (DSPy module then endpoint). `6.4` + `6.5` chain on comprehension. `6.6` waits on `6.1`, `6.3`, `6.5` (needs all three exercise types live to widen the grade enum). `6.7` is standalone — it reads the held-out eval set shape, computes Ragas metrics, and writes `eval/ragas_results.jsonl`. `6.8` waits on `6.1` (so the README is honest about RAG-on), `6.3` (match endpoint), `6.5` (comprehension endpoint), `6.6` (3-way grade), `6.7` (Ragas). `6.9` waits on all eight.

### 6.1 — RAG-on schema + cloze prompt template change

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Independent of `6.2`–`6.5`.
- **Why this card exists:** Phase 4.2's cloze prompt is byte-for-byte context-stuffing — the word + the first example sentence. Phase 1's pgvector retrieval is plumbed but unconsumed. Phase 6.1 wires the retrieval-augmented prompt path **opt-in** so the existing offline eval stays reproducible for A/B comparison.
- **Scope:**
  - `backend/app/cloze.py` — extend the DSPy module + `build_prompt` to accept an optional `retrieved_chunks: list[RetrievedChunk] = []` parameter. When the list is non-empty, the user-prompt JSON includes a `retrieved_chunks` array of `{kind: "word" | "example", id: int, text: str}` (text truncated to `RAG_MAX_CHARS_PER_CHUNK = 300` chars). When the list is empty, the JSON is byte-for-byte identical to Phase 4.2's user prompt — **git-diff test asserts this**.
  - `backend/app/cloze.py` — extract `_retrieve_for_cloze(db, word)` that calls `app.retrieval.retrieve(db, query_vec, k=RAG_TOP_K, source="both")` using the word's lemma as the query, returns the top-K results. When `DATABASE_URL` is not Postgres, returns `[]` (graceful — same pattern as `/retrieve`'s 503 fallback, but inline rather than raising so the cloze call still succeeds).
  - `backend/app/schemas.py` — extend `ClozeExerciseOut` with `exercise_type: Literal["cloze"] = "cloze"` (the new shared discriminator field). Add `BaseExerciseFields` mixin if cleaner.
  - **New** `backend/app/schemas.py` — `ClozeGenerateRequest` with `enable_rag: bool = False`. The existing `POST /exercises/cloze` accepts an empty body today; this widens it to accept either `{}` or `{"enable_rag": true}`. Pydantic-default = `False` means existing callers (Phase 4.5 / Phase 5.5 frontend) keep working without changes.
  - `backend/app/main.py` — `generate_cloze_exercise` reads `payload: ClozeGenerateRequest`, passes `enable_rag` through to `generate_cloze`. When `enable_rag=True`, calls `_retrieve_for_cloze` first and passes the chunks to `build_prompt`.
  - `backend/app/cloze.py` — `_trace_cloze` gains `enable_rag: bool` and `retrieved_chunk_count: int` metadata fields (Hard rule #5 / the metadata contract).
  - `backend/tests/test_cloze.py` — extend:
    - `enable_rag=False` → prompt bytes match a stored fixture (the Phase 4.2 prompt, no retrieval call).
    - `enable_rag=True` on Postgres → retrieval call fires; prompt JSON includes `retrieved_chunks` with N items.
    - `enable_rag=True` on SQLite → retrieval returns `[]`; prompt is the same as `enable_rag=False`.
    - `ClozeExerciseOut(exercise_type="matching")` → ValidationError (the new discriminator).
  - `backend/tests/test_schemas.py` — extend:
    - `ClozeGenerateRequest()` → default `enable_rag=False`.
    - `ClozeGenerateRequest(enable_rag=True)` → serialises to `{"enable_rag": true}`.
    - `ClozeExerciseOut` JSON includes `exercise_type: "cloze"`.
- **Out of scope:** the matching / comprehension exercise types (6.2–6.5), the `/exercises/grade` widening (6.6), the Ragas runner (6.7). 6.1 is cloze-only opt-in retrieval.
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_cloze.py tests/test_schemas.py -v` → all green, no network (Postgres tests skipped when `DATABASE_URL` is unset, same Phase 1 pattern).
  - `curl -X POST http://localhost:18700/exercises/cloze -d '{}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `ClozeExerciseOut` with `enable_rag: false, retrieved_chunk_count: 0`. The Langfuse trace shows `enable_rag: false`.
  - `curl -X POST http://localhost:18700/exercises/cloze -d '{"enable_rag": true}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `ClozeExerciseOut` with `enable_rag: true, retrieved_chunk_count: > 0` (on Postgres). The Langfuse trace shows the retrieved chunks + the augmented prompt.
  - `git diff main -- backend/app/cloze.py` shows ONLY: the new `_retrieve_for_cloze` helper, the `retrieved_chunks` parameter on `build_prompt`, and the conditional JSON-include. The Phase 4.2 `build_prompt` system-prompt text is unchanged.
  - `git grep -n "RAG_TOP_K" backend/app/cloze.py` shows exactly one definition + one usage site.
  - No new env vars introduced (the `RAG_TOP_K` constant is hard-coded per Hard rule #9).

### 6.2 — `app/match.py` DSPy module + Pydantic schemas

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Independent of `6.1`; feeds `6.3`. Can run in parallel with `6.4`.
- **Why this card exists:** Phase 4 ships cloze only. Phase 6.2 lands the matching generator module + wire shape; `6.3` exposes the endpoint.
- **Scope:**
  - **New** `backend/app/match.py`:
    - `PROMPT_TEMPLATE_VERSION: str = "match-v1"` (Hard rule #9 type-level guardrail).
    - `MAX_ATTEMPTS: int = 3` (same Phase 4.2 shape).
    - `MATCH_MIN_COUNT: int = 2`, `MATCH_MAX_COUNT: int = 8`, `MATCH_DEFAULT_COUNT: int = 4` — module constants.
    - `MatchingPair` Pydantic model: `left_word_id: int`, `right_word_id: int`, `right_kind: Literal["translation", "synonym"]`.
    - `MatchingExercise` Pydantic model: `target_word_id: int`, `pairs: list[MatchingPair]` (length in `[MATCH_MIN_COUNT, MATCH_MAX_COUNT]`).
    - `MatchingGenerationError(RuntimeError)` — mirrors Phase 4.2's `ClozeGenerationError` shape (same dead-letter fields).
    - `generate_match(db, user_id, *, force_word_id: int | None = None, count: int = MATCH_DEFAULT_COUNT, enable_rag: bool = False) -> MatchingExercise` — picks target via `select_target_word`, builds prompt, calls `llm.complete` wrapped with `instructor` (same pattern as Phase 4.2's `generate_cloze`), returns validated exercise. When `enable_rag=True`, calls `app.retrieval.retrieve(...)` for the target word; the prompt's user-side JSON includes `retrieved_chunks` (same shape as 6.1's cloze-with-RAG).
    - `MatchSignature` + `MatchModule` — DSPy optimization path, mirrors `ClozeSignature` + `ClozeModule` (Phase 4.2). The DSPy adapter (`_DSPyOpenAICompatLM`) is reused from `app/cloze.py`; if it's a private symbol there, 6.2 either extracts it to `app/llm.py` or duplicates the small class.
    - `_trace_match(metadata, latency_ms)` — mirrors Phase 4.3's `_trace_cloze` shape. Span name: `match.generate`. Metadata fields include `exercise_type="matching"`, `target_word_id`, `count`, `enable_rag`, `retrieved_chunk_count`, `model_id`, `prompt_template_version`, `schema_retry_count`, `latency_ms`, `prompt_tokens`, `completion_tokens`. `trace_id` returned; `None` when keys are unset.
  - **New** `backend/app/schemas.py` — `MatchingExerciseOut(BaseModel)` carrying the shared `BaseExerciseFields` + the matching-specific shape: `target_word_id: int`, `pairs: list[MatchingPair]`. The `exercise_type: Literal["matching"] = "matching"` discriminator.
  - **New** `backend/app/schemas.py` — `MatchGenerateRequest(BaseModel)` with `count: int = MATCH_DEFAULT_COUNT` (`Field(ge=MATCH_MIN_COUNT, le=MATCH_MAX_COUNT)`), `enable_rag: bool = False`. The wire is the same shape as 6.1's `ClozeGenerateRequest`.
  - `backend/app/observability.py` — no changes. `_trace_match` uses the existing `get_langfuse()`.
  - `backend/tests/test_match.py` — pytest cases:
    - Happy path: `generate_match(db, user_id)` returns a `MatchingExercise` with `count=4` pairs by default; all `left_word_id` and `right_word_id` are valid `words.id` FKs; `right_kind` is one of `"translation" | "synonym"`.
    - `count=0` → `MatchingExercise` validation error.
    - `count=9` → `MatchingExercise` validation error.
    - `MatchGenerateRequest(count=0)` → 422 (Pydantic).
    - `MatchGenerateRequest(count=20)` → 422.
    - `enable_rag=False` (default) — prompt is byte-for-byte identical to a stored fixture.
    - `enable_rag=True` on Postgres — prompt JSON includes `retrieved_chunks` with N items.
    - Langfuse keys present: span emitted with the metadata keyset.
    - Langfuse keys absent: graceful no-op; `trace_id=None` propagated.
    - All tests run with `respx`-mocked OpenRouter (same Phase 4.2 pattern).
- **Out of scope:** the `/exercises/match` route (6.3), the comprehension module (6.4), the grade-endpoint widening (6.6).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_match.py -v` → all green, no network.
  - `uv run python -c "from app.match import PROMPT_TEMPLATE_VERSION; assert PROMPT_TEMPLATE_VERSION == 'match-v1'"` → exits 0.
  - `git grep -n "MATCH_MIN_COUNT\|MATCH_MAX_COUNT\|MATCH_DEFAULT_COUNT" backend/app/match.py` → exactly one definition + usage sites.
  - `git diff main -- backend/app/cloze.py` is empty (6.2 doesn't touch the cloze path — it only *imports* from `app.cloze` for shared utilities).
  - `git diff main -- backend/app/observability.py` is empty.

### 6.3 — `/exercises/match` endpoint

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 6.2.** Can run in parallel with `6.5`.
- **Why this card exists:** Wire surface for the matching generator. Same shape as `POST /exercises/cloze`.
- **Scope:**
  - `backend/app/main.py` — adds `POST /exercises/match` (auth-gated):
    - Body: `MatchGenerateRequest` (6.2).
    - Calls `generate_match(db, current_user.id, count=payload.count, enable_rag=payload.enable_rag)`.
    - Wraps the activity in `_trace_match` (the Langfuse span from 6.2).
    - Returns `MatchingExerciseOut` carrying `exercise_id` (server-generated `int`), the shared fields, the matching-specific shape.
    - 401 on missing/invalid JWT (Phase 5.3 dependency).
    - 422 on validation error (FastAPI default for Pydantic).
    - 500 on `MatchingGenerationError` — log the user_id + target_word_id + count for triage (mirrors Phase 4.2's 502 dead-letter shape for cloze schema failures).
  - `backend/tests/test_match_endpoint.py` — pytest cases:
    - 200 happy path: payload `{}` returns `MatchingExerciseOut` with `exercise_type="matching"`, `enable_rag=false`, `count=4`.
    - `{"count": 6, "enable_rag": true}` → 200 with `count=6`, `enable_rag=true` (on Postgres).
    - No JWT → 401.
    - `{"count": 0}` → 422.
    - `{"count": 20}` → 422.
    - LLM transport failure → 502 (respx 500 from the OpenRouter mock).
- **Out of scope:** the comprehension endpoint (6.5), the grade widening (6.6), frontend pages.
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_match_endpoint.py -v` → all green, no network.
  - `curl -X POST http://localhost:18700/exercises/match -H "Cookie: lexora_session=$COOKIE" -d '{}'` → 200 + `MatchingExerciseOut`.
  - `curl -X POST http://localhost:18700/exercises/match -H "Cookie: lexora_session=$COOKIE" -d '{"count": 6}'` → 200 + 6 pairs.
  - With Langfuse keys set, `http://localhost:13000` shows a `match.generate` span in the `lexora` project with the metadata keyset.
  - `git diff main -- backend/app/cloze.py backend/app/observability.py` is empty.

### 6.4 — `app/comprehension.py` DSPy module + Pydantic schemas

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Independent of `6.1`–`6.3`; feeds `6.5`.
- **Why this card exists:** The comprehension generator module + wire shape. Mirrors `6.2`'s matching-module shape with passage + multiple-choice.
- **Scope:**
  - **New** `backend/app/comprehension.py`:
    - `PROMPT_TEMPLATE_VERSION: str = "comprehension-v1"`.
    - `MAX_ATTEMPTS: int = 3`.
    - `COMPREHENSION_PASSAGE_MIN_SENTENCES: int = 3`, `COMPREHENSION_PASSAGE_MAX_SENTENCES: int = 5`, `COMPREHENSION_PASSAGE_MAX_CHARS: int = 600`.
    - `ComprehensionChoice = Literal["A", "B", "C", "D"]`.
    - `ComprehensionExercise` Pydantic model: `target_word_id: int`, `passage: str` (`min_length=20`, `max_length=COMPREHENSION_PASSAGE_MAX_CHARS`), `question: str` (`min_length=5`, `max_length=300`), `choices: dict[ComprehensionChoice, str]` (all 4 keys required, each `min_length=1`, `max_length=200`), `correct_choice: ComprehensionChoice`, `rationale: str` (`min_length=1`, `max_length=400`).
    - `ComprehensionGenerationError(RuntimeError)` — same dead-letter shape as Phase 4.2 / 6.2.
    - `generate_comprehension(db, user_id, *, force_word_id: int | None = None, enable_rag: bool = False) -> ComprehensionExercise`. The `force_word_id` knob is here for symmetry with the cloze/matching generators (the Phase 5.4 due-queue may eventually extend to comprehension — Phase 9).
    - `ComprehensionSignature` + `ComprehensionModule` — DSPy optimization path.
    - `_trace_comprehension(metadata, latency_ms)` — span name `comprehension.generate`. Same metadata keyset as `_trace_match` / `_trace_cloze` (exercise_type-scoped).
  - **New** `backend/app/schemas.py` — `ComprehensionExerciseOut(BaseModel)` carrying shared fields + `target_word_id`, `passage`, `question`, `choices`, `correct_choice`, `rationale`. `exercise_type: Literal["comprehension"] = "comprehension"`.
  - **New** `backend/app/schemas.py` — `ComprehensionGenerateRequest(BaseModel)` with `enable_rag: bool = False` (comprehension doesn't have a `count` knob — one passage + one question per call).
  - `backend/tests/test_comprehension.py` — pytest cases mirroring `test_match.py`:
    - Happy path: returns a `ComprehensionExercise` with all 4 choices populated and `correct_choice in {"A","B","C","D"}`.
    - `passage` length in `[20, COMPREHENSION_PASSAGE_MAX_CHARS]`.
    - `enable_rag=False` (default) — prompt bytes match a stored fixture.
    - `enable_rag=True` on Postgres — prompt JSON includes `retrieved_chunks`.
    - Langfuse keys present/absent (graceful-degradation path).
    - All `respx`-mocked OpenRouter.
- **Out of scope:** the `/exercises/comprehension` route (6.5), the grade widening (6.6).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_comprehension.py -v` → all green, no network.
  - `uv run python -c "from app.comprehension import PROMPT_TEMPLATE_VERSION; assert PROMPT_TEMPLATE_VERSION == 'comprehension-v1'"` → exits 0.
  - `git diff main -- backend/app/match.py backend/app/cloze.py` is empty (6.4 doesn't touch the matching or cloze paths).

### 6.5 — `/exercises/comprehension` endpoint

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 6.4.**
- **Why this card exists:** Wire surface for the comprehension generator. Mirrors `6.3`'s matching-endpoint shape (no `count` knob).
- **Scope:**
  - `backend/app/main.py` — adds `POST /exercises/comprehension` (auth-gated):
    - Body: `ComprehensionGenerateRequest` (6.4).
    - Calls `generate_comprehension(...)`.
    - Wraps in `_trace_comprehension`.
    - Returns `ComprehensionExerciseOut`.
    - Same 401 / 422 / 500 handling as 6.3.
  - `backend/tests/test_comprehension_endpoint.py` — pytest cases mirroring `test_match_endpoint.py`:
    - 200 happy path: `{}` returns `ComprehensionExerciseOut` with `exercise_type="comprehension"`.
    - `{"enable_rag": true}` → 200 with `enable_rag=true` (on Postgres).
    - No JWT → 401.
    - LLM transport failure → 502.
- **Out of scope:** grade widening (6.6), frontend pages.
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_comprehension_endpoint.py -v` → all green.
  - `curl -X POST http://localhost:18700/exercises/comprehension -H "Cookie: lexora_session=$COOKIE" -d '{}'` → 200 + `ComprehensionExerciseOut`.
  - With Langfuse keys set, `http://localhost:13000` shows a `comprehension.generate` span.

### 6.6 — `/exercises/grade` 3-way Literal expansion + handler fan-out

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 6.1, 6.3, 6.5** (all three exercise types must be live to test the widening).
- **Why this card exists:** Phase 5.2's `GradeRequest.exercise_type: Literal["cloze"]` is a wire-level guardrail that Phase 6 widens. The 422 path becomes a 3-way literal guard; matching + comprehension handlers are sibling functions to the cloze handler.
- **Scope:**
  - `backend/app/schemas.py` — widen `GradeRequest.exercise_type` to `Literal["cloze", "matching", "comprehension"]`; widen `GradeResponse.exercise_type` the same way. Existing cloze callers are unaffected (Pydantic accepts `"cloze"` as a subset of the union).
  - `backend/app/main.py` — refactor the `/exercises/grade` route:
    - The cloze handler is extracted to `_grade_cloze(db, user_id, exercise_id, grade)` (sibling of the existing inline logic).
    - Two new siblings: `_grade_matching(db, user_id, exercise_id, grade) -> GradeResponse` and `_grade_comprehension(db, user_id, exercise_id, grade) -> GradeResponse`.
    - The cloze handler logic stays byte-for-byte; the matching + comprehension handlers are thin wrappers around the same `apply_grade` + `grade_logs` write path — they differ in the trace span name (`match.grade` / `comprehension.grade` instead of `exercise.grade`) and the `exercise_type` field on the `grade_logs` row.
    - The 3-way `Literal` guardrail is at the Pydantic layer; the route dispatches on `payload.exercise_type` via a `match` statement (Python 3.10+, lexora already requires 3.12).
    - `trace_id` propagation is identical across all three handlers (Phase 5.3's graceful-degradation path).
  - `backend/tests/test_grade.py` — extend:
    - `grade_logs.exercise_type == "matching"` after a match-grade.
    - `grade_logs.exercise_type == "comprehension"` after a comprehension-grade.
    - `GradeRequest(exercise_type="speaking")` → 422.
    - `GradeRequest(exercise_type="cloze")` → 200 (Phase 5.3 behaviour, regression guard).
    - `GradeRequest(exercise_type="matching")` with a mock matching-exercise id → 200 + `GradeResponse` with `exercise_type="matching"`.
  - `backend/app/fsrs.py` — no changes. `apply_grade` is exercise-type-agnostic.
- **Out of scope:** the per-exercise-type Langfuse span names are new but the trace shape stays. No new dependencies, no schema migration (the `grade_logs.exercise_type` column is a `String` from Phase 5.2 — it accepts `"matching"` and `"comprehension"` without an Alembic migration).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_grade.py tests/test_schemas.py -v` → all green.
  - `curl -X POST http://localhost:18700/exercises/grade -H "Cookie: lexora_session=$COOKIE" -d '{"exercise_id": 42, "exercise_type": "matching", "grade": 3}'` → 200 + `GradeResponse` with `exercise_type: "matching"`. `grade_logs` row inserted with `exercise_type: "matching"`.
  - `curl -X POST http://localhost:18700/exercises/grade -d '{"exercise_id": 42, "exercise_type": "speaking", "grade": 3}'` → 422.
  - `git diff main -- backend/app/fsrs.py backend/app/cloze.py backend/app/match.py backend/app/comprehension.py` is empty (6.6 doesn't touch the generator modules — only `schemas.py` + `main.py`).
  - `git grep -n "exercise_type.*Literal" backend/app/schemas.py` shows the widened 3-way union.

### 6.7 — Ragas offline runner + held-out set extensions

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Standalone (doesn't depend on 6.1–6.6 to ship, but 6.8 needs it for the README). Can run in parallel with 6.1–6.5.
- **Why this card exists:** The lexora spec §"Eval" mandates Ragas as the regression-detector layer for retrieval + generation. Phase 4.4's hand-labeled cloze judgments are the primary eval signal; Ragas is layered on top with the four metrics named in the spec.
- **Scope:**
  - **New** `backend/app/eval/__init__.py` — empty package init.
  - **New** `backend/app/eval/ragas.py` — module-level constants (Hard rule #9):
    - `RAGAS_MIN_CONTEXT_PRECISION: float = 0.6`
    - `RAGAS_MIN_CONTEXT_RECALL: float = 0.5`
    - `RAGAS_MIN_FAITHFULNESS: float = 0.7`
    - `RAGAS_MIN_ANSWER_RELEVANCE: float = 0.6`
    - `RAG_TOP_K: int = 5` (the cloze-with-RAG top-K)
    - `RAG_MAX_CHARS_PER_CHUNK: int = 300`
    - `RAG_MAX_CHARS: int = 1500` (total context budget)
    - `RAGAS_DRY_RUN_MIN_OVERALL: float = 0.6` (the floor for the CI smoke path)
    - Helper functions: `format_retrieved_chunks(items: list) -> str` (truncates to per-chunk + total budgets), `build_ragas_sample(question, answer, contexts, ground_truth) -> dict` (shapes a Ragas sample).
  - **New** `backend/scripts/eval_ragas.py` — the offline runner. Mirrors `backend/scripts/eval_cloze.py`'s shape:
    - Loads `eval/cloze_judgments.jsonl` (Phase 4.4), `eval/match_judgments.jsonl` (new in 6.7 — 40 rows, hand-labeled or template-based fallback per the Phase 4.4 deviation), `eval/comprehension_judgments.jsonl` (new — 40 rows, same shape).
    - `--dry-run` exits 0 and prints `OK` without contacting OpenRouter or Ragas. Computes a deterministic proxy that mirrors Ragas's metrics against the held-out set's own `judgment` columns (same pattern as `eval_cloze.py --dry-run`).
    - `--live` (or `--predictions FILE`): runs the Ragas library against the held-out set + the corresponding generator's predictions. Requires `OPENROUTER_API_KEY` + `RAGAS_API_KEY` (Ragas can use OpenAI as the judge model).
    - Adds `ragas` to `pyproject.toml` deps. **No** `RAGAS_API_KEY` literal in the repo — env-derived per Hard rule #7.
    - Writes `eval/ragas_results_<timestamp>.jsonl` + a stable-path symlink `eval/ragas_results.jsonl` (latest run). One JSON object per row: `{exercise_type, row_id, context_precision, context_recall, faithfulness, answer_relevance, pass: bool}`.
    - The CI smoke path (`make eval-ragas-dry`) runs `--dry-run` and asserts `overall >= RAGAS_DRY_RUN_MIN_OVERALL`.
  - **New** `eval/match_judgments.jsonl` — 40 rows. Each row: `{word_id, target_word, word_type, expected_pairs: [{left_word_id, right_word_id, right_kind}], judgment: "accept" | "reject"}`. Template-based fallback per Phase 4.4's deviation pattern (the Phase 4 review flagged OpenRouter's chat-model guardrail as data, not code; the matching eval set mirrors this).
  - **New** `eval/comprehension_judgments.jsonl` — 40 rows. Each row: `{word_id, target_word, word_type, expected_passage: str, expected_question: str, expected_choices: {A, B, C, D}, expected_correct_choice: Literal["A","B","C","D"], judgment: "accept" | "reject"}`.
  - `backend/tests/test_eval/test_ragas.py` — pytest cases:
    - `format_retrieved_chunks` truncates to per-chunk + total budgets.
    - `build_ragas_sample` produces the expected dict shape.
    - `scripts.eval_ragas --dry-run` exits 0 on the new match + comprehension eval sets (Postgres tests skipped when `DATABASE_URL` is unset).
    - The `eval/ragas_results.jsonl` symlink points at the latest run.
- **Out of scope:** running Ragas against live OpenRouter (that's the `--live` flag, not exercised in CI; documented as the manual QA path). Tuning Ragas metrics (Hard rule #9: constants are committed, not configurable).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_eval/test_ragas.py -v` → all green.
  - `uv run python -m scripts.eval_ragas --dry-run` → exits 0, prints `OK`, writes a results file with `overall >= RAGAS_DRY_RUN_MIN_OVERALL`.
  - `git grep -n "RAGAS_MIN_CONTEXT_PRECISION\|RAGAS_MIN_FAITHFULNESS\|RAG_TOP_K" backend/app/eval/ragas.py` → exactly one definition + usage sites.
  - No `RAGAS_API_KEY` or similar literal in the repo (`git grep -n "RAGAS_API_KEY=*** -- ':!*.example'` returns nothing).
  - `eval/match_judgments.jsonl` and `eval/comprehension_judgments.jsonl` exist with 40 rows each.

### 6.8 — README Limitations + Phase 6 status update

- **Assignee:** `athena`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 6.1, 6.3, 6.5, 6.6, 6.7.**
- **Why this card exists:** The Phase 5 README Limitations line ("no retrieval-augmented prompts, no Ragas eval") becomes dishonest the moment 6.1 and 6.7 land. Athena owns the README copy (the lexora team's writer profile; same pattern as Phase 4.7 / 5.x docs updates).
- **Scope:**
  - `README.md`:
    - Update the Limitations section: the "no retrieval-augmented prompts, no Ragas eval" line becomes "RAG-on wired (opt-in via `enable_rag` on `/exercises/cloze`); Ragas regression in place (`make eval-ragas-dry`); A/B lift numbers reported against the v80 cloze set". Add a sentence stating the A/B lift is reported in `eval/ragas_results.jsonl` and is a regression detector, not the primary optimization signal (the Phase 4.4 hand-labeled set remains primary).
    - Add a "Phase 6 — three exercise types" section in the Exercise-types overview:
      - **Cloze** (Phase 4.2 + 5.x + 6.1): fill-in-the-blank. `POST /exercises/cloze`. Opt-in retrieval via `enable_rag`.
      - **Matching** (Phase 6.2 + 6.3): DE word ↔ EN translation / synonym pairs. `POST /exercises/match`.
      - **Comprehension** (Phase 6.4 + 6.5): 3–5 sentence passage + multiple-choice. `POST /exercises/comprehension`.
    - Update the API table: add `POST /exercises/match` + `POST /exercises/comprehension` rows.
    - Update the API table: add the `enable_rag` query parameter to `POST /exercises/cloze`.
    - Update the "Eval set + offline runner" section: add a Ragas subsection that points at `backend/scripts/eval_ragas.py` + `eval/ragas_results.jsonl` + the four-metric keyset.
  - **No new code or schema changes.** Athena's deliverable is the README.
- **Out of scope:** any code change, any new file outside `README.md`. The doc-only nature mirrors Phase 4.7 / 5.x doc updates.
- **Acceptance:**
  - `git diff main -- README.md` is the only file changed.
  - The new "Matching" + "Comprehension" sections are present with curl examples (the curl shape mirrors Phase 4.2's `/exercises/cloze` curl).
  - The Limitations section no longer claims RAG-on or Ragas are deferred.

### 6.9 — Phase 6 review (Helena)

- **Assignee:** `helena`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Parent-linked to all eight build/doc cards (6.1–6.8)** — picks up when all eight are `done`.
- **Scope:** Same shape as `t_2cd86c93` (Phase 5.7 review). Verifies, with severity-tagged findings (critical / major / minor / nit) and a final PASS / FAIL verdict:
  1. **RAG-on is opt-in.** `git grep -n "enable_rag" backend/app/cloze.py` shows the default `False`. A `--dry-run` test on `/exercises/cloze?enable_rag=false` produces a prompt byte-for-byte identical to a Phase 4.2 fixture. A `enable_rag=true` test produces a prompt with `retrieved_chunks` populated. SQLite fallback returns `retrieved_chunks=[]` gracefully.
  2. **`/retrieve` is consumed as-is.** No new routes, no new vector stores, no new embedding model. `git diff main -- backend/app/retrieval.py backend/app/embeddings.py` is empty.
  3. **Three exercise types only.** `git grep -n "exercise_type.*Literal" backend/app/schemas.py` shows `Literal["cloze", "matching", "comprehension"]`. The `/exercises/grade` handler asserts and 422s otherwise (covered by a test). `GradeRequest(exercise_type="speaking")` → 422.
  4. **Single LLM provider (OpenRouter).** No new keys, no LangChain, no new model registry. `git diff main -- backend/app/llm.py` shows ONLY the `_DSPyOpenAICompatLM` extraction (if 6.2 needed it); no other transport changes.
  5. **Every state-mutating call is traced.** `get_langfuse()` is called before each generator (`cloze.generate` / `match.generate` / `comprehension.generate`) and each grade (`exercise.grade`). Tests cover both Langfuse-keys-present and -absent paths.
  6. **Pydantic v2 validated input/output.** `ClozeGenerateRequest.enable_rag` defaults to `False`. `MatchGenerateRequest.count` is bounded `[MATCH_MIN_COUNT, MATCH_MAX_COUNT]`. `ComprehensionGenerateRequest.enable_rag` defaults to `False`. `GradeRequest.exercise_type` widens to the 3-way union; out-of-range → 422. Tests cover all paths.
  7. **No committed secrets.** Same Phase 4 + 5 check: `git grep -n "OPENROUTER_API_KEY=*** -- ':!*.example'` returns nothing. `RAGAS_API_KEY` (if added by 6.7) follows the same rule.
  8. **Offline-capable tests.** `cd backend && uv run pytest` exits 0 with no network. `cd backend && uv run pytest tests/test_match.py tests/test_match_endpoint.py tests/test_comprehension.py tests/test_comprehension_endpoint.py tests/test_eval/test_ragas.py` exits 0 with no network.
  9. **Type-level guardrails on thresholds.** `git grep -n "RAGAS_MIN_CONTEXT_PRECISION\|RAGAS_MIN_FAITHFULNESS\|RAG_TOP_K\|RAG_MAX_CHARS" backend/app/eval/ragas.py` shows the constants. No env-derived Ragas parameters anywhere in the codebase (`git grep -n "getenv.*RAGAS\|getenv.*RAG"` returns nothing).
  10. **All Phase 6 work on the lexora board.** `git diff main -- backend/app/match.py backend/app/comprehension.py backend/app/eval/ragas.py` shows only the Phase 6 deliverable files; no out-of-scope refactors.
  11. **Existing callers stay byte-for-byte unchanged.** `git diff main -- backend/app/fsrs.py backend/app/observability.py backend/app/retrieval.py backend/app/embeddings.py` is empty (none of 6.1–6.7 touched these files; 6.6 only touches `schemas.py` + `main.py`).
  12. **Ragas smoke green.** `cd backend && uv run python -m scripts.eval_ragas --dry-run` exits 0, writes a results file with `overall >= RAGAS_DRY_RUN_MIN_OVERALL`.
  13. **README Limitations is honest.** The Phase 5 "no RAG-on / no Ragas" line is gone; the Phase 6 status is described accurately.
  14. **End-to-end smoke.** With the dev stack up (`docker compose up -d`):
      - `curl -X POST http://localhost:18700/exercises/cloze -d '{}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `ClozeExerciseOut` with `enable_rag: false`.
      - `curl -X POST http://localhost:18700/exercises/cloze -d '{"enable_rag": true}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `ClozeExerciseOut` with `enable_rag: true` (Postgres only; SQLite returns `[]` chunks gracefully).
      - `curl -X POST http://localhost:18700/exercises/match -d '{}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `MatchingExerciseOut`.
      - `curl -X POST http://localhost:18700/exercises/comprehension -d '{}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `ComprehensionExerciseOut`.
      - `curl -X POST http://localhost:18700/exercises/grade -d '{"exercise_id": 42, "exercise_type": "matching", "grade": 3}' -H "Cookie: lexora_session=$COOKIE"` → 200 + `GradeResponse` with `exercise_type: "matching"`. `grade_logs.exercise_type == "matching"`.
      - `curl -X POST http://localhost:18700/exercises/grade -d '{"exercise_id": 42, "exercise_type": "speaking", "grade": 3}' -H "Cookie: lexora_session=$COOKIE"` → 422.
- **Out of scope:** collocations / prepositional-objects review (Phase 7), idiom entries review (Phase 8).
- **Acceptance (Helena's verdict is the gate):**
  - Severity-tagged findings posted as a comment on `t_6_9_id`.
  - Verdict line: `VERDICT: PASS` or `VERDICT: FAIL`. FAIL triggers a remediation card (parent-linked to the failing build card), per Phase 3 / Phase 4 / Phase 5 pattern.

## Verification at end of Phase 6

The exact commands the team runs (and the expected output) to declare Phase 6 done. Every line should pass cleanly.

### Backend (in `/home/ody/workspace/lexora/backend`)

```bash
# 1. Full pytest sweep — no network, mocked via respx
uv run pytest -v
# Expected: all tests pass, including the new test_match.py +
# test_match_endpoint.py + test_comprehension.py +
# test_comprehension_endpoint.py + test_eval/test_ragas.py +
# the extended test_cloze.py + test_grade.py + test_schemas.py +
# all Phase 0–5 tests.

# 2. RAG-on opt-in is byte-for-byte stable for the False case
git grep -n "enable_rag" backend/app/cloze.py
# Expected: exactly one definition + usage sites, default False.

# 3. Three exercise types only
git grep -n "exercise_type.*Literal" backend/app/schemas.py
# Expected: Literal["cloze", "matching", "comprehension"].

# 4. /retrieve is consumed as-is
git diff main -- backend/app/retrieval.py backend/app/embeddings.py
# Expected: no output.

# 5. Single LLM provider
git grep -n "OpenAI\(\)" backend/app/
# Expected: only the existing app/cloze.py:OpenAI(...) lazy construction.

# 6. Ragas constants are hard-coded
git grep -n "RAGAS_MIN_CONTEXT_PRECISION\|RAGAS_MIN_FAITHFULNESS\|RAG_TOP_K"
# Expected: exactly one definition site in app/eval/ragas.py.

# 7. No env-derived Ragas params
git grep -n "getenv.*RAGAS\|getenv.*RAG"
# Expected: no output.

# 8. Ragas dry-run green
uv run python -m scripts.eval_ragas --dry-run
# Expected: exits 0, prints OK, eval/ragas_results.jsonl has overall >= RAGAS_DRY_RUN_MIN_OVERALL.

# 9. Alembic round-trip (no new migration expected — 6.6 uses
#    Phase 5.2's grade_logs.exercise_type String column directly)
uv run alembic upgrade head
DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic downgrade -1
DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic upgrade head
# Expected: each command exits 0; idempotent.

# 10. No retrieval on grading path
git grep -n "from app.retrieval" backend/app/main.py
# Expected: only the existing /retrieve endpoint usage, not the grade endpoint.

# 11. Langfuse graceful-degradation path on the three generators + grade endpoint
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
uv run python -c "from app.main import app; print('OK')"
# Expected: imports cleanly; routes mounted.
```

### End-to-end smoke (with `docker compose up -d`)

```bash
COOKIE=$(cat ~/.lexora/.test-session-cookie)

# 12. RAG-on: cloze without retrieval
curl -s -X POST http://localhost:18700/exercises/cloze \
  -H "Cookie: lexora_session=$COOKIE" -d '{}' | jq .enable_rag
# Expected: false.

# 13. RAG-on: cloze with retrieval (Postgres only)
curl -s -X POST http://localhost:18700/exercises/cloze \
  -H "Cookie: lexora_session=$COOKIE" -d '{"enable_rag": true}' \
  | jq '.enable_rag, .retrieved_chunk_count'
# Expected: true, > 0 (on Postgres). On SQLite: true, 0.

# 14. Matching endpoint
curl -s -X POST http://localhost:18700/exercises/match \
  -H "Cookie: lexora_session=$COOKIE" -d '{}' | jq .exercise_type
# Expected: "matching".

# 15. Comprehension endpoint
curl -s -X POST http://localhost:18700/exercises/comprehension \
  -H "Cookie: lexora_session=$COOKIE" -d '{}' | jq .exercise_type
# Expected: "comprehension".

# 16. /exercises/grade 3-way guardrail
curl -s -X POST http://localhost:18700/exercises/grade \
  -H "Cookie: lexora_session=$COOKIE" \
  -d '{"exercise_id": 42, "exercise_type": "matching", "grade": 3}' \
  | jq .exercise_type
# Expected: "matching".

curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  http://localhost:18700/exercises/grade \
  -H "Cookie: lexora_session=$COOKIE" \
  -d '{"exercise_id": 42, "exercise_type": "speaking", "grade": 3}'
# Expected: 422.

# 17. Langfuse: three generator spans land in the lexora project
# Manual: open http://localhost:13000 → lexora project → confirm
# cloze.generate, match.generate, comprehension.generate spans from
# the curls above, each with the metadata keyset.

# 18. grade_logs row has the right exercise_type
psql -h localhost -p 25432 -U lexora lexora \
  -c "SELECT exercise_type, count(*) FROM grade_logs WHERE user_id = <test_user_id> GROUP BY exercise_type"
# Expected: at least one row per exercise_type exercised.

# 19. Ragas dry-run results file is recent
ls -la eval/ragas_results.jsonl
jq -r '.overall' eval/ragas_results.jsonl | head -1
# Expected: file exists, overall >= RAGAS_DRY_RUN_MIN_OVERALL (0.6).
```

### Repo hygiene

```bash
# 20. No secrets in repo
git grep -n "OPENROUTER_API_KEY=*** -- ':!*.example'
git grep -n "RAGAS_API_KEY=*** -- ':!*.example'
# Expected: no output.

# 21. Phase 6 files are documented
ls docs/PHASE-6.md
# Expected: file exists, the file you're reading.

# 22. Eval set extensions exist
wc -l eval/match_judgments.jsonl eval/comprehension_judgments.jsonl
# Expected: 40 each (template-based fallback, same as Phase 4.4's cloze set).
```

When all 22 checks pass, the Phase 6 review card flips to PASS and Phase 6 is done. Phase 7 (collocations + prepositional-objects schema + corpus extension) unblocks.

## File map (what lands on disk in Phase 6)

```
lexora/
├── docs/
│   └── PHASE-6.md                     NEW (this file; doubles as the Phase 6 plan + post-mortem)
├── README.md                         MODIFIED (6.8 — Limitations + exercise-types + API table)
├── backend/
│   ├── app/
│   │   ├── cloze.py                  MODIFIED (6.1 — _retrieve_for_cloze + enable_rag plumbing)
│   │   ├── match.py                  NEW (6.2 — MatchingPair/MatchingExercise + generate_match + DSPy module)
│   │   ├── comprehension.py          NEW (6.4 — ComprehensionExercise + generate_comprehension + DSPy module)
│   │   ├── eval/
│   │   │   ├── __init__.py           NEW (6.7 — empty package init)
│   │   │   └── ragas.py              NEW (6.7 — module constants + format/build helpers)
│   │   ├── schemas.py                MODIFIED (6.1, 6.2, 6.4, 6.6 — request/response models + widened Literal)
│   │   ├── main.py                   MODIFIED (6.1, 6.3, 6.5, 6.6 — three endpoints + grade handler fan-out)
│   │   └── ...                       (existing — fsrs.py, llm.py, observability.py, embeddings.py,
│   │                                 retrieval.py, bootstrap.py, anki_builder.py UNCHANGED)
│   ├── scripts/
│   │   ├── eval_ragas.py             NEW (6.7 — offline runner)
│   │   └── ...                       (existing — eval_cloze.py, optimize_cloze.py UNCHANGED)
│   ├── tests/
│   │   ├── test_match.py             NEW (6.2 — happy path + 422 + Langfuse on/off + RAG opt-in)
│   │   ├── test_match_endpoint.py    NEW (6.3 — 200/401/422/502 cases)
│   │   ├── test_comprehension.py     NEW (6.4 — happy path + Langfuse + RAG opt-in)
│   │   ├── test_comprehension_endpoint.py  NEW (6.5 — 200/401/502 cases)
│   │   ├── test_cloze.py             MODIFIED (6.1 — RAG opt-in + byte-for-byte stability)
│   │   ├── test_grade.py             MODIFIED (6.6 — 3-way guardrail + matching/comprehension handlers)
│   │   ├── test_schemas.py           MODIFIED (6.1, 6.2, 6.4, 6.6 — new request/response shapes)
│   │   └── test_eval/
│   │       └── test_ragas.py         NEW (6.7 — formatter + sample builder + dry-run smoke)
│   ├── pyproject.toml                MODIFIED (6.7 — ragas dep added)
│   └── uv.lock                       MODIFIED (lockfile update from pyproject.toml change)
└── eval/
    ├── match_judgments.jsonl         NEW (6.7 — 40 rows, template-based fallback)
    └── comprehension_judgments.jsonl NEW (6.7 — 40 rows, template-based fallback)
```

## What Phase 7 picks up

The Phase 7 plan card (not created yet) will land on the lexora board when Phase 6 review flips PASS. It will:

- Add the `collocations` + `verb_prepositions` schema (the `collocations.py` + `prepositions.py` loaders from `lexora-data/loader/`).
- Extend the corpus with collocations + prepositional objects, re-run the embedding backfill (Phase 1's `scripts.backfill_embeddings`).
- Possibly expand the RAG-on context to include collocations + prepositional-objects chunks (a 6.7 follow-on if Ragas reports low context recall on the held-out set).
- The exercise generator prompts may need refresh — Phase 6's `match-v1` / `comprehension-v1` templates are cloze-era; Phase 7 may add a `match-v2` / `comprehension-v2` A/B cohort.

## Gotchas anticipated (the lessons learned)

These are the patterns the team has hit in earlier phases that this plan encodes around:

1. **`/retrieve` shape is the contract.** Phase 6.1's `_retrieve_for_cloze` calls `app.retrieval.retrieve(db, query_vec, k=RAG_TOP_K, source="both")`. If a future maintainer changes the return shape, 6.1's `format_retrieved_chunks` (6.7's helper) breaks. The contract is "list of dicts with `kind`, `id`, `text` fields" — locked here.
2. **`enable_rag=False` is byte-for-byte stable.** A `git diff` of the prompt bytes between Phase 4.2 (no RAG) and Phase 6.1 (RAG-off) shows zero differences. The test that asserts this lives in `test_cloze.py` and is a Helena review criterion. If a future maintainer accidentally rephrases the system prompt while adding the RAG branch, the diff surfaces.
3. **Harness redaction on `.env.example`** mangles `KEY` / `SECRET` / `TOKEN` var names when written via `patch` or `write_file`. 6.x adds `RAGAS_API_KEY` to `.env.example` (the Ragas judge-model key); the file must be written via a Python script that reconstructs variable names from non-triggering fragments, same as Phase 1's `OPENROUTER_API_KEY` workaround (NOTES.md §"Gotchas hit"). The literal bytes on disk are correct; the terminal output is just display-redacted.
4. **Hard-coded Docker paths** broke Phase 1's pytest harness. All Phase 6 env vars are read via `os.getenv` with sensible defaults; no `/app/...` literals. The RAG-related constants (`RAG_TOP_K`, `RAG_MAX_CHARS`) are module constants, not env-derived — Hard rule #9.
5. **`notify-subscribe` cap.** All workers self-send at the end of their turn via `hermes send`, never `kanban notify-subscribe` to Anurag's Discord/Telegram. Standing rule, restated here for the 6.x workers.
6. **DSPy adapter reuse.** Phase 4.2's `_DSPyOpenAICompatLM` is a private class in `app/cloze.py`. If 6.2 + 6.4 need it, the cleanest path is to extract it to `app/llm.py` (one shared utility). The alternative — duplicating it in `app/match.py` and `app/comprehension.py` — is more code to maintain and harder to keep in sync. 6.2's body instructs the coder to extract.
7. **`exercise_type` widening is wire-level, not just docstring.** Phase 6.6 widens `Literal["cloze"]` to `Literal["cloze", "matching", "comprehension"]` on BOTH `GradeRequest` and `GradeResponse`. Existing cloze callers are unaffected (Pydantic accepts the narrower value as a subset). The `grade_logs.exercise_type` column is already a `String` from Phase 5.2 — no Alembic migration is needed for the widening.
8. **Ragas `--dry-run` proxy.** Per Phase 4.4's deviation pattern, the held-out eval sets (`eval/match_judgments.jsonl`, `eval/comprehension_judgments.jsonl`) are template-based fallback, not AI-generated. The Phase 4 review flagged OpenRouter's chat-model guardrail as a data issue; the Phase 6 mirror uses the same template-based approach to keep CI smoke hermetic. If a future LLM-judge-based eval becomes available, it's a follow-on card that swaps `--dry-run`'s proxy for the real Ragas library path.
9. **Frontend is out of scope for Phase 6.** The match + comprehension endpoints are wire-only; the SPA rendering is Phase 9 (study-session mixing). A future maintainer reading this doc should not "fix" the missing `/exercises/match` page — it's deferred on purpose. The endpoints are curl-exercisable for QA today.
10. **`fsrs_cards` row shape stays Phase-0.** The 3-way grade widening in 6.6 doesn't touch `fsrs_cards` — the schema columns Phase 5.1 defined are the final shape. The `exercise_type` discriminator is on the `grade_logs` row, not the `fsrs_cards` row. Per Hard rule #11 (existing callers stay byte-for-byte unchanged), no new column on `fsrs_cards`.
11. **The first-login gate stays cloze-only.** Phase 5.6 routes users with due cards to `/exercises/due`. Phase 6 doesn't extend this — the `fsrs_cards.due_date` is the gate's only signal, and it's a per-word index, not an exercise-type index. Phase 9's study-session mixing will revisit this (e.g. "show me one cloze + one matching + one comprehension"). For Phase 6, the matching/comprehension exercises are available via direct navigation, not via the gate.
12. **No study-session mixing.** Even though `grade_logs.exercise_type` is now a 3-way literal, the `/exercises/due` route stays cloze-only (Phase 5.4's contract). Mixing is Phase 9+. A future maintainer adding a `?exercise_type=` query parameter to `/exercises/due` is doing Phase 9 work, not Phase 6 work.