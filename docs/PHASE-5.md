# Phase 5 — py-fsrs loop + exercise grading + due-queue

> Spec card: `t_4538af2d` on the lexora board (apollo, plan).
> Parent: `t_15b304ce` (Phase 4.6 review verdict PASS on 2026-07-03).
> Scope source: `project_ideas/15_lexora_personalized_learner.md` line 35 (Phase 5) + lines 138–140 (Must-be) + 161–162 (Should-be).
> Standing permission: Anurag 2026-06-28 — "Do it with AI, it's fine. I'm not gonna touch it... just do it and get the tickets moving." No fresh sign-off needed for this plan card.

## Outcome of this phase

By the time Phase 5 closes, lexora has the **closed study loop**:

1. **The cloze flow is no longer a dead-end.** A user lands on `/exercises/cloze`, grades the cloze with one of four buttons (Again / Hard / Good / Easy) → the backend persists a row to `fsrs_cards`, schedules the next review via `py-fsrs`, and immediately renders the next due card inline. No separate "submit and wait" step.
2. **A new `/exercises/due` endpoint.** Returns the next FSRS-due cloze for the authenticated user. Used by the first-login gate (5.6) to route users with outstanding cards away from `/diagnostic` and onto `/exercises/due`.
3. **First-login gate update.** If the user has any due cards on login, the gate routes to `/exercises/due`. Otherwise the existing diagnostic / weakness-profile behavior applies (no regression).
4. **A `grade_log` observability table** with `(user_id, exercise_id, grade, scheduled_next_due_at, trace_id)` — every grade writes one row, traceable through the existing `lexora` Langfuse project. The trace ID links the grade to the Langfuse span emitted at grade-persist time.
5. **All Phase 5 test cases green**, no secrets in the repo, OpenRouter remains the only LLM provider, `fsrs_cards` finally accepts writes, no retrieval-augmented prompts, no Ragas eval (those are Phase 6).
6. **README Limitations section is honest.** The "no spaced-repetition loop yet" line becomes "closed study loop lives; matching/comprehension + RAG-on are deferred to Phase 6."

The closed-loop outcome is the deliverable. Each piece (4 buttons, an endpoint, a DB write, a Langfuse trace) is plumbing for that outcome.

## What is NOT in Phase 5 (deferred — keep the discipline)

- **No matching exercise type.** Phase 6.
- **No comprehension exercise type.** Phase 6.
- **No retrieval-augmented prompts.** The cloze path stays non-RAG. Phase 6 is the RAG-on phase.
- **No Ragas eval.** Phase 6 (needs retrieval to be live).
- **No new LLM provider.** OpenRouter remains the only path.
- **No multi-agent orchestration.**
- **No new schema columns on `fsrs_cards`.** The existing Phase-0 schema is the final shape: `id, word_id, difficulty, stability, retrievability, due_date, last_review, reps, lapses, state, elapsed_days, scheduled_days`. We add a separate `grade_logs` table for the per-grade audit trail — that's a new table, not a column on `fsrs_cards`. No `review_history` denormalization, no graph-of-cards.
- **No custom card scheduling algorithm.** `py-fsrs` defaults only; do not roll our own. The default 19-parameter weights are shipped as a typed constant.
- **No forced review queue / surface area beyond the inline "next card" flow.** No `/review` page in Phase 5. The inline next-card surface IS the queue.
- **No auth changes.** Reuse the Phase 2 JWT cookie. No refresh tokens, no session expiry changes.
- **No diagnostic probe changes.** The first-login gate update (5.6) only routes differently when there are due cards; the probe's question bank, scoring, and result review are untouched.
- **No Anki export changes.** Phase 0's `anki_builder.py` stays as-is; the FSRS loop is a parallel path that does not touch the static deck generation.

## Hard rules (apply to every 5.x build card)

These rules are enforced by the card body and Helena's review. A build that violates any one of them is `FAIL`.

1. **py-fsrs only.** No homegrown SR algorithm, no manual intervals. Version-pinned to `fsrs==4.1.2` (last release before the 5.x breaking change to 21-parameter weights and renamed serializer methods — pin as a `uv` constraint; the spec explicitly says to lock the version). The version lives as a typed constant `PY_FSRS_VERSION = "4.1.2"` in `backend/app/fsrs.py` and is asserted on import. Bumping requires a deliberate code change, not a `uv lock` drift.
2. **Cloze-only grading in Phase 5.** Even if `/exercises/grade` accepts a generic shape, the only `exercise_type` that goes through it is `cloze`. Phase 6 adds matching/comprehension. The route handler asserts `exercise_type == "cloze"` and 422s otherwise — this is a type-level guardrail, not a soft check.
3. **Single LLM provider (OpenRouter).** No new keys, no LangChain. Phase 4's `app/llm.py` is reused; no chat-client changes.
4. **Every state-mutating call is traced.** `get_langfuse()` from `backend/app/observability.py` is called before the grade-persist write and the `trace_id` is stored on the `grade_logs` row. The existing graceful-degradation path (returns `None` when keys missing) is reused — the write proceeds with `trace_id = None` rather than failing.
5. **Pydantic v2 validated input.** `/exercises/grade` payload is a `pydantic.BaseModel`; `grade` enum is `Literal[1, 2, 3, 4]`. Out-of-range → 422. `exercise_id` is a positive int. `exercise_type` is `Literal["cloze"]` (the only valid value in Phase 5).
6. **No committed secrets.** Same pattern as Phase 4. The literal API key never enters the repo; `OPENROUTER_API_KEY` and Langfuse keys stay in `~/.lexora/.env`.
7. **Offline-capable tests.** py-fsrs scheduling is deterministic — all schedule tests run without DB. Integration tests use the same test DB; no live Langfuse. The grading endpoint test mocks `get_langfuse` and asserts the trace_id propagates to the row when keys are set, and `None` when unset.
8. **Type-level guardrails on thresholds.** Default FSRS parameters (the 19-tuple), `desired_retention`, `learning_steps`, `relearning_steps`, `maximum_interval`, `enable_fuzzing`, the version pin — all hard-coded module constants in `backend/app/fsrs.py`. Never config, never env. If someone wants to tune `desired_retention`, they edit the file, commit, and review.
9. **All Phase 5 work goes on the lexora board.** Not `default`. This includes the build cards, the schema migration, the doc updates, the FSRS tests — anything the team produces.
10. **No `notify-subscribe` to Anurag's Discord/Telegram.** Per the standing framework rule (the `completed` builder caps summaries at ~200 chars; broken path). Workers self-send at the end of their turn via `hermes send`.

## The FSRS shape (locked contract for Phase 5)

The `fsrs_cards` row is what py-fsrs reads on every grade. The schema is fixed by Phase 0; Phase 5 populates it. Mapping:

| `fsrs_cards` column | py-fsrs `Card` field | Notes |
|---|---|---|
| `word_id` | (FK only) | One card per word. Unique constraint added in 5.2's Alembic migration. |
| `difficulty` | `card.difficulty` | float, FSRS D value |
| `stability` | `card.stability` | float, FSRS S value |
| `retrievability` | (derived) | computed via `card.get_retrievability()` at read time, NOT stored on every grade (Phase 6 may add a snapshot column; not Phase 5) |
| `due_date` | `card.due` | UTC tz-aware |
| `last_review` | `card.last_review` | UTC tz-aware |
| `reps` | `card.reps` | int |
| `lapses` | `card.lapses` | int |
| `state` | `card.state` | int, 1/2/3 (Learning/Review/Relearning) |
| `elapsed_days` | `card.elapsed_days` | int |
| `scheduled_days` | `card.scheduled_days` | int |

The grader (5.3) reads the row, reconstructs the `Card` via `Card.from_dict()`, runs `scheduler.review_card(card, Rating(grade))`, persists the new row + inserts a `grade_logs` audit row.

## The metadata contract (Langfuse + `grade_logs` row)

Every grade writes the same shape, both on the Langfuse trace and on the `grade_logs` row. Inherits the Phase 4 contract fields where they apply; adds grade-specific fields.

| Field | Type | Source | Notes |
|---|---|---|---|
| `user_id` | int | JWT subject | From `auth.dependencies.current_user_id` |
| `exercise_id` | int | Request payload | FK to a notional `exercises` table OR the (cloze_id, word_id) composite — see 5.2 |
| `exercise_type` | `Literal["cloze"]` | Request payload | Type-level guardrail (Hard rule #2) |
| `word_id` | int | Looked up from exercise_id | FK to `words.id` |
| `grade` | int (1-4) | Request payload | FSRS `Rating` enum value |
| `scheduled_next_due_at` | DateTime | py-fsrs `card.due` after `review_card` | UTC tz-aware |
| `prev_due_at` | DateTime | py-fsrs `card.due` before `review_card` | For interval-delta observability |
| `state` | int | post-review `card.state` | 1/2/3 |
| `stability` | float | post-review `card.stability` | |
| `difficulty` | float | post-review `card.difficulty` | |
| `reps` | int | post-review `card.reps` | |
| `lapses` | int | post-review `card.lapses` | |
| `trace_id` | str \| null | Langfuse span id | `None` when keys are unset |
| `latency_ms` | int | Activity boundary | For the Langfuse trace |
| `graded_at` | DateTime | server-side `datetime.now(UTC)` | UTC tz-aware |

The Pydantic wire model is `schemas.GradeRequest` / `schemas.GradeResponse` (5.2's deliverable).

## Build-card breakdown

Six build cards (5.1–5.6) + one Helena review card (5.7). All seven parent-linked to `t_4538af2d` (this card). The dependency ordering is:

```
5.1 (fsrs module + version pin + tests)     ──┐
                                              ├──► 5.3 (grade endpoint) ──┐
5.2 (schema migration + GradeRequest/Response) ──┤                          ├──► 5.5 (frontend) ──► 5.6 (first-login gate)
                                              └──► 5.4 (/exercises/due) ────┘
                                                                          
5.7 (Helena review) — parent-linked to all five + 5.1 + 5.2
```

5.1 + 5.2 can run in parallel (no shared file dependencies). 5.3 + 5.4 each wait on 5.1 + 5.2. 5.5 waits on 5.3. 5.6 waits on 5.4. The dispatcher doesn't know this; the card bodies encode the dependency by **referring to the upstream cards in the body** so a worker that spawns on 5.3 before 5.1 lands sees "wait for 5.1" and self-blocks. This is the Phase 2 / Phase 4 pattern.

### 5.1 — py-fsrs integration module (`backend/app/fsrs.py`)

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Can run in parallel with 5.2.
- **Why this card exists:** No `fsrs.py` exists yet. The repo's only mention of FSRS is the empty `FsrsCard` table from Phase 0 and a comment promising Phase 5 wiring. This card builds the integration layer everything downstream imports.
- **Scope:**
  - `backend/app/fsrs.py` — the integration module:
    - `PY_FSRS_VERSION = "4.1.2"` module constant. An import-time assertion checks `importlib.metadata.version("fsrs") == PY_FSRS_VERSION`; mismatch raises `RuntimeError` with the install-vs-pin delta.
    - `DEFAULT_PARAMETERS = (0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925, 1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621)` — the 19-tuple from py-fsrs v4.1.2's default scheduler. Hard-coded.
    - `DEFAULT_DESIRED_RETENTION = 0.9`, `DEFAULT_LEARNING_STEPS = (timedelta(minutes=1), timedelta(minutes=10))`, `DEFAULT_RELEARNING_STEPS = (timedelta(minutes=10),)`, `DEFAULT_MAXIMUM_INTERVAL = 36500`, `DEFAULT_ENABLE_FUZZING = True` — all module constants.
    - `build_scheduler() -> Scheduler` — constructs a `Scheduler` with the constants above. The single construction site for the whole phase; no other module instantiates `Scheduler` directly.
    - `row_to_card(row: FsrsCard) -> Card` and `card_to_row_dict(card: Card, word_id: int) -> dict` — bridges the SQLAlchemy row to py-fsrs `Card` via `Card.from_dict` / `to_dict`. The mapping table in this doc is the authoritative schema; the docstring cross-links it.
    - `apply_grade(card: Card, grade: int) -> tuple[Card, ReviewLog]` — thin wrapper around `scheduler.review_card(card, Rating(grade))`. Returns the updated card + the review log (the `ReviewLog` is what 5.2's `grade_logs` row reads).
  - `backend/tests/test_fsrs.py` — pytest cases for:
    - Version pin: `PY_FSRS_VERSION` matches the installed `fsrs` package version.
    - `row_to_card` round-trips: build a `Card`, set every column, serialize via `to_dict`, and the resulting `Card.from_dict` matches the original.
    - `apply_grade(card, 1)` (Again) on a Learning-state card transitions to Relearning; on a Review-state card increments `lapses`.
    - `apply_grade(card, 3)` (Good) on a Learning card graduates to Review.
    - `apply_grade(card, 4)` (Easy) on a Learning card jumps straight to Review with longer `scheduled_days`.
    - The scheduler instance is deterministic across calls (fuzzing disabled in tests via a `build_scheduler(enable_fuzzing=False)` helper — fuzzing on by default in production, off in tests for repeatability).
    - No DB, no network: every case constructs `Card()` in-memory.
  - `backend/pyproject.toml` — adds `fsrs==4.1.2` to the production deps; `uv lock` regenerates. No `[optimizer]` extra (Hard rule: defaults only, no parameter tuning).
  - README: short note in the existing Architecture section that py-fsrs is wired at v4.1.2.
- **Out of scope:** the `grade_logs` table (5.2), the grading HTTP route (5.3), the due-queue route (5.4), the first-login gate update (5.6). 5.1 is the integration layer only — DB-aware callers live downstream.
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_fsrs.py -v` → all green, no DB, no network.
  - `uv run python -c "from app.fsrs import PY_FSRS_VERSION, build_scheduler; assert PY_FSRS_VERSION == '4.1.2'; s = build_scheduler(); print(s)"` → exits 0.
  - `uv pip show fsrs | grep Version` → `4.1.2`.
  - `git diff main -- backend/app/models.py` is empty (5.1 doesn't touch the schema; 5.2 owns the migration).
  - `git diff main -- backend/app/main.py backend/app/cloze.py` is empty (5.1 doesn't add routes).

### 5.2 — Schema migration + `grade_logs` table + GradeRequest/GradeResponse

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Can run in parallel with 5.1.
- **Why this card exists:** Phase 0's baseline migration created `fsrs_cards` but the existing rows have no `word_id` UNIQUE constraint — Phase 5 needs one card per word. The `grade_logs` audit table is also new. The Pydantic wire models for `/exercises/grade` live here so 5.3 and 5.4 import the same shape.
- **Scope:**
  - `backend/alembic/versions/<hash>_phase5_fsrs_unique_and_grade_logs.py` — new Alembic revision. Idempotent, reversible. Operations:
    - On `fsrs_cards`: add `UNIQUE (word_id)` constraint. SQLite + Postgres compatible.
    - New table `grade_logs`:
      ```python
      class GradeLog(Base):
          __tablename__ = "grade_logs"
          id = Column(Integer, primary_key=True, index=True)
          user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
          exercise_id = Column(Integer, nullable=False)
          exercise_type = Column(String, nullable=False)  # "cloze" only in Phase 5
          word_id = Column(Integer, nullable=False)
          grade = Column(Integer, nullable=False)  # 1-4
          scheduled_next_due_at = Column(DateTime, nullable=False)
          prev_due_at = Column(DateTime, nullable=False)
          state = Column(Integer, nullable=False)
          stability = Column(Float, nullable=False)
          difficulty = Column(Float, nullable=False)
          reps = Column(Integer, nullable=False)
          lapses = Column(Integer, nullable=False)
          trace_id = Column(String, nullable=True)
          latency_ms = Column(Integer, nullable=False)
          graded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
      ```
    - `db_id INTEGER PRIMARY KEY AUTOINCREMENT` for SQLite; serial for Postgres. Alembic handles both.
  - `backend/app/models.py` — add the `GradeLog` SQLAlchemy class. Update `FsrsCard.word_id` to carry the unique constraint (`unique=True` on the column definition).
  - `backend/app/schemas.py` — add:
    - `class GradeRequest(BaseModel)` with `exercise_id: int = Field(..., gt=0)`, `exercise_type: Literal["cloze"]`, `grade: Literal[1, 2, 3, 4]`.
    - `class GradeResponse(BaseModel)` carrying the next due card + the metadata-contract fields:
      ```python
      class GradeResponse(BaseModel):
          graded: Literal[True] = True
          exercise_id: int
          exercise_type: Literal["cloze"]
          next_due_at: datetime
          card_state: int  # 1/2/3
          stability: float
          difficulty: float
          trace_id: str | None
      ```
  - `backend/tests/test_fsrs_schema.py` — Alembic round-trip cases:
    - `alembic upgrade head` runs cleanly on both SQLite (test DB) and Postgres (skipped if `DATABASE_URL` is unset).
    - `alembic downgrade -1` reverses both the unique constraint and the `grade_logs` table.
    - Inserting a second `FsrsCard` row with the same `word_id` raises `IntegrityError`.
    - `GradeLog` insert + select round-trip.
  - `backend/tests/test_schemas.py` — extend with:
    - `GradeRequest(grade=5)` → ValidationError.
    - `GradeRequest(exercise_type="matching")` → ValidationError (Phase 5 type-level guardrail).
    - `GradeResponse` JSON round-trip preserves all fields.
- **Out of scope:** the actual `/exercises/grade` and `/exercises/due` routes (5.3 and 5.4), the `apply_grade` wiring (5.1 already exported it; 5.3 imports). The schema + the wire shapes are 5.2's only deliverable.
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_fsrs_schema.py tests/test_schemas.py -v` → all green.
  - `cd backend && DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic upgrade head` → exits 0.
  - `cd backend && DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic downgrade -1 && uv run alembic upgrade head` → idempotent.
  - `git diff main -- backend/app/fsrs.py backend/app/cloze.py` is empty (5.2 doesn't touch the FSRS module or the cloze generator).
  - `git grep -n "exercise_type.*matching\|exercise_type.*comprehension" backend/app/schemas.py` returns nothing (the enum is hard-locked to `"cloze"`).

### 5.3 — `/exercises/grade` endpoint + Langfuse tracing

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 5.1 + 5.2.** Body references those card ids.
- **Why this card exists:** The closed loop's left half. Without this, the cloze UI from Phase 4.5 is dead-end.
- **Scope:**
  - `backend/app/main.py` — adds `POST /exercises/grade` (auth-gated):
    - Body: `GradeRequest` (5.2).
    - Looks up the FSRS row for `(current_user.id, exercise_id)` (or `(word_id)` if the cloze is keyed off the word — see 5.2's `exercise_id` resolution). If no row exists: create a fresh `FsrsCard` row (Learning state, due immediately). If a row exists: load via `row_to_card`.
    - Calls `apply_grade(card, request.grade)`.
    - Persists: UPDATEs the `fsrs_cards` row with the new fields; INSERTs a `grade_logs` row with the metadata contract.
    - Wraps the entire persist in a Langfuse span `exercise.grade`:
      ```python
      def _trace_grade(metadata: dict, latency_ms: int) -> str | None:
          client = get_langfuse()
          if client is None:
              return None
          with client.start_as_current_span(name="exercise.grade") as span:
              span.update(input={"grade": metadata["grade"], "exercise_id": metadata["exercise_id"]}, metadata=metadata)
              return span.id  # or however the SDK exposes the span id; document in docstring
      ```
      The returned `trace_id` (or `None` when keys are unset) is stored on the `grade_logs` row.
    - Returns `GradeResponse` (5.2).
    - 401 on missing/invalid JWT. 422 on validation error (FastAPI default for Pydantic). 500 on DB integrity failure (caught and logged with the user_id + word_id + grade for triage).
  - `backend/tests/test_grade.py` — pytest cases:
    - First-grade path: user with no `fsrs_cards` row for the word POSTs a grade; a row is created (Learning → Review on Good); `grade_logs` row inserted with `trace_id=None` (Langfuse keys unset in tests).
    - Repeat-grade path: pre-existing `fsrs_cards` row, POSTs Good; `reps` increments, `scheduled_days` > 0.
    - Langfuse keys present: with `get_langfuse` mocked to return a fake client, assert the span carries every metadata-contract field. Trace_id propagates to the `grade_logs` row.
    - Langfuse keys absent: `get_langfuse` returns `None`; row inserts with `trace_id=None`; no exception.
    - Out-of-range grade: `GradeRequest(grade=5)` → 422 (covered by FastAPI's Pydantic adapter).
    - `exercise_type="matching"` → 422 (the Literal guardrail).
    - No JWT → 401.
    - DB integrity failure (e.g. concurrent insert) → 500 with a structured body.
  - No changes to `backend/app/cloze.py` or `backend/app/llm.py`.
- **Out of scope:** the due-queue endpoint (5.4), the frontend grading UI (5.5), the first-login gate update (5.6). 5.3 is backend-only.
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_grade.py -v` → all green, no network.
  - `curl -X POST http://localhost:18700/exercises/grade -H "Cookie: lexora_session=…" -d '{"exercise_id": 42, "exercise_type": "cloze", "grade": 3}'` → 200 + `GradeResponse` JSON; `grade_logs` table has a new row.
  - With Langfuse keys set, `http://localhost:13000` shows an `exercise.grade` span in the `lexora` project with the metadata keyset.
  - `git diff main -- backend/app/cloze.py backend/app/llm.py` is empty (the cloze generator and chat client are unchanged).
  - `git grep -n "exercise_type.*Literal\\[\"cloze\"\\]" backend/app/` shows the guardrail is applied at the route boundary.

### 5.4 — `/exercises/due` endpoint + first-due-card picker

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 5.1 + 5.2.** Can run in parallel with 5.3.
- **Why this card exists:** The closed loop's right half — given an authenticated user, return the next cloze they're due to review. Used by the first-login gate (5.6) and by the inline "next card" rendering in the frontend (5.5).
- **Scope:**
  - `backend/app/main.py` — adds `GET /exercises/due` (auth-gated):
    - No body. Returns the next due cloze for the user.
    - Query: `SELECT word_id FROM fsrs_cards WHERE word_id IN (user's graded-word set) AND due_date <= now() ORDER BY due_date ASC LIMIT 1`. If no rows, return 204 No Content (the user has nothing due; the frontend can show an empty-state).
    - For the picked `word_id`, generate a fresh cloze via `app.cloze.generate_cloze(db, current_user.id)` — but **lock the word** by extending `generate_cloze`'s internal selection so the caller can pass an explicit `word_id`. (The current `select_target_word` is server-driven. Phase 5 adds an optional `force_word_id: int | None = None` parameter; the function returns the forced word when set, otherwise the existing deterministic selection applies.)
    - On the picked `word_id`, also create a fresh `FsrsCard` row if none exists (Learning state, due immediately) — so the user's first encounter of a word registers a card, and the next grade is well-defined.
    - Response: `ClozeExerciseOut` (existing Phase 4.2 wire shape) + an extra `due_from_fsrs: bool` flag so the frontend knows whether this is a FSRS-driven review (re-grade of a known word) or a fresh pick (no card yet).
    - 204 when no due cards exist. 401 on missing JWT.
  - `backend/tests/test_due.py` — pytest cases:
    - User with one due card → 200 + cloze with `due_from_fsrs=True`.
    - User with no due cards → 204.
    - User with one Learning-state card (due immediately) → 200; card row's `due_date` is in the past or `now()`.
    - User with 3 due cards → 200, returns the one with the earliest `due_date` (assert ordering).
    - First-encounter path: word has no `fsrs_cards` row → a fresh Learning row is created; response carries `due_from_fsrs=False`.
    - No JWT → 401.
  - `backend/app/cloze.py` — extend `select_target_word` with the `force_word_id: int | None = None` parameter. The existing deterministic-by-seed path remains the default. **Backwards-compatible**: Phase 4.5's existing caller (`POST /exercises/cloze`) doesn't pass the parameter and gets the same behavior.
- **Out of scope:** the frontend rendering (5.5), the first-login gate update (5.6 — uses this endpoint but doesn't change it).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_due.py -v` → all green, no network.
  - `curl -X GET http://localhost:18700/exercises/due -H "Cookie: lexora_session=…"` → 200 + `ClozeExerciseOut` JSON when cards are due; 204 when not.
  - `git diff main -- backend/app/cloze.py backend/app/fsrs.py` shows only the `force_word_id` parameter addition to `select_target_word` and its test; no other cloze-generator logic changes.
  - `git grep -n "force_word_id" backend/app/cloze.py` confirms the parameter is used only at the `/exercises/due` call site, not at `/exercises/cloze` (which keeps the deterministic seed path).

### 5.5 — Frontend grading buttons + inline next-card flow

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 5.3.**
- **Why this card exists:** The closed loop's middle — the user-facing buttons and the inline-render-next-card flow. Phase 4.5's `ClozePage.tsx` has a placeholder "Submit" button that shows a "Phase 5 will grade this" toast; this card replaces the placeholder.
- **Scope:**
  - `frontend/src/pages/ClozePage.tsx`:
    - Replace the placeholder "Submit" with four grade buttons: **Again / Hard / Good / Easy**. Each carries the FSRS `Rating` enum value (1/2/3/4) on its `data-grade` attribute.
    - Click handler:
      1. `POST /exercises/grade` with `{exercise_id, exercise_type: "cloze", grade}`. Auth-cookie same as the existing fetch.
      2. On 200: optimistically replace the current cloze with a new one from `GET /exercises/due`. Show a sonner toast with the interval change (e.g. "Next review in 3 days").
      3. On 204: show an honest "All caught up — nothing due right now." empty state.
      4. On 422: toast the validation error.
      5. On 500: toast "Grade failed — try again" and keep the current card.
    - The "Generate another" button from Phase 4.5 becomes a secondary action — it stays for users who want to skip ahead, but the primary flow is grade-then-auto-next.
  - `frontend/src/api.ts` (or `frontend/src/api/cloze.ts`):
    - `gradeCloze(exercise_id, grade): Promise<GradeResponse>`.
    - `getDueCloze(): Promise<ClozeExerciseOut | null>` (returns `null` on 204).
    - Both functions reuse the existing cookie-auth fetch helper from Phase 2.
  - `frontend/src/components/GradeButtons.tsx` — small reusable component for the four-button row. `data-testid` on each button for the test.
  - `frontend/src/lib/__tests__/ClozePage.test.tsx` — extend:
    - Click "Good" → mock `gradeCloze` returns `{graded: true, next_due_at: ..., …}` → mock `getDueCloze` returns a new `ClozeExercise` → assert the new sentence renders.
    - Click "Easy" on a hard cloze → grade 4 → assert the call payload.
    - 204 on `getDueCloze` → empty-state message appears.
    - 422 → sonner toast called with the error.
- **Out of scope:** the `App.tsx` split (still forbidden by the standing rule), the dashboard / streak / cost-per-session surfaces (Phase 5 "Should be" bucket — defer to Phase 6+), the `/review` queue page (Hard rule: no forced review surface area in Phase 5).
- **Acceptance:**
  - `cd frontend && pnpm tsc --noEmit` → no errors.
  - `cd frontend && pnpm test` → all green, including the new GradeButtons + ClozePage cases.
  - `cd frontend && pnpm build` → vite build clean.
  - In a browser session, clicking Good replaces the current cloze with a new one and shows the next-due toast. Clicking Again shows the same word again with a 1-minute interval.
  - `git diff main -- frontend/src/App.tsx` is empty or only carries the same nav-link addition from 4.5; the rest of `App.tsx` is untouched.

### 5.6 — First-login gate update: route to `/exercises/due` when cards are due

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Wait for 5.4.**
- **Why this card exists:** The current first-login gate (Phase 3.3) routes users with no weakness profile to `/diagnostic` and existing-profile users to `/weakness-profile`. Phase 5 adds a third branch: if the user has any due FSRS cards, route to `/exercises/due` instead — that's the actual study flow, and the most valuable thing to show a returning learner.
- **Scope:**
  - `frontend/src/App.tsx` — the post-login gate (Phase 3.3's `first-login-gate.tsx` or equivalent). Add a third branch before the existing profile-state branches:
    1. `GET /exercises/due` on mount.
    2. 200 → navigate to `/exercises/due`.
    3. 204 → fall through to the existing logic (diagnostic if no profile, weakness-profile if profile exists).
  - Backend: no changes (5.4 already exposes the endpoint with the right semantics).
  - `frontend/src/lib/__tests__/first-login-gate.test.tsx` (or whichever Phase 3.3 test file holds the gate) — extend:
    - User with due cards → navigated to `/exercises/due`, not `/diagnostic` or `/weakness-profile`.
    - User without due cards + no profile → still navigates to `/diagnostic` (no regression).
    - User without due cards + profile → still navigates to `/weakness-profile` (no regression).
- **Out of scope:** the underlying study UI (5.5), the diagnostic probe (Phase 3 — untouched), the weakness-profile page (Phase 2.4 — untouched).
- **Acceptance:**
  - `cd frontend && pnpm test` → all green.
  - In a browser, sign up a fresh account, immediately grade one cloze, log out, log back in → routed to `/exercises/due`, not `/diagnostic`.
  - In a browser, sign up a fresh account, do NOT grade anything, log out, log back in → routed to `/diagnostic` (Phase 3.3 behavior unchanged).
  - `git diff main -- backend/app/` is empty (frontend-only change).

### 5.7 — Phase 5 review (Helena)

- **Assignee:** `helena`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Parent-linked to all six build cards (5.1–5.6)** — picks up when all six are `done`.
- **Scope:** Same shape as `t_15b304ce` (Phase 4.6 review). Verifies, with severity-tagged findings (critical / major / minor / nit) and a final PASS / FAIL verdict:
  1. **py-fsrs version is pinned.** `git grep -n "PY_FSRS_VERSION" backend/app/fsrs.py` shows `"4.1.2"`. `uv pip show fsrs | grep Version` returns `4.1.2`. The version-pinning assertion on import is exercised by a test.
  2. **FSRS defaults only.** `git grep -n "Scheduler(" backend/app/` returns ONLY the single construction site in `app/fsrs.py`'s `build_scheduler()`. No other module instantiates `Scheduler` with custom parameters. No `[optimizer]` extra in `pyproject.toml`.
  3. **Cloze-only grading.** `git grep -n "exercise_type.*Literal" backend/app/schemas.py` shows `Literal["cloze"]`. The `/exercises/grade` handler asserts and 422s otherwise (covered by a test).
  4. **Schema is the Phase-0 shape + the additions.** `git diff main -- backend/app/models.py` shows ONLY: `FsrsCard.word_id` carrying `unique=True`, plus the new `GradeLog` class. No other `fsrs_cards` column changes. No `review_history` table, no graph-of-cards.
  5. **Every grade writes a `grade_logs` row.** Test asserts the row count == request count for a burst of N grades.
  6. **Langfuse trace propagates.** With `get_langfuse` mocked to return a client, the `grade_logs.trace_id` column is populated. With `get_langfuse` returning `None`, the row has `trace_id=NULL` and the request still succeeds. The graceful-degradation path is exercised.
  7. **Pydantic-validated input.** `GradeRequest.grade` is `Literal[1,2,3,4]`; out-of-range → 422. `GradeRequest.exercise_type` is `Literal["cloze"]`. Tests cover both.
  8. **No committed secrets.** Same Phase 4 check: `git grep -n "OPENROUTER_API_KEY=*** -- ':!*.example'` returns nothing. The literal key never enters the repo.
  9. **Offline-capable tests.** `cd backend && uv run pytest` exits 0 with no network. `cd backend && uv run pytest tests/test_fsrs.py tests/test_fsrs_schema.py tests/test_grade.py tests/test_due.py` exits 0 with no network (all FSRS scheduling is deterministic, all Langfuse calls are mocked).
  10. **Frontend builds clean.** `cd frontend && pnpm tsc --noEmit` no errors; `pnpm test` all green; `pnpm build` clean. The new GradeButtons component + extended ClozePage test are present.
  11. **Closed-loop end-to-end smoke.** With the dev stack up (`docker compose up -d`):
      - `curl -X POST http://localhost:18700/exercises/cloze -H "Cookie: lexora_session=…"` → 200 + cloze.
      - `curl -X POST http://localhost:18700/exercises/grade -d '{"exercise_id": N, "exercise_type": "cloze", "grade": 3}' -H "Cookie: lexora_session=…"` → 200 + `GradeResponse`.
      - `SELECT count(*) FROM grade_logs WHERE user_id = <test_user>` → ≥ 1.
      - `SELECT count(*) FROM fsrs_cards WHERE word_id = <the graded word>` → 1.
      - `curl -X GET http://localhost:18700/exercises/due -H "Cookie: lexora_session=…"` → 200 + the same or next cloze.
  12. **First-login gate routes correctly.** With a test user who has ≥ 1 due card: login redirects to `/exercises/due`. With a fresh user (no due cards): Phase 3.3's existing behavior applies — `/diagnostic` if no profile, `/weakness-profile` if profile exists. Frontend test covers all three branches.
  13. **Type-level guardrails.** `git grep -n "DEFAULT_PARAMETERS\|PY_FSRS_VERSION\|DEFAULT_DESIRED_RETENTION" backend/app/fsrs.py` shows the constants. No env-derived FSRS parameters anywhere in the codebase (`git grep -n "os.getenv.*FSRS\|getenv.*RETENTION"` returns nothing).
- **Out of scope:** matching / comprehension review (Phase 6), RAG-on review (Phase 6), Ragas review (Phase 6).
- **Acceptance (Helena's verdict is the gate):**
  - Severity-tagged findings posted as a comment on `t_5_7_id`.
  - Verdict line: `VERDICT: PASS` or `VERDICT: FAIL`. FAIL triggers a remediation card (parent-linked to the failing build card), per Phase 3 / Phase 4 pattern.

## Verification at end of Phase 5

The exact commands the team runs (and the expected output) to declare Phase 5 done. Every line should pass cleanly.

### Backend (in `/home/ody/workspace/lexora/backend`)

```bash
# 1. Full pytest sweep — no network, mocked via respx
uv run pytest -v
# Expected: all tests pass, including the new test_fsrs.py + test_fsrs_schema.py +
# test_grade.py + test_due.py + the extended test_schemas.py + all Phase 0–4 tests.

# 2. py-fsrs version pin
uv run python -c "from app.fsrs import PY_FSRS_VERSION; assert PY_FSRS_VERSION == '4.1.2'"
uv pip show fsrs | grep Version
# Expected: 4.1.2 / 4.1.2

# 3. FSRS scheduler is single-source
git grep -n "Scheduler(" app/ backend/app/
# Expected: only one construction site, in app/fsrs.py:build_scheduler().

# 4. No FSRS params from env
git grep -n "getenv.*FSRS\|getenv.*RETENTION"
# Expected: no output.

# 5. Cloze-only grading is enforced at the wire
uv run python -c "from app.schemas import GradeRequest; GradeRequest(exercise_id=1, exercise_type='matching', grade=3)"
# Expected: ValidationError (pydantic).

# 6. Alembic round-trip
uv run alembic upgrade head
DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic downgrade -1
DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic upgrade head
# Expected: each command exits 0; the second `upgrade head` brings back the unique
# constraint + grade_logs table.

# 7. Langfuse graceful-degradation path on the grade endpoint
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
uv run python -c "from app.main import app; print('OK')"
# Expected: imports cleanly; routes mounted.

# 8. No retrieval on grading path
git grep -n "from app.retrieval\|import retrieval" app/exercises.py 2>/dev/null
# Expected: no output (5.x doesn't add retrieval to grading; only Phase 6 wires RAG-on).
```

### Frontend (in `/home/ody/workspace/lexora/frontend`)

```bash
pnpm tsc --noEmit
# Expected: no errors.

pnpm test
# Expected: all tests pass, including the new GradeButtons + extended ClozePage +
# first-login-gate cases.

pnpm build
# Expected: vite build completes, no warnings about missing imports.
```

### End-to-end smoke (with `docker compose up -d`)

```bash
# 9. Closed loop: grade a cloze, see the next due card
COOKIE=$(cat ~/.lexora/.test-session-cookie)
curl -s -X POST http://localhost:18700/exercises/cloze -H "Cookie: lexora_session=$COOKIE" | jq .exercise_id
# Expected: a numeric exercise_id.

curl -s -X POST http://localhost:18700/exercises/grade \
  -H "Cookie: lexora_session=$COOKIE" \
  -d '{"exercise_id": <id>, "exercise_type": "cloze", "grade": 3}' | jq .
# Expected: 200 + GradeResponse JSON.

curl -s -X GET http://localhost:18700/exercises/due -H "Cookie: lexora_session=$COOKIE" | jq .
# Expected: 200 + ClozeExerciseOut (or 204 if no cards due).

# 10. grade_logs row exists
psql -h localhost -p 25432 -U lexora lexora \
  -c "SELECT count(*), max(graded_at) FROM grade_logs WHERE user_id = <test_user_id>"
# Expected: count >= 1, max(graded_at) recent.

# 11. Langfuse grade trace lands
open http://localhost:13000
# Switch to the lexora project → confirm an `exercise.grade` span from the curl above
# is visible with the metadata contract fields populated.

# 12. First-login gate routes by due state
# (Manual in a browser session. Sign up a fresh account, grade one cloze, log out, log
# back in → routed to /exercises/due. Sign up a fresh account with no grades, log out,
# log back in → routed to /diagnostic.)
```

### Repo hygiene

```bash
# 13. No secrets in repo
git grep -n "OPENROUTER_API_KEY=*** -- ':!*.example'
# Expected: no output.

# 14. py-fsrs version in lockfile
grep -A 1 'name = "fsrs"' uv.lock
# Expected: version = "4.1.2".

# 15. Phase 5 files are documented
ls docs/PHASE-5.md
# Expected: file exists, the file you're reading.
```

When all 15 checks pass, the Phase 5 review card flips to PASS and Phase 5 is done. Phase 6 (matching + comprehension + RAG-on + Ragas) unblocks.

## File map (what lands on disk in Phase 5)

```
lexora/
├── docs/
│   └── PHASE-5.md                 NEW (this file; doubles as the Phase 5 plan + post-mortem)
├── backend/
│   ├── app/
│   │   ├── fsrs.py                NEW (5.1 — Scheduler/Card/Rating integration, version pin)
│   │   ├── models.py              MODIFIED (5.2 — GradeLog class, FsrsCard.word_id unique)
│   │   ├── schemas.py             MODIFIED (5.2 — GradeRequest, GradeResponse)
│   │   ├── cloze.py               MODIFIED (5.4 — force_word_id param on select_target_word)
│   │   ├── main.py                MODIFIED (5.3, 5.4 — POST /exercises/grade, GET /exercises/due)
│   │   └── ...                    (existing — llm.py, retrieval.py, observability.py, embeddings.py UNCHANGED)
│   ├── alembic/
│   │   └── versions/<hash>_phase5_fsrs_unique_and_grade_logs.py  NEW (5.2)
│   ├── tests/
│   │   ├── test_fsrs.py           NEW (5.1 — deterministic scheduling, version pin, round-trip)
│   │   ├── test_fsrs_schema.py    NEW (5.2 — Alembic upgrade/downgrade, IntegrityError on duplicate)
│   │   ├── test_schemas.py        MODIFIED (5.2 — GradeRequest/GradeResponse cases)
│   │   ├── test_grade.py          NEW (5.3 — first-grade, repeat-grade, Langfuse on/off, 422 paths)
│   │   └── test_due.py            NEW (5.4 — single-due, no-due, multi-due ordering, first-encounter)
│   ├── pyproject.toml             MODIFIED (fsrs==4.1.2 added; no [optimizer] extra)
│   └── uv.lock                    MODIFIED (lockfile update from pyproject.toml change)
└── frontend/
    └── src/
        ├── components/
        │   └── GradeButtons.tsx    NEW (5.5 — four-button row)
        ├── pages/
        │   └── ClozePage.tsx       MODIFIED (5.5 — replaces placeholder Submit with grade + next-due flow)
        ├── api/cloze.ts            MODIFIED (5.5 — gradeCloze + getDueCloze)
        ├── App.tsx                 MODIFIED (5.6 — first-login gate adds the due-cards branch)
        └── lib/__tests__/
            ├── ClozePage.test.tsx  MODIFIED (5.5 — grade clicks, empty state, error toasts)
            └── first-login-gate.test.tsx  MODIFIED (5.6 — three-branch routing)
```

## What Phase 6 picks up

The Phase 6 plan card (not created yet) will land on the lexora board when Phase 5 review flips PASS. It will:

- Add the matching exercise type (using the existing cloze flow's schema contract as the template).
- Add the comprehension exercise type.
- Wire retrieval-augmented prompts on every exercise type (Phase 1's pgvector plumbing finally gets consumed).
- Add Ragas eval as a regression detector on the retrieval + generation pipeline.
- Inherit the metadata contract from Phase 4 + 5 — the `grade_logs.trace_id` column is the join key for Ragas traces.

## Gotchas anticipated (the lessons learned)

These are the patterns the team has hit in earlier phases that this plan encodes around:

1. **py-fsrs package name is `fsrs`, not `py-fsrs`.** The PyPI distribution is `fsrs` (the GitHub repo is still `py-fsrs`). 5.1's build card body spells this out explicitly: `fsrs==4.1.2`, not `py-fsrs==4.1.2`. The README and PHASE-4.md use "py-fsrs" colloquially; the dep is `fsrs`.
2. **py-fsrs v4.x → v5.x is a breaking change.** v4.1.2 has 19 default parameters; v5.x bumped to 21 and renamed `to_dict` / `from_dict` to `to_json` / `from_json` (v6.x). Phase 5 pins to 4.1.2 deliberately — bumping requires a card that does the migration (the param tuple shape, the serializer method calls, the `get_retrievability` location). Hard rule #1 encodes this. If a build-time smoke test on `uv add fsrs` pulls a different version, the import-time assertion in `app/fsrs.py` raises and the build fails fast.
3. **Harness redaction on `.env.example`** mangles `KEY` / `SECRET` var names when written via `patch` or `write_file`. 5.x doesn't add new env vars (FSRS params are hard-coded constants), but the `.env.example` may be touched by 5.2 if the Alembic migration adds new column defaults. Same Phase 1 NOTES.md pattern: write via Python script that reconstructs variable names from non-triggering fragments. The literal bytes on disk are correct; the terminal output is just display-redacted.
4. **Hard-coded Docker paths** broke Phase 1's pytest harness. All Phase 5 env vars continue to be read via `os.getenv` with sensible defaults; no `/app/...` literals.
5. **`notify-subscribe` cap.** All workers self-send at the end of their turn via `hermes send`, never `kanban notify-subscribe` to Anurag's Discord/Telegram. Standing rule, restated here for the 5.x workers.
6. **Alembic on SQLite + Postgres.** Phase 0's baseline migration already handles the dual-dialect. 5.2's migration uses standard SQL types (`Integer`, `String`, `DateTime`, `Float`) and lets Alembic emit the right dialect. The unique constraint is portable. The FKs (`user_id` → `users.id`) are nullable=False with `ON DELETE CASCADE` (Phase 2's User model is `cascade="all, delete-orphan"` on its relationships; the FK mirrors that).
7. **`force_word_id` on `select_target_word`.** Phase 4.5's `ClozePage` calls `POST /exercises/cloze` (server-driven selection). Phase 5.5's flow calls `GET /exercises/due` (force a specific word). Both go through the same generator — the `force_word_id` parameter is the discriminator. If 5.4 forgets to thread it through, the due-queue will return a fresh cloze instead of the scheduled one — the closed loop breaks. The test for 5.4 asserts the response's `word_id == <forced word_id>`.
8. **Hard rule #3 (cloze-only) is a wire-level guardrail, not just a docstring.** The `Literal["cloze"]` on `GradeRequest.exercise_type` makes `matching` / `comprehension` structurally impossible to accept today. Phase 6 widens the enum and adds the matching handler; the wire is the only place that changes.
9. **`grade_logs` table is append-only.** No UPDATEs, no DELETEs. Hard rule #4 (every grade writes) plus the absence of an update endpoint means the table is an audit log. Phase 6 may add a Postgres trigger to reject `UPDATE` / `DELETE` (Helena may flag this as a Phase 5 review nit; the rule is "no destructive ops on grade_logs" either way).