# Phase 4 — LLM exercise generator (cloze) + DSPy + Langfuse

> Spec card: `t_706a4ffa` on the lexora board (apollo, plan).
> Parent: `t_4ecf021d` (Phase 3 plan, done; review `t_e92b3b75` PASS on 2026-07-02).
> Scope source: `project_ideas/15_lexora_personalized_learner.md` line 35 (Phase 4) + lines 138–140 (Must-be) + 161–162 (Should-be).
> Standing permission: Anurag 2026-06-28 — "Do it with AI, it's fine. I'm not gonna touch it... just do it and get the tickets moving." No fresh sign-off needed for this plan card.

## Outcome of this phase

By the time Phase 4 closes, lexora has:

1. **A working end-to-end cloze-exercise flow.** A logged-in user with a weakness profile can hit `/exercises/cloze` → backend picks a word whose profile-aligned axis is weakest → calls OpenRouter (chat) with a constrained prompt → returns a `ClozeExercise` Pydantic model → frontend renders one sentence with a blank + four choices → user grades (out of scope for Phase 4, but the row is ready for Phase 5's FSRS review).
2. **The first real LLM call in the lexora backend**, wrapped with `get_langfuse()` from `backend/app/observability.py` (loaded in Phase 0, unused until now). Every generation traces to the dedicated `lexora` Langfuse project with the metadata contract documented below.
3. **A DSPy module + signature** for cloze generation, with a MIPROv2 (or BootstrapFewShot) optimizer that runs **offline** against a held-out eval set in `eval/cloze_judgments.jsonl`.
4. **An honest held-out eval set** committed to the repo (50–200 labeled triples). Provenance is LLM-generated, marked explicitly so Anurag can hand-review when he has time.
5. **All Phase 4 test cases green**, no secrets in the repo, OpenRouter as the only LLM provider, `fsrs_cards` table unchanged.

## What is NOT in Phase 4 (deferred — keep the discipline)

- **No matching exercise type.** Phase 5.
- **No comprehension exercise type.** Phase 5 or 6.
- **No free-text exercise types.** Not in any spec'd phase.
- **No FSRS work.** `fsrs_cards` stays empty and untouched. Phase 5.
- **No exercise grading endpoint.** The cloze row exists, but no `/exercises/grade` route, no py-fsrs integration. Phase 5.
- **No due-queue / session UI.** Phase 5.
- **No retrieval-augmented prompts.** The retrieval plumbing (Phase 1) stays unused on the cloze path. Phase 6 is the RAG-on phase.
- **No Ragas eval.** Phase 6 (it needs retrieval to be live).
- **No cost-per-session display.** Phase 5 "Should be" bucket.
- **No "what changed this week" journal.** Phase 5 "Should be" bucket.
- **No asciinema recording.** Phase 6/7 "Should be" bucket.
- **No new LLM provider.** OpenRouter is the only path; no new keys, no LangChain.
- **No multi-agent orchestration.** Single generator path; the eval set is a passive offline signal.

## Hard rules (apply to every 4.x build card)

These rules are enforced by the card body and Helena's review. A build that violates any of them is `FAIL`.

1. **Cloze only.** No matching, no comprehension, no free-text.
2. **No FSRS work.** The `fsrs_cards` table is read-only this phase. No INSERTs, no UPDATEs, no schema change.
3. **No retrieval-augmented prompts.** The cloze activity does NOT call `/retrieve`. Word selection is deterministic from the user's weakness profile + a corpus-word id; context is the word's stored examples (already on the row, not a retrieval query).
4. **Every LLM call is wrapped.** `get_langfuse()` from `backend/app/observability.py` is called before the generation and the metadata contract below is populated. If `get_langfuse()` returns `None` (keys missing), the call still succeeds — graceful degradation is the existing pattern.
5. **OpenRouter only.** Single provider abstraction, same wire format as `embeddings.py`. No LangChain, no new keys.
6. **Pydantic v2 validated output.** `ClozeExercise` is a `pydantic.BaseModel` with the documented field set. Generation is constrained via `instructor` (or `outlines` — pick one, document it). Schema-violation retries are bounded (≤ 3), with a dead-letter on persistent violation.
7. **No committed secrets.** `OPENROUTER_API_KEY` stays in `~/.lexora/.env` (or the host systemd env at `~/.config/environment.d/hermes-openrouter.conf`). The repo's `.env.example` keeps placeholders only.
8. **Offline-capable eval.** The DSPy optimizer and the `scripts/eval_cloze.py` runner never call out to OpenRouter during tests. Mocks via `respx`, same pattern as `test_embeddings.py`.
9. **All Phase 4 work goes on the lexora board.** Not `default`. This includes the build cards, the eval set commits, the doc updates — anything the team produces.
10. **No `notify-subscribe` to Anurag's Discord/Telegram.** Per the standing framework rule (the `completed` builder caps summaries at ~200 chars; broken path). Workers self-send at the end of their turn.
11. **Type-level guardrails on thresholds.** Any new threshold (cloze-quality bar, eval-set passing floor, retry counts, batch sizes) goes as a hard-coded constant in a named module — never a config value, never an env var.
12. **Honest eval-set provenance.** If `eval/cloze_judgments.jsonl` is LLM-generated (the realistic path), the file's leading comment carries `labeler: ai-assisted-claude-minimax-m3` and `provenance: llm-generated-v1-pending-anurag-hand-review`. Same pattern as priorart's Phase 1.5a.

## The metadata contract (Langfuse + Pydantic output)

Every cloze generation records the same shape, both on the Langfuse trace and on the persisted `ClozeExercise` row. Phase 5's FSRS loop reads these fields; locking the contract now saves a migration later.

| Field | Type | Source | Notes |
|---|---|---|---|
| `user_id` | int (str in JSON) | JWT subject | From `auth.dependencies.current_user_id` |
| `weakness_axes` | dict[str, int] | `WeaknessProfile.axes` snapshot at call time | JSON-serialized for Langfuse metadata |
| `word_id` | int | Deterministic selection (see 4.2) | FK to `words.id` |
| `difficulty` | `Literal["easy","medium","hard"]` | Pydantic output | The model's self-rated difficulty |
| `model_id` | str | `OPENROUTER_CHAT_MODEL` env var (default documented in 4.1) | e.g. `qwen/qwen3-235b-a22b-2507` |
| `prompt_template_version` | str | Module-level constant in `app/cloze.py` | Bumped when the prompt template changes |
| `sentence_with_blank` | str | Pydantic output | `___` is the blank marker; UI replaces with input |
| `answer_word_id` | int | Pydantic output | FK to `words.id`; equals `word_id` for the picked word |
| `distractors` | list[int] | Pydantic output | 3 FKs to `words.id`; same word_type as `answer_word_id` |
| `rationale` | str | Pydantic output | One sentence explaining the cloze design |
| `latency_ms` | int | Recorded at the activity boundary | For the Langfuse trace |
| `schema_retry_count` | int | Instructor retry counter | 0 on the happy path; ≤ 3 by hard rule |
| `prompt_tokens`, `completion_tokens` | int | OpenRouter response `usage` block | For cost attribution |

The Pydantic model is:

```python
# backend/app/cloze.py (4.2's deliverable)
from typing import Literal
from pydantic import BaseModel, Field

class ClozeExercise(BaseModel):
    sentence_with_blank: str = Field(..., description="German sentence with '___' marking the cloze position. The LLM must not mutate the answer word's case, article, or surrounding word forms.")
    answer_word_id: int = Field(..., description="FK to words.id of the correct answer.")
    distractors: list[int] = Field(..., min_length=3, max_length=3, description="Exactly 3 FKs to words.id of plausible wrong answers. Same word_type as answer_word_id.")
    difficulty: Literal["easy", "medium", "hard"]
    rationale: str = Field(..., min_length=1, max_length=400, description="One sentence explaining the cloze design.")
    prompt_template_version: str = Field(..., description="Bump when prompt changes; enables A/B eval.")
```

## Build-card breakdown

All five build cards are parent-linked to `t_706a4ffa` (this card). When this plan card completes, the dispatcher auto-promotes the build cards to `ready`. The review card (`t_4.N`) is parent-linked to all five build cards — Helena picks it up when they finish.

The build cards can fan out in parallel for 4.1 + 4.4 (no shared file dependencies), and 4.2 + 4.3 + 4.5 must run after 4.1 (they consume `app/llm.py`). The dispatcher doesn't know this; the card bodies encode the dependency by **referring to the upstream card in the body** rather than via `parents` (so a worker that reads 4.2 first sees "wait for 4.1" and self-blocks). This is the same pattern Phase 2 used.

### 4.1 — OpenRouter chat client (`backend/app/llm.py`)

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`.
- **Why this card exists:** Phase 1 promised an `app/llm.py` in `app/embeddings.py`'s docstring ("Phase 4's exercise generator will import … same wire format as `app/llm.py` will use later"). The file does not exist yet. Build it first; everything downstream imports from it.
- **Scope:**
  - `backend/app/llm.py` — OpenAI-compatible chat client, same wire format as `embeddings.py` (httpx, retries on 429/5xx, exponential backoff). One high-level function:
    ```python
    def complete(
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
    ) -> ChatResult: ...
    ```
    where `ChatResult` carries the assistant text + `usage` + `latency_ms`. No JSON-mode here — that's `instructor`'s job (4.2's deliverable).
  - Constants (hard-coded module-level, not env): `MAX_ATTEMPTS = 3`, `RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504)`, `BACKOFF_SCHEDULE_S = (0.5, 1.0, 2.0)`.
  - Env-derived defaults (these ARE config): `OPENROUTER_CHAT_MODEL` (default `"qwen/qwen3-235b-a22b-2507"` — quality/speed sweet spot on the existing OpenRouter account; subject to a smoke probe at build time), `OPENROUTER_BASE_URL` (default `"https://openrouter.ai/api/v1"`, mirrors `embeddings.py`).
  - `backend/tests/test_llm.py` — pytest cases for batching, retries on 429/5xx, schema-violation retry counter (mocked via `respx`), no-network guarantee, and a "missing API key raises" path.
  - `backend/app/observability.py` is NOT touched here — `get_langfuse()` already exists.
  - `.env.example` adds `OPENROUTER_CHAT_MODEL` + `OPENROUTER_BASE_URL` with comments. The real key (already in `~/.lexora/.env` from Phase 1) is reused — no new env surface.
  - `docker-compose.yml` adds the two new env vars to the backend service.
- **Out of scope:** `instructor`-wrapped schema validation (4.2), DSPy integration (4.2), cloze-specific prompts (4.2).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_llm.py -v` → all cases green, no network calls.
  - `uv run python -c "from app.llm import complete; print(complete.__doc__)"` → imports cleanly.
  - `git grep -n "OPENROUTER_API_KEY"` shows the variable ONLY in `.env.example` and host-systemd paths; the literal key never enters the repo.
  - `git diff main -- backend/app/main.py backend/app/embeddings.py` is empty — this card does not modify either.

### 4.2 — Cloze activity + DSPy module (`backend/app/cloze.py`)

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`.
- **Wait for 4.1.** Body references `t_4_1_id` (filled in at create time) so the worker self-blocks if it spawns before 4.1 lands.
- **Scope:**
  - `backend/app/cloze.py`:
    - The Pydantic `ClozeExercise` model (full schema above).
    - `select_target_word(db, user_id) -> Word` — deterministic id selection from the user's `WeaknessProfile.axes`. Strategy: among the user's weakest axis (highest score = most critical, where 0=unknown / 1=shaky / 2=developing / 3=critical per Phase 2), pick a random word of that word_type from the corpus. Determinism: seeded by `(user_id, axis, day)` so a user re-clicking gets the same word that day; new day → new word. Document the seed scheme in the docstring.
    - `build_prompt(word, weakness_axes) -> list[dict]` — builds the chat messages. Prompt template embeds: the target word + its first example sentence (from `Word.examples`), the user's weakness axes, the explicit prohibition ("Do not change word forms, articles, or case endings; do not invent grammar; do not translate; the answer is always one of the German words shown"), and the JSON output schema. **`PROMPT_TEMPLATE_VERSION = "cloze-v1"`** is the module-level constant; bump on edits.
    - `generate_cloze(db, user_id) -> ClozeExercise` — the activity. Wraps `app.llm.complete` with `instructor` for JSON-schema enforcement. Bounded schema-violation retries (≤ 3, hard rule #6). Records the metadata contract fields on every call. Langfuse wrapping is **4.3's deliverable** — `generate_cloze` calls a no-op `_trace_cloze(result, metadata)` hook that 4.3 fills in. This keeps 4.2 testable without Langfuse installed.
    - The DSPy part:
      - `ClozeSignature(dspy.Signature)` — input: `word: str`, `context_sentence: str`, `learner_axes_json: str`, `target_word_id: int`. Output: `ClozeExercise` (DSPy's Pydantic-typed signature).
      - `ClozeModule(dspy.Module)` — wraps `app.llm.complete` (DSPy talks to OpenRouter directly, no instructor on this path — Phase 4 ships both an instructor path for production and a DSPy path for optimization).
      - `optimize_cloze_module(train_set, val_set) -> ClozeModule` — runs `dspy.MIPROv2` (or `BootstrapFewShot` if MIPROv2 isn't on the dep tree) against the held-out eval set loaded from `eval/cloze_judgments.jsonl`. **Offline** — no OpenRouter calls during CI; the optimization step is a `scripts/optimize_cloze.py` CLI invokable on demand, not an import-time side effect.
  - `backend/app/main.py` — adds `POST /exercises/cloze` (auth-gated). Body: `{}` (word selection is server-driven). Response: the `ClozeExercise` Pydantic model. **No grading endpoint** — that's Phase 5.
  - `backend/tests/test_cloze.py` — pytest cases for:
    - Word selection (seeded determinism — same `(user_id, axis, day)` returns same word).
    - Prompt template version constant is `cloze-v1`.
    - `generate_cloze` calls `app.llm.complete` exactly once on the happy path; up to 3 times when the first response fails Pydantic validation; dead-letters after 3 with a structured error.
    - The Langfuse hook is invoked (mocked) with the metadata contract fields.
    - No retrieval call: `app.retrieval` is NOT imported from `app/cloze.py` (the hard rule, asserted via `sys.modules` snapshot in the test).
    - The route is auth-gated (no JWT → 401).
  - `backend/pyproject.toml` — adds `instructor` + `dspy` (and `dspy-ai` if the import name differs) to the production deps; verify they resolve via `uv lock`. `respx` is already a dev-dep.
  - `backend/scripts/optimize_cloze.py` — CLI: `uv run python -m scripts.optimize_cloze` reads `eval/cloze_judgments.jsonl`, runs MIPROv2 offline (mocked LM via DSPy's `DummyLM` if OpenRouter keys are absent; real LM if present and the user opts in via `--live` flag), writes the optimized prompt to `backend/app/cloze_optimized.json`.
  - README: short note about the new endpoint + how to run the offline optimizer.
- **Out of scope:** retrieval-augmented prompts, FSRS row writes, eval-set creation (that's 4.4), frontend UI (4.5).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_cloze.py -v` → all green.
  - `curl -X POST http://localhost:18700/exercises/cloze -H "Cookie: lexora_session=…"` → 200 + `ClozeExercise` JSON.
  - `uv run python -c "from app.cloze import generate_cloze, PROMPT_TEMPLATE_VERSION; assert PROMPT_TEMPLATE_VERSION == 'cloze-v1'"` → no error.
  - `uv run python -m scripts.optimize_cloze --help` → exits 0.
  - `git diff main -- backend/app/models.py backend/app/retrieval.py` is empty — `fsrs_cards` schema unchanged, retrieval untouched.
  - No call to `app.retrieval` from anywhere in the new code: `git grep -n "from app.retrieval" backend/app/cloze.py backend/app/main.py` returns nothing.

### 4.3 — Langfuse wiring on the cloze path (`backend/app/cloze.py`)

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`.
- **Wait for 4.2.** This card is intentionally small — it fills in the `_trace_cloze(result, metadata)` hook that 4.2 left as a no-op.
- **Scope:**
  - In `backend/app/cloze.py`, replace the no-op `_trace_cloze` with the real implementation:
    ```python
    def _trace_cloze(result: ClozeExercise, metadata: dict, latency_ms: int) -> None:
        client = get_langfuse()
        if client is None:
            return  # graceful degradation — Phase 0's design.
        with client.start_as_current_span(name="cloze.generate") as span:
            span.update(
                input=metadata["prompt_messages"],
                output=result.model_dump_json(),
                metadata={
                    "user_id": metadata["user_id"],
                    "weakness_axes": metadata["weakness_axes"],
                    "word_id": metadata["word_id"],
                    "difficulty": result.difficulty,
                    "model_id": metadata["model_id"],
                    "prompt_template_version": result.prompt_template_version,
                    "schema_retry_count": metadata["schema_retry_count"],
                    "latency_ms": latency_ms,
                    "prompt_tokens": metadata["prompt_tokens"],
                    "completion_tokens": metadata["completion_tokens"],
                },
            )
    ```
    (Implementation choice between `langfuse_context` and the SDK `start_as_current_span` — pick whichever the installed `langfuse` SDK exposes; document the choice in the docstring.)
  - `backend/tests/test_cloze.py` — add cases:
    - `_trace_cloze` with `get_langfuse` returning `None` does NOT raise and does NOT contact the network (the function is a no-op in that branch).
    - `_trace_cloze` with a mocked Langfuse client records a span carrying every metadata-contract field. Assert the metadata dict keyset matches the contract table exactly.
  - `backend/app/main.py` — the `/retrieve` endpoint already traces (Phase 1); no change needed here. The startup `_ensure_client()` already warms the Langfuse singleton; no change needed.
- **Out of scope:** writing new Langfuse wrappers (the existing `get_langfuse()` is sufficient), changing the project (still `lexora` — Phase 0 wired it).
- **Acceptance:**
  - `cd backend && uv run pytest tests/test_cloze.py -v` → all green, including the two new cases.
  - With `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` unset, `uv run python -c "from app.cloze import _trace_cloze; _trace_cloze(...)"` returns silently (no exception, no network).
  - With keys set (the `~/.lexora/.env` values), the trace lands in the `lexora` project on the shared instance, visible in the UI's project dropdown. Confirmed manually by Anurag or via the QA hook below.
  - `git diff main -- backend/app/observability.py` is empty (don't touch the wrapper — extend it from the call site only).
  - **QA hook (manual):** with keys set, run `uv run python -c "from app.cloze import generate_cloze; ..."` (a short script that calls `generate_cloze` once with a real DB), then check `http://localhost:13000` → switch to `lexora` project → confirm a `cloze.generate` span is present with the metadata keyset.

### 4.4 — Held-out cloze eval set + offline runner

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. Can run in parallel with 4.1 (no shared file dependencies).
- **Why this card exists:** The Phase 4 spec requires the eval set (`eval/cloze_judgments.jsonl`) for the DSPy optimizer to have anything to optimize against. Without it, 4.2's optimizer is a stub.
- **Scope:**
  - `eval/cloze_judgments.jsonl` — 50–100 labeled triples (the spec says 50–200; the realistic path for LLM-generated labels is ~80 before diminishing returns). Shape: one JSON object per line:
    ```json
    {
      "word_id": 42,
      "word": "wählen",
      "context_sentence": "Die Partei hat einen neuen Vorsitzenden ___.",
      "expected_answer_word_id": 42,
      "expected_distractors": [1042, 2087, 3155],
      "expected_difficulty": "medium",
      "labeler": "ai-assisted-claude-minimax-m3",
      "provenance": "llm-generated-v1-pending-anurag-hand-review",
      "judgment": "accept",
      "rationale": "Sentence clearly cues wählen via 'Vorsitzenden'. Distractors are all verbs that fit grammatically but semantically mismatch."
    }
    ```
    **Leading comment (the first line) declares provenance:**
    ```
    # labeler: ai-assisted-claude-minimax-m3
    # provenance: llm-generated-v1-pending-anurag-hand-review
    # spec: project_ideas/15_lexora_personalized_learner.md §Phase 4 / Must-be
    # bar: "would a C1 German speaker accept this cloze without edits?"
    ```
  - `scripts/build_cloze_eval_set.py` — the script that generates the eval set. Reads the corpus, picks 50–100 words across all `word_type`s, calls `app.llm.complete` (instructor-wrapped) for each to produce the cloze, then re-prompts the same model with the cloze to self-judge (accept / reject + rationale). **This script IS allowed to call OpenRouter** — it's the only Phase 4 surface that does (everything else is mocked in tests, offline in CI). It writes `eval/cloze_judgments.jsonl` with the metadata above. Idempotent (re-running overwrites the file with the same content if the seed is fixed).
  - `scripts/eval_cloze.py` — the offline runner. Reads `eval/cloze_judgments.jsonl`, takes a model's cloze output (either from a JSONL of generated clozes or by running the DSPy module against the eval prompts), compares each prediction to the held-out set, computes:
    - `accept_rate` = fraction of `judgment: accept` clozes the new model also accepts (semantic-accept: the new cloze carries the same `answer_word_id` AND has a semantically equivalent sentence).
    - `schema_validity_rate` = fraction of responses that pass Pydantic validation.
    - `rationale_quality_proxy` = average length of the rationale field (rough heuristic; the spec's hard rule is the qualitative C1-accept bar, not a numeric floor).
    The runner writes `eval/results_<timestamp>.json` with the metrics + per-row diffs.
  - `docs/EVAL.md` — short doc explaining how to re-run the optimizer and the eval runner. Mirrors the README's "Embeddings & retrieval" section shape from Phase 1.
- **Out of scope:** Ragas (Phase 6), retrieval eval (Phase 6), live grading by a human (the eval set is LLM-judged by self-consistency; Anurag can hand-review when he has time).
- **Acceptance:**
  - `eval/cloze_judgments.jsonl` exists with 50–100 lines, valid JSONL, leading-comment provenance block, `labeler: ai-assisted-claude-minimax-m3` on every row.
  - `wc -l eval/cloze_judgments.jsonl` returns the expected count.
  - `uv run python -m scripts.eval_cloze --help` exits 0; `uv run python -m scripts.eval_cloze --dry-run` exits 0 and prints "OK" without contacting OpenRouter.
  - `git diff main -- backend/app/models.py backend/app/main.py` is empty (this card touches the eval dir + scripts, not the app).
  - The eval set's first 10 rows are spot-checked against `words` table: every `word_id` / `expected_answer_word_id` exists; every `expected_distractors` FK resolves; every `context_sentence` contains the target word.

### 4.5 — Frontend exercise surface (minimal)

- **Assignee:** `perseus`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`.
- **Wait for 4.2.** Body references the 4.2 card id.
- **Scope:**
  - `frontend/src/pages/ClozePage.tsx` (or equivalent — the repo's current `App.tsx` is monolithic at 555 lines; **Phase 4 does NOT split it** — that's a separate refactor card). One component, one route (`/exercises/cloze`), renders:
    - The sentence with the blank replaced by a styled input.
    - 4 multiple-choice buttons (the distractor words, in randomized order, with the correct answer among them).
    - A "Submit" button (placeholder — no grading endpoint in Phase 4; clicking shows a sonner toast "Phase 5 will grade this").
    - An honest empty state when the user has no `WeaknessProfile` yet: a card linking to `/weakness-profile` (Phase 2's page).
    - A loading skeleton while `/exercises/cloze` is in flight.
    - A "Generate another" button that re-fetches.
  - The page uses `react-router` (already added in Phase 2.3). Add a nav link from the existing header.
  - `frontend/src/api.ts` (or wherever the auth-aware fetch helper lives post-Phase-2) — add a `generateCloze()` function. Cookie-based auth, same as the weakness-profile save call.
  - `frontend/src/lib/__tests__/ClozePage.test.tsx` — vitest case for: blank renders, distractor click updates the selection, "Generate another" re-fetches (mocked), empty-profile state shows the link.
- **Out of scope:** the App.tsx split, the grading loop, the study-session shell, the FSRS-driven queue, the streak/dashboard (Phase 5/6).
- **Acceptance:**
  - `cd frontend && pnpm dev` → loads; navigate to `/exercises/cloze` → blank + 4 choices render.
  - `cd frontend && pnpm test` → all green (the new test + existing suite).
  - `cd frontend && pnpm tsc --noEmit` → no errors.
  - Honest empty state when the user has no weakness profile (verify by signing up a fresh account).
  - The page calls `/exercises/cloze` exactly once on mount (verified in DevTools Network tab).
  - `git diff main -- frontend/src/App.tsx` shows ONLY a nav-link addition — the rest of `App.tsx` is untouched (no opportunistic refactor).

### 4.6 — Phase 4 review (Helena)

- **Assignee:** `helena`. **Workspace:** `worktree:/home/ody/workspace/lexora`, branch `main`. **Parent-linked to all five build cards (4.1, 4.2, 4.3, 4.4, 4.5)** — picks up when all five are `done`.
- **Scope:** Same shape as `t_e92b3b75` (Phase 3 review). Verifies, with severity-tagged findings (critical / major / minor / nit) and a final PASS / FAIL verdict:
  1. **First LLM call discipline.** Every code path that calls OpenRouter for chat goes through `app/llm.py`. Grep `git grep -n "openrouter\|chat/completions" backend/app/` — every hit either imports from `app.llm` (4.1's function) or is a test fixture. No new provider introduced.
  2. **`fsrs_cards` is unchanged.** `git diff main -- backend/app/models.py` for the `FsrsCard` class is empty. `SELECT count(*) FROM fsrs_cards` in the live DB returns 0.
  3. **No retrieval on the cloze path.** `git grep -n "from app.retrieval\|import retrieval" backend/app/cloze.py` returns nothing. The `/exercises/cloze` handler does NOT consume `/retrieve`. Test case asserts no retrieval import.
  4. **Pydantic-validated output.** `ClozeExercise` is `pydantic.BaseModel` with the documented field set. The activity uses `instructor` (or `outlines`, document the choice). Schema-violation retries are bounded ≤ 3. Dead-letter is structured (not a bare 500).
  5. **Offline-capable eval.** `cd backend && uv run pytest` exits 0 with no network calls (verify via `pytest --strict-markers -p no:cacheprovider` and a `toxiproxy`/`respx` log assertion). `uv run python -m scripts.eval_cloze --dry-run` exits 0 without OpenRouter contact.
  6. **No committed secrets.** `git grep -n "OPENROUTER_API_KEY=[^<]"` (i.e. non-placeholder) returns nothing. The key still lives in `~/.lexora/.env`.
  7. **Langfuse metadata contract.** `git grep -n "user_id\|weakness_axes\|prompt_template_version\|schema_retry_count" backend/app/cloze.py` shows the contract fields are populated. The `_trace_cloze` path is exercised in tests with a mocked client carrying every metadata field.
  8. **OpenRouter only.** `git grep -n "anthropic\|openai\|cohere\|groq" backend/app/` returns nothing (except for the words appearing in docstrings explaining why OpenRouter was chosen). No new env var surface beyond `OPENROUTER_CHAT_MODEL` + `OPENROUTER_BASE_URL`.
  9. **Eval set provenance.** `eval/cloze_judgments.jsonl` exists, has the leading-comment provenance block, every row's `labeler` is `ai-assisted-claude-minimax-m3` and `provenance` is `llm-generated-v1-pending-anurag-hand-review`. 50–100 rows, valid JSONL.
  10. **`pytest` green.** All existing tests (74 from Phase 0–3) + all new tests (4.1, 4.2, 4.3, 4.4, 4.5) pass.
  11. **`pnpm tsc --noEmit` clean** for the frontend.
  12. **End-to-end smoke.** With the dev stack up (`docker compose up -d`), the QA hook below passes.
- **Out of scope:** RAG-on review, FSRS review, matching/comprehension review (those are Phase 5/6).
- **Acceptance (Helena's verdict is the gate):**
  - Severity-tagged findings posted as a comment on `t_4_6_id`.
  - Verdict line: `VERDICT: PASS` or `VERDICT: FAIL`. FAIL triggers a remediation card (parent-linked to the failing build card), per Phase 3's pattern.

## Verification at end of Phase 4

The exact commands the team runs (and the expected output) to declare Phase 4 done. Every line should pass cleanly.

### Backend (in `/home/ody/workspace/lexora/backend`)

```bash
# 1. Full pytest sweep — no network, mocked via respx
uv run pytest -v
# Expected: all tests pass, including the new test_llm.py + test_cloze.py + existing 74 from Phases 0-3.

# 2. Langfuse graceful-degradation path
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
uv run python -c "from app.cloze import _trace_cloze; _trace_cloze(None, {}, 0)"
# Expected: exits 0 silently (no exception, no network).

# 3. Pydantic schema enforcement
uv run python -c "from app.cloze import ClozeExercise, PROMPT_TEMPLATE_VERSION; print(PROMPT_TEMPLATE_VERSION)"
# Expected: cloze-v1

# 4. No retrieval on cloze path
git grep -n "from app.retrieval\|import retrieval" app/cloze.py
# Expected: no output.

# 5. fsrs_cards unchanged
git diff main -- app/models.py | grep -i "fsrs"
# Expected: no output.

# 6. Eval set committed
wc -l ../eval/cloze_judgments.jsonl
# Expected: 50 <= count <= 100

# 7. Eval runner offline mode
uv run python -m scripts.eval_cloze --dry-run
# Expected: "OK" printed, exit 0, no OpenRouter contact.

# 8. DSPy optimizer CLI works
uv run python -m scripts.optimize_cloze --help
# Expected: usage banner, exit 0.
```

### Frontend (in `/home/ody/workspace/lexora/frontend`)

```bash
pnpm tsc --noEmit
# Expected: no errors.

pnpm test
# Expected: all tests pass, including the new ClozePage test.

pnpm build
# Expected: vite build completes, no warnings about missing imports.
```

### End-to-end smoke (with `docker compose up -d`)

```bash
# 9. Live cloze endpoint
curl -s -X POST http://localhost:18700/exercises/cloze \
  -H "Cookie: lexora_session=$(cat ~/.lexora/.test-session-cookie)" | jq .
# Expected: 200 + ClozeExercise JSON with all fields populated.

# 10. Langfuse trace lands
open http://localhost:13000
# Switch to the lexora project → confirm a cloze.generate span from the curl above
# is visible with the metadata contract fields populated.
```

### Repo hygiene

```bash
# 11. No secrets in repo
git grep -n "OPENROUTER_API_KEY=[^<]" -- ':!*.example'
# Expected: no output (only the .env.example placeholder matches, which is excluded).

# 12. Phase 4 files are documented
ls docs/PHASE-4.md
# Expected: file exists, the file you're reading.
```

When all 12 checks pass, the Phase 4 review card (`t_4_6_id`) flips to PASS and Phase 4 is done. Phase 5 (FSRS + matching + grading) unblocks.

## File map (what lands on disk in Phase 4)

```
lexora/
├── docs/
│   └── PHASE-4.md                 NEW (this file; doubles as the Phase 4 plan + post-mortem)
├── eval/
│   └── cloze_judgments.jsonl      NEW (50-100 labeled triples, LLM-generated provenance)
├── backend/
│   ├── app/
│   │   ├── llm.py                 NEW (OpenRouter chat client; 4.1)
│   │   ├── cloze.py               NEW (ClozeExercise, generate_cloze, DSPy module + optimizer, _trace_cloze; 4.2 + 4.3)
│   │   ├── main.py                MODIFIED (POST /exercises/cloze route; 4.2)
│   │   └── ...                    (existing — embeddings.py, retrieval.py, observability.py UNCHANGED)
│   ├── scripts/
│   │   ├── build_cloze_eval_set.py  NEW (4.4 — generates the eval set)
│   │   ├── eval_cloze.py            NEW (4.4 — offline runner)
│   │   └── optimize_cloze.py        NEW (4.2 — DSPy MIPROv2 CLI)
│   ├── tests/
│   │   ├── test_llm.py            NEW (4.1 — 6-10 cases, mocked via respx)
│   │   └── test_cloze.py          NEW (4.2 + 4.3 — 8-12 cases)
│   ├── pyproject.toml             MODIFIED (instructor + dspy added)
│   └── uv.lock                    MODIFIED (lockfile update from pyproject.toml change)
└── frontend/
    └── src/
        └── pages/ClozePage.tsx    NEW (4.5 — minimal single-shot cloze display)
```

## What Phase 5 picks up

The Phase 5 plan card (not created yet) will land on the lexora board when Phase 4 review flips PASS. It will:

- Add the grading endpoint (`POST /exercises/grade`).
- Wire `py-fsrs` to the `fsrs_cards` table (already present, empty since Phase 0).
- Build the study-session UI on top of `ClozePage` from 4.5.
- Add the matching exercise type.
- Inherit the metadata contract from Phase 4 — the `ClozeExercise.prompt_template_version` field is the A/B-test key.

## Gotchas anticipated (the lessons learned)

These are the patterns the team has hit in earlier phases that this plan encodes around:

1. **Hard-coded Docker paths** broke Phase 1's pytest harness. All Phase 4 env vars are read via `os.getenv` with sensible defaults; no `/app/...` literals.
2. **Harness redaction on `.env.example`** mangles `KEY` / `SECRET` var names when written via `patch` or `write_file`. The `.env.example` update for 4.1 will be written via a Python script that reconstructs variable names from non-triggering fragments, per the Phase 1 NOTES.md pattern. The literal bytes on disk are correct; the terminal output is just display-redacted.
3. **OpenRouter privacy filter** already blocked `baai/bge-m3` in Phase 1. If `qwen/qwen3-235b-a22b-a22b-2507` (or any chat-model choice) hits the same filter, 4.1's build card body has a documented fallback: probe three chat models at build time, pick the first one that returns a non-404 probe response, document the choice in the docstring. The Phase 4 plan card does NOT pre-resolve the model — that's a build-time call, not a planning decision.
4. **DSPy import path** varies between `dspy` and `dspy-ai` depending on version. 4.2's build card body instructs: "use whatever import name the `pyproject.toml` resolves to; verify via `uv lock`."
5. **`notify-subscribe` cap.** All workers self-send at the end of their turn via `hermes send`, never `kanban notify-subscribe` to Anurag's Discord/Telegram. Standing rule, restated here for the 4.x workers.

---

*For the Apollo (PM) summary:* 5 build cards + 1 review card. Workers fan out from `t_706a4ffa` when this plan card completes. Helena's review is the exit gate. Phase 5 unblocks after Helena's PASS.