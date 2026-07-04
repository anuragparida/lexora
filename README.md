# Lexora

Full-stack German vocabulary application: browse, search, filter, and
generate beautiful dark-themed Anki decks from a curated word corpus.

- **Backend:** FastAPI + SQLAlchemy + Postgres (pgvector) + Alembic
- **Frontend:** React 19 + Vite + TypeScript + Tailwind CSS
- **Observability:** Langfuse (dedicated `lexora` project on the shared
  instance at `localhost:13000`)
- **Infrastructure:** Docker Compose

The vocabulary corpus is shipped as a pre-built SQLite database in
`backend/data/vocabeo_words.db`. On first boot, the backend
container applies Alembic migrations to the Postgres DB and seeds
the corpus from the SQLite file if the `words` table is empty. The
SQLite file remains in the repo as a dev fallback.

The data ingestion pipeline that produces this database is maintained
separately and is not part of this repository.

## Architecture (Phase 1)

```
┌────────────────────────────────────────────────────────────┐
│ Docker Compose                                             │
│                                                            │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐          │
│  │  frontend  │  │  backend   │  │   postgres   │          │
│  │  React/Vite│→ │ FastAPI    │→ │ pgvector/pg16│          │
│  │  :18701    │  │  :8000     │  │  :25432→5432 │          │
│  └────────────┘  └─────┬──────┘  └──────────────┘          │
│                        │                                   │
│                        ├─ OpenRouter (qwen3-embed, bge-m3) │
│                        └─ host.docker.internal:13000 ──┐   │
└─────────────────────────────────────────────────────────┼───┘
                                                          │
                          ┌──────────────────────────────▼────────┐
                          │ Langfuse (shared, multi-project)       │
                          │  localhost:13000 — clausecraft,        │
                          │  lexora, ...                           │
                          └────────────────────────────────────────┘
```

## Quick start

```bash
docker compose up --build
```

- Frontend: http://localhost:18701
- Backend API: http://localhost:18700
- API docs: http://localhost:18700/docs
- Langfuse (shared): http://localhost:13000

On first boot, the backend container:
1. Waits for `postgres` to become healthy.
2. Runs `alembic upgrade head` (idempotent baseline).
3. Seeds the corpus from `backend/data/vocabeo_words.db` if
   `words` is empty.
4. Starts uvicorn.

## Observability (Langfuse)

The backend is wired to talk to Langfuse. **Phase 1 wires the first
real trace** — every `/retrieve` call emits one span with query
text, latency, result count, and embed/query sub-timings.

1. Open `http://localhost:13000` in a browser.
2. Project dropdown → "New project" → name it `lexora`.
3. Project settings → "API Keys" → create a fresh key pair.
4. Store the keys in `~/.lexora/.env` (NOT in the repo):
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=http://host.docker.internal:13000
   ```

Note: the `LANGFUSE_SECRET_KEY` value is redacted on display but
should be the actual `sk-lf-...` string in your local `.env` file.
5. Re-run `docker compose up`. The keys are read by the backend
   container; missing keys disable tracing gracefully.

Phase 4 wires the exercise generator (the real Langfuse consumer).

Phase 5 (FSRS spaced repetition) wires the grading loop. The
scheduler is built on top of [py-fsrs](https://github.com/open-spaced-repetition/py-fsrs)
**pinned to `fsrs==4.1.2`** (the last release before the v5.x
breaking change to 21-parameter weights and renamed serializer
methods). All FSRS hyperparameters — the 19-tuple
`DEFAULT_PARAMETERS`, `DEFAULT_DESIRED_RETENTION`, learning/relearning
steps, and `DEFAULT_MAXIMUM_INTERVAL` — live as module constants in
`backend/app/fsrs.py`. They are never config, never env
(`git grep -n "getenv.*FSRS\|getenv.*RETENTION"` returns nothing).
The version pin is asserted on import so a wrong-version install
fails fast with `RuntimeError("py-fsrs version drift: ...")`. See
[`backend/app/fsrs.py`](backend/app/fsrs.py) and
[`docs/PHASE-5.md`](docs/PHASE-5.md) for the locked contract.

## Embeddings & retrieval (Phase 1)

Every Word and Example row has a 1024-dim embedding (computed
offline, stored in `pgvector`). The `/retrieve` endpoint embeds
the query on demand and returns top-K nearest neighbours by
cosine distance. Score = 1 - distance, so higher = more similar.

**Endpoint:**
```
GET /retrieve?query=<text>&k=<1-100>&source=<words|examples|both>
```

Default `k=10`, `source=both`. Empty `query` → 422. Invalid
`source` → 422. Non-Postgres backend → 503.

**Example:**
```bash
curl 'http://localhost:18700/retrieve?query=Gl%C3%BCck&k=5'
```

**Embedding model:** `qwen/qwen3-embedding-8b` (OpenRouter). The
spec called for `baai/bge-m3`; that model isn't reachable under
the account's current OpenRouter privacy settings — see
`NOTES.md` §"Phase 1 outcome" for the deviation rationale.

**Backfill (one-time, runs against the live stack):**
```bash
docker compose exec backend uv run python -m scripts.backfill_embeddings
```

Re-running the script skips rows that already have an embedding,
so it's safe to invoke after a partial run. Expected wall time:
~12-15 minutes for the shipped corpus (12,430 words + 29,218
examples) at `EMBEDDING_BATCH_SIZE=32`.

**Why plumbing now, consumer later:** the corpus fits in any
modern context window today, but the cost grows linearly with
context length. Phase 4 (exercise generator) and Phase 6 (RAG-on)
will both consume `/retrieve`; shipping the pipe now keeps the
later phases small.

## Eval set + offline runner (Phase 4.4)

`eval/cloze_judgments.jsonl` is the held-out eval set for the
cloze-exercise generator. 80 rows across all 7 clozable word
types (Noun / Verb / Adjective / Adverb / Pronoun / Preposition /
Conjunction), with `context_sentence`, distractors, difficulty
label, and provenance. The runner (`backend/scripts/eval_cloze.py`)
computes `accept_rate`, `schema_validity_rate`, and
`rationale_quality_proxy`. `--dry-run` exits 0 and prints OK
without contacting OpenRouter — the CI smoke path.

See [`docs/EVAL.md`](docs/EVAL.md) for the full reference:
re-generation, runner usage, metric definitions, and the locked
deviation from the original LLM-generated eval-set spec
(template-based fallback, since all 28 OpenRouter chat models
are blocked by the account's data-policy guardrail).

### Ragas regression runner (Phase 6.7)

[`backend/scripts/eval_ragas.py`](backend/scripts/eval_ragas.py)
scores retrieval + generation against the held-out sets and is
layered on top of the Phase 4.4 cloze runner as a **regression
detector on retrieval-augmented prompts** — not the primary
optimization signal (the Phase 4.4 hand-labeled cloze judgments
remain primary for the cloze generator; matching + comprehension
held-out sets mirror the cloze shape with 40 rows each).

Four metrics, all wired in [`backend/app/eval/ragas.py`](backend/app/eval/ragas.py):

- `context_precision` — fraction of retrieved chunks that are
  actually relevant to the prompt's question.
- `context_recall` — fraction of the ground-truth supporting
  context that was retrieved.
- `faithfulness` — whether the generated answer is grounded in
  the retrieved chunks (no hallucinated facts).
- `answer_relevance` — whether the generated answer actually
  addresses the question.

Per-metric floors are hard-coded module constants
(`RAGAS_MIN_CONTEXT_PRECISION`, `RAGAS_MIN_FAITHFULNESS`, …) —
the `git grep -n "getenv.*RAGAS"` invariant holds by construction
(Hard rule #9). The runner exits non-zero when the overall score
falls below `RAGAS_DRY_RUN_MIN_OVERALL`.

```bash
# Offline — no OpenRouter, no Ragas lib, prints OK and exits 0.
# CI smoke path. Mirrors eval_cloze.py --dry-run.
cd backend
uv run python -m scripts.eval_ragas --dry-run

# Live — fires the real Ragas library when RAGAS_API_KEY is set
# AND `ragas` imports cleanly. The langchain ecosystem is in
# flux on Python 3.12; if the import fails, the runner falls
# back to the dry-run proxy with a clear warning (canonical
# fix documented in docs/EVAL.md as a Phase 6 follow-up).
RAGAS_API_KEY \
  uv run python -m scripts.eval_ragas --live
```

Output lands in `eval/ragas_results.jsonl` (stable symlink to
the most recent `ragas_results_<timestamp>.jsonl`). Each row
carries `exercise_id`, the four-metric keyset, the `prompt_template_version`
A/B key, and the `enable_rag` flag the sample was generated with.
The `grade_logs.exercise_id` ↔ `ragas_results.exercise_id` join
is the A/B lift number: compare non-RAG (`enable_rag=False`)
rows against RAG-on (`enable_rag=True`) rows for the same
`prompt_template_version` to get the lift signal.

## Cloze generation (Phase 4.2 + Phase 6.1)

`POST /exercises/cloze` returns one fill-in-the-blank exercise
tailored to the learner's weakness profile. The route is
auth-gated (cookie session, same as every other authed endpoint).

**What it does.** Picks a target word deterministically from the
learner's highest-scoring weakness axis (e.g. `verbs: 3`), builds
a constrained prompt that includes the word's first example
sentence from the corpus (no retrieval call by default —
Hard rule #3), and sends it to OpenRouter through `instructor`
so the response is validated against a Pydantic `ClozeExercise`
model with bounded retries (≤ 3). The response carries the
answer word id, three distractor word ids of the same word type,
the difficulty label, and a one-sentence rationale. Frontend
(`/exercises/cloze`, Phase 4.5) renders the sentence with a
blank + four randomised choices.

**RAG-on (opt-in).** The cloze request body carries an
`enable_rag: bool = False` flag. When `True`, the generator
augments the prompt with the top-K chunks returned by `/retrieve`
for the target word (the same Phase 1 endpoint shape — consumed
as-is, no new retrieval routes). When `False` (the default), the
prompt template is byte-for-byte identical to the no-RAG fixture
so the offline eval stays reproducible for A/B comparison.
`RAG_TOP_K`, `RAG_MAX_CHARS_PER_CHUNK`, and `RAG_MAX_CHARS` are
hard-coded module constants in `backend/app/cloze.py` — not
env-derived (Hard rule #9). The RAG-on path emits its retrieval
metadata (`retrieved_chunks`, `retrieved_chunk_k`) onto the
Langfuse span so the A/B lift is joinable with `eval/ragas_results.jsonl`.

**Observability (Phase 4.3).** Every generation calls
`_trace_cloze(result, metadata, latency_ms)` which emits a
`cloze.generate` span to the dedicated `lexora` Langfuse project.
The span metadata follows the contract in
[`docs/PHASE-4.md`](docs/PHASE-4.md) §"The metadata contract"
(`user_id`, `weakness_axes`, `word_id`, `difficulty`, `model_id`,
`prompt_template_version`, `schema_retry_count`, `latency_ms`,
`prompt_tokens`, `completion_tokens`). If `LANGFUSE_*_KEY` env vars
are missing, the trace path is a no-op — graceful degradation,
the activity still succeeds. Phase 5 reads the same keyset for the
FSRS grading loop; lock the contract here so the Phase 5
migration isn't needed.

**Curl:**

```bash
# Default — enable_rag=False, byte-for-byte Phase 4.2 prompt.
curl -s -X POST http://localhost:18700/exercises/cloze \
  -H "Cookie: lexora_session=$LEXORA_SESSION" \
  -b cookies.txt -c cookies.txt | jq .

# RAG-on — augment the prompt with retrieval chunks for the target word.
curl -s -X POST http://localhost:18700/exercises/cloze \
  -H "Content-Type: application/json" \
  -H "Cookie: lexora_session=$LEXORA_SESSION" \
  -b cookies.txt -c cookies.txt \
  -d '{"enable_rag": true}' | jq .
```

Returns `200` + `ClozeExercise` JSON, or `502` if the schema
retries are exhausted (the body carries
`schema_retry_count` + `last_validation_error` for triage).

**Run the offline DSPy optimizer** (MIPROv2 / BootstrapFewShot
against the held-out eval set):

```bash
cd backend
# Offline — uses DSPy's DummyLM; safe to run on a CI box with
# no OpenRouter credentials.
uv run python -m scripts.optimize_cloze

# Live — fires the real OpenRouter adapter; only effective when
# OPENROUTER_API_KEY is set, AND you pass --live explicitly.
OPENROUTER_API_KEY="$YOUR_OPENROUTER_KEY" uv run python -m scripts.optimize_cloze --live
```

The CLI writes the optimised prompt instructions to
`backend/app/cloze_optimized.json` (gitignored). The path is
the production wiring point for Phase 5+.

## Matching generation (Phase 6.2 + Phase 6.3)

`POST /exercises/match` returns one concentration-style
exercise: N word pairs (default 4, range `[2, 8]`) the user
connects by dragging each German word to its translation or
synonym. The route is auth-gated (cookie session, same as every
other authed endpoint) and is a thin wrapper over
`backend/app/match.py`'s `generate_match` — all generation logic
lives in the DSPy module, the route only translates transport
errors to HTTP and locks `prompt_template_version` on the way out.

**What it does.** Picks a target word deterministically from the
learner's highest-scoring weakness axis (same `select_target_word`
helper cloze uses), then asks the LLM to produce `count` pairs
where each pair is `{left_word_id, right_word_id, right_kind}`
with `right_kind: Literal["translation", "synonym"]`. The LLM
sees the target word + the word's first example sentence (no
retrieval by default — Hard rule #3), and the response is
validated against a Pydantic `MatchingExercise` model with
bounded retries (≤ 3). The response shape is `MatchingExerciseOut`
(subclass of `BaseExerciseFields`) with `exercise_type="matching"`,
a server-minted `exercise_id` (the same id re-appears on the
`grade_logs` row for Ragas join determinism), the target word id,
the `prompt_template_version` (`match-v1`), the `enable_rag` echo,
and the `pairs` list.

**RAG-on (opt-in).** The request body carries
`enable_rag: bool = False` and `count: int = 4`. When
`enable_rag=True`, the generator augments the prompt with
top-K chunks returned by `/retrieve` for the target word.
When `False` (the default), the prompt template is
byte-for-byte identical to the no-RAG fixture so the offline
eval stays reproducible.

**Curl:**

```bash
# Default — enable_rag=False, count=4.
curl -s -X POST http://localhost:18700/exercises/match \
  -H "Content-Type: application/json" \
  -H "Cookie: lexora_session=$LEXORA_SESSION" \
  -b cookies.txt -c cookies.txt \
  -d '{}' | jq .

# RAG-on, 6 pairs.
curl -s -X POST http://localhost:18700/exercises/match \
  -H "Content-Type: application/json" \
  -H "Cookie: lexora_session=$LEXORA_SESSION" \
  -b cookies.txt -c cookies.txt \
  -d '{"count": 6, "enable_rag": true}' | jq .
```

Returns `200` + `MatchingExerciseOut` JSON, or `422` if `count`
is outside `[2, 8]` (Pydantic-level rejection), or `502` if the
schema retries are exhausted (the body carries
`schema_retry_count` + `last_validation_error` for triage).

## Comprehension generation (Phase 6.4 + Phase 6.5)

`POST /exercises/comprehension` returns one reading-comprehension
exercise: a 3–5 sentence German passage on the target word's
topic plus a multiple-choice question with four options A–D.
The route is auth-gated and mirrors `/exercises/match` and
`/exercises/cloze` — thin wrapper over
`backend/app/comprehension.py`'s `generate_comprehension`, all
generation logic in the DSPy module.

**What it does.** Picks a target word deterministically (same
`select_target_word` helper cloze + matching use), then asks the
LLM to produce a `passage` (3–5 sentences, 20–600 chars), a
`question` (5–300 chars), four `choices` keyed A/B/C/D (each
1–200 chars), a `correct_choice: Literal["A","B","C","D"]`, and
a one-sentence `rationale` (1–400 chars). The Pydantic
`ComprehensionExercise` model enforces all bounds — out-of-range
fields are rejected at the schema layer, not as a runtime
mismatch downstream. The response shape is
`ComprehensionExerciseOut` (subclass of `BaseExerciseFields`)
with `exercise_type="comprehension"`, the same server-minted
`exercise_id` shape matching ships, and the
`prompt_template_version` (`comprehension-v1`).

**RAG-on (opt-in).** The request body carries only
`enable_rag: bool = False` — no `count` knob (comprehension
generates one passage + one question per call, mirroring cloze,
not matching). When `enable_rag=True`, the comprehension
generator calls `/retrieve` for the target word and embeds the
chunks in the prompt (and uses a retrieved chunk as the passage
seed when one is available — falls back to LLM-generated content
otherwise). When `False` (the default), the prompt template is
byte-for-byte identical to the no-RAG fixture.

**Curl:**

```bash
# Default — enable_rag=False.
curl -s -X POST http://localhost:18700/exercises/comprehension \
  -H "Content-Type: application/json" \
  -H "Cookie: lexora_session=$LEXORA_SESSION" \
  -b cookies.txt -c cookies.txt \
  -d '{}' | jq .

# RAG-on — passage seeded from a retrieved chunk for the target word.
curl -s -X POST http://localhost:18700/exercises/comprehension \
  -H "Content-Type: application/json" \
  -H "Cookie: lexora_session=$LEXORA_SESSION" \
  -b cookies.txt -c cookies.txt \
  -d '{"enable_rag": true}' | jq .
```

Returns `200` + `ComprehensionExerciseOut` JSON, or `502` if the
schema retries are exhausted (the body carries
`schema_retry_count` + `last_validation_error` for triage).

## Development

### Backend (against the bundled Postgres)

```bash
cd backend
uv sync
DATABASE_URL=postgresql+psycopg://lexora:lexora@localhost:25432/lexora \
  uv run alembic upgrade head
uv run python main.py
```

### Backend (SQLite fallback)

```bash
cd backend
uv sync
DATABASE_URL=sqlite:///./data/vocabeo_words.db uv run alembic upgrade head
uv run python main.py
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/words` | List words, pagination + filter by word type / frequency |
| GET | `/words/search?q=...` | Substring search over German words |
| GET | `/words/filters/options` | Distinct word types + frequency levels |
| GET | `/words/{id}` | Single word with examples + verb conjugation |
| POST | `/decks/generate` | Build an `.apkg` deck from filtered words |
| GET | `/decks/list` | List previously generated decks |
| GET | `/retrieve?query=&k=&source=` | Top-K nearest neighbours by cosine distance (Phase 1, Postgres + pgvector only) |
| POST | `/exercises/cloze` | Generate one cloze exercise for the logged-in learner (Phase 4.2 + Phase 6.1 + Phase 7.3 collocation + Phase 7.4 partner_lang; auth-gated; accepts optional `{"enable_rag": bool, "collocation": bool, "partner_lang": "de"|"en"}` body) |
| POST | `/exercises/match` | Generate one matching exercise (`count` pairs, default 4, range `[2, 8]`; accepts optional `{"count": int, "enable_rag": bool, "partner_lang": "de"|"en"}` body; Phase 6.2 + Phase 6.3 + Phase 7.4 bilingual; auth-gated) |
| POST | `/exercises/comprehension` | Generate one comprehension exercise (3–5 sentence passage + multiple-choice question; accepts optional `{"enable_rag": bool}` body; Phase 6.4 + Phase 6.5; auth-gated) |
| POST | `/exercises/grade` | Persist a grade for a generated exercise; `exercise_type` widened to `Literal["cloze", "matching", "comprehension"]` in Phase 6.6 (Phase 5.3 route + Phase 6.6 widening; auth-gated) |

## Limitations (Phase 7)

Phase 7 (folded): collocations + prepositional-objects schema
+ retrieval-quality A/B; bilingual exercise opt-in
(`partner_lang`); bge-m3 one-env-var swap via `EMBEDDING_MODEL`.
Builds landed on `main` via commits `ebcb534` (7.1) + `a2447df`
(7.2) + `a48dd0c` (7.3) + `8b4609d` (7.4) + `4895e94` (7.5); see
`docs/PHASE-7.md` for the spec. The honest constraints at this
snapshot:

- **Retrieval-quality A/B verdict (Phase 7.5):** `no_significant_lift`
  on `context_precision`, `context_recall`, `faithfulness`,
  `answer_relevance` for the v80 held-out cloze set. Run
  `make eval-retrieval-compare` to refresh both per-row CSVs
  (`eval/retrieval_compare/current_per_row.csv` +
  `bge_m3_per_row.csv`) and the markdown comparison table
  (`eval/retrieval_compare/retrieval_compare_report.md`). The
  `RETRIEVAL_MIN_QUALITY_FLOOR = 0.05` is a hard-coded module
  constant in `backend/app/eval/retrieval_compare.py` (Hard rule
  #7 — no env-derived thresholds). The A/B currently runs in
  dry-run (deterministic-template) mode because OpenRouter's
  privacy filter blocks `bge-m3` as a chat-model call;
  `bge-m3` loads from the local `sentence-transformers` cache
  on a warm HuggingFace pull (~2.3GB on first download).
- **Phase 7 schema-curated, not LLM-learned:** collocations +
  prepositional-objects are hand-curated from DWDS + Wiktionary
  subsets (≥200 rows each). Generator reads, never writes. No
  DSPy optimizer path touches these tables; no `INSERT`, no
  `UPDATE`, no `ON CONFLICT` writes from runtime — the seed
  scripts in 7.1 are the only path that touches them outside
  Alembic migrations.
- **Phase 7 wire-level only:** the matching + cloze
  `partner_lang="en"` endpoints are exercisable via
  curl/Postman. Frontend rendering is Phase 9 (study-session
  mixing). The cloze `collocation=true` flag is the same story
  — it's a cloze variant that renders inside the existing
  cloze surface; bilingual exercises render inside the existing
  match surface. Wire-level ≠ UI.
- **A/B lift is a regression detector, not the primary signal.**
  The Phase 4.4 hand-labeled cloze judgments (`eval/cloze_judgments.jsonl`,
  80 rows across all 7 clozable word types) remain the primary
  optimization signal for the cloze generator. Ragas lift numbers
  are reported against the v80 cloze set plus the 40-row matching
  and 40-row comprehension held-out sets; the Phase 7.5
  retrieval-quality A/B reports against the same held-out cloze
  set under the Phase 6.7 Ragas metrics. Both catch regressions
  without supplanting the hand-labeled grader.
- **Retrieval-quality disclaimer stands.** Phase 1's retrieval
  relies on whatever embedding model is currently pinned
  (OpenRouter `qwen/qwen3-embedding-8b` as of Phase 1, with the
  original `bge-m3` swap documented as a one-env-var fallback in
  `NOTES.md` and now live in `backend/app/embeddings.py` as the
  `EMBEDDING_MODEL` env var per Phase 1 Hard rule #6). Phase 7.5
  measures the lift on retrieval + faithfulness; if retrieval
  quality is poor, the lift reflects it on `faithfulness` and
  `context_recall`.

## Generating Anki decks

The frontend sidebar has a "Generate Deck" button that POSTs to
`/decks/generate`. Filters (word type, frequency, card direction) are
applied server-side. The response returns the deck filename; the file
itself lands in `backend/generated_decks/`.

Each note in the deck produces two cards (German→English and
English→German) using genanki's native bidirectional model, so Anki
tracks review state per direction correctly.

## Project structure

```
lexora/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI routes + lifespan bootstrap
│   │   ├── models.py        SQLAlchemy ORM (Word, Example, VerbConjugation, FsrsCard)
│   │   ├── schemas.py       Pydantic response models
│   │   ├── crud.py          Query helpers
│   │   ├── database.py      Engine + session
│   │   ├── bootstrap.py     One-time corpus seeder (SQLite → Postgres)
│   │   ├── observability.py Langfuse client wrapper
│   │   ├── anki_builder.py  genanki deck builder with dark CSS
│   │   ├── llm.py           OpenRouter chat-completions client (Phase 4.1)
│   │   ├── cloze.py         Cloze exercise generator + DSPy module (Phase 4.2 + Phase 6.1 RAG-on)
│   │   ├── match.py         Matching exercise generator + DSPy module (Phase 6.2)
│   │   ├── comprehension.py Comprehension exercise generator + DSPy module (Phase 6.4)
│   │   ├── eval/            Ragas runner + per-metric floor constants (Phase 6.7)
│   ├── alembic/             Migrations (baseline = words/examples/verb/fsrs)
│   ├── data/
│   │   └── vocabeo_words.db Pre-built SQLite corpus (dev fallback)
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── alembic.ini
│   └── main.py              uvicorn entrypoint
├── postgres/
│   └── init/                Init scripts for first Postgres boot
├── frontend/
│   └── src/                 React app (App.tsx)
├── docker-compose.yml       postgres + backend + frontend
├── .env.example             Documented env vars
├── NOTES.md                 Agent / future-self onboarding notes
└── README.md
```

## License

Application code in this repository is provided as-is. The vocabulary
data contained in the shipped SQLite database is the output of an
internal curation pipeline; it is included for app convenience and
should not be redistributed as a standalone dataset.