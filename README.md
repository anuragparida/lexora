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
│   │   └── anki_builder.py  genanki deck builder with dark CSS
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