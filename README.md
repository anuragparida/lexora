# Lexora

Full-stack German vocabulary application: browse, search, filter, and
generate beautiful dark-themed Anki decks from a curated word corpus.

- **Backend:** FastAPI + SQLAlchemy + SQLite (`backend/`)
- **Frontend:** React 19 + Vite + TypeScript + Tailwind CSS (`frontend/`)
- **Infrastructure:** Docker Compose

The vocabulary corpus is shipped as a pre-built SQLite database in
`backend/data/vocabeo_words.db` so the app runs out of the box. The
data ingestion pipeline that produces this database is maintained
separately and is not part of this repository.

## Quick start

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Development

### Backend

```bash
cd backend
uv sync
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
│   │   ├── main.py          FastAPI routes
│   │   ├── models.py        SQLAlchemy ORM (Word, Example, VerbConjugation)
│   │   ├── schemas.py       Pydantic response models
│   │   ├── crud.py          Query helpers
│   │   ├── database.py      Engine + session + ad-hoc migrations
│   │   └── anki_builder.py  genanki deck builder with dark CSS
│   ├── data/
│   │   └── vocabeo_words.db Pre-built SQLite corpus
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── main.py              uvicorn entrypoint
├── frontend/
│   └── src/                 React app (App.tsx)
├── docker-compose.yml
├── NOTES.md                 Agent / future-self onboarding notes
└── README.md
```

## License

Application code in this repository is provided as-is. The vocabulary
data contained in the shipped SQLite database is the output of an
internal curation pipeline; it is included for app convenience and
should not be redistributed as a standalone dataset.
