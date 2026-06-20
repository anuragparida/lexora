# Lexora backend

FastAPI service for the Lexora vocabulary app.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

The API listens on `:8000`. Interactive docs at `/docs`.

## Data

The pre-built SQLite database lives at `data/vocabeo_words.db`. The
schema is created automatically on first run; no separate migration
step is needed.

Tables:

- `words` — vocabulary entries with metadata (word type, frequency,
  CEFR level, translations, additional info, completion flag, optional
  FK to verb conjugation)
- `examples` — example sentences per word (German + English)
- `verb_conjugations` — distinct verb entries (infinitive + principal
  parts), deduped
- `fsrs_cards` — schema in place for future in-app spaced-repetition;
  empty for now

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/words` | List words, paginated, filterable by `word_types` / `frequencies` |
| GET | `/words/search?q=...` | Substring search over German words |
| GET | `/words/filters/options` | Distinct values for filter UI |
| GET | `/words/{id}` | Single word + examples + verb conjugation |
| POST | `/decks/generate` | Build `.apkg` from filtered words (`direction` = `both` / `de-en` / `en-de`) |
| GET | `/decks/list` | List generated `.apkg` files |

## Anki deck builder

`app/anki_builder.py` produces `.apkg` decks using the `genanki`
library. Each word becomes one note with two card templates
(German→English and English→German), styled with a dark theme that
matches the web UI.

Generated files land in `/app/generated_decks/` inside the container
(mounted from `./generated_decks/` on the host).
