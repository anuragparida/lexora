# Lexora — agent notes

> Read these before any change. Repo lives at `/home/ody/workspace/lexora/`,
> pushed from `https://github.com/anuragparida/lexora.git`.

---

## Repo layout (after provenance split, June 2026)

```
lexora/                            PUBLIC (this repo)
├── backend/                       FastAPI + SQLAlchemy + SQLite. The API + .apkg generator.
├── frontend/                      React 19 + Vite + Tailwind 4. Dark theme, one big App.tsx.
├── docker-compose.yml
├── NOTES.md                       This file.
└── README.md

lexora-data/                       PRIVATE (separate repo)
├── vocabeo/                       Raw scraping artifacts (JSON, scraper.py, etc.)
├── anki_reference/                Third-party Anki deck used as reference
├── loader/loader.py               vocabeo JSON → SQLite DB
└── scripts/                       reverse_cards.py, fix_reversed_cards.py
```

The split exists because the public app should not advertise a scraping
provenance. The private `lexora-data` repo holds the ingestion pipeline
and raw inputs; the public `lexora` ships with a pre-built
`backend/data/vocabeo_words.db` that came out of that loader. Anyone
cloning the public repo gets a working app with no surface connection
back to the source.

---

## What this thing actually is

German vocab app, three layers:

- **Backend** (FastAPI + SQLAlchemy + SQLite, ~750 lines across 6 small files). 7 endpoints. Clean.
- **Frontend** (React 19 + Vite + Tailwind 4, dark). One monolithic `App.tsx` at 555 lines. Search + filter sidebar + word list + Anki-deck builder.
- **Data layer** (one file: `backend/data/vocabeo_words.db`). Pre-built. Schema is generic; content is a curated corpus.

**The product shape.** Browse/search/filter German vocabulary in a web UI → generate Anki `.apkg` decks with genuinely nice dark-themed cards (1 note, 2 cards per word via genanki's native bidirectional model). The cards are the *real* product here — they look better than 90% of community decks.

---

## Backend (`backend/app/`)

Six small files, ~750 lines total.

| File | Lines | What it does |
|---|---|---|
| `main.py` | 127 | FastAPI app. 7 endpoints: `/`, `/health`, `/words`, `/words/search`, `/words/filters/options`, `/words/{id}`, `POST /decks/generate`, `GET /decks/list` |
| `models.py` | 53 | `Word`, `Example`, `VerbConjugation` SQLAlchemy models. CASCADE delete on examples. |
| `schemas.py` | 55 | Pydantic v2 (`from_attributes=True`) response models. |
| `database.py` | 38 | Engine + sessionmaker, plus a hand-rolled `run_migrations()` that just does `ALTER TABLE words ADD COLUMN is_complete` via PRAGMA. |
| `crud.py` | 67 | `get_words`, `get_word`, `search_words` (LIKE on `word` column), `get_word_types`, `get_frequencies`. |
| `anki_builder.py` | 489 | The interesting one. Builds `.apkg` files with `genanki`. |

**Notable details:**

- **`anki_builder.py` is the heart.** ~190 lines of hand-rolled CSS for dark Anki cards. Two card templates per note: German→English and English→German. Uses Anki's native bidirectional generation (`"1 note, 2 cards"`), which is the right pattern.
- The "pair" logic in `create_anki_deck` is hacky: it groups by `tuple(sorted([word, translations]))` and assumes pair length == 2 means one DE card + one EN card. It uses `german_chars = set("äöüßÄÖÜ")` to discriminate direction. Works for >95% of cases but will mis-pair rare words that happen to contain no umlauts (e.g. "Tor", "See", "Wahl").
- `deck_id = random.randrange(1 << 30, 1 << 31)` — random deck IDs per generation. Stable model ID (1607392319) so Anki updates overwrite previous decks cleanly.
- `run_migrations()` is a stopgap — when a real schema system (Alembic) shows up, this dies.

---

## Frontend (`frontend/src/`)

**One file: `App.tsx` at 555 lines.** StrictMode shell, inline types, no router, no state library.

- `WordList` — grid of word cards with badges + conjugation box + 2 examples
- `WordTypeCheckboxes` / `FrequencyCheckboxes` — filter pills (hidden inputs, clickable labels)
- `App` — owns all state: words, page, filters, search, decks, sidebar toggle

**What's missing on the frontend:** no download button for generated decks, no auth, no review/session UI, no flashcard study mode. The Anki cards exist only as `.apkg` files downloaded out-of-band.

---

## Data pipeline (now lives in `lexora-data`)

Two phases, but both happen in the private repo now:

1. **Scrape** (`vocabeo/scraper.py`): Playwright-driven scrape of `vocabeo.com/browse`. ~6,200 words + examples + verb conjugations.
2. **Load** (`loader/loader.py`): converts the raw JSON into the SQLite DB the public repo consumes.

To regenerate the public DB:

```bash
cd lexora-data
python3 loader/loader.py
cp dist/vocabeo_words.db ../lexora/backend/data/vocabeo_words.db
```

If you want bidirectional cards in the public app (every word has a
matching EN→DE twin), run `lexora-data/scripts/reverse_cards.py` once
against the freshly-loaded DB. The currently-shipped DB does **not**
have reversals — only the 6,215 originals. Easy to flip on later.

---

## The hidden gem: `fsrs_cards` table

Found in the SQLite schema, **not referenced anywhere in the source code**. Empty.

```sql
CREATE TABLE fsrs_cards (
    id, word_id, difficulty, stability, retrievability,
    due_date, last_review, reps, lapses, state,
    elapsed_days, scheduled_days
)
```

This is the **FSRS (Free Spaced Repetition Scheduler)** schema — the modern successor to SM-2/Anki's old algorithm. The columns map exactly to `ts-fsrs` library's `Card` dataclass.

**Translation:** there was a planned or half-built feature to do spaced-repetition review *inside* the app (no Anki required), using FSRS. The schema was created but never wired to a model, route, or frontend. The repo contains zero references to "fsrs", "spaced", "review", or "ts-fsrs". It's a planted flag for "the next direction."

---

## Database state (current commit)

The shipped DB contains **originals only** (no reversals).

| Table | Rows |
|---|---|
| `words` | 6,215 |
| `examples` | ~14,600 (avg 2.4 per word) |
| `verb_conjugations` | 1,452 (4 dupes in raw JSON de-duped on load) |
| `fsrs_cards` | 0 (table exists, no rows, no code) |

Word type distribution: 3,177 Nouns, 1,456 Verbs, 990 Adjectives, 348 Adverbs, 63 Pronouns, 62 Numbers, 57 Prepositions, 34 Conjunctions, 26 Interjections, 2 Articles.

Frequency distribution (vocabeo's 1-5 scale): heavy skew toward 3-4.

---

## What's wrong / low-effort wins

1. **`App.tsx` is 555 lines.** Not a blocker yet, but a sign that the next feature will hurt. Natural split points: a `hooks/` directory for data fetching, a `components/` for `WordCard`, `Sidebar`, `DeckBuilder`.
2. **CORS is `allow_origins=["*"]`** in `main.py`. Fine for local, but mark it explicitly as a "dev-only" setting before this ever faces the internet.
3. **`run_migrations()` is a band-aid.** It only handles the `is_complete` column. Alembic is overdue.
4. **No tests.** Not even a smoke test against `/health`. There are 7 endpoints and ~50 unit-level functions in the backend, none covered.
5. **The pair-matching logic in `anki_builder.py` is fragile.** The `german_chars` heuristic misses "Tor" (gate vs. fool), "See" (lake vs. sea), and other common homographs. If `direction == 'both'`, ~5-10% of pairs will be wrong.
6. **`is_complete = False` for almost all words.** Whatever this flag means, it's never been set for almost any card. Either drop the column or wire it up to something real (it looks like it was meant for "user marked this card as fully understood" but never got UI).
7. **No download UX.** Generated `.apkg` files exist on disk but the frontend just alerts the filename — no link, no way to grab them.

---

## What's actually good

- The Anki deck output is genuinely high quality. Dark-themed CSS, three conjugation slots, examples, badges — this is a real product, not a toy.
- `anki_builder.py` correctly uses genanki's native bidirectional model (1 note, 2 cards) instead of duplicating notes. This means Anki handles review state correctly across directions.
- Backend is small enough to grok in 5 minutes. No premature abstraction.
- Schema is normalized. Verb conjugations are deduped, examples have a proper FK. Reasonable for the size.
- The split between public (app) and private (data) repos keeps the product surface clean while preserving the ingestion pipeline for future corpus updates.
- Anki approach sidesteps the "build a full SRS in-app" problem elegantly. The .apkg export is a real distribution channel.

---

## Future direction — three credible paths

**A. In-app spaced repetition (kill Anki dependency).**
The empty `fsrs_cards` table is a tell. Add `ts-fsrs` (or `py-fsrs` server-side), wire up the table, build a review screen in the frontend. Real product, but you now own the review loop forever.

**B. Better data layer — vocab that's actually learned, not just scraped.**
No audio, no IPA, no native-speaker sentences, no collocations, no frequency-by-genre. The word list is the ceiling on this app's value. Pro: 10x corpus quality. Con: expensive data work.

**C. Personalization + progress tracking.**
Auth, mark words as known/unknown, track which examples they got wrong, generate review sessions from "unknown" + "due" cards. The infrastructure for this (fsrs table, is_complete flag, conjugation_id) is *already in the schema* — it just needs the surface.

**My pick if you're energy-constrained:** C. Pair it with the App.tsx split. The schema is already pointing there.
