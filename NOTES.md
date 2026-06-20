# Lexora — agent notes

> Read these before any change. Repo lives at `/home/ody/workspace/lexora/`,
> pushed from `https://github.com/anuragparida/lexora.git`.
> Single `init` commit on `main`, no other branches, no remote tracking work.

---

## What this thing actually is

German vocabulary app with three logically distinct layers in one repo:

```
lexora/
├── backend/          FastAPI + SQLAlchemy + SQLite. The API + .apkg generator.
├── frontend/         React 19 + Vite + Tailwind 4. Dark theme, one big App.tsx.
├── data/             Off-app tooling: vocabeo.com scraper, JSON dumps, two notebooks.
├── scripts/          One-off DB migration scripts (reverse cards, fix reverse cards).
└── docker-compose.yml Two services: backend (8000) and frontend (5173).
```

**The product story in one line:** scrape ~6,200 German words + examples + verb conjugations from vocabeo.com → store in SQLite → expose browse/search API + filter UI → generate Anki `.apkg` decks with rich, dark-themed cards.

---

## Backend (`backend/app/`)

Six small files, ~750 lines total. Clean and readable.

| File | Lines | What it does |
|---|---|---|
| `main.py` | 127 | FastAPI app. 7 endpoints: `/`, `/health`, `/words`, `/words/search`, `/words/filters/options`, `/words/{id}`, `POST /decks/generate`, `GET /decks/list` |
| `models.py` | 53 | `Word`, `Example`, `VerbConjugation` SQLAlchemy models. CASCADE delete on examples. |
| `schemas.py` | 55 | Pydantic v2 (`from_attributes=True`) response models. |
| `database.py` | 38 | Engine + sessionmaker, plus a hand-rolled `run_migrations()` that just does `ALTER TABLE words ADD COLUMN is_complete` via PRAGMA. |
| `crud.py` | 67 | `get_words`, `get_word`, `search_words` (LIKE on `word` column), `get_word_types`, `get_frequencies`. |
| `anki_builder.py` | 489 | The interesting one. Builds `.apkg` files with `genanki`. |

**Notable details:**

- **`anki_builder.py` is the heart.** ~190 lines of hand-rolled CSS for dark Anki cards (matches the app's dark theme). Two card templates per note: German→English and English→German. Uses Anki's native bidirectional generation (`"1 note, 2 cards"`), which is the right pattern.
- The "pair" logic in `create_anki_deck` is hacky: it groups by `tuple(sorted([word, translations]))` and assumes pair length == 2 means one DE card + one EN card. It uses `german_chars = set("äöüßÄÖÜ")` to discriminate direction. Works for >95% of cases but will mis-pair rare words that happen to contain no umlauts (e.g. "Tor", "See", "Wahl").
- `deck_id = random.randrange(1 << 30, 1 << 31)` — random deck IDs per generation. Stable model ID (1607392319) so Anki updates overwrite previous decks cleanly. Good.
- `run_migrations()` is a stopgap — when a real schema system (Alembic) shows up, this dies.

---

## Frontend (`frontend/src/`)

**One file: `App.tsx` at 555 lines.** StrictMode shell, inline types, no router, no state library.

Component structure (all in App.tsx):
- `WordList` — grid of word cards with badges + conjugation box + 2 examples
- `WordTypeCheckboxes` / `FrequencyCheckboxes` — filter pills (hidden inputs, clickable labels)
- `App` — owns all state: words, page, filters, search, decks, sidebar toggle

Features:
- Sidebar with search + word-type filter + frequency filter + card-direction dropdown + deck builder + deck history + clear
- Main pane: paginated word list (20/page), Previous/Next
- POST to `/decks/generate` creates an .apkg; alert with filename (no download UX)
- GET `/decks/list` shows previously generated decks (filename + size only — no download button)

**What's missing on the frontend:** no download button for generated decks, no auth, no review/session UI, no flashcard study mode. The Anki cards exist only as `.apkg` files downloaded out-of-band.

---

## Data pipeline (`data/`)

Two phases, two projects, both broken in their own way.

### Phase 1: scrape vocabeo.com

`data/vocabeo/scraper.py` (331 lines): Playwright-driven scraper of `vocabeo.com/browse`.

- 6,260-word target list. Uses **keyboard arrow-down navigation** instead of DOM-walking the virtual list — clever workaround for vocabeo's virtualized scroll container.
- Extracts: word, word_type, frequency, level, translations (list), conjugation, examples (list of {german, english}), additional_info (dict with plural/gender).
- Headless=False in `scrape()` despite being scripted — this is meant to be run interactively with a visible browser so you can Ctrl+C.
- Saves to `vocabeo_words.json` (current) and `vocabeo_progress_NNN.json` (checkpoints every 50 words).
- Rate: variable. The arrow-down + 50ms delay caps it around ~15-20 words/sec.

**Two JSONs committed:**
- `vocabeo_words.json` — **6,215 entries**, 4.4 MB. This is the canonical dataset.
- `vocabeo_backup.json` — 4,704 entries, 3.4 MB. Smaller; unclear why both exist.

**Verb conjugations:** `verb_conjugations.json` — 1,456 entries with `{infinitive, present_3rd_person, simple_past, participle}`. Loaded as a separate table, joined via `Word.conjugation_id`.

### Phase 2: prep for Anki

`data/anki/extract-and-filter.ipynb` (164 cells, broken kernel "Python 3.14.3 is no longer available"):

- Unzips `Deutsch_4000_German_Words_by_Frequency.apkg` (48 MB committed to repo ⚠️) into `apkg_extracted/` (62 MB committed ⚠️).
- Reads `collection.anki2` to extract notes, parse them via the model JSON, build a flashcards DataFrame.

**Notebook is dead.** The kernel it's pinned to (3.14.3) doesn't exist on this machine. Whoever wrote this never went back and re-ran / cleaned it up. The output `apkg_extracted/` is checked in even though it's regeneratable from the `.apkg`.

---

## Scripts (`scripts/`)

Two DB migration scripts that were run once and left behind:

- `reverse_cards.py` — duplicates every word as its reverse (DE→EN becomes EN→DE) by swapping `word` ↔ `translations` fields and swapping example languages. Original card ID 1-N, reversal ID N+1-2N.
- `fix_reversed_cards.py` — undoes `reverse_cards.py` for cards with id > N and re-runs it correctly. The 90-line preamble docstring is a stream-of-consciousness of the author working out the semantics live.

**The current DB state is consistent.** 12,430 words = 6,215 × 2 (originals + reversals). No `is_complete = True` except 13 rows, which means most cards were generated when those were added and never re-touched. Looks like the reverse-cards experiment was tested on the live DB and then never cleaned.

---

## The hidden gem: `fsrs_cards` table

Found in the SQLite schema, **not referenced anywhere in the source code**. Empty (0 rows).

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

| Table | Rows |
|---|---|
| `words` | 12,430 (6,215 originals + 6,215 reversals) |
| `examples` | 29,218 (avg 2.6 examples/word) |
| `verb_conjugations` | 1,452 |
| `fsrs_cards` | 0 (table exists, no rows, no code) |

Word type distribution: 6,354 Nouns, 2,912 Verbs, 1,980 Adjectives, 696 Adverbs, 126 Pronouns, 124 Numbers, 114 Prepositions, 68 Conjunctions, 52 Interjections, 4 Articles.

Frequency distribution (vocabeo's 1-5 scale): heavy skew toward 3-4 (3,433 + 1,762 = 5,195 of 6,215).

CEFR levels: 827 A1, 683 A2, 1,689 B1, 3,016 unknown (level not extracted from vocabeo).

---

## What's wrong / low-effort wins

These are the calls I'd make before any "build the future" conversation:

1. **Repo hygiene.** 110MB+ on disk because:
   - `data/anki/apkg_extracted/` (62 MB, 4,200 files) — extract of an .apkg, regeneratable.
   - `data/anki/Deutsch_4000_German_Words_by_Frequency.apkg` (48 MB).
   - `backend/data/vocabeo_words.db` (5.7 MB committed SQLite DB).
   - `backend/generated_decks/*.apkg` (6 .apkg files, ~550 KB total).
   - `__pycache__/*.pyc` (6 files), `.DS_Store` (3), `database.cpython-314.pyc` (orphan from a 3.14 environment).
   - `data/vocabeo/vocabeo_backup.json` (3.4 MB — same as vocabeo_words.json but smaller, unclear why committed alongside).

   **One root `.gitignore` would fix 90% of this.** Add: `__pycache__/`, `.DS_Store`, `*.pyc`, `backend/data/*.db`, `backend/generated_decks/`, `data/anki/apkg_extracted/`, `data/anki/*.apkg`, `data/vocabeo/vocabeo_backup.json`. Run `git rm -r --cached` on the rest.

2. **`vocabeo_backup.json` is unclear.** Either it's redundant (drop it) or it's a meaningful earlier scrape with different coverage (document why). Right now it looks like both.

3. **The dead notebook** (`data/anki/extract-and-filter.ipynb`) has a broken kernel pin and was clearly abandoned mid-experiment. Either restore it or drop it. The `apkg_extracted/` directory it's supposed to produce is the *bigger* problem — committing an unzipped archive is almost never right.

4. **`App.tsx` is 555 lines.** Not a blocker yet, but a sign that the next feature will hurt. Natural split points: a `hooks/` directory for data fetching (`useWords`, `useDecks`, `useFilterOptions`), a `components/` for `WordCard`, `Sidebar`, `DeckBuilder`.

5. **CORS is `allow_origins=["*"]`** in `main.py`. Fine for local, but mark it explicitly as a "dev-only" setting before this ever faces the internet.

6. **`run_migrations()` is a band-aid.** It only handles the `is_complete` column. The moment another column is added, this stops working silently. Alembic is overdue.

7. **No tests.** Not even a smoke test against `/health`. There are 7 endpoints and ~50 unit-level functions in the backend, none covered.

8. **The pair-matching logic in `anki_builder.py` is fragile.** The `german_chars` heuristic misses "Tor" (gate vs. fool), "See" (lake vs. sea), and other common homographs. If `direction == 'both'`, ~5-10% of pairs will be wrong.

9. **`is_complete = False` for 12,417 of 12,430 words.** Whatever this flag means, it's never been set for almost any card. Either drop the column or wire it up to something real (it looks like it was meant for "user marked this card as fully understood" but never got UI).

---

## What's actually good

- The Anki deck output is genuinely high quality. Dark-themed CSS, three conjugation slots, examples, badges — this is a real product, not a toy.
- `anki_builder.py` correctly uses genanki's native bidirectional model (1 note, 2 cards) instead of duplicating notes. This means Anki handles review state correctly across directions.
- Backend is small enough to grok in 5 minutes. No premature abstraction.
- The vocabeo scraper is a real engineering solution to a real problem (virtualized scroll + keyboard nav). It would be reasonable to keep using this for future scrapes if vocabeo's UI doesn't change.
- Schema is normalized. Verb conjugations are deduped, examples have a proper FK. Reasonable for the size.
- Anki approach sidesteps the "build a full SRS in-app" problem elegantly. The .apkg export is a real distribution channel.

---

## Future direction — the actual take

The honest read: **this app's value proposition right now is "Anki deck generator for vocabeo's word list."** That's not nothing — the CSS is genuinely nice, and the cards look better than 90% of community decks — but it's a one-shot generator, not a product.

Three directions the project could credibly go, ranked by effort/value:

**A. In-app spaced repetition (kill Anki dependency).**
The empty `fsrs_cards` table is a tell. The intent was there. Add `ts-fsrs` (or `py-fsrs` server-side), wire up the table, build a review screen in the frontend, and you've got a standalone vocab app that doesn't need Anki at all. Pros: way better UX (instant feedback, no export/import step), real retention signal. Cons: you now own the review loop forever, and you'll never beat Anki at it.

**B. Better data layer — vocab that's actually learned, not just scraped.**
The 6,215 words are a static scrape. No audio, no IPA, no native-speaker sentences, no images, no collocations, no frequency-by-genre (news vs. spoken vs. formal). The word list is the ceiling on this app's value. Adding a content pipeline (TTS, IPA, wikidata-derived examples, deepl-translated collocations) would 10x the corpus quality. Pros: every feature downstream improves. Cons: expensive data work, easy to fall into content-trap hell.

**C. Personalization + progress tracking.**
Hook up auth (even just local accounts), let users mark words as known/unknown, track which examples they got wrong, generate review sessions from "unknown" + "due" cards. The infrastructure for this (fsrs table, is_complete flag, conjugation_id) is *already in the schema* — it just needs the surface. Pros: feels like a real product to a single user (you, mostly). Cons: lots of frontend work for a single-user app.

**My honest pick if you're energy-constrained:** C. It's where your own schema is already pointing, the work is mostly UI, and the payoff is "I actually use this daily" which is the only way a side project stays alive. Pair it with one or two cleanups from "What's wrong" (especially the gitignore pass and dropping `vocabeo_backup.json`) so the repo doesn't feel cluttered when you come back to it.

A and B are "build a company" tracks. C is "build a tool for myself" track. Different time horizons.