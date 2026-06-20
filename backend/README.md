# German Vocabulary API

FastAPI backend for the German vocabulary learning app.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /words` - List words (pagination with skip/limit)
- `GET /words/search?q=...` - Search words
- `GET /words/{id}` - Get single word
