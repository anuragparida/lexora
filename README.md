# German Vocabulary App

Full-stack German vocabulary learning application.

## Architecture

- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Frontend**: React + Vite + TypeScript + Tailwind CSS
- **Infrastructure**: Docker Compose

## Quick Start

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

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

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /words` - List words with pagination
- `GET /words/search?q=...` - Search words
- `GET /words/{id}` - Get single word
