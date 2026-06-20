FROM python:3.12-slim

# pgvector/pgvector:pg16 ships postgres client libs but the backend needs
# to talk to it. libpq5 is the runtime dependency for psycopg; curl is
# used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync

COPY . .

ENV DATABASE_URL=postgresql+psycopg://lexora:lexora@postgres:5432/lexora

EXPOSE 8000

# Apply Alembic migrations on every boot so schema always matches code.
# The baseline migration is idempotent (CREATE TABLE IF NOT EXISTS),
# so this is safe on both fresh Postgres and the SQLite fallback.
CMD ["sh", "-c", "uv run alembic upgrade head && uv run python main.py"]