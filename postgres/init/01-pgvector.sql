-- Enable pgvector on the lexora database.
-- Runs once on first container boot via docker-entrypoint-initdb.d.
-- The lexora DB already exists (created by POSTGRES_DB env); this just
-- attaches the vector extension to it.

\connect lexora

CREATE EXTENSION IF NOT EXISTS vector;