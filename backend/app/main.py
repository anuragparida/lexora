import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Literal
from app import crud, models, schemas, anki_builder, bootstrap, retrieval
from app.database import get_db, engine
from app.embeddings import embed_one, EmbeddingError
from app.observability import _ensure_client, get_langfuse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lexora.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure schema (Alembic owns migrations), seed corpus if
    Postgres is empty, and warm the Langfuse client (no tracing yet).
    Phase 4 wires real observability call sites."""
    logger.info("startup: ensuring schema via Base.metadata.create_all")
    models.Base.metadata.create_all(bind=engine)
    logger.info("startup: seeding corpus from SQLite if Postgres is empty")
    bootstrap.seed_corpus()
    logger.info("startup: warming Langfuse client (Phase 4 will use it)")
    _ensure_client()
    yield


app = FastAPI(title="German Vocabulary API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "German Vocabulary API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/words", response_model=schemas.WordListResponse)
def read_words(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    word_types: List[str] = Query(None),
    frequencies: List[str] = Query(None),
    db: Session = Depends(get_db),
):
    return crud.get_words(
        db,
        skip=skip,
        limit=limit,
        word_types=word_types,
        frequencies=frequencies,
    )


@app.get("/words/search", response_model=schemas.WordListResponse)
def search_words(
    q: str = Query(..., min_length=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    word_types: List[str] = Query(None),
    frequencies: List[str] = Query(None),
    db: Session = Depends(get_db),
):
    return crud.search_words(
        db,
        query=q,
        skip=skip,
        limit=limit,
        word_types=word_types,
        frequencies=frequencies,
    )


@app.get("/words/filters/options")
def get_filter_options(db: Session = Depends(get_db)):
    return {
        "word_types": crud.get_word_types(db),
        "frequencies": crud.get_frequencies(db),
    }


@app.get("/words/{word_id}", response_model=schemas.WordResponse)
def read_word(word_id: int, db: Session = Depends(get_db)):
    word = crud.get_word(db, word_id=word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="Word not found")
    return word


@app.get("/retrieve")
def retrieve_endpoint(
    query: str = Query(..., min_length=1, description="Text to embed and search for"),
    k: int = Query(10, ge=1, le=100, description="Max items to return per source"),
    source: Literal["words", "examples", "both"] = Query(
        "both", description="Which table(s) to search"
    ),
    db: Session = Depends(get_db),
):
    """Top-k nearest neighbours by cosine distance.

    The query text is embedded on demand via OpenRouter; the result
    is the top-k rows whose precomputed embeddings are closest to
    that vector. Score is 1 - cosine_distance (higher = more similar).

    The endpoint is plumbing: no consumer wires retrieval into a
    prompt yet. Phase 4's exercise generator and Phase 6's RAG
    prompt both depend on this shape.

    On the SQLite dev fallback the endpoint returns 503 — pgvector
    has no analogue on SQLite, and a "no results" response would
    hide the configuration mismatch.
    """
    if not retrieval._is_postgres_target():
        raise HTTPException(
            status_code=503,
            detail=(
                "/retrieve requires Postgres + pgvector. The active "
                "DATABASE_URL points at a non-Postgres backend."
            ),
        )

    started = time.perf_counter()
    try:
        query_vec = embed_one(query)
    except EmbeddingError as exc:
        logger.error("retrieve: embedding failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"embedding provider error: {exc}")
    embed_ms = int((time.perf_counter() - started) * 1000)

    started_q = time.perf_counter()
    items = retrieval.retrieve(db, query_vec, k=k, source=source)
    query_ms = int((time.perf_counter() - started_q) * 1000)
    total_ms = int((time.perf_counter() - started) * 1000)

    # Langfuse trace — first real consumer of the observability
    # wrapper from Phase 0. Skip silently when keys are missing so
    # local dev (no keys) doesn't break the endpoint.
    _trace_retrieval(
        query=query,
        k=k,
        source=source,
        embed_ms=embed_ms,
        query_ms=query_ms,
        total_ms=total_ms,
        result_count=len(items),
    )

    return {
        "query": query,
        "source": source,
        "k": k,
        "result_count": len(items),
        "latency_ms": total_ms,
        "items": items,
    }


def _trace_retrieval(
    *,
    query: str,
    k: int,
    source: str,
    embed_ms: int,
    query_ms: int,
    total_ms: int,
    result_count: int,
) -> None:
    """Emit one Langfuse trace per retrieval call (best-effort).

    Phase 1 fix (t_2e386ba9 / Helena review §7): use the v2 SDK's
    ``client.span(...)`` + ``span.update(...)`` + ``span.end()``
    sequence. The earlier ``client.start_as_current_observation``
    call is a v3-only API; on the v2.60.10 SDK (the floor pinned by
    pyproject.toml to match the v2.95.11 server) the method does
    not exist and the call was silently swallowed by the
    ``except Exception`` below, so every retrieval was untraced.
    The non-context-manager shape is the canonical v2 pattern:
    span() returns a handle, update() merges metadata, end()
    closes the observation, flush() pushes to the ingestion API.
    """
    client = get_langfuse()
    if client is None:
        # Keys missing — already warned at startup. Don't spam per call.
        return
    span = None
    try:
        span = client.span(name="lexora.retrieval")
        span.update(
            metadata={
                "query_text": query,
                "query_len_chars": len(query),
                "k": k,
                "source": source,
                "result_count": result_count,
                # v2 ``span.update`` does not accept a separate
                # ``metrics`` kwarg — latency is encoded as metadata
                # keys with ``_ms`` suffixes so it's filterable in
                # the UI. Phase 4 will wire the v3 generation API
                # for true usage/cost metrics.
                "embed_latency_ms": embed_ms,
                "query_latency_ms": query_ms,
                "total_latency_ms": total_ms,
            },
        )
        span.end()
        # Langfuse buffers traces and flushes asynchronously. Force
        # a flush so the trace is queryable in the UI before the
        # request returns — important for QA validation, less
        # important for steady-state traffic.
        client.flush()
    except Exception as exc:
        # Tracing failures must never break the request. Log and move on.
        logger.warning("retrieve: Langfuse trace failed (non-fatal): %s", exc)
        # Make sure a half-opened span is closed even on failure so
        # we don't leak a background flush task.
        if span is not None:
            try:
                span.end()
            except Exception:
                pass


@app.post("/decks/generate")
def generate_deck(
    word_types: Optional[List[str]] = Query(None),
    frequencies: Optional[List[str]] = Query(None),
    direction: str = Query("both"),
    db: Session = Depends(get_db),
):
    """Generate an Anki deck from filtered words."""
    try:
        filepath = anki_builder.create_anki_deck(
            db,
            word_types=word_types,
            frequencies=frequencies,
            direction=direction,
        )
        filename = filepath.split("/")[-1]
        return {
            "message": "Deck generated successfully",
            "filename": filename,
            "filepath": filepath,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decks/list")
def list_decks():
    """List all generated decks."""
    decks = []
    if os.path.exists(anki_builder.DECKS_DIR):
        for filename in os.listdir(anki_builder.DECKS_DIR):
            if filename.endswith(".apkg"):
                filepath = os.path.join(anki_builder.DECKS_DIR, filename)
                decks.append(
                    {
                        "filename": filename,
                        "created": os.path.getctime(filepath),
                        "size": os.path.getsize(filepath),
                    }
                )
    return {"decks": sorted(decks, key=lambda x: x["created"], reverse=True)}
