import os
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typing import Any, List, Optional, Literal
from app import auth, crud, models, schemas, anki_builder, bootstrap, retrieval
from app.database import get_db, engine
from app.diagnostic.routes import router as diagnostic_router
from app.embeddings import embed_one, EmbeddingError
from app.observability import _ensure_client, get_langfuse
from app.passwords import hash_password

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


# ---------------------------------------------------------------------------
# Phase 2.2 — Auth + gated weakness-profile routes (card t_74c3aa1e)
#
# Replaces the Phase 2.1 routes:
#   - ``POST /users`` (raw ``password_hash``)  →  ``POST /auth/signup`` (bcrypt)
#   - ``GET /users/me`` (501 placeholder)      →  ``GET /auth/me`` (JWT decode)
#   - ``GET/PUT /weakness-profile/{user_id}``  →  same routes, now auth-gated
#
# New surface added in this card:
#   - ``POST /auth/signup`` — bcrypt hash, issue JWT cookie, return
#     ``{access_token, user}``.
#   - ``POST /auth/login``  — verify bcrypt, issue JWT cookie.
#   - ``POST /auth/logout`` — clear the cookie. 204.
#   - ``GET  /auth/me``     — decode cookie or Bearer header, return
#     the ``User`` row.
#
# Existing endpoints (``/words``, ``/decks/generate``, ``/retrieve``,
# etc.) stay untouched and unauthenticated. Only ``/auth/*`` and
# ``/weakness-profile/*`` route through ``get_current_user``.
# ---------------------------------------------------------------------------


def _issue_token_and_respond(
    user: models.User, db: Session
) -> schemas.AuthResponse:
    """Mint a JWT for the user and shape the response body.

    The cookie is set as a side-effect on the FastAPI ``Response``
    by the route — this helper only builds the body so a future
    caller (e.g. a CLI script) can reuse the auth shape without
    needing a Response object.
    """
    token = auth.create_access_token(user.id)
    user_out = schemas.UserOut.model_validate(user)
    return schemas.AuthResponse(access_token=token, user=user_out)


@app.post(
    "/auth/signup",
    response_model=schemas.AuthResponse,
    status_code=201,
)
def signup(
    payload: schemas.SignupRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Create a new user, hash the password with bcrypt, issue a
    JWT cookie, return ``{access_token, user}``.

    Duplicate emails return 409 (pre-check + IntegrityError
    fallback, same shape). Email format is validated by
    ``EmailStr`` (Pydantic); password length is validated by the
    field bounds ``[8, 128]``. 422 on either failure.
    """
    # Pre-check the email — fast path. The unique constraint is
    # still the source of truth so two concurrent signups with the
    # same email both surface 409 cleanly.
    if crud.get_user_by_email(db, payload.email) is not None:
        raise HTTPException(status_code=409, detail="email already registered")
    pw_hash = hash_password(payload.password)
    try:
        user = crud.create_user(
            db, email=payload.email, password_hash=pw_hash
        )
    except IntegrityError:
        # Concurrent insert beat us to it. Roll back the in-flight
        # session and surface the same 409 the pre-check would have.
        db.rollback()
        raise HTTPException(status_code=409, detail="email already registered")

    auth.set_auth_cookie(response, auth.create_access_token(user.id))
    return _issue_token_and_respond(user, db)


@app.post(
    "/auth/login",
    response_model=schemas.AuthResponse,
)
def login(
    payload: schemas.LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Verify the (email, password) pair and issue a JWT cookie.

    Returns 401 on any failure — the response body does not
    distinguish "no such email" from "wrong password" so a
    username-enumeration probe gets the same shape either way.
    """
    user = crud.authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )
    auth.set_auth_cookie(response, auth.create_access_token(user.id))
    return _issue_token_and_respond(user, db)


@app.post("/auth/logout", status_code=204)
def logout(response: Response):
    """Clear the ``lexora_token`` cookie. Always 204 — logout
    is idempotent (a second logout is a no-op).

    Note: the route MUST return the injected ``response`` itself
    (not a fresh ``Response(status_code=204)``) — FastAPI's
    response-cycle replaces the response object on return, and a
    fresh response would drop the Set-Cookie header that
    ``clear_auth_cookie`` just added.
    """
    auth.clear_auth_cookie(response)
    response.status_code = 204
    return response


@app.get("/auth/me", response_model=schemas.MeOut)
def read_me(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Return the authenticated user plus the two fields the
    post-signup first-login gate needs: ``weakness_profile`` and
    ``diagnostic_state``.

    401 if no / invalid / expired token (raised by the dependency,
    same opaque shape for all failure modes — never leaks whether
    the cookie was missing vs. the signature was bad vs. the user
    was deleted).

    Phase 3.3 (card t_ff6fa637) added the two new fields. The
    response shape is non-breaking for clients that only read
    ``id`` / ``email`` — the header, the protected-route gate, and
    the weakness-profile page all keep working without changes.

    Computation:

    - ``weakness_profile``: a single ``GET /weakness-profile``
      lookup. ``None`` if the user has no profile row yet (we
      deliberately do NOT auto-create here — the
      ``/weakness-profile/{user_id}`` route is the only place that
      auto-creates; the ``/auth/me`` payload is a read-only
      projection). The frontend's gate treats ``None`` as empty.
    - ``diagnostic_state``: a single query against
      ``diagnostic_sessions`` for the user's most-recent row.
      ``"never"`` when no row exists, otherwise the literal status
      of the latest row. The status string is already constrained
      to the four expected values by the model layer; we still
      defensively fall back to ``"never"`` on any unexpected value
      so a future bug never reaches the SPA as a 500.
    """
    profile_row = crud.get_weakness_profile(db, current_user.id)
    if profile_row is None:
        weakness_profile_payload = None
    else:
        weakness_profile_payload = {
            "id": profile_row.id,
            "user_id": profile_row.user_id,
            "axes": crud.serialize_weakness_profile_axes(profile_row),
            "updated_at": profile_row.updated_at,
        }

    sessions = crud.list_diagnostic_sessions(db, current_user.id)
    if not sessions:
        diagnostic_state: schemas.DiagnosticState = "never"
    else:
        # ``list_diagnostic_sessions`` returns newest-first; the
        # gate on the client only cares about the most recent
        # session's status, so we read row zero and trust the
        # status string verbatim (``in_progress`` | ``completed``
        # | ``applied``; the model also allows ``skipped`` which
        # the gate treats as "user has decided to set axes
        # manually" — same routing as ``completed`` / ``applied``).
        latest_status = sessions[0].status
        if latest_status in ("in_progress", "completed", "applied"):
            diagnostic_state = latest_status  # type: ignore[assignment]
        else:
            # ``skipped`` (or any future status) → map to
            # ``completed`` so the gate doesn't re-route the user
            # back to the probe they've already opted out of. The
            # literal type is open; the gate only branches on
            # the two "send to diagnostic" values
            # (``never`` / ``in_progress``).
            diagnostic_state = "completed"

    return schemas.MeOut(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
        weakness_profile=weakness_profile_payload,
        diagnostic_state=diagnostic_state,
    )


@app.get(
    "/weakness-profile/{user_id}",
    response_model=schemas.WeaknessProfileOut,
)
def read_weakness_profile(
    user_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Return the weakness profile for a user.

    Phase 2.2 auth: the caller must be authenticated. The JWT
    subject (``current_user.id``) must match ``user_id`` — a
    logged-in user can only read their own profile, and probing
    another user's ``user_id`` returns 403.

    Auto-creates an empty default profile on first read so a
    fresh user always sees a stable response shape. Returns 404
    if the user_id doesn't exist (defence in depth — the
    dependency already verified ``current_user.id`` is a real
    row, but the route guards in case the URL parameter is
    edited by hand).
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=403, detail="cannot read another user's profile"
        )
    if crud.get_user_by_id(db, user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    profile = crud.get_weakness_profile(db, user_id)
    if profile is None:
        profile = crud.create_empty_weakness_profile(db, user_id)
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "axes": crud.serialize_weakness_profile_axes(profile),
        "updated_at": profile.updated_at,
    }


@app.put(
    "/weakness-profile/{user_id}",
    response_model=schemas.WeaknessProfileOut,
)
def write_weakness_profile(
    user_id: int,
    payload: schemas.WeaknessProfileUpdate,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Upsert the weakness profile for a user.

    Phase 2.2 auth: same as ``GET /weakness-profile/{user_id}``.
    The ``WeaknessProfileUpdate`` Pydantic model already validates
    ``axes`` (each value must be an int in [0, 3]) so a 422
    surfaces on bad input. Returns 404 if the user_id doesn't
    exist (defence in depth — the dependency verified
    ``current_user.id`` is real, but the route guards against
    hand-edited URLs).
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=403, detail="cannot write another user's profile"
        )
    if crud.get_user_by_id(db, user_id) is None:
        raise HTTPException(status_code=404, detail="user not found")
    profile = crud.upsert_weakness_profile(db, user_id, payload.axes)
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "axes": crud.serialize_weakness_profile_axes(profile),
        "updated_at": profile.updated_at,
    }


# ---------------------------------------------------------------------------
# Phase 4.2 — Cloze exercise route (card t_bdd9ffbe)
#
# Single auth-gated endpoint, ``POST /exercises/cloze``. Body: ``{}``
# (word selection is server-driven from the user's weakness profile).
# Response: the ``ClozeExercise`` Pydantic model.
#
# No grading endpoint — Phase 5 wires ``py-fsrs`` + the matching
# exercise type + a ``POST /exercises/grade`` route on top of this.
#
# Errors:
# - 401: missing / invalid JWT (raised by ``get_current_user``).
# - 502: LLM transport / provider error or persistent schema
#   violation (``ClozeGenerationError``). The body carries the
#   structured fields so an operator can triage without re-running.
# - 500: corpus inconsistency (e.g. ``select_target_word`` raised
#   ``ValueError`` because the mapped ``word_type`` has zero rows).
# ---------------------------------------------------------------------------


@app.post("/exercises/cloze", response_model=schemas.ClozeExerciseOut)
def generate_cloze_exercise(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Generate one cloze exercise for the logged-in learner.

    The route is intentionally thin — all logic lives in
    ``app.cloze.generate_cloze``. We import lazily inside the
    handler so the module-level ``main.py`` stays import-cheap and
    so the existing test suite (which imports ``app.main`` before
    the cloze module) doesn't pay the OpenAI / instructor import
    cost.

    The body is an empty ``BaseModel`` — we don't accept any input.
    The frontend re-fetches on every ``Generate another`` click, so
    a fixed-shape request keeps the surface stable.
    """
    from app.cloze import (
        ClozeGenerationError,
        generate_cloze,
        PROMPT_TEMPLATE_VERSION,
    )
    from app.llm import LLMError

    try:
        exercise = generate_cloze(db, current_user.id)
    except LLMError as exc:
        logger.error(
            "cloze: LLM transport failure for user_id=%d: %s",
            current_user.id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"cloze generation failed: {exc}",
        )
    except ClozeGenerationError as exc:
        logger.error(
            "cloze: schema dead-letter for user_id=%d after %d attempt(s): %s",
            current_user.id,
            exc.schema_retry_count,
            exc.last_validation_error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "cloze_generation_failed",
                "schema_retry_count": exc.schema_retry_count,
                "last_validation_error": exc.last_validation_error,
            },
        )
    except ValueError as exc:
        # Corpus inconsistency (e.g. axis-mapped word_type has zero
        # rows). 500 — operator needs to look at the seed data.
        logger.error("cloze: corpus inconsistency for user_id=%d: %s",
                     current_user.id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Lock the prompt_template_version on the way out so a future
    # ``generate_cloze`` that forgets to set it can't desync the
    # contract.
    return exercise.model_copy(
        update={"prompt_template_version": PROMPT_TEMPLATE_VERSION}
    )


# ---------------------------------------------------------------------------
# Phase 5.4 — ``GET /exercises/due`` (card t_e8548d6d)
#
# The closed study loop's right half. Given an authenticated user,
# return the next cloze they're due to review (FSRS-driven) OR a fresh
# word they've never seen (first-encounter warm-up).
#
# Wire contract:
#   - Auth: 401 if no / invalid JWT (raised by ``get_current_user``).
#   - 204 No Content if no ``fsrs_cards`` rows are due AND no fresh
#     words exist in the corpus (the user has nothing to study; the
#     frontend shows an honest empty state).
#   - 200 + ``ClozeDueExerciseOut`` (the cloze payload plus a
#     ``due_from_fsrs: bool`` discriminator) otherwise.
#   - 502 if the cloze LLM call fails (transport / persistent schema
#     violation); 500 on corpus inconsistency (the due-queue picked
#     a word id that's no longer in the words table).
#
# Single-user dev assumption:
#   Phase 5's ``fsrs_cards`` table has no ``user_id`` column, so the
#   query is global — any due card in the system is returned to any
#   authenticated user. Phase 6 may add the column for per-user
#   scoping. The model assumes a single-user dev environment; the
#   route is documented as such.
#
# Two branches:
#   1. **FSRS-driven (due_from_fsrs=True).** Pick the row from
#      ``fsrs_cards`` whose ``due_date <= now()`` and that has the
#      earliest due_date. The picked word has a row in fsrs_cards
#      (we just queried it) — we don't need to create one. Pass its
#      word_id to ``generate_cloze(force_word_id=...)`` so the cloze
#      is built for THIS word, not a fresh selection.
#   2. **First-encounter (due_from_fsrs=False).** No card is due.
#      Pick the lowest-id ``Word`` row that has NO ``fsrs_cards`` row
#      yet — i.e. the user has never seen this word. Create a fresh
#      Learning row inline (state=1, due=now) so the next
#      ``POST /exercises/grade`` has a row to update. Pass its
#      word_id to ``generate_cloze(force_word_id=...)``.
#
# The route is intentionally thin — all word selection + cloze
# generation logic lives in ``app.cloze``. The route is responsible
# only for the due-queue SQL + the inline Learning-row creation.
# ---------------------------------------------------------------------------


@app.get("/exercises/due", response_model=schemas.ClozeDueExerciseOut)
def get_due_exercise(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Return the next cloze the authenticated user is due to review.

    See the section header above for the full wire contract. The
    two-branch logic is documented inline below.
    """
    from app.cloze import (
        ClozeGenerationError,
        generate_cloze,
        PROMPT_TEMPLATE_VERSION,
    )
    from app.llm import LLMError

    now = datetime.utcnow()

    # Branch 1: FSRS-driven. Pick the earliest-due fsrs_cards row.
    # The query is intentionally global (no user_id filter) per the
    # single-user dev assumption documented in the section header.
    # Tie-break on ``id`` ASC so the deterministic ordering survives
    # any clock skew between request time and ``due_date``.
    due_row = (
        db.query(models.FsrsCard)
        .filter(models.FsrsCard.due_date <= now)
        .order_by(
            models.FsrsCard.due_date.asc(),
            models.FsrsCard.id.asc(),
        )
        .first()
    )
    if due_row is not None:
        picked_word_id: int = due_row.word_id
        due_from_fsrs = True
    else:
        # Branch 2: First-encounter. Pick a corpus word that has no
        # fsrs_cards row yet. The ``NOT IN`` subquery is the simplest
        # portable shape; both SQLite and Postgres support it. We use
        # ``select(...)`` explicitly (SQLAlchemy 2.x's preferred
        # form) rather than passing the subquery object directly to
        # ``.in_()`` — the latter emits a SAWarning and is deprecated.
        graded_word_ids_subq = (
            select(models.FsrsCard.word_id)
            .where(models.FsrsCard.word_id.isnot(None))
            .scalar_subquery()
        )
        fresh_word = (
            db.query(models.Word)
            .filter(~models.Word.id.in_(graded_word_ids_subq))
            .order_by(models.Word.id.asc())
            .first()
        )
        if fresh_word is None:
            # No due cards AND no fresh words in the corpus. The user
            # has literally nothing to study. 204 keeps the contract
            # clean — the frontend's "All caught up" empty state is
            # the right surface, not a fake exercise.
            return Response(status_code=204)

        picked_word_id = fresh_word.id
        # Inline-create a fresh Learning row so the next
        # ``POST /exercises/grade`` (Phase 5.3) has a row to update.
        # ``state=1`` is py-fsrs's ``State.Learning`` (1/2/3 in the
        # 4.1.2 enum); ``due_date=now`` makes the card immediately
        # due so a same-session grade works. All other numeric
        # columns start at 0 / None — py-fsrs will populate them on
        # the first ``review_card`` call.
        new_card = models.FsrsCard(
            word_id=picked_word_id,
            difficulty=None,
            stability=None,
            retrievability=None,
            due_date=now,
            last_review=None,
            reps=0,
            lapses=0,
            state=1,
            elapsed_days=0,
            scheduled_days=0,
        )
        db.add(new_card)
        try:
            db.commit()
        except IntegrityError:
            # Concurrent insert beat us to it — another request just
            # created an fsrs_cards row for this word. The unique
            # constraint (Phase 5.2's ``UNIQUE(word_id)``) fires here.
            # Roll back and re-query: the row exists now, so the
            # *next* /exercises/due call will see it via Branch 1.
            # For THIS call, we still want to give the user a cloze
            # for the picked word — keep due_from_fsrs=False because
            # semantically this is still a first-encounter from the
            # user's perspective. The fresh Learning row exists now
            # regardless of who won the race.
            db.rollback()
            logger.info(
                "exercises/due: concurrent fsrs_cards insert for "
                "word_id=%d; rolling back the inline create",
                picked_word_id,
            )
        due_from_fsrs = False

    # Generate the cloze for the picked word. ``force_word_id`` is
    # the 5.4-only knob on ``generate_cloze``; the deterministic
    # seed path stays untouched for the 4.5 ``POST /exercises/cloze``
    # caller.
    try:
        exercise = generate_cloze(
            db, current_user.id, force_word_id=picked_word_id
        )
    except LLMError as exc:
        logger.error(
            "exercises/due: LLM transport failure for user_id=%d: %s",
            current_user.id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"cloze generation failed: {exc}",
        )
    except ClozeGenerationError as exc:
        logger.error(
            "exercises/due: schema dead-letter for user_id=%d after %d "
            "attempt(s): %s",
            current_user.id,
            exc.schema_retry_count,
            exc.last_validation_error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "cloze_generation_failed",
                "schema_retry_count": exc.schema_retry_count,
                "last_validation_error": exc.last_validation_error,
            },
        )
    except ValueError as exc:
        # Corpus inconsistency — the picked word_id is no longer in
        # the words table. The route picked it from fsrs_cards (a
        # join that should never go stale) but defensive: surface
        # 500 so the operator notices.
        logger.error(
            "exercises/due: corpus inconsistency for user_id=%d "
            "word_id=%d: %s",
            current_user.id,
            picked_word_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc))

    # Lock the prompt_template_version (same as POST /exercises/cloze)
    # and attach the due_from_fsrs discriminator. FastAPI's
    # ``response_model=schemas.ClozeDueExerciseOut`` validates the
    # returned dict against the merged Pydantic schema.
    exercise = exercise.model_copy(
        update={"prompt_template_version": PROMPT_TEMPLATE_VERSION}
    )
    return {
        **exercise.model_dump(),
        "due_from_fsrs": due_from_fsrs,
    }


# ---------------------------------------------------------------------------
# Phase 3.1 — Diagnostic probe router (card t_41d85c32)
#
# Four auth-gated routes mounted at the ``/diagnostic`` prefix:
#   - POST /diagnostic/start   — create / resume a session, return the
#                                stripped question bank.
#   - POST /diagnostic/answer  — record one (question, choice) answer.
#   - GET  /diagnostic/result  — recompute the deterministic score.
#   - POST /diagnostic/apply   — UPSERT the score into the caller's
#                                WeaknessProfile (existing helper).
#
# Handlers live in ``app.diagnostic.routes``. NO LLM call, NO Langfuse
# tracing — the probe is fully deterministic. Existing endpoints stay
# untouched and unauthenticated.
# ---------------------------------------------------------------------------

app.include_router(
    diagnostic_router, prefix="/diagnostic", tags=["diagnostic"]
)

