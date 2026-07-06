"""Tests for Phase 10.3 — ``POST /exercises/phrase_match`` route (card t_13bb48d2).

Coverage map (mirrors the card body's "End-to-end test" section):

1. **200 happy path** — ``{"word_id": <existing>}`` →
   ``PhraseMatchExerciseOut`` with ``exercise_type="phrase_match"``,
   server-minted ``exercise_id`` (int, non-zero), ``word_id``
   echoed, ``target_word_id == word_id``, ``prompt_template_version
   == "phrase-match-v1"``, all phrase-match-specific fields
   populated (phrase_a, phrase_b, relation, relation_rationale,
   source_attribution).

2. **200 with ``enable_rag=True``** — payload ``{"word_id": ...,
   "enable_rag": true}`` → 200 with ``enable_rag=True`` echoed on
   the response. Verifies the route forwards the flag to the
   generator AND the ``_retrieve_phrase_pair_neighbors`` helper
   was called (the prompt was built with the retrieval hook —
   monkeypatched to a known list of neighbors for offline
   determinism).

3. **401 — no JWT cookie** → 401 (auth-gated route; the
   ``get_current_user`` dependency raises).

4. **422 — malformed body** (``{}`` — missing required
   ``word_id``) → 422 (Pydantic body validation).

5. **422 — bad ``enable_rag`` type** (``{"word_id": 1,
   "enable_rag": "true"}`` — string instead of bool) → 422.

6. **422 — self-pair gate** (the resolved ``phrase_pairs`` row has
   ``phrase_a_id == phrase_b_id``) → 422 with structured
   ``self_pair_rejected`` detail. We seed a self-pair row directly
   in the DB to exercise the route's defense-in-depth gate; the
   10.2 Pydantic layer also rejects this on the input side.

7. **422 — pre-flight DB gate** (the resolved ``Phrase`` row for
   ``phrase_a_id`` or ``phrase_b_id`` doesn't exist in the planted
   ``phrases`` table) → 422 with structured
   ``phrase X not found in planted phrases table`` detail.

8. **404 — empty ``phrase_pairs``** → 404 with the card body's
   detail string. Mirrors the 8.4 idiom ``IdiomNotFoundError`` 404
   discipline.

9. **200 — OpenAPI schema regeneration** — the
   ``/exercises/phrase_match`` route is listed at
   ``openapi.json#/paths/~1exercises~1phrase_match`` with the
   correct request/response shapes.

Hermetic: a fresh temp SQLite DB + a temp JWT secret per
test. The OpenRouter chat-completions call is mocked via
``monkeypatch.setattr("app.phrase_match._openai_client", ...)``
so no network is touched. The 8.1 ``Phrase`` rows + 10.1
``phrase_pairs`` rows are seeded inline via
``database.Base.metadata.create_all`` plus SQLAlchemy inserts.

Phase 10.3's offline / DummyLM discipline: NO live LLM in CI.
The ``_openai_client`` monkeypatch returns a
``MockTransport``-backed OpenAI client whose ``chat.completions``
yields an instructor-validated ``PhraseMatchExercise`` payload.

Run from ``backend/``::

    uv run pytest -q tests/test_phrase_match_endpoint.py
"""
from __future__ import annotations

import json
import secrets
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Top-level ``app.phrase_match`` import — registers the local
# ``PhrasePair`` mirror with ``Base.metadata`` BEFORE the
# ``db_session`` fixture calls ``Base.metadata.create_all``.
# Without this, the table wouldn't exist when the fixture
# seeds phrase_pairs rows (the route lazy-imports
# ``app.phrase_match`` so the table wouldn't be registered
# until the first POST hits the route).
from app import phrase_match  # noqa: F401 — table registration side-effect


# ---------------------------------------------------------------------------
# Fixtures — mirror the per-test SQLite + JWT secret pattern from
# test_idiom_endpoint / test_comprehension_endpoint / test_match_endpoint.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path, monkeypatch):
    """Fresh temp SQLite DB + a JWT secret so ``app.auth``'s import-time
    check passes. Same pattern as the comprehension / matching /
    idiom endpoint test suites.
    """
    db_path = tmp_path / "test_phrase_match_endpoint.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("LEXORA_DECKS_DIR", str(tmp_path / "decks"))
    monkeypatch.setenv("JWT_SECRET", secrets.token_hex(32))
    return str(db_path)


@pytest.fixture
def client(sqlite_db_path):
    """``TestClient`` wired to a fresh per-test SQLite DB.

    ``database.Base.metadata.create_all`` registers the 8.1
    ``Phrase`` model + the 10.2 ``phrase_pairs`` local mirror
    on the same metadata object as the rest of the corpus, so
    the 404 / 422 gates (which query ``phrases`` /
    ``phrase_pairs``) find the tables populated.
    """
    from app import database
    from app.main import app

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(sqlite_db_path):
    """A SQLAlchemy session for the per-test SQLite DB.

    Mirrors test_idiom_endpoint / test_comprehension_endpoint.
    """
    from sqlalchemy.orm import sessionmaker

    from app import database

    database.reconfigure_for_test(f"sqlite:///{sqlite_db_path}")
    database.Base.metadata.create_all(bind=database.engine)
    engine = database.engine
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_phrase_pair_fixture(
    session,
    *,
    n_phrase_rows: int = 5,
    n_pair_rows: int = 5,
) -> None:
    """Insert ``n_phrase_rows`` Phrase rows + ``n_pair_rows`` PhrasePair
    rows so ``select_phrase_pair`` has stable selection material.

    Mirrors the test_phrase_match.py ``db_session`` fixture shape
    byte-for-byte — same slugs, same FK references, same
    attested_pair=1 default. The phrase-pair FK chains point at
    unique ``phrases.id`` slugs so a 5-row pair fixture can
    reference 5 unique phrase rows.
    """
    from app.phrase_match import Phrase, PhrasePair

    phrase_rows = [
        Phrase(
            id=f"phrase-{i:02d}",
            phrase=f"phrase number {i:02d} surface text",
            definition=f"definition {i:02d}",
            example_usage=f"example {i:02d} usage.",
            source_attribution="dwds",
            frequency_band="high" if i % 2 == 0 else "mid",
            dwds_url=None,
            attested_quote=None,
            attested_source=None,
        )
        for i in range(1, n_phrase_rows + 1)
    ]
    for row in phrase_rows:
        session.add(row)
    session.flush()

    pair_rows = [
        PhrasePair(
            id=i,
            phrase_a_id=f"phrase-0{i}",
            phrase_b_id=f"phrase-0{i + 1 if i < n_pair_rows else 1}",
            relation="equivalent" if i == 1 else (
                "paraphrase" if i == 2 else (
                    "related" if i == 3 else (
                        "unrelated" if i == 4 else "equivalent"
                    )
                )
            ),
            attested_pair=1,
        )
        for i in range(1, n_pair_rows + 1)
    ]
    for row in pair_rows:
        session.add(row)
    session.commit()


def _signup(client: TestClient, email: str = "ada@example.com") -> dict:
    """POST /auth/signup and return the parsed body."""
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": "supersecret"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# OpenAI stub — mirrors test_idiom_endpoint's stub. Returns a
# real OpenAI client whose httpx transport is a MockTransport so
# ``instructor.from_openai`` accepts it. The assistant message
# content is a JSON payload that validates against
# ``app.phrase_match.PhraseMatchExercise``.
# ---------------------------------------------------------------------------


def _make_stub_instructor_client(
    payload: str,
    *,
    model: str = "qwen/qwen3-235b-a22b-2507",
    prompt_tokens: int = 30,
    completion_tokens: int = 12,
) -> Any:
    """Build a stub OpenAI client that returns ``payload`` as the
    assistant message content. The stub is order-independent: it
    bypasses respx so OpenAI's custom httpx pool doesn't matter.
    """
    import httpx
    from openai import OpenAI

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-phrase-match-endpoint-001",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": payload,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    return OpenAI(
        api_key="test-key-not-real",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _phrase_match_payload(
    *,
    phrase_a: str = "ins Blaue hinein",
    phrase_b: str = "ohne festes Ziel",
    relation: str = "equivalent",
    relation_rationale: str = (
        "Beide Ausdruecke bezeichnen eine planlose Handlung."
    ),
    source_attribution: str = "dwds",
) -> str:
    """Build a valid ``PhraseMatchExercise`` JSON payload for a test.

    Field bounds are enforced by ``PhraseMatchExercise``:

    - ``phrase_a`` / ``phrase_b``: 5..200 chars (we use a stub
      well above 5 and well below 200).
    - ``relation``: closed 4-way literal (we use "equivalent").
    - ``relation_rationale``: 1..400 chars.
    - ``source_attribution``: closed literal subset
      (we use "dwds"; route layer adds "bge-m3-cosine" when
      ``enable_rag=True``).
    - ``prompt_template_version``: must equal
      ``app.phrase_match.PROMPT_TEMPLATE_VERSION`` ("phrase-match-v1")
      so the generated row matches what the offline
      DummyLM-stub pool ships.
    """
    return json.dumps(
        {
            "exercise_id": 44444441,
            "phrase_a": phrase_a,
            "phrase_b": phrase_b,
            "relation": relation,
            "relation_rationale": relation_rationale,
            "source_attribution": source_attribution,
            "prompt_template_version": "phrase-match-v1",
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 1. 200 happy path — default enable_rag=False, single word_id
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_happy_path_default(
    client, db_session, monkeypatch
):
    """``POST /exercises/phrase_match`` with ``{"word_id": <existing>}``
    returns a ``PhraseMatchExerciseOut`` carrying the wire metadata
    contract:

    - ``exercise_type="phrase_match"`` (``PhraseMatchExerciseOut``
      default).
    - ``exercise_id`` is a server-minted int (not None, not 0).
    - ``word_id`` is echoed on the response.
    - ``target_word_id`` equals ``word_id`` (same numerical
      value; the response carries both for cross-exercise
      consumer symmetry with the comprehension / matching /
      idiom responses).
    - ``prompt_template_version == "phrase-match-v1"``.
    - ``phrase_a`` / ``phrase_b`` / ``relation`` /
      ``relation_rationale`` / ``source_attribution`` all
      populated.
    - There is no ``count`` knob on the wire — phrase-match
      generates one exercise per call by design (mirrors idiom).
    - ``enable_rag`` defaults to ``False`` (Hard rule #1:
      opt-in).
    - ``source_attribution`` does NOT carry the
      ``"bge-m3-cosine"`` token when ``enable_rag=False`` (Hard
      rule #7 — the token is added only on RAG-on).
    """
    from app.phrase_match import PROMPT_TEMPLATE_VERSION

    _seed_phrase_pair_fixture(db_session)
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _phrase_match_payload()
    monkeypatch.setattr(
        phrase_match,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post(
        "/exercises/phrase_match", json={"word_id": 1}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Wire metadata contract.
    assert body["exercise_type"] == "phrase_match"
    assert isinstance(body["exercise_id"], int)
    assert body["exercise_id"] != 0
    assert body["word_id"] == 1
    assert body["target_word_id"] == 1
    assert body["prompt_template_version"] == PROMPT_TEMPLATE_VERSION

    # Phrase-match-specific fields populated.
    assert body["phrase_a"]
    assert 5 <= len(body["phrase_a"]) <= 200
    assert body["phrase_b"]
    assert 5 <= len(body["phrase_b"]) <= 200
    assert body["relation"] in {
        "equivalent", "paraphrase", "related", "unrelated"
    }
    assert body["relation_rationale"]
    assert 1 <= len(body["relation_rationale"]) <= 400
    # ``source_attribution`` validates against the closed
    # literal subset — the stub returns "dwds". On
    # ``enable_rag=False`` the route must NOT add the
    # ``bge-m3-cosine`` token.
    tokens = [
        t.strip() for t in body["source_attribution"].split(",") if t.strip()
    ]
    valid = {"dwds", "goethe", "schiller", "bge-m3-cosine"}
    assert all(tok in valid for tok in tokens)
    assert "bge-m3-cosine" not in tokens

    # No ``count`` knob on the wire — phrase-match generates
    # one exercise per call by design (mirrors idiom).
    assert "count" not in body
    assert "pairs" not in body
    assert "passage" not in body  # not a comprehension
    assert "phrase" not in body  # not an idiom (idiom has singular "phrase")

    # No regression: existing 4 endpoints' callers still parse.
    assert body["enable_rag"] is False  # default


# ---------------------------------------------------------------------------
# 2. 200 with overrides — enable_rag=True threaded through to generator
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_enable_rag_appends_cosine_token(
    client, db_session, monkeypatch
):
    """``{"word_id": ..., "enable_rag": true}`` → 200 with
    ``enable_rag=True`` echoed + ``bge-m3-cosine`` appended to
    ``source_attribution`` even when no neighbor rows surface
    on the SQLite fallback path.

    The 10.3 retrieval helper returns ``[]`` on SQLite (no
    pgvector, no embedding column populated by the offline
    backfill). The route layer still stamps
    ``bge-m3-cosine`` on the response so the cohort split
    between RAG-on / RAG-off callers is observable from
    outside the generator.
    """

    _seed_phrase_pair_fixture(db_session)
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    payload = _phrase_match_payload()
    monkeypatch.setattr(
        phrase_match,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post(
        "/exercises/phrase_match",
        json={"word_id": 1, "enable_rag": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # ``enable_rag`` is echoed on the response.
    assert body["enable_rag"] is True

    # ``source_attribution`` carries ``bge-m3-cosine`` — the
    # route layer appends the token on RAG-on calls so the
    # response correctly attributes the cohort even when no
    # neighbor rows surface (SQLite fallback).
    tokens = [
        t.strip() for t in body["source_attribution"].split(",") if t.strip()
    ]
    assert "bge-m3-cosine" in tokens
    # Token-by-token validator on
    # ``PhraseMatchExerciseOut._validate_source_attribution``
    # guarantees the closed-literal invariant survived.
    valid = {"dwds", "goethe", "schiller", "bge-m3-cosine"}
    assert all(tok in valid for tok in tokens)


def test_post_exercises_phrase_match_enable_rag_calls_retrieval_helper(
    client, db_session, monkeypatch
):
    """``enable_rag=True`` triggers ``_retrieve_phrase_pair_neighbors``
    on the generator side. We spy on the helper to confirm the
    boolean gate is the only branching point.

    The card body says: "verify the DSPy module received few-shot
    context (assert via the call args)." We monkeypatch the
    helper to return a known list of neighbors, then capture the
    prompt messages via a wrapper around ``build_prompt`` to
    confirm the neighbors were embedded.
    """

    _seed_phrase_pair_fixture(db_session)
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    # Spy on the retrieval helper. On the SQLite path it would
    # return []; we inject a known list to assert the route
    # actually threaded the call through.
    neighbors_payload = [
        {
            "kind": "phrase_pair",
            "id": 999,
            "phrase_a_id": "phrase-99",
            "phrase_b_id": "phrase-100",
            "phrase_a": "eine bekannte Redewendung",
            "phrase_b": "ein aehnlicher Ausdruck",
            "relation": "paraphrase",
            "score": 0.92,
        },
        {
            "kind": "phrase_pair",
            "id": 998,
            "phrase_a_id": "phrase-77",
            "phrase_b_id": "phrase-78",
            "phrase_a": "noch eine Redewendung",
            "phrase_b": "ein verwandter Ausdruck",
            "relation": "related",
            "score": 0.81,
        },
    ]

    call_args: dict[str, Any] = {"called": 0, "top_k": None, "pair_id": None}

    def _spy_retrieval(db, pair, *, top_k=phrase_match.RAG_TOP_K):
        call_args["called"] += 1
        call_args["top_k"] = top_k
        call_args["pair_id"] = pair.id
        return neighbors_payload

    monkeypatch.setattr(
        phrase_match,
        "_retrieve_phrase_pair_neighbors",
        _spy_retrieval,
    )

    # Capture the prompt messages to assert neighbors were embedded.
    # IMPORTANT: save a reference to the real ``build_prompt``
    # BEFORE monkeypatch.setattr replaces it on the module —
    # otherwise our spy would recursively call itself via
    # ``phrase_match.build_prompt``.
    real_build_prompt = phrase_match.build_prompt
    captured_messages: dict[str, Any] = {}

    def _spy_build_prompt(pair, phrase_a_row, phrase_b_row, **kwargs):
        captured_messages["kwargs"] = kwargs
        messages = real_build_prompt(
            pair, phrase_a_row, phrase_b_row, **kwargs
        )
        captured_messages["messages"] = messages
        return messages

    monkeypatch.setattr(phrase_match, "build_prompt", _spy_build_prompt)
    payload = _phrase_match_payload()
    monkeypatch.setattr(
        phrase_match,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post(
        "/exercises/phrase_match",
        json={"word_id": 1, "enable_rag": True},
    )
    assert resp.status_code == 200, resp.text

    # Retrieval helper was called exactly once, with the
    # ``RAG_TOP_K=3`` default. The ``pair`` argument is a
    # transient ``PhrasePair`` view (``_make_transient_pair``
    # from the generator) so ``pair.id`` is ``None`` — the
    # spy captures the call count + the top_k arg, not the
    # transient pair's autoincrement PK (which would only be
    # set after a session flush + commit).
    assert call_args["called"] == 1
    assert call_args["top_k"] == phrase_match.RAG_TOP_K

    # The neighbors were embedded in the user prompt — the
    # ``retrieved_neighbors_json`` kwarg is a JSON string
    # carrying the injected list.
    kwargs = captured_messages["kwargs"]
    assert kwargs.get("retrieved_neighbors_json") is not None
    embedded = json.loads(kwargs["retrieved_neighbors_json"])
    assert embedded == neighbors_payload


def test_post_exercises_phrase_match_enable_rag_false_skips_retrieval(
    client, db_session, monkeypatch
):
    """``enable_rag=False`` (default) skips the retrieval helper
    entirely. The boolean gate is the only branching point —
    the curated-only path is byte-for-byte stable for A/B
    comparison.
    """

    _seed_phrase_pair_fixture(db_session)
    _signup(client)

    monkeypatch.setattr("app.llm.BACKOFF_SCHEDULE_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    call_args: dict[str, Any] = {"called": 0}

    def _spy_retrieval(db, pair, *, top_k=phrase_match.RAG_TOP_K):
        call_args["called"] += 1
        return []

    monkeypatch.setattr(
        phrase_match,
        "_retrieve_phrase_pair_neighbors",
        _spy_retrieval,
    )

    captured_kwargs: dict[str, Any] = {}

    # IMPORTANT: save a reference to the real ``build_prompt``
    # BEFORE monkeypatch.setattr replaces it on the module —
    # otherwise our spy would recursively call itself via
    # ``phrase_match.build_prompt``.
    real_build_prompt_2 = phrase_match.build_prompt

    def _spy_build_prompt(pair, phrase_a_row, phrase_b_row, **kwargs):
        captured_kwargs.update(kwargs)
        return real_build_prompt_2(
            pair, phrase_a_row, phrase_b_row, **kwargs
        )

    monkeypatch.setattr(phrase_match, "build_prompt", _spy_build_prompt)

    payload = _phrase_match_payload()
    monkeypatch.setattr(
        phrase_match,
        "_openai_client",
        lambda: _make_stub_instructor_client(payload),
    )

    resp = client.post(
        "/exercises/phrase_match", json={"word_id": 1}
    )
    assert resp.status_code == 200, resp.text

    # Retrieval helper was NEVER called on the curated-only path.
    assert call_args["called"] == 0

    # The prompt was built WITHOUT a ``retrieved_neighbors_json``
    # kwarg (None) — the generator branches on
    # ``if enable_rag:`` so the call site passes None on the
    # False branch.
    assert captured_kwargs.get("retrieved_neighbors_json") is None


# ---------------------------------------------------------------------------
# 3. 401 — no JWT cookie
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_unauthenticated_returns_401(
    client, db_session
):
    """``POST /exercises/phrase_match`` with no JWT cookie
    returns 401 — same auth gate as the 4 prior exercise
    endpoints (Hard rule #1).
    """
    _seed_phrase_pair_fixture(db_session)

    resp = client.post(
        "/exercises/phrase_match", json={"word_id": 1}
    )
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# 4. 422 — malformed body (missing word_id)
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_missing_word_id_returns_422(
    client, db_session
):
    """``POST /exercises/phrase_match`` with an empty body
    (``{}``) returns 422 because ``word_id`` is required.
    The Pydantic body validator surfaces the missing field
    with the standard FastAPI 422 envelope.
    """
    _seed_phrase_pair_fixture(db_session)
    _signup(client)

    resp = client.post("/exercises/phrase_match", json={})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 5. 422 — bad enable_rag type (string instead of bool)
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_string_enable_rag_returns_422(
    client, db_session
):
    """``{"word_id": 1, "enable_rag": "true"}`` (string) → 422.
    ``enable_rag`` is a ``StrictBool`` (Phase 7 hard rule #5);
    string ``"true"`` and integer ``1`` are rejected at the
    Pydantic layer.
    """
    _seed_phrase_pair_fixture(db_session)
    _signup(client)

    resp = client.post(
        "/exercises/phrase_match",
        json={"word_id": 1, "enable_rag": "true"},
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 6. 422 — self-pair gate (phrase_a_id == phrase_b_id)
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_self_pair_returns_422(
    client, db_session, monkeypatch
):
    """``POST /exercises/phrase_match`` with a resolved pair row
    whose ``phrase_a_id == phrase_b_id`` → 422 with structured
    ``self_pair_rejected`` detail.

    The card body explicitly demands this gate; the 10.1 DB
    CHECK constraint + the 10.2 Pydantic layer both enforce
    it on the input side. The 10.3 route's pre-flight gate
    is the defense-in-depth backstop — even if a future
    migration relaxes the DB CHECK, the wire shape holds.
    """
    from app.phrase_match import Phrase, PhrasePair

    # Seed only the two phrase rows + a self-pair (both
    # pointing at phrase-01). The fixture skips the standard
    # 5-row seeding.
    session = db_session
    for i in (1, 2):
        session.add(
            Phrase(
                id=f"phrase-0{i}",
                phrase=f"phrase number 0{i} surface text",
                definition=f"definition 0{i}",
                example_usage=f"example 0{i} usage.",
                source_attribution="dwds",
                frequency_band="high",
                dwds_url=None,
                attested_quote=None,
                attested_source=None,
            )
        )
    session.flush()
    # Self-pair: phrase_a_id == phrase_b_id == phrase-01.
    session.add(
        PhrasePair(
            id=1,
            phrase_a_id="phrase-01",
            phrase_b_id="phrase-01",
            relation="related",
            attested_pair=1,
        )
    )
    session.commit()

    _signup(client)

    resp = client.post(
        "/exercises/phrase_match", json={"word_id": 1}
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json().get("detail", {})
    if isinstance(detail, dict):
        assert detail.get("error") == "self_pair_rejected"
        assert detail.get("phrase_a_id") == "phrase-01"
        assert detail.get("phrase_b_id") == "phrase-01"


# ---------------------------------------------------------------------------
# 7. 422 — pre-flight DB gate (phrase X not found in planted phrases table)
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_missing_phrase_returns_422(
    client, db_session, monkeypatch
):
    """``POST /exercises/phrase_match`` with a ``phrase_pairs``
    row whose ``phrase_a_id`` (or ``phrase_b_id``) doesn't
    exist in the planted ``phrases`` table → 422 with the
    card body's structured detail.

    Mirrors the 8.4 idiom endpoint's "no Phrase row exists
    for word_id=N" pre-flight gate. The route translates to
    422, not 404 — the operator needs to see which slug is
    missing, not just that the corpus is broken.
    """
    from app.phrase_match import Phrase, PhrasePair

    # Seed only one Phrase row + a phrase_pairs row whose
    # ``phrase_b_id`` points at a slug that doesn't exist.
    # ``phrase_a_id`` is valid; ``phrase_b_id`` is missing.
    session = db_session
    session.add(
        Phrase(
            id="phrase-01",
            phrase="phrase number 01 surface text",
            definition="definition 01",
            example_usage="example 01 usage.",
            source_attribution="dwds",
            frequency_band="high",
            dwds_url=None,
            attested_quote=None,
            attested_source=None,
        )
    )
    session.flush()
    session.add(
        PhrasePair(
            id=1,
            phrase_a_id="phrase-01",
            phrase_b_id="phrase-DOES-NOT-EXIST",
            relation="related",
            attested_pair=1,
        )
    )
    session.commit()

    _signup(client)

    resp = client.post(
        "/exercises/phrase_match", json={"word_id": 1}
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json().get("detail", "")
    assert "phrase-DOES-NOT-EXIST" in detail
    assert "not found in planted phrases table" in detail


# ---------------------------------------------------------------------------
# 8. 404 — empty phrase_pairs table
# ---------------------------------------------------------------------------


def test_post_exercises_phrase_match_empty_phrase_pairs_returns_404(
    client, db_session
):
    """``POST /exercises/phrase_match`` with an empty
    ``phrase_pairs`` table → 404 with the card body's
    structured detail string.

    Mirrors the 8.4 idiom endpoint's ``IdiomNotFoundError``
    404 discipline — the card body explicitly demands 404
    here (not 500). The route layer translates
    ``PhraseMatchNotFoundError`` to 404 with
    ``"no phrase_pairs row exists for word_id=N"``.
    """
    # No seeding — the phrase_pairs table is empty (10.1
    # seed hasn't run yet). This is the canonical 404 path.
    _signup(client)

    resp = client.post(
        "/exercises/phrase_match", json={"word_id": 1}
    )
    assert resp.status_code == 404, resp.text
    detail = resp.json().get("detail", "")
    assert "no phrase_pairs row exists for word_id=1" in detail


# ---------------------------------------------------------------------------
# 9. OpenAPI schema regeneration
# ---------------------------------------------------------------------------


def test_phrase_match_route_in_openapi_schema(client):
    """The ``/exercises/phrase_match`` route is included in the
    auto-generated OpenAPI JSON.

    Verifies the route is listed at
    ``openapi.json#/paths/~1exercises~1phrase_match`` with a
    POST method and the correct request/response shapes.
    """
    schema = client.app.openapi()
    paths = schema.get("paths", {})
    assert "/exercises/phrase_match" in paths
    pm_path = paths["/exercises/phrase_match"]
    assert "post" in pm_path

    post = pm_path["post"]
    # Request body references the PhraseMatchGenerateRequest schema.
    request_body = post.get("requestBody", {})
    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    schema_ref = json_content.get("schema", {}).get("$ref", "")
    assert "PhraseMatchGenerateRequest" in schema_ref

    # Response 200 references the PhraseMatchExerciseOut schema.
    responses = post.get("responses", {})
    response_200 = responses.get("200", {})
    response_content = response_200.get("content", {})
    response_json = response_content.get("application/json", {})
    response_ref = response_json.get("schema", {}).get("$ref", "")
    assert "PhraseMatchExerciseOut" in response_ref

    # The 5-way additive widening is enforced at the Python level
    # (Pydantic v2 doesn't expose the inherited ``BaseExerciseFields``
    # as a separate OpenAPI component — it inlines the fields into
    # each subclass). Verify the source-level invariant directly
    # so a future refactor that drops the widening fails loudly.
    from app.schemas import BaseExerciseFields
    annotation = BaseExerciseFields.model_fields["exercise_type"].annotation
    # ``typing.Literal["cloze","matching","comprehension","idiom","phrase_match"]``
    # — get_args yields the literals in order.
    import typing
    args = typing.get_args(annotation)
    arg_values = tuple(typing.get_args(a) if typing.get_args(a) else a for a in args)
    flat = []
    for a in arg_values:
        sub = typing.get_args(a)
        if sub:
            flat.extend(sub)
        else:
            flat.append(a)
    assert "phrase_match" in flat, flat
    assert "cloze" in flat
    assert "matching" in flat
    assert "comprehension" in flat
    assert "idiom" in flat
    assert len(flat) == 5