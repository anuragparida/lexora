"""Unit tests for the OpenRouter embedding client.

Mocks httpx so the test never touches the network — even though
``OPENROUTER_API_KEY`` is set on the dev box, we don't want CI or
local pytest runs to consume OpenRouter quota. The mock mirrors the
shape of OpenRouter's actual response::

    {"data": [{"index": 0, "embedding": [...]}, ...]}

so the test also covers the sort-by-index code path.

Run from ``backend/``::

    uv run pytest -q tests/test_embeddings.py
"""
from __future__ import annotations

import os

import pytest
import respx
from httpx import Response

from app.embeddings import embed, embed_one, EmbeddingError


# Make sure the env-derived defaults don't blow up. The fixtures set
# OPENROUTER_API_KEY via the test runner; if not, skip.
@pytest.fixture(autouse=True)
def require_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")


def _mock_openrouter_response(vectors: list[list[float]], model: str = "qwen/qwen3-embedding-8b") -> dict:
    """Build an OpenRouter-shaped response body."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _unit_vector(seed: int, dim: int = 1024) -> list[float]:
    """Deterministic unit-ish vector for tests. Not a real unit
    vector, but the embedding client doesn't normalize — that's the
    provider's job. We just need 1024 floats per call."""
    return [0.001 * ((seed * (i + 1)) % 1000 - 500) for i in range(dim)]


@respx.mock
def test_embed_returns_one_vector_per_input():
    """Single batch: 3 inputs, 3 vectors out, in order."""
    vectors = [_unit_vector(seed=1), _unit_vector(seed=2), _unit_vector(seed=3)]
    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(200, json=_mock_openrouter_response(vectors))
    )
    out = embed(["a", "b", "c"])
    assert len(out) == 3
    for vec in out:
        assert len(vec) == 1024


@respx.mock
def test_embed_batches_when_input_exceeds_batch_size(monkeypatch):
    """Force a tiny batch_size to exercise the chunking loop.

    Note: ``DEFAULT_BATCH_SIZE`` is read at module import. Patch the
    env var BEFORE the app package loads in any other test that runs
    in the same process — for a single test session we patch the
    module-level attribute directly.
    """
    import app.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "DEFAULT_BATCH_SIZE", 2)
    # 5 inputs, batch=2 -> 3 batches (2+2+1)
    all_vectors = [_unit_vector(seed=i) for i in range(5)]
    # The client should call /embeddings 3 times.
    route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        side_effect=[
            Response(200, json=_mock_openrouter_response(all_vectors[0:2])),
            Response(200, json=_mock_openrouter_response(all_vectors[2:4])),
            Response(200, json=_mock_openrouter_response(all_vectors[4:5])),
        ]
    )
    out = embed(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert route.call_count == 3


@respx.mock
def test_embed_empty_input_short_circuits():
    """No network call when there is nothing to embed."""
    route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(200, json={})
    )
    out = embed([])
    assert out == []
    assert route.call_count == 0


@respx.mock
def test_embed_retries_on_429():
    """A single 429 then 200 — should succeed without raising."""
    vectors = [_unit_vector(seed=42)]
    route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        side_effect=[
            Response(429, json={"error": {"message": "rate limited"}}),
            Response(200, json=_mock_openrouter_response(vectors)),
        ]
    )
    out = embed(["probe"])
    assert len(out) == 1
    assert route.call_count == 2


@respx.mock
def test_embed_gives_up_after_max_attempts(monkeypatch):
    """Three consecutive 429s — should raise EmbeddingError."""
    import app.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "MAX_ATTEMPTS", 3)
    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(503, json={"error": "down"})
    )
    with pytest.raises(EmbeddingError):
        embed(["probe"])


@respx.mock
def test_embed_one_unwraps_single_element_list():
    """embed_one returns a flat list, not [[...]]."""
    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(200, json=_mock_openrouter_response([_unit_vector(seed=1)]))
    )
    vec = embed_one("hello")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert len(vec) == 1024


@respx.mock
def test_embed_passes_dimensions_param(monkeypatch):
    """When EMBEDDING_DIM is set, the request body includes
    ``dimensions`` so Matryoshka-capable providers (like
    qwen3-embedding-8b) project to the schema-matching dim."""
    import app.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "DEFAULT_DIM", 1024)
    route = respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(200, json=_mock_openrouter_response([_unit_vector(seed=1)]))
    )
    embed(["probe"])
    # Inspect the request body to confirm ``dimensions`` was sent.
    request = route.calls.last.request
    import json
    body = json.loads(request.content)
    assert body["dimensions"] == 1024
    assert body["model"] == "qwen/qwen3-embedding-8b"


@respx.mock
def test_embed_preserves_input_order_even_if_provider_reorders():
    """OpenRouter's response uses index-based ordering. Even if the
    provider returns the ``data`` array in a different order than the
    input list, ``out[i]`` must correspond to ``texts[i]``."""
    a_vec = _unit_vector(seed=0)
    b_vec = _unit_vector(seed=1)
    c_vec = _unit_vector(seed=2)
    # Mock returns data array in a non-sorted order: [c, b, a] instead
    # of [a, b, c]. The ``index`` field on each row correctly maps to
    # the original input position, so sorting by index should restore
    # input order.
    reordered = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 2, "embedding": c_vec},
            {"object": "embedding", "index": 1, "embedding": b_vec},
            {"object": "embedding", "index": 0, "embedding": a_vec},
        ],
        "model": "qwen/qwen3-embedding-8b",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(200, json=reordered)
    )
    out = embed(["a", "b", "c"])
    assert out[0] == a_vec
    assert out[1] == b_vec
    assert out[2] == c_vec


def test_missing_api_key_raises(monkeypatch):
    """No OPENROUTER_API_KEY -> EmbeddingError, no network call."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="OPENROUTER_API_KEY"):
        embed(["anything"])