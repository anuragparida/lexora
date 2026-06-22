"""Embedding client for Lexora.

Phase 1 plumbing — every Word and Example row gets a 1024-dim
embedding (computed offline via the backfill script; served via
``/retrieve`` at request time). Phase 4's exercise generator will
import ``embed()`` for on-demand query embedding; Phase 6 will use
the stored embeddings for retrieval-augmented prompts.

The transport is OpenAI-compatible (OpenRouter) — same wire format
as ``app/llm.py`` will use later. No LangChain.

## Why qwen/qwen3-embedding-8b and not baai/bge-m3?

The Phase 1 spec asked for ``baai/bge-m3`` (1024-d). The OpenRouter
probe at build time returned::

    {"error": {"message": "No endpoints available matching your
    guardrail restrictions and data policy. Configure:
    https://openrouter.ai/settings/privacy", "code": 404}}

i.e. the OpenRouter account's data-policy filter excludes that
provider. ``qwen/qwen3-embedding-8b`` is available on the same
account (it's already used by the honcho project per memory), returns
1024-dim vectors (matches the schema), and the cosine-distance
plumbing is identical. The model id is wired through ``EMBEDDING_MODEL``
so swapping back to bge-m3 later is a one-env-var change once the
account policy is relaxed.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


# Defaults live in env vars so the backend container picks them up
# without a code change. The .env.example documents the full list.
DEFAULT_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
DEFAULT_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# The spec'd batch size. OpenRouter rate-limits generously; 32 is
# small enough to keep per-request latency low, large enough to make
# the backfill ~10 minutes for the 12k-word + 12k-example corpus.
DEFAULT_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Retry policy: 3 attempts on 429/5xx with exponential backoff
# (0.5s, 1s, 2s). Below the max-attempts threshold we just bubble the
# exception — the caller decides whether to fail the whole backfill or
# skip the failing batch.
MAX_ATTEMPTS = int(os.getenv("EMBEDDING_MAX_ATTEMPTS", "3"))
RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504)


class EmbeddingError(RuntimeError):
    """Raised when the OpenRouter embedding endpoint fails unrecoverably."""


def _get_api_key() -> str:
    """Resolve the OpenRouter API key from env.

    The var name ``OPENROUTER_API_KEY`` is intentional — env var
    names containing ``SECRET`` / ``TOKEN`` trigger harness redaction
    on this account even when the value isn't a secret. See the
    devops/hermes-ecosystem-gotchas skill for the full list.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EmbeddingError(
            "OPENROUTER_API_KEY is not set. Add it to ~/.lexora/.env "
            "and restart the backend container."
        )
    return key


def _embed_one_batch(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    dimensions: int | None = None,
) -> list[list[float]]:
    """POST one batch to /embeddings with retries on transient errors.

    Returns vectors in the same order as the input list. Raises
    ``EmbeddingError`` on non-retryable failures or after the retry
    budget is exhausted.

    The optional ``dimensions`` parameter requests a Matryoshka
    projection of the embedding (qwen3-embedding-8b supports this).
    Most embedding models emit a fixed dim; only providers that
    accept the ``dimensions`` field will honor it. The default
    ``EMBEDDING_DIM`` env var is passed in here so the schema's
    ``vector(N)`` and the actual vector dimension always agree.
    """
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {"model": model, "input": texts}
    if dimensions is not None:
        payload["dimensions"] = dimensions

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.post(url, json=payload, headers=headers, timeout=60.0)
            if resp.status_code in RETRYABLE_STATUS:
                raise EmbeddingError(
                    f"retryable HTTP {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            data = resp.json()
            # OpenRouter returns {"data": [{"embedding": [...]}, ...]}
            # keyed by index. Sort by index so the caller's input
            # order is preserved even if the provider reorders.
            items = sorted(data["data"], key=lambda d: d["index"])
            return [item["embedding"] for item in items]
        except (httpx.HTTPError, EmbeddingError) as exc:
            last_exc = exc
            if attempt >= MAX_ATTEMPTS:
                break
            backoff = 0.5 * (2 ** (attempt - 1))
            logger.warning(
                "embed: attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                exc,
                backoff,
            )
            time.sleep(backoff)

    raise EmbeddingError(
        f"embed: gave up after {MAX_ATTEMPTS} attempts; last error: {last_exc}"
    )


def embed(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """Embed a list of strings.

    Returns a list of float vectors (1024-d by default) in the same
    order as ``texts``. Empty input returns ``[]`` without making a
    network call.

    Parameters
    ----------
    texts
        Strings to embed. Empty strings are passed through (the
        provider returns a valid vector for them) — callers that want
        to skip empties should filter first.
    model, batch_size, base_url
        Override the env-derived defaults. Mostly useful for tests.
    """
    if not texts:
        return []

    api_key = _get_api_key()
    use_model = model or DEFAULT_MODEL
    use_batch = batch_size or DEFAULT_BATCH_SIZE
    use_base = base_url or DEFAULT_BASE_URL

    out: list[list[float]] = []
    # Reuse one HTTP client across batches — connection pooling pays
    # off during the backfill (~750 batches for 12k words + examples).
    with httpx.Client() as client:
        for start in range(0, len(texts), use_batch):
            chunk = texts[start : start + use_batch]
            vectors = _embed_one_batch(
                client,
                use_base,
                api_key,
                use_model,
                chunk,
                dimensions=DEFAULT_DIM,
            )
            out.extend(vectors)
            logger.info(
                "embed: %d/%d texts embedded (batch %d)",
                len(out),
                len(texts),
                (start // use_batch) + 1,
            )
    return out


def embed_one(text: str, **kwargs) -> list[float]:
    """Convenience: embed a single string.

    Used by ``/retrieve`` at request time. The provider accepts a
    one-element list and returns a one-element list; we unwrap here.
    """
    return embed([text], **kwargs)[0]