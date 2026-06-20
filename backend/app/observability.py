"""Langfuse observability wrapper for the Lexora backend.

Phase 0 wires the client and configures it from env vars, but no
call sites use it yet. Phase 4 (LLM exercise generator) imports
``get_langfuse()`` and decorates generation paths.

The dedicated ``lexora`` Langfuse project must exist on the shared
instance with project-scoped API keys in ``~/.lexora/.env`` (NOT
in the repo). See README §"Observability" for the setup steps.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _has_keys() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def get_langfuse():
    """Return a configured Langfuse client, or ``None`` if not configured.

    Returning ``None`` (rather than raising) keeps the rest of the
    app bootable when keys are missing — useful for local dev and for
    Phase 0's "plumbing loaded but unused" state. Phase 4 callers
    should treat ``None`` as "tracing disabled" and proceed without
    LLM observability.
    """
    if not _has_keys():
        logger.warning(
            "Langfuse keys missing — observability disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in "
            "~/.lexora/.env to enable tracing."
        )
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.error(
            "langfuse package not installed; pip install langfuse to enable tracing."
        )
        return None

    host = os.getenv("LANGFUSE_HOST", "http://langfuse-web:3000")
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=host,
    )


# Module-level singleton, lazily initialized on first import. Importing
# this module alone is safe — it does not contact Langfuse until
# ``get_langfuse()`` is called or ``client`` is touched.
client: Optional[object] = None


def _ensure_client():
    global client
    if client is None:
        client = get_langfuse()
    return client