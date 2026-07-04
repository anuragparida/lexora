"""Lexora eval package.

Phase 6.7 deliverable. Houses the Ragas offline runner constants +
helpers, plus the Phase 7.5 retrieval-quality A/B runner. The
actual runners live under ``backend/scripts/`` (``eval_ragas`` +
``eval_retrieval_compare``) to mirror the Phase 4.4
``eval_cloze.py`` shape (script under ``scripts/``, library under
``app/``).
"""
from app.eval import ragas, retrieval_compare  # noqa: F401
