# Lexora backend Makefile.
#
# Operational entry points for the offline eval runners. The CLI
# scripts under ``scripts/`` are the source of truth; these
# aliases keep the operational surface consistent (Phase 6.7 +
# Phase 7.5 spec, PHASE-7.md gotcha #11).
#
# Run from the repo root::
#
#     make eval-ragas                # cloze+matching+comprehension, dry-run
#     make eval-retrieval-compare    # current vs bge-m3, dry-run
#
# Or pass --live for the live path (requires OPENROUTER_API_KEY
# + a warm bge-m3 cache for the retrieval compare).

.PHONY: eval-ragas eval-retrieval-compare

# Phase 6.7 Ragas regression detector.
# Default mode (--dry-run) is hermetic; --live requires the keys.
eval-ragas:
	cd backend && uv run python -m scripts.eval_ragas --dry-run

# Phase 7.5 Retrieval-quality A/B runner (current vs bge-m3).
# Default mode (--dry-run) is hermetic; --live requires the bge-m3
# cache + the OpenRouter key. Mirrors eval-ragas shape.
# The cloze held-out JSONL lives at the repo root's eval/ dir; the
# runner is invoked from backend/ (so ``app.eval.retrieval_compare``
# resolves), so we resolve the judgments path relative to the
# parent of backend/.
eval-retrieval-compare:
	cd backend && uv run python -m scripts.eval_retrieval_compare \
	    --judgments ../eval/cloze_judgments.jsonl \
	    --out ../eval/retrieval_compare/