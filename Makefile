# Lexora backend Makefile.
#
# Operational entry points for the offline eval runners. The CLI
# scripts under ``scripts/`` are the source of truth; these
# aliases keep the operational surface consistent (Phase 6.7 +
# Phase 7.5 spec, PHASE-7.md gotcha #11).
#
# Run from the repo root::
#
#     make eval-ragas                  # cloze+matching+comprehension, dry-run
#     make eval-retrieval-compare      # current vs bge-m3, dry-run
#     make eval-optimize-cloze         # Phase 4.4 MIPROv2 optimizer (offline)
#     make eval-optimize-match         # Phase 9.3 MIPROv2 optimizer (offline)
#     make eval-optimize-comprehension # Phase 9.4 MIPROv2 optimizer (offline)
#     make eval-optimize-phrase-match  # Phase 10.7 optimizer (offline)
#     make eval-optimize-all           # chain cloze + match + comprehension + phrase_match
#
# Or pass --live for the live path (requires OPENROUTER_API_KEY
# + a warm bge-m3 cache for the retrieval compare). Live-LLM runs
# for the optimizers are gated on the Phase 6.7 Ragas floor; do
# not pass --live until that floor is green.

.PHONY: eval-ragas eval-retrieval-compare \
        eval-optimize-cloze eval-optimize-match eval-optimize-comprehension \
        eval-optimize-phrase-match \
        eval-optimize-all

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

# Phase 4.4 / 9.3 / 9.4 — DSPy prompt optimizers.
# Default mode (no --live) is the DummyLM offline path so the CLI
# plumbing runs end-to-end without network. Pass --live only after
# the Phase 6.7 Ragas floor is green. Each per-type target mirrors
# the eval-ragas shape (one script invocation, exit code is the
# gate).
eval-optimize-cloze:
	cd backend && uv run python -m scripts.optimize_cloze

eval-optimize-match:
	cd backend && uv run python -m scripts.optimize_match

eval-optimize-comprehension:
	cd backend && uv run python -m scripts.optimize_comprehension

# Phase 10.7 — DSPy phrase_match optimizer.
# Default mode (no --live) is the DummyLM offline path so the CLI
# plumbing runs end-to-end without network. Pass --live only after
# the Phase 6.7 Ragas floor is green. Mirrors the eval-optimize-match
# shape: same DummyLM discipline, same --live gate, same artifact
# path (``backend/app/phrase_match_optimized.json``).
eval-optimize-phrase-match:
	cd backend && uv run python -m scripts.optimize_phrase_match

# Chain cloze + match + comprehension + phrase_match. ``make -k``
# keeps going on a per-target failure so one broken script doesn't
# block the others (the operator inspects each artifact
# individually anyway).
eval-optimize-all:
	$(MAKE) -k eval-optimize-cloze
	$(MAKE) -k eval-optimize-match
	$(MAKE) -k eval-optimize-comprehension
	$(MAKE) -k eval-optimize-phrase-match