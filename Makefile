# Lexora backend Makefile.
#
# Operational entry points for the offline eval runners. The CLI
# scripts under ``scripts/`` are the source of truth; these
# aliases keep the operational surface consistent (Phase 6.7 +
# Phase 7.5 spec, PHASE-7.md gotcha #11).
#
# Run from the repo root::
#
#     make eval-ragas                       # cloze+matching+comprehension, dry-run
#     make eval-retrieval-compare           # current vs bge-m3, dry-run
#     make eval-optimize-match              # offline DSPy optimizer (DummyLM)
#     make eval-optimize-all                # cloze + matching + comprehension wrappers
#
# Or pass --live for the live path (requires OPENROUTER_API_KEY
# + a warm bge-m3 cache for the retrieval compare). Live optimizer
# runs are gated on the Phase 6.7 Ragas floor (card t_52ef2d50 +
# card t_bdd9ffbe).

.PHONY: eval-ragas eval-retrieval-compare \
        eval-optimize-cloze eval-optimize-match eval-optimize-comprehension \
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

# Phase 4.2 — Cloze DSPy optimizer CLI (card t_bdd9ffbe).
# Offline by default; DummyLM is used unless OPENROUTER_API_KEY is
# set AND --live is passed. Phase 4.4's optimizer artefact lands at
# backend/app/cloze_optimized.json (gitignored).
eval-optimize-cloze:
	cd backend && uv run python -m scripts.optimize_cloze

# Phase 9.3 — Matching DSPy optimizer CLI (card t_52ef2d50).
# Offline by default; DummyLM is used unless OPENROUTER_API_KEY is
# set AND --live is passed. Reuses the Phase 6 match_judgments.jsonl
# held-out set (40 accept rows, template-based per Phase 6 hard
# rule #8). Optimizer artefact lands at backend/app/match_optimized.json
# (gitignored).
eval-optimize-match:
	cd backend && uv run python -m scripts.optimize_match

# Phase 9.4 — Comprehension DSPy optimizer CLI (companion card,
# not yet shipped).
# Stub entry point that fails fast with a clear pointer; once
# scripts/optimize_comprehension.py lands, the body flips to
# match the eval-optimize-match shape line-for-line (card body
# scope explicitly defers 9.4 to a sibling card).
eval-optimize-comprehension:
	@echo "eval-optimize-comprehension: Phase 9.4 not yet shipped." >&2
	@echo "scripts/optimize_comprehension.py is the companion card; this" >&2
	@echo "target becomes live in Phase 9.4." >&2
	@exit 1

# Phase 9 — Run every per-type optimizer. The cloze target predates
# 9.3 but is wired here for symmetry. The comprehension target is
# the Phase 9.4 stub above; running ``make eval-optimize-all``
# before 9.4 lands fails fast on the comprehension step.
eval-optimize-all: eval-optimize-cloze eval-optimize-match eval-optimize-comprehension