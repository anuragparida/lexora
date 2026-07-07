# Phase 10.1 — test-path stub for the seed script's similarity function.
#
# The production similarity path uses bge-m3 cosine via the local
# sentence-transformers cache (Phase 1.3 offline path Phase 7.5 A/B
# sets up). The CI / dev path doesn't have that dependency, so
# ``tests/test_seed_phrase_pairs.py`` invokes the seed script with
# ``--similarity-fn tests.stub_seed_similarity:stub_similarity``.
#
# The stub returns 0.7 for any input pair — above the 0.55 threshold
# the seed script enforces, so every (i, j) phrase pair lands in the
# candidate pool. The bucket assignment is then driven entirely by
# (seed, sorted input) and the deterministic rank-quartile split
# runs as designed.
"""Stub similarity function for offline seed-script tests."""


def stub_similarity(text_a: str, text_b: str) -> float:
    """Return a constant 0.7 for any input pair.

    Phase 10.1's bge-m3 cosine lives behind
    ``--similarity-fn custom_module:fn`` so the CI / test
    path doesn't need sentence-transformers. 0.7 is
    deliberately above the SIMILARITY_THRESHOLD (0.55) so
    every (i, j) phrase pair enters the candidate pool.
    """
    # Ignore text inputs — the function is intentionally
    # constant. The point of the test path is to exercise
    # the seed script's bucket assignment / ordering /
    # idempotency logic, not the embedding model.
    return 0.7
