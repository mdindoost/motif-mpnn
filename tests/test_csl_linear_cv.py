"""Reproduces the paper's Section III CSL separability claim as an actual experiment:
a LINEAR classifier on per-graph cycle-spectrum features, 5-fold CV, clean and under
2% multiplicative noise. CSL is a synthetic, theory-defined expressivity construct.

This test asserts only separability + robustness (deterministic); it does NOT target
the draft's specific 99.5% figure — the script reports the actual numbers.
"""
import pytest

pytestmark = pytest.mark.slow

from scripts.expressivity.csl_linear_cv import run


def test_csl_cycle_spectrum_linearly_separable():
    res = run()
    assert res["n_classes"] == 10
    assert res["n_features"] == 6  # cycle lengths 3..8
    # The 10 classes share one 1-WL coloring but have 10 distinct cycle-spectrum
    # signatures, so a linear classifier separates them exactly when clean.
    assert res["clean_mean"] == 1.0
    # Robust to the draft's 2% multiplicative noise (honest floor, not a 99.5% target).
    assert res["noisy_mean"] >= 0.99
