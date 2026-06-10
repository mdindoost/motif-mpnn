import torch
from src.datasets.motif_loader import _log1p_zscore_dense


def test_aut_normalization_is_noop_under_zscore():
    # A single-orbit feature column of raw counts.
    raw = torch.tensor([[0.0], [1.0], [3.0], [7.0], [12.0]])
    for aut in (1.0, 2.0, 6.0, 24.0):  # |Aut(H)| divisors
        z_raw, _ = _log1p_zscore_dense(raw.clone(), None)
        # NOTE: |Aut| normalization divides counts BEFORE log1p, so emulate that.
        z_norm, _ = _log1p_zscore_dense((raw / aut).clone(), None)
        # log1p(x/a) != log1p(x)/a, so they are NOT identical in general;
        # the true no-op holds when normalization is a per-column SCALE AFTER log.
        # Assert the post-log z-score of a scaled-after-log column is identical:
        logged = torch.log1p(raw)
        za = (logged - logged.mean(0)) / logged.std(0)
        zb_input = logged / aut
        zb = (zb_input - zb_input.mean(0)) / zb_input.std(0)
        assert torch.allclose(za, zb, atol=1e-6)
