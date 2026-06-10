from src.utils.wl import wl_graph_signature
from src.datasets.expressivity import make_csl_single, make_shrikhande, make_rook_4x4


def test_csl_classes_share_one_signature():
    # Any two CSL skips collapse to the same 1-WL signature (all vertices one color).
    sigs = [wl_graph_signature(make_csl_single(41, s), 41) for s in (2, 3, 16)]
    assert sigs[0] == sigs[1] == sigs[2]
    # And that single color spans all 41 vertices
    assert sum(sigs[0].values()) == 41
    assert len(sigs[0]) == 1


def test_shrikhande_and_rook_share_signature():
    sh = wl_graph_signature(make_shrikhande(), 16)
    rk = wl_graph_signature(make_rook_4x4(), 16)
    assert sh == rk


def test_wl_distinguishes_obviously_different_graphs():
    # path P3 (0-1-2) vs triangle (0-1-2-0) must differ
    import torch
    p3 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    tri = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    assert wl_graph_signature(p3, 3) != wl_graph_signature(tri, 3)
