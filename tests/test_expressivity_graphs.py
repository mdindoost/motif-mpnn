import torch
from src.datasets.expressivity import (
    make_csl, make_csl_single, make_shrikhande, make_rook_4x4, _edge_index_to_igraph,
)


def _degrees(edge_index, n):
    deg = torch.zeros(n, dtype=torch.long)
    for u, v in edge_index.t().tolist():
        deg[u] += 1
    return deg


def test_csl_single_graph_is_4_regular():
    ei = make_csl_single(n=41, s=3)
    n = 41
    deg = _degrees(ei, n)
    assert deg.tolist() == [4] * n  # undirected, each row counted once per direction


def test_csl_dataset_shape():
    graphs = make_csl(n=41, copies_per_class=15, seed=0)
    assert len(graphs) == 150
    labels = sorted({lbl for _, lbl in graphs})
    assert labels == list(range(10))
    # 15 copies per class
    counts = {}
    for _, lbl in graphs:
        counts[lbl] = counts.get(lbl, 0) + 1
    assert all(c == 15 for c in counts.values())


def test_shrikhande_is_srg_16_6_2_2():
    ei = make_shrikhande()
    g = _edge_index_to_igraph(ei, 16)
    assert g.vcount() == 16
    assert all(d == 6 for d in g.degree())
    # 4-clique count must be 0
    assert len(g.cliques(min=4, max=4)) == 0


def test_rook_is_srg_16_6_2_2():
    ei = make_rook_4x4()
    g = _edge_index_to_igraph(ei, 16)
    assert g.vcount() == 16
    assert all(d == 6 for d in g.degree())
    # rook has exactly 8 four-cliques (4 rows + 4 columns)
    assert len(g.cliques(min=4, max=4)) == 8
