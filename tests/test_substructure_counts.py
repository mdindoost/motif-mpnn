import networkx as nx
from src.datasets.substructure_counts import (
    triangles_per_node, four_cliques_per_node, simple_cycles_per_node,
)
from src.datasets.expressivity import make_shrikhande, make_rook_4x4
from src.datasets.expressivity import _edge_index_to_igraph  # reuse igraph helper


def _ig(ei, n):
    return _edge_index_to_igraph(ei, n)


def test_triangles_on_k4():
    g = nx.complete_graph(4)
    tpn = triangles_per_node(g, 4)
    # each vertex of K4 is in C(3,2)=3 triangles
    assert tpn == {0: 3, 1: 3, 2: 3, 3: 3}


def test_four_cliques_on_k4():
    g = nx.complete_graph(4)
    fpn = four_cliques_per_node(g, 4)
    assert fpn == {0: 1, 1: 1, 2: 1, 3: 1}


def test_simple_cycles_on_c5():
    g = nx.cycle_graph(5)
    spn = simple_cycles_per_node(g, 5, l_max=5)
    # only one simple cycle, length 5, through every node
    for v in range(5):
        assert spn[v].get(5, 0) == 1
        assert spn[v].get(3, 0) == 0
        assert spn[v].get(4, 0) == 0


def test_srg_pair_triangles_equal_cliques_differ():
    import torch
    sh = make_shrikhande(); rk = make_rook_4x4()
    g_sh = nx.Graph(sorted({tuple(sorted((u, v))) for u, v in sh.t().tolist() if u != v}))
    g_rk = nx.Graph(sorted({tuple(sorted((u, v))) for u, v in rk.t().tolist() if u != v}))
    # total triangles: 32 each
    assert sum(triangles_per_node(g_sh, 16).values()) // 3 == 32
    assert sum(triangles_per_node(g_rk, 16).values()) // 3 == 32
    # 4-cliques: 0 vs 8 total; per-node 0 vs 2
    assert sum(four_cliques_per_node(g_sh, 16).values()) == 0
    assert sum(four_cliques_per_node(g_rk, 16).values()) == 8 * 4  # 8 cliques * 4 nodes
    assert set(four_cliques_per_node(g_rk, 16).values()) == {2}
