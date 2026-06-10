import torch
import networkx as nx
from src.datasets.expressivity import make_shrikhande, make_rook_4x4
from src.utils.wl import wl_graph_signature
from src.datasets.substructure_counts import triangles_per_node, four_cliques_per_node
from src.models.gcn import GCN


def _nx(ei, n):
    g = nx.Graph(); g.add_nodes_from(range(n))
    for u, v in ei.t().tolist():
        if u != v:
            g.add_edge(int(u), int(v))
    return g


def test_1wl_cannot_separate_but_4clique_can():
    sh, rk = make_shrikhande(), make_rook_4x4()
    # 1-WL identical
    assert wl_graph_signature(sh, 16) == wl_graph_signature(rk, 16)
    g_sh, g_rk = _nx(sh, 16), _nx(rk, 16)
    # triangles identical (32 each)
    assert sum(triangles_per_node(g_sh, 16).values()) // 3 == 32
    assert sum(triangles_per_node(g_rk, 16).values()) // 3 == 32
    # 4-cliques separate them: 0 vs 8
    assert sum(four_cliques_per_node(g_sh, 16).values()) == 0
    assert sum(four_cliques_per_node(g_rk, 16).values()) // 4 == 8


def test_gcn_embeddings_identical_on_constant_features():
    # A GCN with constant node features yields identical graph readouts for the pair
    # (consequence of 1-WL equivalence): pooled embeddings must match closely.
    torch.manual_seed(0)
    sh, rk = make_shrikhande(), make_rook_4x4()
    model = GCN(in_dim=1, out_dim=4, hidden_dim=8, num_layers=2,
                dropout=0.0, layer_norm=False, residual=False, task="graph")
    model.eval()
    def pooled(ei):
        from torch_geometric.nn import global_mean_pool
        x = torch.ones(16, 1)
        batch = torch.zeros(16, dtype=torch.long)
        h = model.encode(x, ei)
        return global_mean_pool(h, batch)
    with torch.no_grad():
        e_sh, e_rk = pooled(sh), pooled(rk)
    assert torch.allclose(e_sh, e_rk, atol=1e-5)
