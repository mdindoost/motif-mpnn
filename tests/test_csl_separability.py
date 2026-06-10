import pytest
import torch
from src.datasets.expressivity import make_csl_single, CSL_SKIPS
from src.datasets.substructure_counts import simple_cycles_per_node
import networkx as nx

pytestmark = pytest.mark.slow

L_MAX = 8  # keep in sync with generate_motifs.CYCLE_L_MAX; bump if this test fails


def _graph_cycle_vector(s, l_max):
    ei = make_csl_single(41, s)
    g = nx.Graph()
    g.add_nodes_from(range(41))
    for u, v in ei.t().tolist():
        if u != v:
            g.add_edge(int(u), int(v))
    spn = simple_cycles_per_node(g, 41, l_max=l_max)
    # graph-level cycle spectrum: total participations per length / length
    vec = [0.0] * (l_max + 1)
    for _, by_len in spn.items():
        for L, c in by_len.items():
            vec[L] += c
    return torch.tensor([vec[L] / max(L, 1) for L in range(3, l_max + 1)])


def test_cycle_spectrum_separates_all_csl_classes():
    vecs = [_graph_cycle_vector(s, L_MAX) for s in CSL_SKIPS]
    mat = torch.stack(vecs)
    # all 10 class signatures must be pairwise distinct
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            assert not torch.allclose(mat[i], mat[j]), (
                f"classes {i},{j} collide at L_MAX={L_MAX}; increase L_MAX")


def test_short_cycles_alone_do_not_separate_all_classes():
    # Bounded library (cycles of length <= 4) leaves some CSL classes colliding.
    vecs = [_graph_cycle_vector(s, l_max=4) for s in CSL_SKIPS]
    mat = torch.stack(vecs)
    collisions = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if torch.allclose(mat[i], mat[j]):
                collisions += 1
    # At least one pair collides at L<=4 — bounded structures are insufficient.
    assert collisions > 0, "short cycles unexpectedly separated all classes; revise the characterization"
