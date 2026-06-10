import numpy as np
import networkx as nx
import pytest

from src.datasets import orca_orbits as oo


def test_constants_present():
    assert oo.N_ORBITS == {4: 15, 5: 73}
    assert oo.ORCA_ORBIT_BASE == 2000
    assert oo.DEGREE_ORBIT == 0
    # triangle and 4-clique orbit indices are the documented ORCA size-4 numbering;
    # Task 3 validates them against hand-computed graphs (the rigor gate).
    assert oo.TRIANGLE_ORBIT == 3
    assert oo.CLIQUE4_ORBIT == 14


def test_orbit_graphlet_size_table():
    t = oo.ORBIT_GRAPHLET_SIZE
    assert len(t) == 15
    assert set(t) <= {2, 3, 4}
    assert t[oo.DEGREE_ORBIT] == 2      # edge endpoint -> 2-node graphlet
    assert t[oo.TRIANGLE_ORBIT] == 3    # triangle -> 3-node graphlet
    assert t[oo.CLIQUE4_ORBIT] == 4     # K4 -> 4-node graphlet


def test_orbit_to_km():
    # degree orbit -> (k=2, motif_id=2000); 4-clique orbit -> (k=4, motif_id=2014)
    assert oo.orbit_to_km(0) == (2, 2000)
    assert oo.orbit_to_km(oo.TRIANGLE_ORBIT) == (3, 2003)
    assert oo.orbit_to_km(oo.CLIQUE4_ORBIT) == (4, 2014)


def test_ensure_orca_built_idempotent():
    p1 = oo.ensure_orca_built()
    assert p1.exists()
    p2 = oo.ensure_orca_built()  # second call is a no-op
    assert p2 == p1 and p2.exists()


def _edges(g):
    return list(g.edges())


def test_k4_orbits():
    # K4: every node has degree 3, is in C(3,2)=3 triangles, and 1 four-clique.
    g = nx.complete_graph(4)
    m = oo.count_orbits(_edges(g), 4)
    assert m.shape == (4, 15)
    assert np.all(m[:, oo.DEGREE_ORBIT] == 3)
    assert np.all(m[:, oo.TRIANGLE_ORBIT] == 3)
    assert np.all(m[:, oo.CLIQUE4_ORBIT] == 1)


def test_c5_orbits():
    # 5-cycle: degree 2 everywhere; no triangles, no 4-cliques.
    g = nx.cycle_graph(5)
    m = oo.count_orbits(_edges(g), 5)  # 2nd arg is num_nodes (=5), NOT graphlet_size
    assert m.shape == (5, 15)
    assert np.all(m[:, oo.DEGREE_ORBIT] == 2)
    assert np.all(m[:, oo.TRIANGLE_ORBIT] == 0)
    assert np.all(m[:, oo.CLIQUE4_ORBIT] == 0)


def test_degree_orbit_matches_networkx():
    # Cross-check orbit 0 against an independent tool on a random small graph.
    g = nx.gnp_random_graph(20, 0.3, seed=7)
    m = oo.count_orbits(_edges(g), 20)
    deg = np.array([d for _, d in sorted(g.degree())], dtype=np.int64)
    assert np.array_equal(m[:, oo.DEGREE_ORBIT], deg)


def test_isolated_node_is_zero_row():
    # Node 3 is isolated; its entire orbit row must be zero.
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])  # triangle on 0,1,2
    m = oo.count_orbits(_edges(g), 4)
    assert np.all(m[3, :] == 0)
    assert m[0, oo.TRIANGLE_ORBIT] == 1


@pytest.mark.slow
def test_srg_method_validation():
    # The method-validation oracle: Shrikhande vs 4x4-rook are 1-WL-identical with
    # equal triangle counts but differ in 4-cliques (0 vs 2 per node).
    from src.datasets.expressivity import make_shrikhande, make_rook_4x4

    def edges_of(ei):
        return [(int(u), int(v)) for u, v in ei.t().tolist() if u != v]

    sh = oo.count_orbits(edges_of(make_shrikhande()), 16)
    rk = oo.count_orbits(edges_of(make_rook_4x4()), 16)
    # triangle orbit: equal totals, both nonzero
    assert sh[:, oo.TRIANGLE_ORBIT].sum() == rk[:, oo.TRIANGLE_ORBIT].sum()
    assert sh[:, oo.TRIANGLE_ORBIT].sum() > 0
    # 4-clique orbit separates the cospectral pair
    assert np.all(sh[:, oo.CLIQUE4_ORBIT] == 0)
    assert np.all(rk[:, oo.CLIQUE4_ORBIT] == 2)
