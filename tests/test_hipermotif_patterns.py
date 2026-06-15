"""Local unit tests for the HiPerMotif orbit-feature backend.

arkouda/arachne are not installed locally, so the cluster path itself cannot run here.
These tests validate everything around it: the pattern library (|Aut|, orbit partitions,
ORCA-orbit mapping), the per-vertex normalization math (corrected contract: collapse
positions then / |Aut(H)|), the FULL 15-orbit mapping against ORCA as oracle on real
graphs (not just the 0/3/14 anchors), and the import-guard.
"""
import numpy as np
import networkx as nx
import pytest

from src.datasets import hipermotif_patterns as hp
from src.datasets.orca_orbits import count_orbits as orca_count, DEGREE_ORBIT, TRIANGLE_ORBIT, CLIQUE4_ORBIT


# --- |Aut(H)| values (known) -------------------------------------------------
EXPECTED_AUT = {"edge": 2, "p3": 2, "triangle": 6, "p4": 2, "claw": 6,
                "paw": 2, "c4": 8, "diamond": 4, "k4": 24}


def test_automorphism_counts():
    for name, exp in EXPECTED_AUT.items():
        naut, _ = hp.automorphism_count_and_orbits(name)
        assert naut == exp, f"{name}: |Aut|={naut} expected {exp}"


def test_orbit_table_is_bijection_to_0_14_with_aut_divisor():
    table = hp.ORBIT_TABLE
    assert [s.orca_orbit for s in table] == list(range(15))
    # divisor stored is the FULL |Aut(H)| (corrected contract), and motif_id/k consistent
    for s in table:
        assert s.n_aut == EXPECTED_AUT[s.pattern]
        assert s.motif_id == 2000 + s.orca_orbit
        assert s.k == hp.PATTERNS[s.pattern][1]


def test_anchor_orbits_match_orca_constants():
    by_orca = {s.orca_orbit: s for s in hp.ORBIT_TABLE}
    assert by_orca[DEGREE_ORBIT].pattern == "edge"
    assert by_orca[TRIANGLE_ORBIT].pattern == "triangle"
    assert by_orca[CLIQUE4_ORBIT].pattern == "k4"


# --- mocked-normalization: K4 calibration ------------------------------------
def test_mocked_normalization_k4_anchors():
    """Feed induced embeddings (the mock of subgraph_isomorphism output) for K4 and
    assert the corrected normalization yields ORCA's K4 anchors: degree=3, triangle=3,
    4-clique=1 per vertex."""
    K4 = nx.complete_graph(4)

    edge_emb, edge_map = hp.induced_embeddings(K4, "edge")   # 6 edges x 2 orderings = 12
    assert edge_emb.shape == (12, 2)
    deg = hp.normalize_embeddings("edge", edge_emb, 4, edge_map)
    assert np.array_equal(deg[DEGREE_ORBIT], np.full(4, 3))

    tri_emb, tri_map = hp.induced_embeddings(K4, "triangle")  # 4 triangles x 6 = 24
    assert tri_emb.shape == (24, 3)
    tri = hp.normalize_embeddings("triangle", tri_emb, 4, tri_map)
    assert np.array_equal(tri[TRIANGLE_ORBIT], np.full(4, 3))

    k4_emb, k4_map = hp.induced_embeddings(K4, "k4")          # 1 clique x 24 = 24
    assert k4_emb.shape == (24, 4)
    clq = hp.normalize_embeddings("k4", k4_emb, 4, k4_map)
    assert np.array_equal(clq[CLIQUE4_ORBIT], np.full(4, 1))


# --- the MANDATORY check: all 15 orbits == ORCA on real graphs ---------------
@pytest.mark.parametrize("gname,G", [
    ("K4", nx.complete_graph(4)),
    ("C5", nx.cycle_graph(5)),
    ("rand_a", nx.gnp_random_graph(9, 0.45, seed=1)),
    ("rand_b", nx.gnp_random_graph(11, 0.35, seed=7)),
    ("petersen", nx.petersen_graph()),
])
def test_all_15_orbits_equal_orca(gname, G):
    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()
    emb = {name: hp.induced_embeddings(G, name) for name in hp.PATTERN_NAMES}
    X = hp.orbit_matrix_from_embeddings(emb, N)             # HiPerMotif-normalization path
    O = orca_count([(int(u), int(v)) for u, v in G.edges()], N)
    assert X.shape == O.shape == (N, 15)
    assert np.array_equal(X, O), f"{gname}: mismatch at {np.argwhere(X != O)[:5].tolist()}"


# --- the regression test that would have caught the Wulver gate failure ------
@pytest.mark.parametrize("pattern", ["p3", "paw", "p4", "claw", "diamond"])
def test_remap_required_for_multiorbit_patterns(pattern):
    """reorder_type="structural" permutes the pattern's vertices, so output column j holds
    original pattern vertex mapper[j], NOT j. For multi-orbit patterns this crosses orbits;
    ignoring the mapper mixes orbit tallies (the bug that failed the gate on Cora/Shrikhande
    while the old identity-ordered mock passed). This asserts: (a) the mock perm is genuinely
    non-identity so the remap is exercised; (b) WITH the mapper the counts match ORCA; (c)
    WITHOUT the mapper the result is detectably wrong (raises on |Aut| divisibility, or differs
    from ORCA) — i.e. the bug cannot pass silently."""
    perm = hp.mock_structural_perm(pattern)
    assert not np.array_equal(perm, np.arange(len(perm))), \
        f"{pattern} mock perm is identity — the test would not exercise the remap"

    G = nx.convert_node_labels_to_integers(nx.gnp_random_graph(10, 0.45, seed=3))
    N = G.number_of_nodes()
    emb, mapper = hp.induced_embeddings(G, pattern)
    assert emb.size > 0, f"test host has no induced {pattern}; pick a different graph"
    O = orca_count([(int(u), int(v)) for u, v in G.edges()], N)

    # (b) WITH the mapper: every orbit of this pattern matches the ORCA oracle.
    got = hp.normalize_embeddings(pattern, emb, N, mapper)
    for orca_id, vec in got.items():
        assert np.array_equal(vec, O[:, orca_id]), \
            f"{pattern} ORCA orbit {orca_id} mismatches with mapper applied"

    # (c) WITHOUT the mapper (the bug): must NOT silently match ORCA.
    try:
        bad = hp.normalize_embeddings(pattern, emb, N)  # identity column assumption
    except ValueError:
        return  # divisibility guard fired — the bug is detected, as intended
    assert any(not np.array_equal(bad[o], O[:, o]) for o in bad), \
        f"{pattern}: ignoring the mapper still matched ORCA — remap not actually exercised"


# --- import-guard ------------------------------------------------------------
def test_backend_imports_without_arkouda():
    """The backend module must import cleanly without arkouda/arachne present, and only
    fail (with a clear Wulver message) when the cluster path is actually invoked."""
    import importlib
    mod = importlib.import_module("src.datasets.hipermotif_backend")
    assert hasattr(mod, "count_orbits_hipermotif")
    with pytest.raises(RuntimeError, match="Wulver|arkouda|arachne"):
        mod.count_orbits_hipermotif([(0, 1), (1, 2), (0, 2)], 3)
