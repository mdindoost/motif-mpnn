"""Size-4 graphlet-orbit pattern library for the HiPerMotif backend.

Maps each of ORCA's 15 size-4 orbits (indices 0..14) to a (graphlet pattern,
automorphism-orbit) pair, and provides the per-vertex normalization that converts
HiPerMotif's induced isomorphism embeddings into ORCA-equivalent orbit counts.

Pure numpy/networkx — imports WITHOUT arkouda/arachne, so it is unit-testable
locally. (The arkouda calls live in hipermotif_backend.py, import-guarded.)

ORBIT -> GRAPHLET CORRESPONDENCE WAS VERIFIED AGAINST ORCA, not trusted from the
Hocevar-Demsar (2014) figure: tests/test_hipermotif_patterns.py recomputes per-vertex
orbit counts on real graphs through normalize_embeddings() and asserts byte-exact
equality with ORCA for all 15 columns. (The design-doc's figure-based guesses for
orbits 8-13 were wrong; e.g. the 4-cycle is orbit 8, not 10.)

PER-VERTEX NORMALIZATION (corrected contract). HiPerMotif's induced subgraph_isomorphism
returns every induced copy of a pattern H exactly |Aut(H)| times (once per ordering).
For each ORCA orbit O_j of H we COLLAPSE (sum) the per-vertex embedding tallies over all
pattern positions in O_j, then divide by the FULL |Aut(H)| -- NOT |Aut(H)|/|O_j|.
Worked K4 calibration (each = ORCA truth):
    degree   : collapsed 6  / |Aut(K2)|=2  = 3
    triangle : collapsed 18 / |Aut(C3)|=6  = 3
    4-clique : collapsed 24 / |Aut(K4)|=24 = 1
(The |Aut|/|orbit| form over-counts by |orbit|; it escaped notice because graph-level
K4 checks and |orbit|=1 orbits mask it.)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

ORCA_ORBIT_BASE = 2000
N_ORBITS_SIZE4 = 15

# pattern name -> (undirected edge list, num pattern vertices)
PATTERNS: Dict[str, Tuple[List[Tuple[int, int]], int]] = {
    "edge":     ([(0, 1)], 2),
    "p3":       ([(0, 1), (1, 2)], 3),
    "triangle": ([(0, 1), (0, 2), (1, 2)], 3),
    "p4":       ([(0, 1), (1, 2), (2, 3)], 4),
    "claw":     ([(0, 1), (0, 2), (0, 3)], 4),
    "paw":      ([(0, 1), (1, 2), (1, 3), (2, 3)], 4),   # pendant 0 on triangle {1,2,3}
    "c4":       ([(0, 1), (1, 2), (2, 3), (3, 0)], 4),
    "diamond":  ([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)], 4),  # K4 minus edge (0,2)
    "k4":       ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4),
}


def pattern_graph(name: str) -> nx.Graph:
    edges, n = PATTERNS[name]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


def automorphism_count_and_orbits(name: str) -> Tuple[int, List[List[int]]]:
    """Return (|Aut(H)|, orbit_groups) for pattern H, computed exactly via networkx.

    orbit_groups is a list of sorted vertex lists; the orbit of vertex v is the set of
    vertices v is mapped to under some automorphism.
    """
    g = pattern_graph(name)
    autos = list(GraphMatcher(g, g).isomorphisms_iter())
    seen, groups = set(), []
    for v in range(g.number_of_nodes()):
        s = frozenset(a[v] for a in autos)
        if s not in seen:
            seen.add(s)
            groups.append(sorted(s))
    return len(autos), groups


# VERIFIED (ORCA orbit index, pattern, orbit_id) — orbit_id indexes into
# automorphism_count_and_orbits(pattern)[1]. Discovered by matching per-vertex orbit
# vectors to ORCA on random graphs; re-asserted in tests against ORCA on real graphs.
_MAPPING: List[Tuple[int, str, int]] = [
    (0,  "edge",     0),   # degree (edge endpoint)
    (1,  "p3",       0),   # path-P3 end
    (2,  "p3",       1),   # path-P3 center (wedge apex)
    (3,  "triangle", 0),   # triangle
    (4,  "p4",       0),   # 4-path end
    (5,  "p4",       1),   # 4-path inner
    (6,  "claw",     1),   # 3-star (claw) leaf
    (7,  "claw",     0),   # 3-star (claw) center
    (8,  "c4",       0),   # 4-cycle
    (9,  "paw",      0),   # paw pendant (tail)
    (10, "paw",      2),   # paw triangle base (deg-2 triangle vertices)
    (11, "paw",      1),   # paw hub (deg-3 vertex joining tail + triangle)
    (12, "diamond",  0),   # diamond deg-2 vertices
    (13, "diamond",  1),   # diamond deg-3 vertices
    (14, "k4",       0),   # 4-clique
]


@dataclass(frozen=True)
class OrbitSpec:
    orca_orbit: int            # ORCA orbit index 0..14
    pattern: str               # graphlet name
    positions: Tuple[int, ...] # pattern-vertex positions forming this automorphism orbit
    n_aut: int                 # |Aut(H)| — the per-vertex divisor (corrected contract)
    k: int                     # graphlet node count
    motif_id: int              # 2000 + orca_orbit


def build_orbit_table() -> List[OrbitSpec]:
    table: List[OrbitSpec] = []
    for orca, name, oid in _MAPPING:
        naut, groups = automorphism_count_and_orbits(name)
        table.append(OrbitSpec(
            orca_orbit=orca, pattern=name, positions=tuple(groups[oid]),
            n_aut=naut, k=PATTERNS[name][1], motif_id=ORCA_ORBIT_BASE + orca,
        ))
    return table


ORBIT_TABLE: List[OrbitSpec] = build_orbit_table()

# (pattern, orbit_id) -> ORCA orbit index, for the normalizer.
_PATTERN_ORBIT_TO_ORCA: Dict[Tuple[str, int], int] = {
    (name, oid): orca for orca, name, oid in _MAPPING
}

# patterns that actually contribute orbits (all 9), in a stable order
PATTERN_NAMES: List[str] = list(PATTERNS.keys())


def normalize_embeddings(pattern: str, embeddings: np.ndarray, num_host_nodes: int) -> Dict[int, np.ndarray]:
    """Convert induced isomorphism embeddings of `pattern` into ORCA-orbit per-vertex counts.

    `embeddings` is an integer array of shape [num_embeddings, n_pattern_vertices]; column j
    holds the host vertex mapped to pattern vertex j (HiPerMotif `return_isos_as="vertices"`
    reshaped). Returns {orca_orbit_index: int64 per-vertex count array of length num_host_nodes}.

    For each automorphism orbit of the pattern: collapse (sum) the per-vertex tallies over the
    orbit's positions, then divide by |Aut(pattern)| (the corrected contract). Raises if the
    collapsed tally is not divisible by |Aut| (a malformed / non-induced embedding set).
    """
    emb = np.asarray(embeddings, dtype=np.int64)
    naut, groups = automorphism_count_and_orbits(pattern)
    if emb.ndim != 2 or (emb.size and emb.shape[1] != PATTERNS[pattern][1]):
        raise ValueError(
            f"embeddings shape {emb.shape} != [*, {PATTERNS[pattern][1]}] for pattern {pattern!r}")
    out: Dict[int, np.ndarray] = {}
    for oid, positions in enumerate(groups):
        tally = np.zeros(num_host_nodes, dtype=np.int64)
        if emb.size:
            for p in positions:
                np.add.at(tally, emb[:, p], 1)
        if np.any(tally % naut != 0):
            raise ValueError(
                f"pattern {pattern!r} orbit {oid}: collapsed tally not divisible by "
                f"|Aut|={naut} — embedding set is malformed or not induced.")
        out[_PATTERN_ORBIT_TO_ORCA[(pattern, oid)]] = tally // naut
    return out


def orbit_matrix_from_embeddings(embeddings_by_pattern: Dict[str, np.ndarray],
                                 num_host_nodes: int) -> np.ndarray:
    """Assemble the full [num_host_nodes, 15] ORCA-orbit matrix from per-pattern embeddings."""
    X = np.zeros((num_host_nodes, N_ORBITS_SIZE4), dtype=np.int64)
    for name in PATTERN_NAMES:
        emb = embeddings_by_pattern.get(name)
        if emb is None:
            continue
        for orca, vec in normalize_embeddings(name, emb, num_host_nodes).items():
            X[:, orca] = vec
    return X


def induced_embeddings(G: nx.Graph, pattern: str) -> np.ndarray:
    """LOCAL ORACLE / SIMULATION of HiPerMotif's induced subgraph_isomorphism output.

    Enumerates every induced copy of `pattern` in host graph G and every isomorphism of
    that copy onto the pattern (i.e. |Aut| orderings per copy), returning embeddings of
    shape [num_embeddings, n_pattern_vertices] with ORIGINAL host vertex IDs — exactly the
    semantics the Wulver `ar.subgraph_isomorphism(..., return_isos_as="vertices",
    algorithm_type="si", reorder_type="structural")` call produces. Pure networkx; used by
    tests and the equivalence gate's local side. G must have integer node labels 0..N-1.
    """
    H = pattern_graph(pattern)
    n = H.number_of_nodes()
    rows: List[List[int]] = []
    nodes = list(G.nodes())
    for combo in itertools.combinations(nodes, n):
        sub = G.subgraph(combo)
        if sub.number_of_edges() != H.number_of_edges():
            continue
        for phi in GraphMatcher(sub, H).isomorphisms_iter():  # host_vertex -> pattern_vertex
            inv = [None] * n
            for u, pv in phi.items():
                inv[pv] = u
            rows.append(inv)
    return np.asarray(rows, dtype=np.int64).reshape(-1, n)
