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


def normalize_embeddings(pattern: str, embeddings: np.ndarray, num_host_nodes: int,
                         mapper: "np.ndarray | None" = None) -> Dict[int, np.ndarray]:
    """Convert induced isomorphism embeddings of `pattern` into ORCA-orbit per-vertex counts.

    `embeddings` is an integer array of shape [num_embeddings, n_pattern_vertices].

    COLUMN ORDER IS NOT THE ORIGINAL PATTERN-VERTEX ORDER. HiPerMotif
    `return_isos_as="vertices"` returns two arrays: result[0] (the host-vertex embeddings,
    reshaped into `embeddings`) and result[1] (`isoMapper` = the structurally-reordered
    pattern-vertex order `nodeMapGraphG2`, repeated per embedding). Because
    `reorder_type="structural"` permutes the pattern's vertices internally, output column j
    holds the host vertex mapped to ORIGINAL pattern vertex `mapper[j]` — NOT pattern vertex j.
    `mapper` must be the length-n_pattern permutation (one row of result[1]); pass None only
    when column order is known to be identity (e.g. an already-canonical local mock).

    Ignoring `mapper` silently mixes columns across automorphism orbits whenever the structural
    reorder crosses orbits (e.g. p3 center vs ends, paw hub vs base) — which made the collapsed
    tally not divisible by |Aut| on the cluster while the identity-ordered local mock passed.

    For each automorphism orbit of the pattern: collapse (sum) the per-vertex tallies over the
    orbit's positions (mapped through `mapper` to the right output columns), then divide by
    |Aut(pattern)| (the corrected contract). Raises if the collapsed tally is not divisible by
    |Aut| (a malformed / non-induced embedding set, or a wrong/missing mapper).
    """
    emb = np.asarray(embeddings, dtype=np.int64)
    n_pat = PATTERNS[pattern][1]
    naut, groups = automorphism_count_and_orbits(pattern)
    if emb.ndim != 2 or (emb.size and emb.shape[1] != n_pat):
        raise ValueError(
            f"embeddings shape {emb.shape} != [*, {n_pat}] for pattern {pattern!r}")
    # mapper[j] = ORIGINAL pattern vertex held by output column j (None => identity)
    if mapper is None:
        mapper = np.arange(n_pat, dtype=np.int64)
    else:
        mapper = np.asarray(mapper, dtype=np.int64).ravel()
    if mapper.shape != (n_pat,) or sorted(mapper.tolist()) != list(range(n_pat)):
        raise ValueError(
            f"mapper {mapper.tolist()} is not a permutation of 0..{n_pat - 1} "
            f"for pattern {pattern!r}")
    # inverse: original pattern vertex p -> the output column carrying it
    col_of_vertex = np.empty(n_pat, dtype=np.int64)
    col_of_vertex[mapper] = np.arange(n_pat, dtype=np.int64)
    out: Dict[int, np.ndarray] = {}
    for oid, positions in enumerate(groups):
        tally = np.zeros(num_host_nodes, dtype=np.int64)
        if emb.size:
            for p in positions:                       # p is an ORIGINAL pattern vertex
                np.add.at(tally, emb[:, int(col_of_vertex[p])], 1)
        if np.any(tally % naut != 0):
            orca = _PATTERN_ORBIT_TO_ORCA[(pattern, oid)]
            raise ValueError(
                f"pattern {pattern!r} local orbit {oid} (ORCA orbit {orca}, "
                f"positions {tuple(positions)}): collapsed tally not divisible by "
                f"|Aut|={naut} — embedding set is malformed, not induced, or the "
                f"column->pattern-vertex mapper is wrong.")
        out[_PATTERN_ORBIT_TO_ORCA[(pattern, oid)]] = tally // naut
    return out


def orbit_matrix_from_embeddings(embeddings_by_pattern: "Dict[str, object]",
                                 num_host_nodes: int) -> np.ndarray:
    """Assemble the full [num_host_nodes, 15] ORCA-orbit matrix from per-pattern embeddings.

    Each value in `embeddings_by_pattern` is either a `(embeddings, mapper)` pair (the real
    HiPerMotif contract: result[0] and the column->pattern-vertex permutation from result[1])
    or a bare `embeddings` array (treated as identity column order for back-compat).
    """
    X = np.zeros((num_host_nodes, N_ORBITS_SIZE4), dtype=np.int64)
    for name in PATTERN_NAMES:
        item = embeddings_by_pattern.get(name)
        if item is None:
            continue
        if isinstance(item, tuple):
            emb, mapper = item
        else:
            emb, mapper = item, None  # identity column order
        for orca, vec in normalize_embeddings(name, emb, num_host_nodes, mapper).items():
            X[:, orca] = vec
    return X


# Deterministic structural-reorder permutations for the LOCAL mock. mapper[j] = original
# pattern vertex held by output column j. For multi-orbit patterns these are deliberately
# NON-identity AND NOT automorphisms (they move a vertex into a column another orbit would
# occupy), so the column->pattern-vertex remap is actually exercised by the unit tests — the
# exact case the old identity-ordered mock missed while Wulver failed. Single-orbit / fully
# symmetric patterns use identity (any permutation is an automorphism, so it is invariant).
_MOCK_STRUCTURAL_PERM: Dict[str, Tuple[int, ...]] = {
    "edge":     (0, 1),
    "p3":       (1, 0, 2),        # center -> col 0 (crosses center/ends orbits)
    "triangle": (0, 1, 2),
    "p4":       (1, 0, 2, 3),     # inner vertex 1 -> col 0 (crosses inner/ends orbits)
    "claw":     (1, 0, 2, 3),     # a leaf -> col 0 (crosses center/leaves orbits)
    "paw":      (1, 0, 2, 3),     # hub -> col 0 (crosses hub/base orbits)
    "c4":       (0, 1, 2, 3),
    "diamond":  (1, 0, 2, 3),     # deg-3 -> col 0 (crosses deg-2/deg-3 orbits)
    "k4":       (0, 1, 2, 3),
}


def mock_structural_perm(pattern: str) -> np.ndarray:
    """The mapper (column->original-pattern-vertex permutation) the local mock simulates."""
    return np.asarray(_MOCK_STRUCTURAL_PERM[pattern], dtype=np.int64)


def induced_embeddings(G: nx.Graph, pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    """LOCAL ORACLE / SIMULATION of HiPerMotif's induced subgraph_isomorphism output.

    Enumerates every induced copy of `pattern` in host graph G and every isomorphism of
    that copy onto the pattern (i.e. |Aut| orderings per copy), and returns
    `(embeddings, mapper)` — exactly the two-array shape the Wulver
    `ar.subgraph_isomorphism(..., return_isos_as="vertices", algorithm_type="si",
    reorder_type="structural")` call produces:
      * embeddings : [num_embeddings, n_pattern_vertices], ORIGINAL host vertex IDs, with
                     columns ordered by the (simulated) structural reorder — output column j
                     holds the host vertex mapped to original pattern vertex `mapper[j]`.
      * mapper     : the length-n_pattern permutation (result[1]'s repeated `nodeMapGraphG2`).
    The mock applies a fixed non-identity permutation for multi-orbit patterns (see
    `_MOCK_STRUCTURAL_PERM`) so callers MUST honor `mapper` to recover correct orbit counts.
    Pure networkx; used by tests and the equivalence gate's local side. G must have integer
    node labels 0..N-1.
    """
    H = pattern_graph(pattern)
    n = H.number_of_nodes()
    perm = mock_structural_perm(pattern)               # mapper[j] = original vertex at col j
    rows: List[List[int]] = []
    nodes = list(G.nodes())
    for combo in itertools.combinations(nodes, n):
        sub = G.subgraph(combo)
        if sub.number_of_edges() != H.number_of_edges():
            continue
        for phi in GraphMatcher(sub, H).isomorphisms_iter():  # host_vertex -> pattern_vertex
            inv = [None] * n
            for u, pv in phi.items():
                inv[pv] = u                            # inv[original pattern vertex] = host
            rows.append([inv[int(perm[j])] for j in range(n)])  # col j carries vertex perm[j]
    emb = np.asarray(rows, dtype=np.int64).reshape(-1, n)
    return emb, perm
