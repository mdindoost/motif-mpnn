"""HiPerMotif (Arachne / Arkouda) orbit-feature backend — RUNS ON WULVER ONLY.

arkouda + arachne are NOT installed locally; this module import-guards them so it
imports cleanly for unit testing. The actual subgraph-isomorphism calls require Wulver.
The normalization math (the correctness-critical part) lives in hipermotif_patterns.py
and is fully unit-tested locally against ORCA; this module only wires it to the cluster
engine, producing a drop-in replacement for orca_orbits.count_orbits.

VERIFIED CONVENTIONS (confirmed on Wulver — see CLAUDE.md "HiPerMotif backend"):
  * ar.subgraph_isomorphism is INDUCED matching (do NOT use subgraph_monomorphism).
  * Host and pattern are loaded SYMMETRIZED (each undirected edge in BOTH directions).
  * Returns (induced copies) x |Aut(H)|; normalization divides collapsed per-vertex
    orbit tallies by the FULL |Aut(H)| (hipermotif_patterns.normalize_embeddings).
  * API: ar.subgraph_isomorphism(G, H, return_isos_as="vertices",
    algorithm_type="si", reorder_type="structural"). reorder_type="structural" returns host
    IDs in ORIGINAL numbering (no HOST remap), but it PERMUTES the PATTERN's vertices
    internally. The call returns TWO arrays:
      result[0] = isoArr    : flat host-vertex embeddings, length num_embeddings * n_pattern;
                              reshape to [num_embeddings, n_pattern].
      result[1] = isoMapper : the reordered pattern-vertex order (`nodeMapGraphG2`) repeated
                              once per embedding. Output column j holds the host vertex mapped
                              to ORIGINAL pattern vertex isoMapper[j] — NOT pattern vertex j.
    We extract the (global) permutation from result[1] and pass it as `mapper` to
    hipermotif_patterns so orbit tallies use the correct column per pattern vertex. Ignoring
    isoMapper mixes columns across orbits for multi-orbit patterns (p3, paw, ...) — the bug
    that failed the gate on Cora/Shrikhande while the identity-ordered local mock passed.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.datasets import hipermotif_patterns as hp


def _require_arachne():
    """Import arkouda + arachne or raise a clear Wulver-only error. Keeps this module
    importable (and unit-testable) on machines without the cluster libraries."""
    try:
        import arkouda as ak  # noqa: F401
        import arachne as ar  # noqa: F401
    except Exception as e:  # ImportError locally; other env errors on misconfigured nodes
        raise RuntimeError(
            "HiPerMotif backend requires Wulver (arkouda + arachne), which are not "
            "available here. Run the HiPerMotif path on the cluster; locally use the "
            "ORCA backend (--backend orca)."
        ) from e
    return ak, ar


def _symmetrize(edges: Iterable[Tuple[int, int]], num_nodes: int) -> Tuple[List[int], List[int]]:
    """Return (src, dst) arrays with each undirected, self-loop-free edge in BOTH
    directions and deduplicated. Vertices must be 0..num_nodes-1."""
    seen = set()
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge ({u},{v}) out of range for num_nodes={num_nodes}")
        seen.add((u, v))
        seen.add((v, u))
    if not seen:
        return [], []
    src, dst = zip(*sorted(seen))
    return list(src), list(dst)


def _build_propgraph(ak, ar, src: List[int], dst: List[int]):
    """Build a symmetrized Arachne PropGraph from src/dst host-vertex arrays.

    Uses the verified working Arachne API (confirmed on Wulver 2026-06-15):
    PropGraph().load_edge_attributes(DataFrame{src,dst}, source_column, destination_column).
    The earlier add_edges_from(...) form did not match the installed Arachne version.
    The subgraph_isomorphism call and the normalization are the separately-verified parts.
    """
    g = ar.PropGraph()
    g.load_edge_attributes(
        ak.DataFrame({"src": ak.array(src), "dst": ak.array(dst)}),
        source_column="src", destination_column="dst",
    )
    return g


# CORRECTNESS path for feature extraction. With reorder_type="None" the Chapel server
# (SubgraphSearch.chpl line ~1060 branches on reorderType != "None") does NO pattern-vertex
# reordering, so embeddings come back in our ORIGINAL pattern-vertex order — output column j
# == pattern vertex j, and isoMapper is the identity. This was confirmed on Wulver (C5 wedge
# embeddings put the center in the MIDDLE column; Cora induced wedges match ORCA). We KEEP
# reading isoMapper and ASSERT it is identity here, as a guard against a future change
# silently reintroducing reordering. NOTE: reorder_type="None" may be SLOWER (reordering
# optimizes the search); the scaling/timing experiments are a different need — see CLAUDE.md
# section 13. Pass reorder_type="structural" there, where the non-identity mapper IS applied.
DEFAULT_REORDER_TYPE = "None"


def _run_iso(ar, G, H, reorder_type: str = DEFAULT_REORDER_TYPE) -> Tuple[np.ndarray, np.ndarray]:
    """Call HiPerMotif induced subgraph isomorphism and return BOTH arrays of the
    `return_isos_as="vertices"` contract as flat numpy arrays:
      result[0] = isoArr    : host-vertex embeddings, length n_pattern * num_embeddings.
      result[1] = isoMapper : the pattern-vertex order (`nodeMapGraphG2`) repeated once per
                  embedding. With reorder_type="None" this is the identity; with "structural"
                  it is a (global) permutation the caller must invert to recover orbits."""
    result = ar.subgraph_isomorphism(
        G, H, return_isos_as="vertices", algorithm_type="si", reorder_type=reorder_type,
    )
    iso = np.asarray(result[0].to_ndarray(), dtype=np.int64).ravel()
    mapper = np.asarray(result[1].to_ndarray(), dtype=np.int64).ravel()
    return iso, mapper


def _extract_mapper(mapper_flat: np.ndarray, n_pat: int, num_emb: int,
                    require_identity: bool = False) -> np.ndarray:
    """isoMapper is `nodeMapGraphG2` repeated once per embedding (the runSearch `forall ...
    by numSubgraphVertices` assignment), so the permutation is GLOBAL — identical for every
    embedding. Return the length-n_pat permutation and assert that invariant. When
    `require_identity` (the reorder_type="None" production path) also assert it is the
    identity 0..n_pat-1 — a cheap guard that fires if reordering is ever reintroduced."""
    if num_emb == 0:
        return np.arange(n_pat, dtype=np.int64)  # no embeddings -> mapper irrelevant
    if mapper_flat.size != n_pat * num_emb:
        raise ValueError(
            f"isoMapper length {mapper_flat.size} != n_pat*num_emb={n_pat * num_emb}")
    M = mapper_flat.reshape(num_emb, n_pat)
    if not np.all(M == M[0]):
        raise ValueError(
            "isoMapper differs across embeddings — the reorder is no longer global; "
            "per-embedding column->pattern-vertex remap is required.")
    perm = M[0].astype(np.int64)
    if require_identity and not np.array_equal(perm, np.arange(n_pat)):
        raise ValueError(
            f"reorder_type='None' but isoMapper={perm.tolist()} is not the identity — the "
            f"engine reordered pattern vertices unexpectedly; column->vertex order is not "
            f"trustworthy. Investigate before using these counts.")
    return perm


def count_orbits_hipermotif(edges: Iterable[Tuple[int, int]], num_nodes: int,
                            graphlet_size: int = 4,
                            reorder_type: str = DEFAULT_REORDER_TYPE) -> np.ndarray:
    """Per-vertex ORCA-orbit counts [num_nodes, 15] computed with HiPerMotif (Wulver).

    Drop-in equivalent of orca_orbits.count_orbits: same shape, same orbit indexing,
    same integer counts. Requires arkouda + arachne (raises a clear error otherwise).

    reorder_type="None" (default) is the correctness path: no pattern-vertex permutation,
    column j == pattern vertex j. "structural" is faster but permutes columns; the mapper
    from result[1] is then inverted in normalize_embeddings (used for timing experiments).
    """
    if graphlet_size != 4:
        raise NotImplementedError("HiPerMotif backend currently supports size-4 orbits (15).")
    ak, ar = _require_arachne()

    edge_list = [(int(u), int(v)) for u, v in edges]
    src, dst = _symmetrize(edge_list, num_nodes)
    if not src:  # 0-edge graph -> all-zero orbit matrix (matches ORCA short-circuit)
        return np.zeros((num_nodes, hp.N_ORBITS_SIZE4), dtype=np.int64)
    G = _build_propgraph(ak, ar, src, dst)
    require_identity = (reorder_type == "None")

    embeddings_by_pattern: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in hp.PATTERN_NAMES:
        n_pat = hp.PATTERNS[name][1]
        psrc, pdst = _symmetrize(hp.PATTERNS[name][0], n_pat)
        H = _build_propgraph(ak, ar, psrc, pdst)
        iso_flat, mapper_flat = _run_iso(ar, G, H, reorder_type)
        emb = iso_flat.reshape(-1, n_pat)                       # [num_emb, n_pat]
        mapper = _extract_mapper(mapper_flat, n_pat, emb.shape[0], require_identity)
        # Regression guard: confirm the engine actually returned INDUCED matches (every
        # pattern non-edge maps to a host non-edge). Loud on violation; catches a whole class
        # of monomorphism/adjacency bugs at the source.
        hp.assert_induced_embeddings(name, emb, edge_list, num_nodes, mapper)
        embeddings_by_pattern[name] = (emb, mapper)             # mapper[j] -> orig vertex

    return hp.orbit_matrix_from_embeddings(embeddings_by_pattern, num_nodes)
