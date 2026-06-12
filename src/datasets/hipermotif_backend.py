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
    algorithm_type="si", reorder_type="structural"); reorder_type="structural" returns
    host IDs in ORIGINAL numbering (no remap). result[0] is a FLAT array of length
    num_embeddings * n_pattern_vertices; reshape to [num_embeddings, n_pattern_vertices],
    column j = pattern vertex j.
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

    NOTE (Wulver): this is the one cluster-API touchpoint most likely to need a minor
    adjustment to your Arachne version's PropGraph loader. The subgraph_isomorphism
    call and the normalization are the verified parts.
    """
    g = ar.PropGraph()
    g.add_edges_from(ak.array(src), ak.array(dst))
    return g


def _run_iso(ar, G, H) -> np.ndarray:
    """Call HiPerMotif induced subgraph isomorphism and return embeddings as a numpy
    array [num_embeddings, n_pattern_vertices] of ORIGINAL host vertex IDs."""
    n_pat = H.n_vertices if hasattr(H, "n_vertices") else None  # not relied upon; see reshape
    result = ar.subgraph_isomorphism(
        G, H, return_isos_as="vertices", algorithm_type="si", reorder_type="structural",
    )
    flat = result[0].to_ndarray()  # arkouda pdarray -> numpy
    return flat  # reshaped by the caller using the known pattern vertex count


def count_orbits_hipermotif(edges: Iterable[Tuple[int, int]], num_nodes: int,
                            graphlet_size: int = 4) -> np.ndarray:
    """Per-vertex ORCA-orbit counts [num_nodes, 15] computed with HiPerMotif (Wulver).

    Drop-in equivalent of orca_orbits.count_orbits: same shape, same orbit indexing,
    same integer counts. Requires arkouda + arachne (raises a clear error otherwise).
    """
    if graphlet_size != 4:
        raise NotImplementedError("HiPerMotif backend currently supports size-4 orbits (15).")
    ak, ar = _require_arachne()

    src, dst = _symmetrize(edges, num_nodes)
    if not src:  # 0-edge graph -> all-zero orbit matrix (matches ORCA short-circuit)
        return np.zeros((num_nodes, hp.N_ORBITS_SIZE4), dtype=np.int64)
    G = _build_propgraph(ak, ar, src, dst)

    embeddings_by_pattern: Dict[str, np.ndarray] = {}
    for name in hp.PATTERN_NAMES:
        n_pat = hp.PATTERNS[name][1]
        psrc, pdst = _symmetrize(hp.PATTERNS[name][0], n_pat)
        H = _build_propgraph(ak, ar, psrc, pdst)
        flat = _run_iso(ar, G, H)
        emb = np.asarray(flat, dtype=np.int64).reshape(-1, n_pat)  # [num_emb, n_pat]
        embeddings_by_pattern[name] = emb

    return hp.orbit_matrix_from_embeddings(embeddings_by_pattern, num_nodes)
