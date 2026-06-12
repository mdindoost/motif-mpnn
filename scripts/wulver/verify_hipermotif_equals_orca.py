#!/usr/bin/env python
"""GATING equivalence check: HiPerMotif orbit features == ORCA oracle.  RUN ON WULVER.

This is the correctness gate that licenses cluster numbers for the paper. It computes
per-vertex 15-orbit features BOTH ways on a ladder of test graphs and asserts byte-exact
integer equality:
  - ORCA  : the local oracle, already hand-validated (orbit 0=degree, 3=triangle, 14=4-clique).
  - HiPerMotif : src/datasets/hipermotif_backend (arkouda + arachne, this cluster).

>>> RUN THIS FIRST ON WULVER. Only proceed to any scale experiment if it prints ALL PASS. <<<
It exits nonzero on any mismatch so it can gate a pipeline.

Prerequisites on Wulver:
  1. arkouda server running; note its host/port.
  2. conda env with arkouda + arachne + torch_geometric + networkx + this repo on PYTHONPATH.
  3. g++ available (ORCA oracle builds on first use).

Usage:
  python scripts/wulver/verify_hipermotif_equals_orca.py \
      --ak-host <arkouda_host> --ak-port <arkouda_port> [--skip-large]

On mismatch it prints the exact (graph, vertex, orbit, ORCA, HiPerMotif, ratio) — the ratio
usually reveals a missing/extra |Aut| factor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.orca_orbits import count_orbits as orca_count          # local oracle
from src.datasets.hipermotif_backend import count_orbits_hipermotif      # Wulver engine
from src.datasets.hipermotif_patterns import N_ORBITS_SIZE4


def _edges_n(G: nx.Graph):
    G = nx.convert_node_labels_to_integers(G)
    return [(int(u), int(v)) for u, v in G.edges()], G.number_of_nodes()


def _test_graphs(skip_large: bool):
    """Yield (name, edges, num_nodes) in increasing size order."""
    yield ("K4", *_edges_n(nx.complete_graph(4)))
    yield ("C5", *_edges_n(nx.cycle_graph(5)))

    from src.datasets.expressivity import make_shrikhande, make_rook_4x4

    def _nx_from_ei(ei):
        g = nx.Graph(); g.add_nodes_from(range(16))
        for u, v in ei.t().tolist():
            if u != v:
                g.add_edge(int(u), int(v))
        return g
    yield ("Shrikhande", *_edges_n(_nx_from_ei(make_shrikhande())))
    yield ("4x4-rook", *_edges_n(_nx_from_ei(make_rook_4x4())))

    if skip_large:
        return
    from torch_geometric.datasets import Planetoid, TUDataset
    from torch_geometric.utils import to_networkx
    cora = Planetoid(root=str(REPO / "data/processed"), name="Cora")[0]
    yield ("Cora", *_edges_n(to_networkx(cora, to_undirected=True)))
    prot = TUDataset(root=str(REPO / "data/processed/pyg/proteins"), name="PROTEINS")[0]
    yield ("PROTEINS[0]", *_edges_n(to_networkx(prot, to_undirected=True)))


def _compare(name, edges, n):
    orca = orca_count(edges, n)                      # [n, 15]
    hm = count_orbits_hipermotif(edges, n)           # [n, 15]
    assert orca.shape == (n, N_ORBITS_SIZE4), f"ORCA shape {orca.shape}"
    assert hm.shape == (n, N_ORBITS_SIZE4), f"HiPerMotif shape {hm.shape}"
    if np.array_equal(orca, hm):
        print(f"  [PASS] {name:12} N={n:<6} all 15 orbits match ORCA exactly")
        return True, orca
    print(f"  [FAIL] {name:12} N={n}: HiPerMotif != ORCA")
    rows, cols = np.where(orca != hm)
    for r, c in list(zip(rows, cols))[:20]:
        ov, hv = int(orca[r, c]), int(hm[r, c])
        ratio = hv / ov if ov else float("inf")
        print(f"         vertex {r} orbit {c}: ORCA={ov} HiPerMotif={hv} ratio={ratio:.3f}")
    return False, orca


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ak-host", default="localhost", help="arkouda server host")
    ap.add_argument("--ak-port", default=5555, type=int, help="arkouda server port")
    ap.add_argument("--skip-large", action="store_true",
                    help="skip Cora + PROTEINS (K4/C5/SRG only)")
    args = ap.parse_args()

    import arkouda as ak
    print(f"[gate] connecting to arkouda at {args.ak_host}:{args.ak_port} ...")
    ak.connect(args.ak_host, args.ak_port)

    print("[gate] HiPerMotif vs ORCA orbit-feature equivalence")
    all_pass = True
    srg = {}
    try:
        for name, edges, n in _test_graphs(args.skip_large):
            ok, mat = _compare(name, edges, n)
            all_pass &= ok
            if name in ("Shrikhande", "4x4-rook"):
                srg[name] = mat
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass

    # SRG pair doubles as method validation (paper Section III separation).
    if "Shrikhande" in srg and "4x4-rook" in srg:
        sh, rk = srg["Shrikhande"], srg["4x4-rook"]
        tri_eq = sh[:, 3].sum() == rk[:, 3].sum()
        clique_sep = (sh[:, 14] == 0).all() and (rk[:, 14] == 2).all()
        print(f"\n[gate] SRG method validation: triangle-orbit totals equal={tri_eq} "
              f"(sh={int(sh[:,3].sum())}, rk={int(rk[:,3].sum())}); "
              f"4-clique orbit Shrikhande==0 & rook==2/node = {clique_sep}")
        all_pass &= bool(tri_eq and clique_sep)

    print("\n" + ("=" * 60))
    print("OVERALL: " + ("ALL PASS — safe to proceed to scale experiments."
                         if all_pass else "FAIL — DO NOT proceed; fix the backend first."))
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
