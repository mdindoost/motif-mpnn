#!/usr/bin/env python
"""DIAGNOSTIC for the p4 (open-pattern) embedding-multiplicity gate failure. RUN ON WULVER.

Context: scripts/wulver/verify_hipermotif_equals_orca.py PASSED on K4 and C5, then raised
inside normalize_embeddings on Shrikhande:
    ValueError: pattern 'p4' orbit 0: collapsed tally not divisible by |Aut|=2
("orbit 0" is the LOCAL p4-ends orbit = global ORCA orbit 4, not global orbit 0/degree.)

The local oracle induced_embeddings() emits EXACTLY |Aut(H)| rows per induced copy by
construction (GraphMatcher yields |Aut(H)| isomorphisms when sub == H), so it can never
fail divisibility — it BAKES IN the "x|Aut| orderings" convention. K4/C5 passing proves the
real engine returns x|Aut| orderings AND is induced; Shrikhande p4 failing proves the real
engine is INCONSISTENT for p4 between a sparse host (C5, clean x2) and a dense host
(Shrikhande). This script dumps the RAW engine output vs the mock vs ORCA to decide:

  * real_num_emb / mock_num_emb == 1.0 and rows agree  -> reshape/layout bug in our code
                                                           (count_orbits_hipermotif reshape).
  * ratio == 0.5 on Shrikhande but 1.0 on C5           -> engine dedups/canonicalizes
                                                           orderings host-dependently (cause b).
  * real_num_emb not a clean multiple of mock_num_emb  -> induced edge-case / extra or
                                                           dropped matches on dense hosts (b).
  * real per-column sums asymmetric vs mock            -> flat-array layout != row-major
                                                           [emb0_v0,emb0_v1,...] (reshape).

It does NOT call normalize_embeddings, so it runs to completion even on the failing graph.
It changes NOTHING in the repo and asserts nothing — it only reports. Reuses the EXACT
propgraph-build + iso-call code path from hipermotif_backend so it reproduces the real run.

Usage:
  python scripts/wulver/diagnose_p4_multiplicity.py --ak-host <h> --ak-port <p> \
      [--patterns p4 claw paw c4 diamond] [--graphs C5 Shrikhande 4x4-rook]
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

from src.datasets import hipermotif_patterns as hp
from src.datasets import hipermotif_backend as backend

try:
    from src.datasets.orca_orbits import count_orbits as orca_count
    _HAVE_ORCA = True
except Exception as e:  # pragma: no cover
    _HAVE_ORCA = False
    print(f"[diag] ORCA unavailable ({e}); proceeding without ORCA ground truth.")


def _nx_from_ei(ei, n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v in ei.t().tolist():
        if u != v:
            g.add_edge(int(u), int(v))
    return g


def host_graphs(names):
    avail = {}
    avail["C5"] = nx.cycle_graph(5)            # PASSED in the gate
    avail["K4"] = nx.complete_graph(4)         # PASSED in the gate (control)
    try:
        from src.datasets.expressivity import make_shrikhande, make_rook_4x4
        avail["Shrikhande"] = _nx_from_ei(make_shrikhande(), 16)  # FAILED in the gate
        avail["4x4-rook"] = _nx_from_ei(make_rook_4x4(), 16)
    except Exception as e:
        print(f"[diag] could not build SRG pair ({e}); SRG graphs skipped.")
    for nm in names:
        if nm in avail:
            yield nm, nx.convert_node_labels_to_integers(avail[nm])
        else:
            print(f"[diag] unknown graph {nm!r}; known: {sorted(avail)}")


def col_sums(emb, n_pat):
    if emb.size == 0:
        return [0] * n_pat
    return [int(emb[:, j].sum()) for j in range(n_pat)]


def orbit_divisibility(pattern, emb, num_nodes):
    """Per LOCAL orbit: report (orca_id, positions, |Aut|, collapsed-tally distinct values,
    whether every collapsed per-vertex tally is divisible by |Aut|). Mirrors the gate's check
    WITHOUT raising."""
    naut, groups = hp.automorphism_count_and_orbits(pattern)
    report = []
    for oid, positions in enumerate(groups):
        tally = np.zeros(num_nodes, dtype=np.int64)
        if emb.size:
            for p in positions:
                np.add.at(tally, emb[:, p], 1)
        orca = hp._PATTERN_ORBIT_TO_ORCA[(pattern, oid)]
        divisible = bool(np.all(tally % naut == 0))
        report.append((orca, tuple(positions), naut, sorted(set(tally.tolist()))[:6], divisible))
    return report


def run_real(ak, ar, G_pg, pattern):
    """Exact same path as count_orbits_hipermotif: build symmetrized pattern propgraph,
    run the iso call, reshape row-major to [num_emb, n_pat]."""
    n_pat = hp.PATTERNS[pattern][1]
    psrc, pdst = backend._symmetrize(hp.PATTERNS[pattern][0], n_pat)
    H_pg = backend._build_propgraph(ak, ar, psrc, pdst)
    flat = np.asarray(backend._run_iso(ar, G_pg, H_pg), dtype=np.int64).ravel()
    return flat, n_pat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ak-host", default="localhost")
    ap.add_argument("--ak-port", default=5555, type=int)
    ap.add_argument("--graphs", nargs="+",
                    default=["K4", "C5", "Shrikhande", "4x4-rook"])
    ap.add_argument("--patterns", nargs="+",
                    default=["p4", "claw", "paw", "c4", "diamond"],
                    help="open/dense patterns to probe (default skips edge/p3/triangle/k4)")
    ap.add_argument("--show-rows", type=int, default=8, help="raw rows to print per pattern")
    args = ap.parse_args()

    import arkouda as ak
    import arachne as ar  # noqa: F401  (imported so a missing arachne fails loudly here)
    print(f"[diag] connecting to arkouda at {args.ak_host}:{args.ak_port} ...")
    ak.connect(args.ak_host, args.ak_port)

    try:
        for gname, G in host_graphs(args.graphs):
            n = G.number_of_nodes()
            edges = [(int(u), int(v)) for u, v in G.edges()]
            G_pg = backend._build_propgraph(ak, ar, *backend._symmetrize(edges, n))
            orca = orca_count(edges, n) if _HAVE_ORCA else None
            print("\n" + "=" * 78)
            print(f"HOST {gname}  N={n}  E={len(edges)}")
            print("=" * 78)
            for pattern in args.patterns:
                n_pat = hp.PATTERNS[pattern][1]
                flat, _ = run_real(ak, ar, G_pg, pattern)
                clean_reshape = (flat.size % n_pat == 0)
                real = flat.reshape(-1, n_pat) if clean_reshape else np.empty((0, n_pat), np.int64)
                mock = hp.induced_embeddings(G, pattern)

                r_emb, m_emb = real.shape[0], mock.shape[0]
                ratio = (r_emb / m_emb) if m_emb else float("inf")
                print(f"\n-- pattern {pattern!r} (n_pat={n_pat}, |Aut|="
                      f"{hp.automorphism_count_and_orbits(pattern)[0]}) --")
                print(f"   raw flat len={flat.size}  len%n_pat={flat.size % n_pat}  "
                      f"clean_reshape={clean_reshape}")
                print(f"   REAL num_emb={r_emb:<6} col_sums={col_sums(real, n_pat)}")
                print(f"   MOCK num_emb={m_emb:<6} col_sums={col_sums(mock, n_pat)}")
                print(f"   ratio REAL/MOCK = {ratio:.4f}")
                for orca_id, pos, naut, vals, ok in orbit_divisibility(pattern, real, n):
                    tag = "OK " if ok else "!! NOT DIVISIBLE"
                    print(f"   [REAL] ORCA orbit {orca_id:<2} pos={pos} /|Aut|={naut}: "
                          f"distinct collapsed tallies={vals}  {tag}")
                if orca is not None:
                    for orca_id, _, _, _, _ in orbit_divisibility(pattern, real, n):
                        col = orca[:, orca_id]
                        print(f"   [ORCA] orbit {orca_id:<2} per-vertex distinct={sorted(set(col.tolist()))[:6]}"
                              f"  total={int(col.sum())}")
                if args.show_rows:
                    print(f"   REAL first {args.show_rows} rows:\n{real[:args.show_rows].tolist()}")
                    print(f"   MOCK first {args.show_rows} rows:\n{mock[:args.show_rows].tolist()}")
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass

    print("\n[diag] done. Send the full stdout back for analysis.")


if __name__ == "__main__":
    main()
