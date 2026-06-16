#!/usr/bin/env python
"""RAW dump of every array subgraph_isomorphism returns — settle "wrong array vs race". WULVER.

We are NOT assuming anything here. This calls ar.subgraph_isomorphism with return_isos_as=
"complete" (which returns ALL FOUR arrays: isoArr, isoMapper, srcPerIso, dstPerIso) and also
"vertices", on small KNOWN graphs and a LARGE graph, and prints everything so we can see with
our eyes:

  * len(result) and every slot's length + raw head — is slot 0 vertices, or src/dst?
  * slot0 reshaped by numSubgraphVertices vs slot2/slot3 reshaped by numSubgraphEdges.
  * For each iso: its vertex row AND its src/dst edge rows side by side, and whether the
    edges are exactly the pattern's edges among those vertices (induced check).
  * HYPOTHESIS TEST (Bartosz): for any 'repeated' vertex row in slot0, print that same iso's
    srcPerIso/dstPerIso block — are the repeated numbers actually src or dst values?
  * vertices.slot0 vs complete.slot0 : identical? (are we reading the same array both modes?)
  * DETERMINISM on the large graph: run K times, compare slot0 hash + repeat counts.

Run under SERIAL and PARALLEL servers and compare. Lots of output by design.

Usage (repo root):
  PYTHONPATH=. python scripts/wulver/dump_all_arrays.py --ak-host <h> --ak-port <p>
  PYTHONPATH=. python scripts/wulver/dump_all_arrays.py --ak-host <h> --ak-port <p> \
      --patterns p3 p4 claw triangle --big-n 150 --runs 6
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

from src.datasets import hipermotif_backend as backend
from src.datasets import hipermotif_patterns as hp

np.set_printoptions(threshold=60, linewidth=200)


def edges_n(G):
    G = nx.convert_node_labels_to_integers(G)
    return [(int(u), int(v)) for u, v in G.edges()], G.number_of_nodes()


def host_edge_set(edges, n):
    s = set()
    for u, v in edges:
        s.add((min(u, v), max(u, v)))
    return s


def raw_query(ak, ar, Gpg, Hpg, mode, reorder):
    """Call subgraph_isomorphism and return (len, list-of-ndarrays-or-reprs)."""
    res = ar.subgraph_isomorphism(Gpg, Hpg, return_isos_as=mode,
                                  algorithm_type="si", reorder_type=reorder)
    slots = []
    for k in range(len(res)):
        try:
            slots.append(np.asarray(res[k].to_ndarray(), dtype=np.int64))
        except Exception as e:  # noqa: BLE001
            slots.append(f"<slot {k}: {type(res[k]).__name__} not ndarray: {e}>")
    return len(res), slots


def build(ak, ar, edges, n, pattern):
    src, dst = backend._symmetrize(edges, n)
    Gpg = backend._build_propgraph(ak, ar, src, dst) if src else None
    n_pat = hp.PATTERNS[pattern][1]
    psrc, pdst = backend._symmetrize(hp.PATTERNS[pattern][0], n_pat)
    Hpg = backend._build_propgraph(ak, ar, psrc, pdst)
    n_edges_sym = len(psrc)                      # == numSubgraphEdges (symmetrized) in Chapel
    return Gpg, Hpg, n_pat, n_edges_sym


def dump_graph(ak, ar, name, edges, n, patterns, reorder, show=10):
    print("\n" + "#" * 90)
    print(f"# HOST {name}: N={n} E={len(edges)}   reorder_type={reorder!r}")
    print("#" * 90)
    hes = host_edge_set(edges, n)
    for pat in patterns:
        n_pat = hp.PATTERNS[pat][1]
        pat_edges = hp.PATTERNS[pat][0]
        pat_nonedges = [(a, b) for a in range(n_pat) for b in range(a + 1, n_pat)
                        if (a, b) not in {(min(x, y), max(x, y)) for x, y in pat_edges}]
        Gpg, Hpg, n_pat, n_edges_sym = build(ak, ar, edges, n, pat)
        print(f"\n=== pattern {pat!r}: n_pat={n_pat}  numSubgraphEdges(sym)={n_edges_sym}  "
              f"pat_edges={pat_edges}  pat_nonedges={pat_nonedges} ===")
        if Gpg is None:
            print("  (0-edge host, skipping)")
            continue

        # ---- COMPLETE mode: all four arrays at once -----------------------------------------
        ln, slots = raw_query(ak, ar, Gpg, Hpg, "complete", reorder)
        print(f"  return_isos_as='complete': len(result)={ln}")
        for k, s in enumerate(slots):
            if isinstance(s, np.ndarray):
                print(f"    slot[{k}] len={s.size:<8} head={s[:15].tolist()}")
            else:
                print(f"    slot[{k}] {s}")
        # interpret slots per the Chapel return (isoArr, isoMapper, srcPerIso, dstPerIso)
        isoArr, isoMapper = slots[0], slots[1]
        srcP = slots[2] if len(slots) > 2 and isinstance(slots[2], np.ndarray) else None
        dstP = slots[3] if len(slots) > 3 and isinstance(slots[3], np.ndarray) else None
        numIsos_v = isoArr.size // n_pat if isoArr.size % n_pat == 0 else None
        numIsos_e = (srcP.size // n_edges_sym) if (srcP is not None and n_edges_sym and srcP.size % n_edges_sym == 0) else None
        print(f"    => numIsos from slot0/n_pat={numIsos_v}   from srcPerIso/n_edges={numIsos_e}")

        if numIsos_v:
            V = isoArr.reshape(-1, n_pat)
            rep = [i for i in range(V.shape[0]) if len(set(V[i].tolist())) != n_pat]
            print(f"    slot0 as VERTICES [{V.shape[0]}x{n_pat}]: repeat-rows={len(rep)}; first {show}:")
            for i in range(min(show, V.shape[0])):
                vs = V[i].tolist()
                # induced check: count host edges among this row's vertex set
                cnt = sum(1 for a in range(n_pat) for b in range(a + 1, n_pat)
                          if (min(vs[a], vs[b]), max(vs[a], vs[b])) in hes and vs[a] != vs[b])
                flag = "  <-- REPEAT" if len(set(vs)) != n_pat else ("" if cnt == len(pat_edges) // 1 else "  <-- not induced?")
                print(f"        iso[{i}] verts={vs}  host-edges-among={cnt} (pattern has {len(pat_edges)})"
                      + (("  src="+str(srcP[i*n_edges_sym:(i+1)*n_edges_sym].tolist())) if srcP is not None else "")
                      + (("  dst="+str(dstP[i*n_edges_sym:(i+1)*n_edges_sym].tolist())) if dstP is not None else "")
                      + flag)

            # ---- YOUR HYPOTHESIS: are the 'repeated' numbers actually src/dst? --------------
            if rep:
                print(f"    HYPOTHESIS CHECK — first {min(show,len(rep))} REPEAT rows with their src/dst blocks:")
                for i in rep[:show]:
                    line = f"        iso[{i}] verts(slot0)={V[i].tolist()}"
                    if srcP is not None:
                        line += f"  src(slot2)={srcP[i*n_edges_sym:(i+1)*n_edges_sym].tolist()}"
                    if dstP is not None:
                        line += f"  dst(slot3)={dstP[i*n_edges_sym:(i+1)*n_edges_sym].tolist()}"
                    print(line)

        # ---- VERTICES mode: compare its slot0/slot1 to complete's ---------------------------
        ln2, slots2 = raw_query(ak, ar, Gpg, Hpg, "vertices", reorder)
        same0 = isinstance(slots2[0], np.ndarray) and slots2[0].size == isoArr.size and np.array_equal(slots2[0], isoArr)
        same1 = isinstance(slots2[1], np.ndarray) and slots2[1].size == isoMapper.size and np.array_equal(slots2[1], isoMapper)
        print(f"  return_isos_as='vertices': len(result)={ln2}; "
              f"slot0==complete.slot0? {same0}; slot1==complete.slot1(isoMapper)? {same1}")


def determinism(ak, ar, name, edges, n, pattern, reorder, runs):
    print("\n" + "#" * 90)
    print(f"# DETERMINISM: {name} pattern={pattern!r} reorder={reorder!r}  x{runs} runs (vertices mode)")
    print("#" * 90)
    Gpg, Hpg, n_pat, _ = build(ak, ar, edges, n, pattern)
    hashes, reps, counts = [], [], []
    for _ in range(runs):
        _, slots = raw_query(ak, ar, Gpg, Hpg, "vertices", reorder)
        a = slots[0]
        counts.append(a.size)
        V = a.reshape(-1, n_pat) if a.size % n_pat == 0 else None
        reps.append(0 if V is None else sum(1 for i in range(V.shape[0]) if len(set(V[i].tolist())) != n_pat))
        hashes.append(hash(a.tobytes()))
    print(f"  slot0 sizes : {counts}")
    print(f"  repeat-rows : {reps}")
    print(f"  slot0 identical across runs? {len(set(hashes)) == 1}   "
          f"(stable size, varying contents/repeats => race; all identical => deterministic)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ak-host", default="localhost")
    ap.add_argument("--ak-port", default=5555, type=int)
    ap.add_argument("--patterns", nargs="+", default=["p3", "p4", "claw", "triangle"])
    ap.add_argument("--reorder", default="None", choices=["None", "structural"])
    ap.add_argument("--big-n", type=int, default=150)
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()

    import arkouda as ak
    print(f"[dump] connecting to arkouda at {args.ak_host}:{args.ak_port} ...")
    ak.connect(args.ak_host, args.ak_port)
    import arachne as ar  # noqa: F401
    try:
        cfg = ak.get_config()
        print(f"[dump] numLocales={cfg.get('numLocales','?')} maxTaskPar={cfg.get('maxTaskPar', cfg.get('numPUs','?'))}")
    except Exception:
        pass

    try:
        # small known graphs — see exactly what each slot is
        for name, G in [("P3-path", nx.path_graph(3)), ("C5", nx.cycle_graph(5)),
                        ("K3-triangle", nx.complete_graph(3)), ("C4", nx.cycle_graph(4))]:
            e, n = edges_n(G)
            dump_graph(ak, ar, name, e, n, args.patterns, args.reorder)

        # LARGE graph — where the symptom shows
        big = nx.gnp_random_graph(args.big_n, 0.1, seed=1)
        be, bn = edges_n(big)
        dump_graph(ak, ar, f"gnp(n={args.big_n},p=0.1)", be, bn, args.patterns, args.reorder, show=15)
        determinism(ak, ar, f"gnp(n={args.big_n},p=0.1)", be, bn, "p3", args.reorder, args.runs)

        # Cora too, if cached (no download)
        cora_cache = REPO / "data/processed" / "Cora" / "processed" / "data.pt"
        if cora_cache.exists():
            from torch_geometric.datasets import Planetoid
            from torch_geometric.utils import to_networkx
            cora = Planetoid(root=str(REPO / "data/processed"), name="Cora")[0]
            ce, cn = edges_n(to_networkx(cora, to_undirected=True))
            determinism(ak, ar, "Cora", ce, cn, "p3", args.reorder, args.runs)
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass

    print("\n[dump] done — send the full stdout (ideally one SERIAL run and one PARALLEL run).")


if __name__ == "__main__":
    main()
