"""Beyond-ORCA correctness oracle for the long-cycle experiment (EXP-A gate).

ORCA caps at size-4, so it CANNOT validate C5-C8. This networkx oracle does. For a small
host it reports, per cycle length n in {5,6,7,8}:
  - induced_Cn   : # of n-subsets inducing exactly a chordless cycle (what an INDUCED
                   subgraph-isomorphism engine like ar.subgraph_isomorphism returns), and
  - allsimple_Cn : # of simple cycles of length n (chorded allowed; what nx.simple_cycles
                   and the local CSL accuracy demo count).
  - expected HiPerMotif embeddings (induced) = induced_Cn * |Aut(Cn)| = induced_Cn * 2n.

How to use on Wulver: run HiPerMotif on the SAME small host with --patterns c5 c6 c7 c8,
take n_embeddings for Cn, divide by 2n, and compare to induced_Cn here. If it matches
induced_Cn the engine is induced (expected); if it matches allsimple_Cn it is monomorphism.
Either way this resolves the induced-vs-all-simple semantics before trusting at-scale runs.

  conda run -n motif-mpnn python scripts/wulver/cycle_oracle.py --graph reg --n 200 --d 6
  conda run -n motif-mpnn python scripts/wulver/cycle_oracle.py --edge-file path/to/edges.txt
"""
from __future__ import annotations
import argparse
import itertools
import networkx as nx

LENS = [5, 6, 7, 8]


def induced_and_simple_counts(g: nx.Graph, lens=LENS):
    """Return {n: (induced_count, allsimple_count)} over simple cycles up to max(lens)."""
    induced = {n: 0 for n in lens}
    allsimple = {n: 0 for n in lens}
    for cyc in nx.simple_cycles(g, length_bound=max(lens)):
        n = len(cyc)
        if n not in induced:
            continue
        allsimple[n] += 1
        # chordless (induced) iff the only edges among the n cycle vertices are the n
        # consecutive ones, i.e. no edge between non-adjacent-in-cycle vertices.
        s = set(cyc)
        ring = set(frozenset((cyc[i], cyc[(i + 1) % n])) for i in range(n))
        chord = any(frozenset((u, v)) not in ring and g.has_edge(u, v)
                    for u, v in itertools.combinations(s, 2))
        if not chord:
            induced[n] += 1
    return {n: (induced[n], allsimple[n]) for n in lens}


def _selftest():
    # a pure 8-cycle: exactly 1 induced C8, 0 of C5/C6/C7; embeddings = 1*16 = 16
    g = nx.cycle_graph(8)
    c = induced_and_simple_counts(g)
    assert c[8] == (1, 1), c[8]
    assert c[5] == (0, 0) and c[6] == (0, 0) and c[7] == (0, 0), c
    # K5: every 5-subset is K5; simple 5-cycles exist but NONE are induced (all chorded)
    k5 = nx.complete_graph(5)
    c5 = induced_and_simple_counts(k5)[5]
    assert c5[0] == 0 and c5[1] > 0, c5  # induced C5 = 0, all-simple C5 > 0
    print("[selftest] cycle oracle OK (C8 graph, K5)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="reg", help="reg (random d-regular) | path via --edge-file")
    ap.add_argument("--edge-file", default=None)
    ap.add_argument("--n", type=int, default=200, help="nodes for --graph reg")
    ap.add_argument("--d", type=int, default=6, help="degree for --graph reg")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    _selftest()
    if args.edge_file:
        g = nx.Graph()
        with open(args.edge_file) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    g.add_edge(int(p[0]), int(p[1]))
        name = args.edge_file
    else:
        g = nx.random_regular_graph(args.d, args.n, seed=args.seed)
        name = f"reg(n={args.n},d={args.d},seed={args.seed})"
    g.remove_edges_from(nx.selfloop_edges(g))

    print(f"\nhost = {name}: |V|={g.number_of_nodes()} |E|={g.number_of_edges()}")
    print(f"{'cycle':6} {'induced':>10} {'all-simple':>12} {'expected HM emb (induced*2n)':>30}")
    for n in LENS:
        ind, alls = induced_and_simple_counts(g)[n]
        print(f"  C{n:<4} {ind:>10} {alls:>12} {ind * 2 * n:>30}")


if __name__ == "__main__":
    main()
