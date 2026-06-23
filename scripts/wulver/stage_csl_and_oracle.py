"""Stage the 10 CSL class-representative graphs + a networkx cycle oracle, so HiPerMotif's
C5-C8 extraction can be validated ON CSL (closes the "HiPerMotif is necessary, not just
cycles" gap: the accuracy half computes cycles with networkx; this confirms HiPerMotif
reproduces them and yields the same 10 distinct separating signatures).

Writes, under data/precompute/csl_stage/:
  - csl_s<skip>.txt   : one edge-file per CSL class (skip s), 'u v' per line.
  - csl_cycle_reference.json : per-class induced + all-simple counts for C5..C8, and the
                               distinct-signature check on (C6,C7,C8).

On Wulver, for each csl_s<skip>.txt run HiPerMotif --patterns c5 c6 c7 c8, divide each Cn's
n_embeddings by 2n, and confirm it matches the 'induced' (or 'all_simple') counts here AND
that the 10 (C6,C7,C8) signatures are all distinct.

  conda run -n motif-mpnn python scripts/wulver/stage_csl_and_oracle.py
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.datasets.expressivity import make_csl_single, CSL_SKIPS

CSL_N = 41
LENS = [3, 4, 5, 6, 7, 8]   # full cycle spectrum; C3=triangle, C4=4-cycle (ORCA size-4)
ORCA_COVERABLE = (3, 4, 5)  # cycle lengths within ORCA's <=5-vertex reach
FULL = (3, 4, 5, 6, 7, 8)


def _nx(ei, n):
    g = nx.Graph(); g.add_nodes_from(range(n))
    for u, v in ei.t().tolist():
        if u != v:
            g.add_edge(int(u), int(v))
    return g


def counts(g, lens=LENS):
    ind = {L: 0 for L in lens}; alls = {L: 0 for L in lens}
    for cyc in nx.simple_cycles(g, length_bound=max(lens)):
        L = len(cyc)
        if L not in ind:
            continue
        alls[L] += 1
        ring = set(frozenset((cyc[i], cyc[(i + 1) % L])) for i in range(L))
        chord = any(frozenset((u, v)) not in ring and g.has_edge(u, v)
                    for u, v in itertools.combinations(cyc, 2))
        if not chord:
            ind[L] += 1
    return ind, alls


def main():
    out = ROOT / "data" / "precompute" / "csl_stage"
    out.mkdir(parents=True, exist_ok=True)
    ref = {"n": CSL_N, "skips": list(CSL_SKIPS), "classes": {}}
    ind_sigs, all_sigs = [], []
    print(f"staging {len(CSL_SKIPS)} CSL classes (n={CSL_N}) -> {out}")
    for s in CSL_SKIPS:
        ei = make_csl_single(n=CSL_N, s=s)
        g = _nx(ei, CSL_N)
        # edge-file
        ef = out / f"csl_s{s}.txt"
        with open(ef, "w") as f:
            f.write("\n".join(f"{u} {v}" for u, v in g.edges()) + "\n")
        ind, alls = counts(g)
        ref["classes"][str(s)] = {
            "edge_file": ef.name, "n_edges": g.number_of_edges(),
            "induced":    {f"C{L}": ind[L] for L in LENS},
            "all_simple": {f"C{L}": alls[L] for L in LENS},
            "expected_hipermotif_emb_induced": {f"C{L}": ind[L] * 2 * L for L in LENS},
        }
        ind_sigs.append(tuple(ind[L] for L in FULL))
        all_sigs.append(tuple(alls[L] for L in FULL))
    # distinctness over the FULL spectrum (separating) vs the ORCA-coverable subset
    def ndist(sigs, keep):
        idx = [FULL.index(L) for L in keep]
        return len({tuple(s[i] for i in idx) for s in sigs})
    ref["distinct_signatures"] = {
        "induced_full_C3C8": ndist(ind_sigs, FULL),
        "induced_orca_coverable_C3C5": ndist(ind_sigs, ORCA_COVERABLE),
        "all_simple_full_C3C8": ndist(all_sigs, FULL),
        "all_simple_orca_coverable_C3C5": ndist(all_sigs, ORCA_COVERABLE),
        "of": len(CSL_SKIPS),
    }
    (out / "csl_cycle_reference.json").write_text(json.dumps(ref, indent=2))
    print("\ndistinct signatures over 10 CSL classes (want full=10/10, ORCA-coverable<10):")
    print(f"  induced   : full C3-C8 = {ndist(ind_sigs, FULL)}/10   "
          f"ORCA-coverable C3-C5 = {ndist(ind_sigs, ORCA_COVERABLE)}/10")
    print(f"  all-simple: full C3-C8 = {ndist(all_sigs, FULL)}/10   "
          f"ORCA-coverable C3-C5 = {ndist(all_sigs, ORCA_COVERABLE)}/10")
    print("  => C6-C8 COMPLETE the spectrum to full separation (the beyond-ORCA part).")
    print(f"reference -> {out/'csl_cycle_reference.json'}")


if __name__ == "__main__":
    main()
