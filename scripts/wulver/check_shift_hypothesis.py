#!/usr/bin/env python
"""Is the parallel corruption a RECOVERABLE shift, or non-recoverable interleaving? WULVER.

Tests Bartosz's hypothesis directly: maybe the isos are all present but globally SHIFTED
(reshape boundary offset) -> recoverable. Decisive checks on p3 over gnp(150) (small enough
for a local networkx oracle), all over the WHOLE array:

  1. ELEMENT CONSERVATION: is the multiset of individual ints in the engine's slot0 identical
     to the local oracle's flattened embeddings? (Did the search find exactly the right
     vertices, just grouped wrong?)
  2. EMBEDDING RECOVERY: how many engine rows are valid induced wedges present in the oracle
     set; how many oracle wedges are MISSING; how many engine rows are garbage (repeat/not
     induced)? (Interleaving loses embeddings; a shift loses none.)
  3. GLOBAL-SHIFT recovery: for every roll k in 0..n_pat-1, reshape the rolled flat array and
     count repeats. If any k gives 0 repeats AND recovers the oracle set, it's a recoverable
     uniform shift. If no k helps, it's interleaving.
  4. REPEAT POSITIONS: are repeat rows clustered at regular boundaries (shift) or scattered
     (interleaving)?

Run under the PARALLEL server. Compare to a SERIAL run (should show 0 repeats, perfect match).

Usage: PYTHONPATH=. python scripts/wulver/check_shift_hypothesis.py --ak-host <h> --ak-port <p>
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets import hipermotif_backend as backend
from src.datasets import hipermotif_patterns as hp


def host_edge_set(edges):
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}


def valid_induced(row, hes, n_pat, pat_n_edges):
    if len(set(row.tolist())) != n_pat:
        return False
    cnt = sum(1 for a in range(n_pat) for b in range(a + 1, n_pat)
              if (min(row[a], row[b]), max(row[a], row[b])) in hes)
    return cnt == pat_n_edges


def analyze(name, flat, edges, n_pat, pat, oracle_emb):
    print("\n" + "=" * 80 + f"\n{name}: pattern={pat!r} n_pat={n_pat}\n" + "=" * 80)
    hes = host_edge_set(edges)
    pat_n_edges = len(hp.PATTERNS[pat][0])
    if flat.size % n_pat:
        print(f"  flat size {flat.size} not divisible by n_pat={n_pat} (!)")
    V = flat.reshape(-1, n_pat)
    nrows = V.shape[0]

    # 1. element conservation vs oracle
    eng_mult = Counter(flat.tolist())
    ora_flat = oracle_emb.reshape(-1).tolist()
    ora_mult = Counter(ora_flat)
    conserved = (eng_mult == ora_mult)
    print(f"  rows: engine={nrows}  oracle={oracle_emb.shape[0]}  (counts {'MATCH' if nrows==oracle_emb.shape[0] else 'DIFFER'})")
    print(f"  [1] ELEMENT multiset engine == oracle ? {conserved}"
          + ("" if conserved else f"  (engine-only e.g. {list((eng_mult-ora_mult).items())[:3]}, "
                                   f"oracle-only {list((ora_mult-eng_mult).items())[:3]})"))

    # 2. embedding recovery (as vertex-sets)
    ora_sets = Counter(frozenset(r) for r in oracle_emb.tolist())
    eng_valid = Counter()
    garbage = 0
    for i in range(nrows):
        if valid_induced(V[i], hes, n_pat, pat_n_edges):
            eng_valid[frozenset(V[i].tolist())] += 1
        else:
            garbage += 1
    missing = sum((ora_sets - eng_valid).values())
    extra = sum((eng_valid - ora_sets).values())
    print(f"  [2] garbage rows (repeat/not-induced) = {garbage}")
    print(f"      oracle embeddings MISSING from engine valid rows = {missing}")
    print(f"      engine valid rows NOT in oracle (invented)       = {extra}")

    # 3. global-shift recovery
    print("  [3] global-shift recovery (roll k, count repeat rows):")
    best = None
    for k in range(n_pat):
        rolled = np.roll(flat, k).reshape(-1, n_pat)
        reps = int(np.sum([len(set(rolled[i].tolist())) != n_pat for i in range(rolled.shape[0])]))
        print(f"        roll k={k}: repeat-rows={reps}")
        if best is None or reps < best[1]:
            best = (k, reps)
    print(f"      best roll k={best[0]} -> {best[1]} repeats "
          + ("=> RECOVERABLE by uniform shift" if best[1] == 0 else "=> NOT recovered by any uniform shift"))

    # 4. repeat positions (clustered vs scattered)
    rep_idx = [i for i in range(nrows) if len(set(V[i].tolist())) != n_pat]
    if rep_idx:
        gaps = np.diff(rep_idx)
        print(f"  [4] repeat rows: {len(rep_idx)}; first positions {rep_idx[:15]}")
        print(f"      gap between repeats: min={gaps.min()} max={gaps.max()} mean={gaps.mean():.1f} "
              f"(regular gaps => boundary/shift; irregular => scattered interleaving)")
    else:
        print("  [4] no repeat rows (clean)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ak-host", default="localhost")
    ap.add_argument("--ak-port", default=5555, type=int)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--reorder", default="None", choices=["None", "structural"])
    args = ap.parse_args()

    import arkouda as ak
    ak.connect(args.ak_host, args.ak_port)
    import arachne as ar  # noqa: F401
    try:
        cfg = ak.get_config()
        print(f"[shift] numLocales={cfg.get('numLocales','?')} maxTaskPar={cfg.get('maxTaskPar', cfg.get('numPUs','?'))}")
    except Exception:
        pass

    G = nx.convert_node_labels_to_integers(nx.gnp_random_graph(args.n, 0.1, seed=1))
    edges = [(int(u), int(v)) for u, v in G.edges()]
    pat = "p3"
    n_pat = hp.PATTERNS[pat][1]
    try:
        # local oracle (networkx) — the ground-truth embedding set, deterministic
        oracle_emb, _ = hp.induced_embeddings(G, pat)   # [num, n_pat]
        # engine, parallel
        Gpg, Hpg, _, _ = (lambda e, n: (
            backend._build_propgraph(ak, ar, *backend._symmetrize(e, n)),
            backend._build_propgraph(ak, ar, *backend._symmetrize(hp.PATTERNS[pat][0], n_pat)),
            None, None))(edges, G.number_of_nodes())
        iso, _ = backend._run_iso(ar, Gpg, Hpg, args.reorder)
        analyze(f"gnp(n={args.n},p=0.1)", np.asarray(iso, dtype=np.int64), edges, n_pat, pat, oracle_emb)
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass
    print("\n[shift] done.")


if __name__ == "__main__":
    main()
