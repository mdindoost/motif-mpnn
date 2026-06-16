#!/usr/bin/env python
"""Scale-experiment PREP sanity check for the HiPerMotif orbit-feature pipeline. RUN ON WULVER.

The equivalence gate (verify_hipermotif_equals_orca.py) already passed sequentially and in
parallel on reorder_type="None". This script is the pre-sweep go/no-go: it confirms the
orbit-extraction pipeline works end-to-end on the cluster on real benchmark graphs, AND it
compares the two reorder paths so we know how to run the full sweep:

  * reorder_type="None"        -> the CORRECTNESS path (no pattern-vertex permutation;
                                  output column j == pattern vertex j; isoMapper = identity).
  * reorder_type="structural"  -> faster search, but permutes pattern columns; the backend
                                  inverts the isoMapper (result[1]) to recover orbits.

For each graph it computes ORCA orbits as ground truth (timed), runs HiPerMotif with BOTH
reorder paths (each timed around the extraction only), and checks each against ORCA with
exact integer array equality. The verdict tells us whether we can legitimately TIME both
paths in the real sweep (both correct) or must pin everything to "None".

WHY THE TWO PATHS MATTER FOR THE PAPER: "None" is correctness-critical for the feature
values; "structural" is the faster path we want for the SCALING/TIMING story so HiPerMotif's
performance is not understated (see CLAUDE.md section 13). This script measures both.

reorder_type is a PARAMETER of count_orbits_hipermotif (default = DEFAULT_REORDER_TYPE="None"),
NOT a global we mutate — so we just pass it per call and never leave any module state changed.
We assert DEFAULT_REORDER_TYPE is untouched at the end as a courtesy check.

NO INTERNET: Wulver compute nodes have none. We only load a dataset if its PyG processed
cache is already on disk; otherwise we SKIP it with a clear message (constructing the dataset
would otherwise try to download). The arkouda/arachne dependency is import-guarded inside the
backend, so this file imports fine locally even though it only RUNS on the cluster.

Prerequisites on Wulver:
  1. arkouda server running; note its host/port.
  2. conda env with arkouda + arachne + torch_geometric + networkx + this repo on PYTHONPATH.
  3. g++ available (the ORCA oracle binary builds on first use).

Usage (from repo root):
  PYTHONPATH=. python scripts/wulver/sanity_check.py --ak-host <host> --ak-port <port>
  PYTHONPATH=. python scripts/wulver/sanity_check.py --ak-host <host> --ak-port <port> \
      --paths None structural        # (default: both)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the repo importable whether or not PYTHONPATH=. was set.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# These import cleanly WITHOUT arkouda/arachne: the backend import-guards the cluster libs and
# only raises when count_orbits_hipermotif is actually invoked; ORCA builds its binary on use.
from src.datasets.orca_orbits import count_orbits as orca_count                 # [n,15] oracle
from src.datasets.hipermotif_backend import (                                   # Wulver engine
    count_orbits_hipermotif,
    DEFAULT_REORDER_TYPE,
)
from src.datasets.hipermotif_patterns import N_ORBITS_SIZE4                      # == 15

# Exact on-disk PyG cache locations (mirrors verify_hipermotif_equals_orca.py so we read the
# SAME data). If a cache file is absent we skip that graph rather than triggering a download.
CORA_ROOT = REPO / "data/processed"
CORA_CACHE = CORA_ROOT / "Cora" / "processed" / "data.pt"
PROT_ROOT = REPO / "data/processed/pyg/proteins"
PROT_CACHE = PROT_ROOT / "PROTEINS" / "processed" / "data.pt"


def _edges_n(G):
    """networkx graph -> (edge list with 0..N-1 integer ids, num_nodes). Same convention the
    gate uses, so HiPerMotif and ORCA see identical inputs."""
    import networkx as nx
    G = nx.convert_node_labels_to_integers(G)
    return [(int(u), int(v)) for u, v in G.edges()], G.number_of_nodes()


def load_graphs():
    """Yield (name, edges, num_nodes) for benchmark graphs whose PyG cache is ON DISK.

    Cora is small enough that ORCA is fast; PROTEINS[0] is a single small molecular graph.
    Both are checked for an existing processed cache first (no-internet safety)."""
    # torch_geometric is imported lazily so this module stays importable without it.
    from torch_geometric.utils import to_networkx

    if CORA_CACHE.exists():
        from torch_geometric.datasets import Planetoid
        cora = Planetoid(root=str(CORA_ROOT), name="Cora")[0]
        edges, n = _edges_n(to_networkx(cora, to_undirected=True))
        yield ("Cora", edges, n)
    else:
        print(f"[skip] Cora cache not found at {CORA_CACHE} — skipping (no internet to download).")

    if PROT_CACHE.exists():
        from torch_geometric.datasets import TUDataset
        prot = TUDataset(root=str(PROT_ROOT), name="PROTEINS")[0]
        edges, n = _edges_n(to_networkx(prot, to_undirected=True))
        yield ("PROTEINS[0]", edges, n)
    else:
        print(f"[skip] PROTEINS cache not found at {PROT_CACHE} — skipping (no internet to download).")


def compare_one(name, edges, n, paths):
    """Compute ORCA ground truth (timed) then HiPerMotif for each reorder path (timed),
    and check exact array equality. Returns {reorder_type: ("match"|"mismatch"|"error",
    seconds_or_None, detail_or_None)}."""
    t0 = time.perf_counter()
    O = orca_count(edges, n)                       # [n,15] int64 ground truth
    orca_t = time.perf_counter() - t0
    if O.shape != (n, N_ORBITS_SIZE4):
        raise RuntimeError(f"ORCA returned shape {O.shape}, expected ({n},{N_ORBITS_SIZE4})")

    print(f"\n=== {name}: N={n} E={len(edges)} | ORCA {orca_t:.3f}s ===")
    results = {}
    for rt in paths:
        try:
            t0 = time.perf_counter()
            HM = count_orbits_hipermotif(edges, n, reorder_type=rt)   # per-call param; no globals
            hm_t = time.perf_counter() - t0
        except Exception as ex:  # noqa: BLE001 — report any cluster/engine error, don't crash the run
            print(f"  [ERROR]    reorder_type={rt!r}: {type(ex).__name__}: {ex}")
            results[rt] = ("error", None, f"{type(ex).__name__}: {ex}")
            continue

        if HM.shape == O.shape and np.array_equal(HM, O):
            speed = f"(ORCA/{rt} = {orca_t / hm_t:.2f}x)" if hm_t > 0 else ""
            print(f"  [MATCH]    reorder_type={rt:<11} {hm_t:.3f}s  vs ORCA {orca_t:.3f}s  {speed}")
            results[rt] = ("match", hm_t, None)
        else:
            if HM.shape != O.shape:
                detail = f"shape HM={HM.shape} vs ORCA={O.shape}"
            else:
                d = np.argwhere(HM != O)
                r, c = int(d[0][0]), int(d[0][1])
                detail = f"first diff node {r} orbit {c}: HM={int(HM[r, c])} ORCA={int(O[r, c])}"
            print(f"  [MISMATCH] reorder_type={rt:<11} {hm_t:.3f}s  {detail}")
            results[rt] = ("mismatch", hm_t, detail)
    return results


def final_verdict(all_results, paths):
    """Aggregate per-graph results into a go/no-go for the sweep."""
    print("\n" + "=" * 64)
    if not all_results:
        print("VERDICT: no benchmark graphs available on disk — nothing checked.")
        print("=" * 64)
        return False

    def all_match(rt):
        return all(res.get(rt, ("missing",))[0] == "match" for res in all_results.values())

    none_ok = ("None" in paths) and all_match("None")
    struct_ok = ("structural" in paths) and all_match("structural")

    # mean extraction time per path across graphs (only over graphs that produced a time)
    for rt in paths:
        times = [res[rt][1] for res in all_results.values()
                 if res.get(rt, (None, None))[1] is not None]
        if times:
            print(f"  mean {rt:<11} extraction: {np.mean(times):.3f}s over {len(times)} graph(s)")

    if none_ok and struct_ok:
        print("VERDICT: BOTH reorder paths match ORCA -> both timing paths are valid; the sweep")
        print("         may time 'None' (correctness) AND 'structural' (speed) and report both.")
        ok = True
    elif none_ok:
        print("VERDICT: only 'None' matches ORCA -> use reorder_type='None' EVERYWHERE for the")
        print("         feature sweep. Do NOT trust 'structural' results for correctness.")
        ok = True
    else:
        print("VERDICT: the 'None' (correctness) path did NOT match ORCA -> STOP. Do not run the")
        print("         sweep; investigate the backend before producing any paper numbers.")
        ok = False
    print("=" * 64)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ak-host", default="localhost", help="arkouda server host")
    ap.add_argument("--ak-port", default=5555, type=int, help="arkouda server port")
    ap.add_argument("--paths", nargs="+", default=["None", "structural"],
                    choices=["None", "structural"],
                    help="reorder_type paths to test (default: both)")
    args = ap.parse_args()

    # arkouda imported here (not at module top) so this file stays importable locally.
    import arkouda as ak
    print(f"[sanity] DEFAULT_REORDER_TYPE = {DEFAULT_REORDER_TYPE!r}; testing paths {args.paths}")
    print(f"[sanity] connecting to arkouda at {args.ak_host}:{args.ak_port} ...")
    ak.connect(args.ak_host, args.ak_port)

    all_results = {}
    try:
        for name, edges, n in load_graphs():
            all_results[name] = compare_one(name, edges, n, args.paths)
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass

    # courtesy invariant: we pass reorder_type per call, so the module constant is untouched.
    assert DEFAULT_REORDER_TYPE == "None", "DEFAULT_REORDER_TYPE was mutated unexpectedly"

    ok = final_verdict(all_results, args.paths)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
