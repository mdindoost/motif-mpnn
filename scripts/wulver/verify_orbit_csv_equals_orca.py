#!/usr/bin/env python
"""End-to-end readiness gate: HiPerMotif orbit features == ORCA on the REAL benchmark datasets.

For each dataset it loads the graph(s) via the EXACT same path generate_motifs.py uses, then
computes the per-node 15-orbit rows BOTH ways and asserts they are identical:
  - ORCA       : local oracle (src/datasets/orca_orbits.count_orbits), the validated ground truth.
  - HiPerMotif : cluster engine (src/datasets/hipermotif_backend.count_orbits_hipermotif), which
                 defaults to reorder_type="None" (the correctness path; order-invariant, so a
                 PARALLEL server is fine — orbit counts don't depend on embedding order).

If every dataset MATCHES, the HiPerMotif backend produces byte-identical orbit features to the
ORCA features used for the Phase-1b accuracy tables, i.e. the accuracy results are
HiPerMotif-backed and we are ready for the scale experiments.

RUN ON WULVER (arkouda + arachne + torch_geometric + networkx; g++ for the ORCA oracle).
generate_motifs.py does NOT connect to arkouda; this wrapper does (ak.connect), and the
backend then uses arkouda's module-global client.

Usage (from repo root):
  PYTHONPATH=. python scripts/wulver/verify_orbit_csv_equals_orca.py --ak-host <h> --ak-port <p>
  # subset / quick check:
  PYTHONPATH=. python scripts/wulver/verify_orbit_csv_equals_orca.py --ak-host <h> --ak-port <p> \
      --datasets cora proteins --max-graphs 50
NOTE: HiPerMotif "None" is slow (Cora ~40s; Pubmed and full TU sets can take many minutes).
Start with a subset; use --max-graphs to cap TU graphs per dataset for a fast gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# import generate_motifs.py as a module (it only runs main() under __main__, so importing is safe)
_gm_path = REPO / "scripts" / "preprocess" / "generate_motifs.py"
_spec = importlib.util.spec_from_file_location("generate_motifs", _gm_path)
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)

PLANETOID = ["cora", "citeseer", "pubmed"]
TU = ["proteins", "nci1", "enzymes"]
ALL = PLANETOID + TU


def _rows_to_dict(rows, graph_id):
    """rows: list of (node_id, k, motif_id, count) (nonzero only) -> {(graph_id,node,k,mid):count}."""
    return {(graph_id, int(n), int(k), int(m)): int(c) for (n, k, m, c) in rows}


def _diff(orca: dict, hm: dict):
    """Return (n_match, only_orca, only_hm, val_mismatch, examples)."""
    ko, kh = set(orca), set(hm)
    only_o = ko - kh
    only_h = kh - ko
    common = ko & kh
    mism = [k for k in common if orca[k] != hm[k]]
    examples = []
    for k in list(only_o)[:3]:
        examples.append(f"ORCA-only {k}={orca[k]}")
    for k in list(only_h)[:3]:
        examples.append(f"HM-only {k}={hm[k]}")
    for k in mism[:3]:
        examples.append(f"value {k}: ORCA={orca[k]} HM={hm[k]}")
    return len(common) - len(mism), len(only_o), len(only_h), len(mism), examples


def compare_dataset(ds: str, root: Path, graphlet_size: int, max_graphs: int):
    """Load ds, compute ORCA vs HiPerMotif orbit rows per graph, compare. Returns (ok, msg)."""
    t0 = time.perf_counter()
    if ds in PLANETOID:
        G_nx, n = gm._load_planetoid_as_nx(ds, root)
        graphs = [(0, G_nx, n)]
    elif ds in TU:
        graphs = gm._load_tu_graphs_as_nx(ds, root)
        if max_graphs and max_graphs > 0:
            graphs = graphs[:max_graphs]
    else:
        return False, f"unknown dataset {ds!r}"

    orca_all: dict = {}
    hm_all: dict = {}
    orca_t = hm_t = 0.0
    for gid, g_nx, n in graphs:
        s = time.perf_counter()
        orca_all.update(_rows_to_dict(gm._count_orbits_single(g_nx, n, graphlet_size, "orca"), gid))
        orca_t += time.perf_counter() - s
        s = time.perf_counter()
        hm_all.update(_rows_to_dict(gm._count_orbits_single(g_nx, n, graphlet_size, "hipermotif"), gid))
        hm_t += time.perf_counter() - s

    n_match, only_o, only_h, mism, examples = _diff(orca_all, hm_all)
    ok = (only_o == 0 and only_h == 0 and mism == 0)
    wall = time.perf_counter() - t0
    tag = "[MATCH]   " if ok else "[MISMATCH]"
    msg = (f"{tag} {ds:9} graphs={len(graphs):<5} nonzero_rows(orca/hm)={len(orca_all)}/{len(hm_all)}  "
           f"match={n_match} orca_only={only_o} hm_only={only_h} val_mismatch={mism}  "
           f"[orca {orca_t:.1f}s | hipermotif {hm_t:.1f}s | wall {wall:.1f}s]")
    if not ok and examples:
        msg += "\n           e.g. " + "; ".join(examples)
    return ok, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ak-host", default="localhost")
    ap.add_argument("--ak-port", default=5555, type=int)
    ap.add_argument("--datasets", nargs="+", default=ALL,
                    help=f"subset of {ALL} (default: all 6)")
    ap.add_argument("--max-graphs", type=int, default=0,
                    help="cap graphs per TU dataset for a fast gate (0 = all)")
    ap.add_argument("--graphlet-size", type=int, default=4, choices=[4, 5])
    ap.add_argument("--root", default=str(REPO / "data/processed"),
                    help="PyG dataset root (must already be on disk; no internet on compute nodes)")
    args = ap.parse_args()

    import arkouda as ak
    print(f"[orbit-gate] connecting to arkouda at {args.ak_host}:{args.ak_port} ...")
    ak.connect(args.ak_host, args.ak_port)
    import arachne  # noqa: F401  (fail loudly here if arachne missing)
    try:
        cfg = ak.get_config()
        print(f"[orbit-gate] numLocales={cfg.get('numLocales','?')} "
              f"maxTaskPar={cfg.get('maxTaskPar', cfg.get('numPUs','?'))}")
    except Exception:
        pass

    print(f"[orbit-gate] datasets={args.datasets} max_graphs={args.max_graphs or 'all'} "
          f"graphlet_size={args.graphlet_size}")
    results = {}
    try:
        for ds in args.datasets:
            try:
                ok, msg = compare_dataset(ds, Path(args.root), args.graphlet_size, args.max_graphs)
            except Exception as ex:  # noqa: BLE001 — report and continue
                ok, msg = False, f"[ERROR]    {ds:9} {type(ex).__name__}: {ex}"
            results[ds] = ok
            print("  " + msg)
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass

    all_ok = bool(results) and all(results.values())
    print("\n" + "=" * 64)
    print("OVERALL: " + ("ALL DATASETS MATCH ORCA — HiPerMotif features are ready for the "
                         "accuracy + scale experiments."
                         if all_ok else
                         "MISMATCH — do NOT use HiPerMotif features until resolved. "
                         f"({sum(1 for v in results.values() if v)}/{len(results)} matched)"))
    print("=" * 64)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
