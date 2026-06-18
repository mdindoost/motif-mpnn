#!/usr/bin/env python
"""Scale-experiment timing harness for HiPerMotif orbit extraction. RUN ON WULVER.

Times exact size-4 graphlet-orbit extraction on ONE shared-memory node (HiPerMotif is
shared-memory; not distributed). One graph per invocation; writes raw CSV rows so the figure
script is just a grouping of the data. reorder_type="None" everywhere (the validated correct
path). arkouda/arachne are import-guarded (only touched inside main), so this file imports
locally for inspection but only RUNS on Wulver.

KEY DESIGN POINTS (deliberate):
  * Graph LOADING is timed separately and EXCLUDED from extraction timing (loading is not the
    contribution). No internet on compute nodes: graphs must already be on disk.
  * HiPerMotif: each of the 9 patterns is timed individually around the ar.subgraph_isomorphism
    call, plus a TOTAL row. We read result[0].size (a pdarray attribute) and DO NOT call
    .to_ndarray() — so we never pull the (possibly billions-of-ints) embedding array to the
    client. That times engine work and avoids client-side OOM; the SERVER still materializes it
    (that is the memory ceiling we want to measure).
  * Threads are a SERVER-LAUNCH property (CHPL_RT_NUM_THREADS_PER_LOCALE -> ak.get_config()
    ['maxTaskPar']). This script CANNOT set it; it READS maxTaskPar, records it, and WARNS if it
    != the --threads label. The driver sets it by relaunching the server per thread count.
  * Memory is measured SERVER-SIDE (arkouda), not client RSS (HiPerMotif runs in the server).
  * Robust: any OOM/timeout/exception -> a status=FAILED row with the reason, then continue. We
    WANT the ceiling recorded.
  * Cost axis: per-pattern embedding COUNT (raw isomorphisms) is the work proxy; total over
    patterns is the figure x-axis (NOT |E| — cost is driven by the number of size-4 subgraphs).

Usage:
  PYTHONPATH=. python scripts/wulver/bench_orbits.py --backend hipermotif --threads 128 \
      --graph ogbn-arxiv --runs 3 --out results/scale/bench.csv --ak-host <h> --ak-port <p>
  PYTHONPATH=. python scripts/wulver/bench_orbits.py --backend orca --graph cora --runs 3 \
      --out results/scale/bench.csv
  # arbitrary large host from a staged edge-list (SNAP etc.):
  PYTHONPATH=. python scripts/wulver/bench_orbits.py --backend hipermotif --threads 128 \
      --graph livejournal --edge-file /path/com-lj.ungraph.txt --runs 3 --out ... --ak-host ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# These import WITHOUT arkouda (backend import-guards the cluster libs; ORCA builds on use).
from src.datasets import hipermotif_backend as backend
from src.datasets import hipermotif_patterns as hp
from src.datasets.orca_orbits import count_orbits as orca_count

CSV_FIELDS = ["graph", "n_nodes", "n_edges", "backend", "threads_label", "maxtaskpar_actual",
              "pattern", "run_idx", "py_seconds", "n_embeddings", "server_mem_gb",
              "status", "reason"]


# --------------------------------------------------------------------------------------------
# Graph loading (EXCLUDED from timing). No downloads: everything must be on disk.
# --------------------------------------------------------------------------------------------
def load_graph(name: str, edge_file: str | None, n_synth: int, p_synth: float,
               m_synth: int, seed: int):
    """Return (edges:list[(u,v)] with ids 0..N-1, num_nodes). Loading is not timed."""
    import networkx as nx

    if edge_file:  # arbitrary staged edge-list (SNAP-style): "u v" per line, '#' comments
        raw = set()
        nodes = {}
        def rid(x):
            if x not in nodes:
                nodes[x] = len(nodes)
            return nodes[x]
        with open(edge_file) as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in "#%":
                    continue
                a, b = line.split()[:2]
                u, v = rid(a), rid(b)
                if u != v:
                    raw.add((min(u, v), max(u, v)))
        return [(u, v) for (u, v) in raw], len(nodes)

    if name.startswith("gnp"):
        G = nx.gnp_random_graph(n_synth, p_synth, seed=seed)
    elif name.startswith("ba"):  # Barabasi-Albert: degree skew (realistic load imbalance)
        G = nx.barabasi_albert_graph(n_synth, m_synth, seed=seed)
    elif name in ("cora", "citeseer", "pubmed"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_motifs", REPO / "scripts" / "preprocess" / "generate_motifs.py")
        gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
        G, n = gm._load_planetoid_as_nx(name, REPO / "data/processed")
        G = nx.convert_node_labels_to_integers(G)
        return [(int(u), int(v)) for u, v in G.edges()], n
    elif name.startswith("ogbn-"):
        from ogb.nodeproppred import NodePropPredDataset  # needs ogb + dataset pre-staged
        d = NodePropPredDataset(name=name, root=str(REPO / "data/processed/ogb"))
        ei = d[0][0]["edge_index"]
        n = int(d[0][0]["num_nodes"])
        raw = set()
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            if u != v:
                raw.add((min(int(u), int(v)), max(int(u), int(v))))
        return list(raw), n
    else:
        raise ValueError(f"unknown --graph {name!r}; pass --edge-file for a staged edge-list")
    G = nx.convert_node_labels_to_integers(G)
    return [(int(u), int(v)) for u, v in G.edges()], G.number_of_nodes()


# --------------------------------------------------------------------------------------------
def server_mem_gb(ak) -> float:
    """Arkouda SERVER bytes used -> GB. HiPerMotif runs in the server, not this client process,
    so client RSS is meaningless. Tries known API names; returns -1 if unavailable."""
    for fn in ("get_mem_used",):
        try:
            return float(getattr(ak, fn)()) / 1e9
        except Exception:
            pass
    return -1.0


def predict_embeddings(pattern, deg, arr):
    """Cheap UPPER-BOUND on #embeddings for the degree-driven OPEN patterns (the ones that
    explode on hub-heavy hosts: a degree-d hub alone makes ~C(d,3) claws). Returns None for
    patterns bounded by triangle/clique counts (always run those). float64 avoids int64 overflow
    on huge hubs. Used to SKIP an explosive pattern BEFORE launching it — a client-side timeout
    can't help because the server would keep computing/OOMing."""
    d = deg.astype(np.float64)
    if pattern == "edge":
        return float(d.sum() / 2.0)
    if pattern == "p3":
        return float((d * (d - 1) / 2.0).clip(min=0).sum())
    if pattern == "claw":
        return float((d * (d - 1) * (d - 2) / 6.0).clip(min=0).sum())
    if pattern == "p4" and arr.size:
        return float(((d[arr[:, 0]] - 1) * (d[arr[:, 1]] - 1)).clip(min=0).sum())
    return None


def bench_hipermotif(ak, ar, edges, n, base, emit, measure_mem=True, max_emb=2e8):
    """Per-pattern timed ar.subgraph_isomorphism (reorder='None'); never pulls the array to
    the client (reads pdarray.size). Appends per-pattern + TOTAL rows into `rows`.
    measure_mem=False (the --no-mem escape hatch) skips all server-memory sampling, so a missing
    or slow ak.get_mem_used() can never block or perturb the timing runs (mem column = -1).
    A pattern whose predicted upper-bound #embeddings exceeds `max_emb` is SKIPPED (FAILED row)
    so a hub-driven explosion (e.g. claw on a web graph) records a ceiling instead of OOMing."""
    def _mem():
        return server_mem_gb(ak) if measure_mem else -1.0
    src, dst = backend._symmetrize(edges, n)
    if not src:
        emit({**base, "pattern": "TOTAL", "py_seconds": 0.0, "n_embeddings": 0,
              "server_mem_gb": _mem(), "status": "OK", "reason": ""})
        return
    arr = np.asarray(edges, dtype=np.int64)
    deg = np.bincount(arr.ravel(), minlength=n).astype(np.int64)
    G = backend._build_propgraph(ak, ar, src, dst)
    total_t, total_emb, peak_mem = 0.0, 0, _mem()
    for pat in hp.PATTERN_NAMES:
        n_pat = hp.PATTERNS[pat][1]
        pred = predict_embeddings(pat, deg, arr)
        if pred is not None and pred > max_emb:
            print(f"  [skip] {base['graph']} {pat}: predicted ~{pred:.2e} > cap {max_emb:.1e} "
                  f"-> ceiling row (would be ~{pred/2e5/60:.0f} min at ~200K/s)", flush=True)
            emit({**base, "pattern": pat, "py_seconds": -1, "n_embeddings": -1,
                  "server_mem_gb": _mem(), "status": "FAILED",
                  "reason": f"predicted ~{pred:.2e} embeddings > cap {max_emb:.1e}; "
                            f"skipped (hub-driven explosion, node protection)"})
            continue
        psrc, pdst = backend._symmetrize(hp.PATTERNS[pat][0], n_pat)
        H = backend._build_propgraph(ak, ar, psrc, pdst)
        _pm = f"~{pred:.1e}" if pred is not None else "n/a"
        print(f"  [run ] {base['graph']} {pat} run{base['run_idx']} (predicted {_pm}) ...", flush=True)
        try:
            t0 = time.perf_counter()
            result = ar.subgraph_isomorphism(
                G, H, return_isos_as="vertices", algorithm_type="si", reorder_type="None")
            dt = time.perf_counter() - t0
            n_emb = int(result[0].size) // n_pat      # pdarray.size -> no client transfer
            mem = _mem()
            peak_mem = max(peak_mem, mem)
            emit({**base, "pattern": pat, "py_seconds": round(dt, 6),
                  "n_embeddings": n_emb, "server_mem_gb": round(mem, 3),
                  "status": "OK", "reason": ""})
            total_t += dt
            total_emb += n_emb
            try:
                del result
            except Exception:
                pass
        except Exception as ex:  # OOM / engine error -> record the ceiling, continue
            emit({**base, "pattern": pat, "py_seconds": -1, "n_embeddings": -1,
                  "server_mem_gb": _mem(), "status": "FAILED",
                  "reason": f"{type(ex).__name__}: {str(ex)[:160]}"})
    emit({**base, "pattern": "TOTAL", "py_seconds": round(total_t, 6),
          "n_embeddings": total_emb, "server_mem_gb": round(peak_mem, 3),
          "status": "OK", "reason": ""})


def bench_orca(edges, n, graphlet_size, base, emit):
    """ORCA total time (single-threaded oracle). Per-pattern is not exposed -> TOTAL only."""
    try:
        t0 = time.perf_counter()
        mat = orca_count(edges, n, graphlet_size)
        dt = time.perf_counter() - t0
        # client-side memory here is fine: ORCA is a local subprocess, not the arkouda server.
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB (Linux KB)
        emit({**base, "pattern": "TOTAL", "py_seconds": round(dt, 6),
              "n_embeddings": int(mat.sum()), "server_mem_gb": round(mem, 3),
              "status": "OK", "reason": ""})
    except Exception as ex:
        emit({**base, "pattern": "TOTAL", "py_seconds": -1, "n_embeddings": -1,
              "server_mem_gb": -1, "status": "FAILED",
              "reason": f"{type(ex).__name__}: {str(ex)[:160]}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["hipermotif", "orca"])
    ap.add_argument("--graph", required=True,
                    help="cora|citeseer|pubmed|ogbn-arxiv|ogbn-products|gnp|ba|<name w/ --edge-file>")
    ap.add_argument("--edge-file", default=None, help="staged edge-list path (overrides loader)")
    ap.add_argument("--threads", type=int, default=0, help="label only; must match server maxTaskPar")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--graphlet-size", type=int, default=4, choices=[4])
    ap.add_argument("--out", required=True, help="CSV path (appended; header written if new)")
    ap.add_argument("--ak-host", default="localhost")
    ap.add_argument("--ak-port", default=5555, type=int)
    ap.add_argument("--no-mem", action="store_true",
                    help="skip server-memory sampling entirely (use if ak.get_mem_used is missing/"
                         "slow on your build — memory is optional, never a blocker; column = -1)")
    ap.add_argument("--max-embeddings", type=float, default=2e8,
                    help="skip a pattern whose predicted upper-bound #embeddings exceeds this "
                         "(protects the node from the hub-driven size-4 explosion; recorded as a "
                         "FAILED ceiling row). Default 2e8 ~ 15 min/pattern at the observed "
                         "~200K embeddings/sec; raise it if you have time/RAM headroom.")
    # synthetic params
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print(f"[bench] loading graph {args.graph!r} (NOT timed) ...")
    try:
        edges, n = load_graph(args.graph, args.edge_file, args.n, args.p, args.m, args.seed)
    except FileNotFoundError as ex:
        print(f"[bench] graph not on disk, skipping: {ex}")
        return
    n_edges = len(edges)
    print(f"[bench] {args.graph}: N={n} E={n_edges}")

    ak = ar = None
    maxtask = -1
    if args.backend == "hipermotif":
        import arkouda as ak  # noqa: F811
        ak.connect(args.ak_host, args.ak_port)
        import arachne as ar  # noqa: F401,F811
        try:
            maxtask = int(ak.get_config().get("maxTaskPar", -1))
        except Exception:
            pass
        if args.threads and maxtask != args.threads:
            print(f"[bench] WARNING: --threads={args.threads} but server maxTaskPar={maxtask}. "
                  f"Thread count is a SERVER-LAUNCH property; relaunch the server to change it.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    # INCREMENTAL write: open once, flush every row immediately so a hang/kill never loses data
    # and you get a live progress trail (the lesson from the 17h silent hang).
    fcsv = open(out, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader(); fcsv.flush()
    n_rows = n_fail = 0

    def emit(r):
        nonlocal n_rows, n_fail
        writer.writerow({k: r.get(k, "") for k in CSV_FIELDS}); fcsv.flush()
        n_rows += 1
        if r.get("status") == "FAILED":
            n_fail += 1
        print(f"  [csv ] {r['graph']} {r['pattern']:9} run{r.get('run_idx')} {r['status']:6} "
              f"t={r.get('py_seconds')}s n_emb={r.get('n_embeddings')}"
              f"{(' | ' + r['reason']) if r.get('reason') else ''}", flush=True)

    try:
        for run_idx in range(args.runs):
            base = {"graph": args.graph, "n_nodes": n, "n_edges": n_edges,
                    "backend": args.backend, "threads_label": args.threads,
                    "maxtaskpar_actual": maxtask, "run_idx": run_idx}
            print(f"[bench] run {run_idx+1}/{args.runs} backend={args.backend} graph={args.graph} ...",
                  flush=True)
            if args.backend == "hipermotif":
                bench_hipermotif(ak, ar, edges, n, base, emit, measure_mem=not args.no_mem,
                                 max_emb=args.max_embeddings)
            else:
                bench_orca(edges, n, args.graphlet_size, base, emit)
    except Exception:
        traceback.print_exc()
    finally:
        if args.backend == "hipermotif" and ak is not None:
            try:
                ak.disconnect()
            except Exception:
                pass
        fcsv.close()
    print(f"[bench] wrote {n_rows} rows to {out} ({n_fail} FAILED). done.")


if __name__ == "__main__":
    main()
