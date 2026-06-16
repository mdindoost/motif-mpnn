#!/usr/bin/env python
"""Arachne / HiPerMotif subgraph-isomorphism ENGINE diagnostic. RUN ON WULVER.

Self-contained. Bartosz just runs it — no editing. It probes the engine from several
independent angles to pin the "repeated host vertex in a p3 embedding" symptom (rows like
[1623, 980, 1623] on Cora/PROTEINS) to one of:
  (A) a PARALLEL ASSEMBLY race in the Chapel `forall` that writes isoArr — vertices from
      different embeddings interleave only at scale / under threads, OR
  (B) the engine GENUINELY emitting non-injective (walk-like) matches.

Already established (P3/C5 raw dump): row-major reshape(-1, n_pat) is the correct layout
(0 repeats, valid wedges, center in the middle column, isoMapper = identity under "None").
This script re-confirms that and then attacks the scale/parallel question.

>>> RUN IT TWICE and send BOTH outputs:
      (1) arkouda server launched SERIAL   : 1 locale, CHPL_RT_NUM_THREADS_PER_LOCALE=1
      (2) arkouda server launched PARALLEL : normal (multi-thread / multi-locale)
    The script auto-labels each run from ak.get_config() (numLocales, maxTaskPar). If the
    repeats vanish in the SERIAL run, that is hypothesis (A): an Arachne parallel-assembly
    bug in SubgraphSearch.chpl (the isoArr write), NOT our Python backend (reshape is right).

Usage (from repo root):
  PYTHONPATH=. python scripts/wulver/engine_diagnostic.py --ak-host <host> --ak-port <port>

Prereqs: arkouda server up (note host/port); env with arkouda + arachne + networkx + this
repo on PYTHONPATH; g++ for the ORCA oracle (built on first use). No internet needed: all
core graphs are synthetic; Cora/PROTEINS are used only if their PyG cache is already on disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Import-guarded for arkouda/arachne; these import fine locally and only the cluster calls fail.
from src.datasets import hipermotif_backend as backend
from src.datasets import hipermotif_patterns as hp
from src.datasets.orca_orbits import count_orbits as orca_count

PASS, FAIL, INFO, WARN = "[PASS]", "[FAIL]", "[INFO]", "[WARN]"
_tally = Counter()


def log(tag, msg):
    _tally[tag] += 1
    print(f"  {tag} {msg}")


# ---------------------------------------------------------------------------------------------
# Engine access — reuse the EXACT backend path (symmetrize -> PropGraph -> subgraph_isomorphism)
# so we exercise production code, then reshape row-major like count_orbits_hipermotif does.
# ---------------------------------------------------------------------------------------------
def engine_query(ak, ar, edges, n, pattern, reorder_type):
    """Return (emb_rowmajor [num,n_pat] or None, iso_flat, mapper_flat) for one pattern."""
    n_pat = hp.PATTERNS[pattern][1]
    src, dst = backend._symmetrize(edges, n)
    if not src:
        return np.empty((0, n_pat), np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    G = backend._build_propgraph(ak, ar, src, dst)
    psrc, pdst = backend._symmetrize(hp.PATTERNS[pattern][0], n_pat)
    H = backend._build_propgraph(ak, ar, psrc, pdst)
    iso_flat, mapper_flat = backend._run_iso(ar, G, H, reorder_type)
    emb = iso_flat.reshape(-1, n_pat) if (iso_flat.size % n_pat == 0) else None
    return emb, iso_flat, mapper_flat


def repeat_rows(emb, n_pat):
    """Indices of rows that are NOT injective (a host vertex appears twice)."""
    if emb is None or emb.size == 0:
        return []
    return [i for i in range(emb.shape[0]) if len(set(emb[i].tolist())) != n_pat]


def multiset_of_sets(emb):
    """Counter over frozenset(row) — column-order-independent fingerprint of the embeddings."""
    if emb is None or emb.size == 0:
        return Counter()
    return Counter(frozenset(int(x) for x in row) for row in emb)


def oracle(G_nx, pattern):
    """Local networkx ground truth: (multiset_of_vertex_sets, num_ordered_rows). Small n only."""
    emb, _ = hp.induced_embeddings(G_nx, pattern)        # identity columns; |Aut| rows per copy
    return multiset_of_sets(emb), (0 if emb.size == 0 else emb.shape[0])


def edges_of(G):
    G = nx.convert_node_labels_to_integers(G)
    return [(int(u), int(v)) for u, v in G.edges()], G.number_of_nodes(), G


# ---------------------------------------------------------------------------------------------
# SECTION 1 — Arachne induced-semantics basics (different angles, known answers)
# ---------------------------------------------------------------------------------------------
def section_basics(ak, ar):
    print("\n" + "=" * 78 + "\nSECTION 1 — induced-matching semantics (known counts)\n" + "=" * 78)
    # (pattern, host, expected #ordered embeddings = #induced copies * |Aut(pattern)|)
    cases = [
        ("edge", nx.complete_graph(2), 2),     # 1 edge * |Aut|=2
        ("triangle", nx.complete_graph(3), 6),  # 1 triangle * 6
        ("p3", nx.complete_graph(3), 0),        # INDUCED: K3 has no induced wedge
        ("p4", nx.complete_graph(4), 0),        # INDUCED: K4 has no induced 4-path
        ("k4", nx.complete_graph(4), 24),       # 1 clique * 24
        ("c4", nx.cycle_graph(4), 8),           # 1 4-cycle * |Aut(C4)|=8
    ]
    for pattern, G, expected in cases:
        edges, n, _ = edges_of(G)
        emb, iso, _ = engine_query(ak, ar, edges, n, pattern, "None")
        got = 0 if emb is None else emb.shape[0]
        bad = len(repeat_rows(emb, hp.PATTERNS[pattern][1]))
        ok = (got == expected and bad == 0)
        log(PASS if ok else FAIL,
            f"{pattern:8} on {G.number_of_nodes()}-node host: got {got} embeddings "
            f"(expected {expected}), repeat-rows={bad}")


# ---------------------------------------------------------------------------------------------
# SECTION 2 — layout re-confirmation on P3 & C5 (row-major vs col-major vs oracle)
# ---------------------------------------------------------------------------------------------
def section_layout(ak, ar):
    print("\n" + "=" * 78 + "\nSECTION 2 — layout: row-major vs col-major vs networkx oracle\n" + "=" * 78)
    for label, G in [("P3", nx.path_graph(3)), ("C5", nx.cycle_graph(5))]:
        edges, n, Gn = edges_of(G)
        _, iso, mapper = engine_query(ak, ar, edges, n, "p3", "None")
        n_pat = 3
        if iso.size % n_pat:
            log(FAIL, f"{label}: flat length {iso.size} not divisible by {n_pat}")
            continue
        rm = iso.reshape(-1, n_pat)
        cm = iso.reshape(n_pat, -1).T
        orc_sets, _ = oracle(Gn, "p3")
        rm_ok = (len(repeat_rows(rm, n_pat)) == 0) and (multiset_of_sets(rm) == orc_sets)
        cm_ok = (len(repeat_rows(cm, n_pat)) == 0) and (multiset_of_sets(cm) == orc_sets)
        log(PASS if rm_ok else FAIL, f"{label}: ROW-major matches oracle = {rm_ok} "
                                     f"(repeats {len(repeat_rows(rm, n_pat))})")
        log(INFO, f"{label}: COL-major matches oracle = {cm_ok} "
                  f"(repeats {len(repeat_rows(cm, n_pat))})  [expected False — sanity]")
        log(INFO, f"{label}: isoMapper(first {n_pat}) = {mapper[:n_pat].tolist()}  (identity expected)")


# ---------------------------------------------------------------------------------------------
# SECTION 3 — determinism: same query repeated; do results / repeats move?
# ---------------------------------------------------------------------------------------------
def section_determinism(ak, ar, edges, n, label, pattern="p3", runs=8):
    print("\n" + "=" * 78 + f"\nSECTION 3 — determinism on {label} ({pattern}, {runs} repeats)\n" + "=" * 78)
    n_pat = hp.PATTERNS[pattern][1]
    counts, rep_counts, fingerprints = [], [], []
    for _ in range(runs):
        emb, iso, _ = engine_query(ak, ar, edges, n, pattern, "None")
        counts.append(0 if emb is None else emb.shape[0])
        rep_counts.append(len(repeat_rows(emb, n_pat)))
        fingerprints.append(hash(frozenset(multiset_of_sets(emb).items())))
    log(INFO, f"embedding counts over {runs} runs: {counts}")
    log(INFO, f"repeat-row counts over {runs} runs: {rep_counts}")
    same_count = len(set(counts)) == 1
    same_set = len(set(fingerprints)) == 1
    stable_reps = len(set(rep_counts)) == 1
    log(PASS if same_count else WARN, f"embedding COUNT stable across runs = {same_count}")
    log(PASS if same_set else WARN, f"embedding SET (vertex-sets) stable across runs = {same_set}")
    if max(rep_counts) > 0:
        log(WARN, f"repeats present; stable across runs = {stable_reps}  "
                  f"(NON-stable => parallel race (A); stable => likely genuine (B))")


# ---------------------------------------------------------------------------------------------
# SECTION 4 — size ladder: where do repeats appear? + end-to-end orbits vs ORCA
# ---------------------------------------------------------------------------------------------
def section_ladder(ak, ar, reorder_type):
    print("\n" + "=" * 78 + f"\nSECTION 4 — size ladder (reorder_type={reorder_type!r}): repeats + orbits vs ORCA\n" + "=" * 78)
    graphs = []
    for nn in (8, 12, 16, 20, 30, 50, 80):
        graphs.append((f"gnp(n={nn},p=0.3,seed=1)", nx.gnp_random_graph(nn, 0.3, seed=1)))
    # add cached real graphs if present (no download)
    graphs += _cached_real_graphs()

    first_repro = None
    for label, G in graphs:
        edges, n, Gn = edges_of(G)
        # per-pattern repeat audit + assemble engine orbit matrix
        embs, total_rep, per_pat_rep = {}, 0, {}
        for pat in hp.PATTERN_NAMES:
            emb, iso, mapflat = engine_query(ak, ar, edges, n, pat, reorder_type)
            n_pat = hp.PATTERNS[pat][1]
            r = len(repeat_rows(emb, n_pat))
            if r:
                per_pat_rep[pat] = r
            total_rep += r
            mapper = mapflat[:n_pat] if mapflat.size >= n_pat else np.arange(n_pat)
            embs[pat] = (emb if emb is not None else np.empty((0, n_pat), np.int64), mapper)
        # end-to-end orbit counts vs ORCA (bypasses the guard; tolerates raises)
        orca = orca_count(edges, n)
        try:
            X = hp.orbit_matrix_from_embeddings(embs, n)
            orbits_ok = np.array_equal(X, orca)
            orbit_msg = f"orbits==ORCA: {orbits_ok}"
            if not orbits_ok:
                d = np.argwhere(X != orca)[0]
                orbit_msg += f" (first diff node {d[0]} orbit {d[1]}: HM={X[d[0],d[1]]} ORCA={orca[d[0],d[1]]})"
        except Exception as ex:  # divisibility / permutation guard inside normalize_embeddings
            orbits_ok = False
            orbit_msg = f"orbit assembly RAISED: {type(ex).__name__}: {str(ex)[:90]}"
        tag = FAIL if (total_rep or not orbits_ok) else PASS
        log(tag, f"{label:26} N={n:<5} repeat-rows={total_rep:<4} {dict(per_pat_rep)}  {orbit_msg}")
        if total_rep and first_repro is None:
            first_repro = (label, edges, n)
    return first_repro


def _cached_real_graphs():
    out = []
    cora_cache = REPO / "data/processed" / "Cora" / "processed" / "data.pt"
    prot_cache = REPO / "data/processed/pyg/proteins" / "PROTEINS" / "processed" / "data.pt"
    try:
        from torch_geometric.utils import to_networkx
        if prot_cache.exists():
            from torch_geometric.datasets import TUDataset
            prot = TUDataset(root=str(REPO / "data/processed/pyg/proteins"), name="PROTEINS")[0]
            out.append(("PROTEINS[0]", to_networkx(prot, to_undirected=True)))
        if cora_cache.exists():
            from torch_geometric.datasets import Planetoid
            cora = Planetoid(root=str(REPO / "data/processed"), name="Cora")[0]
            out.append(("Cora", to_networkx(cora, to_undirected=True)))
    except Exception as ex:  # torch_geometric missing or load error -> just skip
        print(f"  {INFO} cached real graphs unavailable ({type(ex).__name__}); using synthetic only")
    return out


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ak-host", default="localhost")
    ap.add_argument("--ak-port", default=5555, type=int)
    args = ap.parse_args()

    import os
    import arkouda as ak
    print(f"[engine-diag] connecting to arkouda at {args.ak_host}:{args.ak_port} ...")
    ak.connect(args.ak_host, args.ak_port)
    import arachne as ar  # noqa: F401  (fail loudly here if arachne missing)

    # --- SECTION 0: runtime config; auto-label SERIAL vs PARALLEL -----------------------------
    print("\n" + "=" * 78 + "\nSECTION 0 — runtime config\n" + "=" * 78)
    cfg = {}
    try:
        cfg = ak.get_config()
    except Exception as ex:
        print(f"  {WARN} ak.get_config() failed: {ex}")
    n_loc = cfg.get("numLocales", "?")
    max_par = cfg.get("maxTaskPar", cfg.get("numPUs", "?"))
    env_thr = os.environ.get("CHPL_RT_NUM_THREADS_PER_LOCALE", "(unset in client env)")
    serial = (str(n_loc) == "1" and str(max_par) == "1")
    label = "SERIAL" if serial else "PARALLEL"
    print(f"  numLocales={n_loc}  maxTaskPar={max_par}  "
          f"CHPL_RT_NUM_THREADS_PER_LOCALE(client env)={env_thr}")
    print(f"  >>> THIS RUN IS LABELED: {label} <<<   (run again under the other config to compare)")

    try:
        section_basics(ak, ar)
        section_layout(ak, ar)
        # determinism on a graph that should be busy enough to matter
        edges, n, _ = edges_of(nx.gnp_random_graph(30, 0.3, seed=1))
        section_determinism(ak, ar, edges, n, "gnp(30,0.3)")
        # ladder on both reorder paths
        repro_none = section_ladder(ak, ar, "None")
        repro_struct = section_ladder(ak, ar, "structural")
        # focused determinism on the first repeats-reproducing graph (if any)
        repro = repro_none or repro_struct
        if repro:
            lab, e, nn = repro
            print(f"\n{INFO} smallest repeats-reproducing graph: {lab} (N={nn}) — determinism on it:")
            section_determinism(ak, ar, e, nn, lab)
    finally:
        try:
            ak.disconnect()
        except Exception:
            pass

    # --- summary ------------------------------------------------------------------------------
    print("\n" + "=" * 78 + f"\nSUMMARY  (run labeled {label})\n" + "=" * 78)
    print(f"  PASS={_tally[PASS]}  FAIL={_tally[FAIL]}  WARN={_tally[WARN]}")
    print(f"  numLocales={n_loc} maxTaskPar={max_par}")
    print("  Interpretation guide:")
    print("   - SECTION 1 all PASS  -> engine does correct INDUCED matching with right counts.")
    print("   - SECTION 2 row-major PASS -> reshape layout is correct (our backend is right).")
    print("   - repeats in SECTION 4 that VANISH in the SERIAL run  -> hypothesis (A):")
    print("     parallel isoArr-assembly race in Arachne SubgraphSearch.chpl (NOT our Python).")
    print("   - repeats STABLE across runs and present even SERIAL    -> hypothesis (B):")
    print("     engine genuinely returns non-injective matches (needs an engine-side filter).")
    print("   - 'orbits==ORCA: False' or 'orbit assembly RAISED' shows the repeats corrupt counts.")
    print("  >>> Send the FULL stdout of BOTH the SERIAL and PARALLEL runs. <<<")
    sys.exit(0)


if __name__ == "__main__":
    main()
