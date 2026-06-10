"""
generate_motifs.py — Motif preprocessing pipeline for Motif-MPNN.

Computes per-node motif participation counts and writes them to the format
consumed by src/datasets/motif_loader.py.  This script is the bridge until
HiPerMotif (the real HPC parallel subgraph isomorphism engine in Arachne /
Chapel / Arkouda) is connected.

OUTPUT FORMAT — identical to what HiPerMotif will produce so swapping engines
requires zero changes downstream:

  Planetoid (node classification):
    data/precompute/<dataset>/node_motifs.csv
    columns: node_id, k, motif_id, count

  TU (graph classification):
    data/precompute/<dataset>/node_motifs.csv
    columns: graph_id, node_id, k, motif_id, count

MOTIF CONVENTION (ALL DATASETS — UNDIRECTED):
All features use UNDIRECTED k=3 subgraph counts (igraph convention).
  k=1, motif_id=0 : degree   (number of undirected neighbors)
  k=3, motif_id=2 : wedge    (node is endpoint of an open path of length 2)
  k=3, motif_id=3 : triangle (node participates in a 3-clique)

motif_id values 2 and 3 match igraph's motifs_undirected() isomorphism class
numbering for 3-node connected undirected subgraphs. Class 0 = single edge +
isolated, class 1 = path of 3 nodes, class 2 = wedge/path-of-2, class 3 = triangle.

HiPerMotif swap: ensure HiPerMotif output uses the same (k, motif_id) pairs
so that motif_loader.py requires no changes.

Usage:
  python scripts/preprocess/generate_motifs.py --dataset cora --tool networkit
  python scripts/preprocess/generate_motifs.py --dataset proteins --tool igraph
  python scripts/preprocess/generate_motifs.py --dataset all
  python scripts/preprocess/generate_motifs.py --dataset cora --force --verify

HiPerMotif swap: when HiPerMotif is connected, replace this script's
counting logic with a single HiPerMotif call that writes the same CSV columns.
No changes needed in motif_loader.py or any model code.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
TU_DATASETS = {"proteins", "nci1", "enzymes"}
SYNTHETIC_DATASETS = {"csl"}
ALL_DATASETS = sorted(PLANETOID_DATASETS | TU_DATASETS | SYNTHETIC_DATASETS)

# HiPerMotif-compatible motif ID scheme for undirected k=3 motifs (igraph convention)
# k=3, motif_id=2 → open triangle (wedge / path of length 2), iso-type 2 in igraph
# k=3, motif_id=3 → closed triangle (3-clique), iso-type 3 in igraph
MOTIF_WEDGE = 2
MOTIF_TRIANGLE = 3

# 4-clique CSV encoding (our scheme; motif_loader treats (k, motif_id) opaquely)
MOTIF_4CLIQUE_K = 4
MOTIF_4CLIQUE_ID = 10
# Simple-cycle CSV encoding: (k=L, motif_id=CYCLE_SENTINEL) for a length-L cycle
CYCLE_SENTINEL = 1000
CYCLE_L_MAX = 8  # empirically bump until all 10 CSL classes separate (see Task 8)


# ---------------------------------------------------------------------------
# Graph loading helpers
# ---------------------------------------------------------------------------

def _load_planetoid_as_nx(dataset: str, root: Path):
    """Load Planetoid graph via PyG and convert to networkx (undirected)."""
    try:
        from torch_geometric.datasets import Planetoid
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: conda run -n motif-mpnn pip install torch_geometric networkx")
        sys.exit(1)

    ds = Planetoid(root=str(root), name=dataset.capitalize())
    data = ds[0]
    G_nx = to_networkx(data, to_undirected=True)
    return G_nx, ds[0].num_nodes


def _load_tu_graphs_as_nx(dataset: str, root: Path):
    """Load TU graphs via PyG and convert each to networkx (undirected)."""
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: conda run -n motif-mpnn pip install torch_geometric")
        sys.exit(1)

    name_upper = dataset.upper()
    tu_root = root / "pyg" / dataset.lower()
    ds = TUDataset(root=str(tu_root), name=name_upper)
    graphs = []
    for i in range(len(ds)):
        g = to_networkx(ds[i], to_undirected=True)
        graphs.append((i, g, int(ds[i].num_nodes)))
    return graphs


def _load_csl_graphs_as_nx(root: Path):
    """Build the 150 CSL graphs (same seed as the dataset) as networkx graphs."""
    import networkx as nx
    import sys as _sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from src.datasets.expressivity import make_csl
    graphs = []
    for gid, (ei, _label) in enumerate(make_csl(seed=0)):
        n = int(ei.max().item()) + 1
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for u, v in ei.t().tolist():
            if u != v:
                g.add_edge(int(u), int(v))
        graphs.append((gid, g, n))
    return graphs


# ---------------------------------------------------------------------------
# Motif counting — NetworKit backend (Planetoid only)
# ---------------------------------------------------------------------------

def _count_motifs_networkit(G_nx, num_nodes: int):
    """
    Compute per-node triangle and wedge counts using NetworKit.
    Returns: list of (node_id, k, motif_id, count)
    """
    try:
        import networkit as nk
    except ImportError:
        print("[ERROR] NetworKit not installed.")
        print("Install with: pip install networkit")
        print("Or use --tool igraph as a fallback.")
        sys.exit(1)

    from tqdm import tqdm

    nk_G = nk.Graph(num_nodes, weighted=False, directed=False)
    for u, v in G_nx.edges():
        if u < num_nodes and v < num_nodes and u != v:
            if not nk_G.hasEdge(u, v):
                nk_G.addEdge(u, v)

    rows = []
    for u in tqdm(range(num_nodes), desc="  degree (NetworKit)", unit="node"):
        deg = nk_G.degree(u)
        if deg > 0:
            rows.append((u, 1, 0, deg))

    lcc_algo = nk.centrality.LocalClusteringCoefficient(nk_G, turbo=True)
    lcc_algo.run()
    lcc = lcc_algo.scores()

    for u in tqdm(range(num_nodes), desc="  triangles (NetworKit)", unit="node"):
        deg = nk_G.degree(u)
        if deg >= 2:
            triangles = int(round(lcc[u] * deg * (deg - 1) / 2))
            wedges = deg * (deg - 1) // 2 - triangles
            if triangles > 0:
                rows.append((u, 3, MOTIF_TRIANGLE, triangles))
            if wedges > 0:
                rows.append((u, 3, MOTIF_WEDGE, wedges))

    return rows


# ---------------------------------------------------------------------------
# Motif counting — igraph backend (Planetoid + TU)
# ---------------------------------------------------------------------------

def _count_motifs_igraph_single(g_nx, num_nodes: int):
    """
    Compute per-node triangle and wedge counts for a single graph using igraph.
    Returns: list of (node_id, k, motif_id, count)
    """
    try:
        import igraph as ig
    except ImportError:
        print("[ERROR] igraph not installed.")
        print("Install with: pip install igraph")
        sys.exit(1)

    edges = [(u, v) for u, v in g_nx.edges() if u < num_nodes and v < num_nodes and u != v]
    g = ig.Graph(n=num_nodes, edges=edges, directed=False)
    g.simplify()

    import collections

    rows = []
    degrees = g.degree()
    for u, deg in enumerate(degrees):
        if deg > 0:
            rows.append((u, 1, 0, deg))

    # igraph 1.x API: list_triangles() returns list of (a, b, c) vertex triples
    tri_per_node = collections.Counter()
    for a, b, c in g.list_triangles():
        tri_per_node[a] += 1
        tri_per_node[b] += 1
        tri_per_node[c] += 1

    for u in range(num_nodes):
        deg = degrees[u]
        tri = tri_per_node.get(u, 0)
        if tri > 0:
            rows.append((u, 3, MOTIF_TRIANGLE, tri))
        wedges = deg * (deg - 1) // 2 - tri
        if wedges > 0:
            rows.append((u, 3, MOTIF_WEDGE, wedges))

    return rows


def _count_substructures_single(g_nx, num_nodes: int):
    """Per-node rows: degree, triangle, wedge, simple cycles (3..CYCLE_L_MAX), 4-cliques."""
    from src.datasets.substructure_counts import (
        triangles_per_node, four_cliques_per_node, simple_cycles_per_node,
    )
    rows = []
    deg = dict(g_nx.degree())
    for u in range(num_nodes):
        d = deg.get(u, 0)
        if d > 0:
            rows.append((u, 1, 0, d))
    tpn = triangles_per_node(g_nx, num_nodes)
    for u in range(num_nodes):
        d = deg.get(u, 0)
        tri = tpn.get(u, 0)
        if tri > 0:
            rows.append((u, 3, MOTIF_TRIANGLE, tri))
        wedges = d * (d - 1) // 2 - tri
        if wedges > 0:
            rows.append((u, 3, MOTIF_WEDGE, wedges))
    # simple cycles by length
    spn = simple_cycles_per_node(g_nx, num_nodes, l_max=CYCLE_L_MAX)
    for u, by_len in spn.items():
        for L, c in by_len.items():
            if c > 0:
                rows.append((u, int(L), CYCLE_SENTINEL, int(c)))
    # 4-cliques
    fpn = four_cliques_per_node(g_nx, num_nodes)
    for u, c in fpn.items():
        if c > 0:
            rows.append((u, MOTIF_4CLIQUE_K, MOTIF_4CLIQUE_ID, int(c)))
    return rows


def _count_orbits_single(g_nx, num_nodes: int, graphlet_size: int = 4):
    """Per-node ORCA orbit rows: (node_id, k, motif_id, count) for nonzero counts.

    Each orbit o -> (k = graphlet node count, motif_id = 2000 + o), disjoint from
    all legacy encodings. ORCA is exact/deterministic (see src/datasets/orca_orbits).
    """
    import sys as _sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from src.datasets.orca_orbits import count_orbits, orbit_to_km

    edges = [(u, v) for u, v in g_nx.edges() if u != v]
    mat = count_orbits(edges, num_nodes, graphlet_size=graphlet_size)
    rows = []
    n_orbits = mat.shape[1]
    for u in range(num_nodes):
        for o in range(n_orbits):
            c = int(mat[u, o])
            if c > 0:
                k, motif_id = orbit_to_km(o)
                rows.append((u, k, motif_id, c))
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_planetoid_csv(rows, out_path: Path, topk: int):
    """Write node_motifs.csv for a Planetoid dataset."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(f"# motif_topk={topk}\n")
        writer = csv.writer(f)
        writer.writerow(["node_id", "k", "motif_id", "count"])
        for node_id, k, motif_id, count in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            writer.writerow([node_id, k, motif_id, count])
    print(f"[OK] Wrote {len(rows)} rows to {out_path}")


def _write_tu_csv(all_rows, out_path: Path, topk: int):
    """Write node_motifs.csv for a TU dataset (multi-graph)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(f"# motif_topk={topk}\n")
        writer = csv.writer(f)
        writer.writerow(["graph_id", "node_id", "k", "motif_id", "count"])
        for graph_id, node_id, k, motif_id, count in sorted(
            all_rows, key=lambda r: (r[0], r[1], r[2], r[3])
        ):
            writer.writerow([graph_id, node_id, k, motif_id, count])
    print(f"[OK] Wrote {len(all_rows)} rows to {out_path}")


def _write_orbit_csv(all_rows, out_path: Path, topk: int, multigraph: bool):
    """Write node_motifs_orbit.csv (graph_id present iff multigraph)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(f"# motif_topk={topk} features=orbit\n")
        writer = csv.writer(f)
        if multigraph:
            writer.writerow(["graph_id", "node_id", "k", "motif_id", "count"])
            for row in sorted(all_rows, key=lambda r: (r[0], r[1], r[2], r[3])):
                writer.writerow(list(row))
        else:
            writer.writerow(["node_id", "k", "motif_id", "count"])
            for row in sorted(all_rows, key=lambda r: (r[0], r[1], r[2])):
                writer.writerow(list(row))
    print(f"[OK] Wrote {len(all_rows)} orbit rows to {out_path}")


# ---------------------------------------------------------------------------
# Verify helper (calls verify_motifs logic inline)
# ---------------------------------------------------------------------------

def _run_verify(out_path: Path):
    """Print quick stats for a freshly written CSV."""
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas not available; skipping --verify stats.")
        return

    print("\n[VERIFY] Quick stats:")
    df = pd.read_csv(out_path, comment="#")
    print(f"  rows={len(df)}, columns={list(df.columns)}")
    if "count" in df.columns:
        print(f"  count: min={df['count'].min()}, max={df['count'].max()}, "
              f"mean={df['count'].mean():.2f}, std={df['count'].std():.2f}")
    if "node_id" in df.columns:
        print(f"  unique nodes: {df['node_id'].nunique()}")
    if "graph_id" in df.columns:
        print(f"  unique graphs: {df['graph_id'].nunique()}")
    motif_ids = df["motif_id"].unique() if "motif_id" in df.columns else []
    print(f"  motif_ids present: {sorted(motif_ids)}")


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def _run_dataset(dataset: str, tool: str, root: Path, out_dir_override: Path | None,
                 force: bool, verify: bool, topk: int,
                 features: str = "legacy", graphlet_size: int = 4):
    out_dir = out_dir_override if out_dir_override else Path("data/precompute") / dataset
    out_path = out_dir / "node_motifs.csv"

    if features == "orbit":
        orbit_path = out_dir / "node_motifs_orbit.csv"
        if orbit_path.exists() and not force:
            print(f"[SKIP] {orbit_path} already exists. Use --force to recompute.")
            return
        print(f"[INFO] ORCA orbit features (graphlet_size={graphlet_size}) for {dataset} ...")
        from tqdm import tqdm
        if dataset in PLANETOID_DATASETS:
            G_nx, num_nodes = _load_planetoid_as_nx(dataset, root)
            rows = _count_orbits_single(G_nx, num_nodes, graphlet_size)
            _write_orbit_csv(rows, orbit_path, topk, multigraph=False)
        else:
            if dataset in TU_DATASETS:
                graphs = _load_tu_graphs_as_nx(dataset, root)
            elif dataset in SYNTHETIC_DATASETS:
                graphs = _load_csl_graphs_as_nx(root)
            else:
                print(f"[ERROR] Unknown dataset: {dataset}")
                sys.exit(1)
            all_rows = []
            for graph_id, g_nx, num_nodes in tqdm(graphs, desc=f"  {dataset} orbits", unit="graph"):
                for node_id, k, motif_id, count in _count_orbits_single(g_nx, num_nodes, graphlet_size):
                    all_rows.append((graph_id, node_id, k, motif_id, count))
            _write_orbit_csv(all_rows, orbit_path, topk, multigraph=True)
        if verify:
            _run_verify(orbit_path)
        print(f"\n[DONE] Orbit CSV written to: {orbit_path}")
        return

    if out_path.exists() and not force:
        print(f"[SKIP] {out_path} already exists. Use --force to recompute.")
        return

    t0 = time.perf_counter()

    if dataset in PLANETOID_DATASETS:
        print(f"[INFO] Loading Planetoid/{dataset} ...")
        G_nx, num_nodes = _load_planetoid_as_nx(dataset, root)
        print(f"[INFO] Graph: {num_nodes} nodes, {G_nx.number_of_edges()} edges")

        if tool == "networkit":
            print("[INFO] Counting motifs with NetworKit ...")
            rows = _count_motifs_networkit(G_nx, num_nodes)
        else:
            if tool != "igraph":
                print(f"[WARN] Unknown tool '{tool}', falling back to igraph.")
            print("[INFO] Counting motifs with igraph ...")
            from tqdm import tqdm
            rows = _count_motifs_igraph_single(G_nx, num_nodes)

        _write_planetoid_csv(rows, out_path, topk)

    elif dataset in TU_DATASETS:
        if tool == "networkit":
            print(f"[WARN] NetworKit TU support is limited; falling back to igraph for {dataset}.")

        print(f"[INFO] Loading TU/{dataset.upper()} ...")
        graphs = _load_tu_graphs_as_nx(dataset, root)
        print(f"[INFO] {len(graphs)} graphs loaded")

        from tqdm import tqdm
        all_rows = []
        for graph_id, g_nx, num_nodes in tqdm(graphs, desc=f"  {dataset} graphs", unit="graph"):
            per_node = _count_motifs_igraph_single(g_nx, num_nodes)
            for node_id, k, motif_id, count in per_node:
                all_rows.append((graph_id, node_id, k, motif_id, count))

        _write_tu_csv(all_rows, out_path, topk)

    elif dataset in SYNTHETIC_DATASETS:
        print(f"[INFO] Building synthetic/{dataset} graphs ...")
        graphs = _load_csl_graphs_as_nx(root)
        print(f"[INFO] {len(graphs)} graphs built")
        from tqdm import tqdm
        all_rows = []
        for graph_id, g_nx, num_nodes in tqdm(graphs, desc=f"  {dataset} graphs", unit="graph"):
            per_node = _count_substructures_single(g_nx, num_nodes)
            for node_id, k, motif_id, count in per_node:
                all_rows.append((graph_id, node_id, k, motif_id, count))
        _write_tu_csv(all_rows, out_path, topk)

    else:
        print(f"[ERROR] Unknown dataset: {dataset}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    if verify:
        _run_verify(out_path)

    print(f"\n[DONE] Motif CSV written to: {out_path}")
    print("Next steps:")
    print("  1. Run your experiment: python -m src.train.run --config configs/experiments/<dataset>_concat.yml")
    print("  2. The motif_loader will auto-build motif_x.pt cache on first run.")
    print()
    print("HiPerMotif swap: replace this script with a HiPerMotif call that writes")
    print("the same CSV columns. No changes needed in motif_loader.py or model code.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-node motif counts for Motif-MPNN datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(PLANETOID_DATASETS | TU_DATASETS | SYNTHETIC_DATASETS) + ["all"],
        help="Dataset to compute motifs for, or 'all' to run every dataset.",
    )
    parser.add_argument(
        "--tool",
        default="igraph",
        choices=["networkit", "igraph"],
        help=(
            "Backend to use for motif counting. "
            "'networkit' is faster for large Planetoid graphs; "
            "'igraph' supports TU multi-graph datasets and is the default fallback."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory override (default: data/precompute/<dataset>/).",
    )
    parser.add_argument(
        "--root",
        default="data/processed",
        help="Root directory for PyG dataset downloads (default: data/processed).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-compute and overwrite CSV even if it already exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After writing CSV, print quick sanity-check stats.",
    )
    parser.add_argument(
        "--k",
        "--topk",
        dest="topk",
        type=int,
        default=10,
        help="motif_topk value written as a header comment in the CSV (default: 10).",
    )
    parser.add_argument(
        "--features",
        default="legacy",
        choices=["legacy", "orbit"],
        help="Feature backend: 'legacy' = degree/wedge/triangle (default); "
             "'orbit' = ORCA per-node graphlet-orbit counts -> node_motifs_orbit.csv.",
    )
    parser.add_argument(
        "--graphlet-size",
        dest="graphlet_size",
        type=int,
        default=4,
        choices=[4, 5],
        help="ORCA graphlet size for --features orbit (default 4; 5 is a smoke-test path).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir_override = Path(args.out_dir) if args.out_dir else None
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset.lower()]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")
        _run_dataset(
            dataset=ds,
            tool=args.tool,
            root=root,
            out_dir_override=out_dir_override,
            force=args.force,
            verify=args.verify,
            topk=args.topk,
            features=args.features,
            graphlet_size=args.graphlet_size,
        )


if __name__ == "__main__":
    main()
