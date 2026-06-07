"""
generate_motifs.py — Motif preprocessing pipeline for Motif-MPNN.

Computes per-node motif participation counts and writes them to the format
consumed by src/datasets/motif_loader.py.  This script is the bridge until
HiPerXplorer (the real HPC motif counter in Chapel/Arkouda) is connected.

OUTPUT FORMAT — identical to what HiPerXplorer will produce so swapping engines
requires zero changes downstream:

  Planetoid (node classification):
    data/precompute/<dataset>/node_motifs.csv
    columns: node_id, k, motif_id, count

  TU (graph classification):
    data/precompute/<dataset>/node_motifs.csv
    columns: graph_id, node_id, k, motif_id, count

MOTIF ID SCHEME (undirected, matches igraph motifs_undirected for k=3):
  k=1, motif_id=0 : degree (number of neighbors)
  k=3, motif_id=2 : wedge  (node participates in an open path of length 2)
  k=3, motif_id=3 : triangle (node participates in a 3-clique)

Usage:
  python scripts/preprocess/generate_motifs.py --dataset cora --tool networkit
  python scripts/preprocess/generate_motifs.py --dataset proteins --tool igraph
  python scripts/preprocess/generate_motifs.py --dataset cora --tool igraph  # igraph fallback

HiPerXplorer swap: when HiPerXplorer is connected, replace this script's
counting logic with a single HiPerXplorer call that writes the same CSV columns.
No changes needed in motif_loader.py or any model code.
"""

import argparse
import csv
import sys
from pathlib import Path

PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
TU_DATASETS = {"proteins", "nci1", "enzymes"}

# HiPerXplorer-compatible motif ID scheme for undirected k=3 motifs (igraph convention)
# k=3, motif_id=2 → open triangle (wedge / path of length 2), iso-type 2 in igraph
# k=3, motif_id=3 → closed triangle (3-clique), iso-type 3 in igraph
MOTIF_WEDGE = 2
MOTIF_TRIANGLE = 3


# ---------------------------------------------------------------------------
# Graph loading helpers
# ---------------------------------------------------------------------------

def _load_planetoid_as_nx(dataset: str):
    """Load Planetoid graph via PyG and convert to networkx (undirected)."""
    try:
        import torch
        from torch_geometric.datasets import Planetoid
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: conda run -n motif-mpnn pip install torch_geometric networkx")
        sys.exit(1)

    root = Path("data/processed")
    ds = Planetoid(root=str(root), name=dataset.capitalize())
    data = ds[0]
    G_nx = to_networkx(data, to_undirected=True)
    return G_nx, ds[0].num_nodes


def _load_tu_graphs_as_nx(dataset: str):
    """Load TU graphs via PyG and convert each to networkx (undirected)."""
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: conda run -n motif-mpnn pip install torch_geometric")
        sys.exit(1)

    name_upper = dataset.upper()
    root = Path("data/processed") / "pyg" / dataset.lower()
    ds = TUDataset(root=str(root), name=name_upper)
    graphs = []
    for i in range(len(ds)):
        g = to_networkx(ds[i], to_undirected=True)
        graphs.append((i, g, int(ds[i].num_nodes)))
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

    # Build NetworKit graph
    nk_G = nk.Graph(num_nodes, weighted=False, directed=False)
    for u, v in G_nx.edges():
        if u < num_nodes and v < num_nodes and u != v:
            if not nk_G.hasEdge(u, v):
                nk_G.addEdge(u, v)

    # Degree (k=1, motif_id=0)
    rows = []
    for u in range(num_nodes):
        deg = nk_G.degree(u)
        if deg > 0:
            rows.append((u, 1, 0, deg))

    # Triangle count per node (k=3, motif_id=3)
    # NetworKit's LocalClusteringCoefficient gives lcc = 2*triangles / (deg*(deg-1))
    # We need raw triangle counts, so use TriangleCount if available, else compute from lcc.
    try:
        algo = nk.sparsification.LocalSimilarityScore(nk_G, list(nk_G.iterEdges()))
    except Exception:
        pass

    # Use triangle counting via LCC
    lcc_algo = nk.centrality.LocalClusteringCoefficient(nk_G, turbo=True)
    lcc_algo.run()
    lcc = lcc_algo.scores()  # per-node LCC in [0,1]

    for u in range(num_nodes):
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

    # Build igraph graph
    edges = [(u, v) for u, v in g_nx.edges() if u < num_nodes and v < num_nodes and u != v]
    g = ig.Graph(n=num_nodes, edges=edges, directed=False)
    g.simplify()  # remove multi-edges and self-loops

    rows = []

    # Degree (k=1, motif_id=0)
    degrees = g.degree()
    for u, deg in enumerate(degrees):
        if deg > 0:
            rows.append((u, 1, 0, deg))

    # Per-node triangle count: igraph triangles() returns number of triangles per vertex
    tri_counts = g.triangles()  # number of triangles each vertex participates in
    for u, tri in enumerate(tri_counts):
        deg = degrees[u]
        if tri > 0:
            rows.append((u, 3, MOTIF_TRIANGLE, tri))
        # Wedges: open triads where u is the center
        # = (deg choose 2) - triangles
        wedges = deg * (deg - 1) // 2 - tri
        if wedges > 0:
            rows.append((u, 3, MOTIF_WEDGE, wedges))

    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_planetoid_csv(rows, out_path: Path):
    """Write node_motifs.csv for a Planetoid dataset."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "k", "motif_id", "count"])
        for node_id, k, motif_id, count in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            writer.writerow([node_id, k, motif_id, count])
    print(f"[OK] Wrote {len(rows)} rows to {out_path}")


def _write_tu_csv(all_rows, out_path: Path):
    """Write node_motifs.csv for a TU dataset (multi-graph)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "node_id", "k", "motif_id", "count"])
        for graph_id, node_id, k, motif_id, count in sorted(
            all_rows, key=lambda r: (r[0], r[1], r[2], r[3])
        ):
            writer.writerow([graph_id, node_id, k, motif_id, count])
    print(f"[OK] Wrote {len(all_rows)} rows to {out_path}")


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
        choices=sorted(PLANETOID_DATASETS | TU_DATASETS),
        help="Dataset to compute motifs for.",
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
    args = parser.parse_args()

    dataset = args.dataset.lower()
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/precompute") / dataset
    out_path = out_dir / "node_motifs.csv"

    if dataset in PLANETOID_DATASETS:
        print(f"[INFO] Loading Planetoid/{dataset} ...")
        G_nx, num_nodes = _load_planetoid_as_nx(dataset)
        print(f"[INFO] Graph: {num_nodes} nodes, {G_nx.number_of_edges()} edges")

        if args.tool == "networkit":
            print("[INFO] Counting motifs with NetworKit ...")
            rows = _count_motifs_networkit(G_nx, num_nodes)
        else:
            if args.tool != "igraph":
                print(f"[WARN] Unknown tool '{args.tool}', falling back to igraph.")
            print("[INFO] Counting motifs with igraph ...")
            rows = _count_motifs_igraph_single(G_nx, num_nodes)

        _write_planetoid_csv(rows, out_path)

    elif dataset in TU_DATASETS:
        if args.tool == "networkit":
            print(f"[WARN] NetworKit TU support is limited; falling back to igraph for {dataset}.")

        print(f"[INFO] Loading TU/{dataset.upper()} ...")
        graphs = _load_tu_graphs_as_nx(dataset)
        print(f"[INFO] {len(graphs)} graphs loaded")

        all_rows = []
        for graph_id, g_nx, num_nodes in graphs:
            if graph_id % 100 == 0:
                print(f"  ... processing graph {graph_id}/{len(graphs)}")
            per_node = _count_motifs_igraph_single(g_nx, num_nodes)
            for node_id, k, motif_id, count in per_node:
                all_rows.append((graph_id, node_id, k, motif_id, count))

        _write_tu_csv(all_rows, out_path)

    else:
        print(f"[ERROR] Unknown dataset: {dataset}")
        sys.exit(1)

    print(f"\n[DONE] Motif CSV written to: {out_path}")
    print("Next steps:")
    print("  1. Run your experiment: python -m src.train.run --config configs/experiments/<dataset>_concat.yml")
    print("  2. The motif_loader will auto-build motif_x.pt cache on first run.")
    print()
    print("HiPerXplorer swap: replace this script with a HiPerXplorer call that writes")
    print("the same CSV columns. No changes needed in motif_loader.py or model code.")


if __name__ == "__main__":
    main()
