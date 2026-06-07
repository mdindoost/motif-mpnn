"""
verify_motifs.py — Sanity-check motif CSVs produced by generate_motifs.py.

Usage:
  python scripts/preprocess/verify_motifs.py --dataset cora
  python scripts/preprocess/verify_motifs.py --dataset proteins --root data/processed
  python scripts/preprocess/verify_motifs.py --dataset cora --plot
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
TU_DATASETS = {"proteins", "nci1", "enzymes"}
ALL_DATASETS = sorted(PLANETOID_DATASETS | TU_DATASETS)


def _locate_csv(dataset: str, root: Path) -> Path:
    return root / dataset / "node_motifs.csv"


def _count_columns(df: pd.DataFrame) -> list[str]:
    """Return column(s) to treat as feature values (everything except id/meta cols)."""
    exclude = {"node_id", "graph_id", "k", "motif_id"}
    return [c for c in df.columns if c not in exclude]


def _print_stats(df: pd.DataFrame):
    print(f"\n  rows       : {len(df)}")
    print(f"  columns    : {list(df.columns)}")

    if "node_id" in df.columns:
        print(f"  unique nodes : {df['node_id'].nunique()}")
    if "graph_id" in df.columns:
        print(f"  unique graphs: {df['graph_id'].nunique()}")

    motif_ids = sorted(df["motif_id"].unique()) if "motif_id" in df.columns else []
    k_vals = sorted(df["k"].unique()) if "k" in df.columns else []
    print(f"  k values   : {k_vals}")
    print(f"  motif_ids  : {motif_ids}")

    if "count" in df.columns:
        col = df["count"]
        print(f"\n  count column stats:")
        print(f"    min  = {col.min()}")
        print(f"    max  = {col.max()}")
        print(f"    mean = {col.mean():.4f}")
        print(f"    std  = {col.std():.4f}")
        zero_rows = (col == 0).sum()
        total_nonzero = (col != 0).sum()
        print(f"    zero entries     : {zero_rows}")
        print(f"    non-zero entries : {total_nonzero}")

    # Build dense node-feature matrix to count all-zero rows
    if "node_id" in df.columns and "motif_id" in df.columns and "count" in df.columns:
        pivot = df.pivot_table(
            index="node_id", columns="motif_id", values="count", aggfunc="sum", fill_value=0
        )
        all_zero_nodes = (pivot == 0).all(axis=1).sum()
        print(f"\n  per-node pivot: {pivot.shape[0]} nodes x {pivot.shape[1]} motif cols")
        print(f"    all-zero node rows : {all_zero_nodes}")
        print(f"    per-motif-col stats:")
        for col_name in pivot.columns:
            s = pivot[col_name]
            print(f"      motif_id={col_name}: min={s.min()}, max={s.max()}, "
                  f"mean={s.mean():.4f}, std={s.std():.4f}")
        return pivot
    return None


def _make_plots(df: pd.DataFrame, dataset: str, root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    if "node_id" not in df.columns or "motif_id" not in df.columns or "count" not in df.columns:
        print("[WARN] Cannot build plots: missing node_id/motif_id/count columns.")
        return

    pivot = df.pivot_table(
        index="node_id", columns="motif_id", values="count", aggfunc="sum", fill_value=0
    )
    motif_cols = list(pivot.columns)
    n = len(motif_cols)
    if n == 0:
        print("[WARN] No motif columns to plot.")
        return

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"Motif count distributions — {dataset}", fontsize=14)

    for idx, col_name in enumerate(motif_cols):
        ax = axes[idx // ncols][idx % ncols]
        data = pivot[col_name]
        ax.hist(data[data > 0], bins=40, color="steelblue", edgecolor="none", log=True)
        ax.set_title(f"motif_id={col_name}")
        ax.set_xlabel("count (non-zero nodes)")
        ax.set_ylabel("frequency (log)")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path = root / dataset / "motif_distributions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n[PLOT] Saved distribution histogram to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check a motif CSV produced by generate_motifs.py."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=ALL_DATASETS,
        help="Dataset whose motif CSV to verify.",
    )
    parser.add_argument(
        "--root",
        default="data/precompute",
        help="Root directory for precomputed motif files (default: data/precompute).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save per-motif histogram PNGs to <root>/<dataset>/motif_distributions.png.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    csv_path = _locate_csv(args.dataset, root)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("Run generate_motifs.py first:")
        print(f"  python scripts/preprocess/generate_motifs.py --dataset {args.dataset}")
        sys.exit(1)

    print(f"[INFO] Verifying: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")

    _print_stats(df)

    if args.plot:
        _make_plots(df, args.dataset, root)


if __name__ == "__main__":
    main()
