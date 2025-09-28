#!/usr/bin/env python3
"""
Export PyTorch Geometric datasets (Planetoid: Cora/Citeseer/Pubmed; TU: PROTEINS/NCI1/ENZYMES)
to flat edge lists "src dst" per graph — ideal for HiPerXplorer and NetworkX.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Set

import torch
from torch_geometric.datasets import Planetoid, TUDataset

def iter_undirected_unique_edges(edge_index: torch.Tensor, dedup: bool):
    ei = edge_index.t().tolist()
    seen: Set[Tuple[int, int]] = set()
    for u, v in ei:
        a, b = (u, v) if u < v else (v, u)
        if a == b:
            continue
        if not dedup or (a, b) not in seen:
            seen.add((a, b))
            yield (a, b)

def iter_directed_edges(edge_index: torch.Tensor, keep_self_loops: bool):
    for u, v in edge_index.t().tolist():
        if not keep_self_loops and u == v:
            continue
        yield (u, v)

def export_planetoid(name, input_root, output_root, undirected, dedup, keep_self_loops):
    ds = Planetoid(root=str(input_root / name), name=name.capitalize())
    data = ds[0]
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "edges.txt").open("w") as f:
        if undirected:
            for u, v in iter_undirected_unique_edges(data.edge_index, dedup):
                f.write(f"{u} {v}\n")
        else:
            for u, v in iter_directed_edges(data.edge_index, keep_self_loops):
                f.write(f"{u} {v}\n")
    with (out_dir / "nodes.txt").open("w") as nf:
        for nid in range(data.num_nodes):
            nf.write(f"{nid}\n")
    print(f"[OK] {name} -> {out_dir}")

def export_tudataset(name, input_root, output_root, undirected, dedup, keep_self_loops):
    ds = TUDataset(root=str(input_root / name), name=name.upper())
    base_dir = output_root / name
    base_dir.mkdir(parents=True, exist_ok=True)
    for gid, data in enumerate(ds):
        gdir = base_dir / str(gid)
        gdir.mkdir(parents=True, exist_ok=True)
        with (gdir / "edges.txt").open("w") as f:
            if undirected:
                for u, v in iter_undirected_unique_edges(data.edge_index, dedup):
                    f.write(f"{u} {v}\n")
            else:
                for u, v in iter_directed_edges(data.edge_index, keep_self_loops):
                    f.write(f"{u} {v}\n")
        with (gdir / "nodes.txt").open("w") as nf:
            for nid in range(data.num_nodes):
                nf.write(f"{nid}\n")
    print(f"[OK] {name} -> {base_dir} ({len(ds)} graphs)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--input-root", type=Path, default=Path("data/processed/pyg"))
    ap.add_argument("--output-root", type=Path, default=Path("data/raw_export"))
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--keep-self-loops", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        lname = name.lower()
        if lname in {"cora", "citeseer", "pubmed"}:
            export_planetoid(lname, args.input_root, args.output_root,
                             args.undirected, args.dedup, args.keep_self_loops)
        else:
            export_tudataset(name, args.input_root, args.output_root,
                             args.undirected, args.dedup, args.keep_self_loops)

if __name__ == "__main__":
    main()
