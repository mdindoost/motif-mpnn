# scripts/preprocess/download_datasets.py
# Purpose: Download a small starter set of node- and graph-classification datasets into data/processed/pyg
# No training here; just caching to a repo-local path so users don’t fetch into their home cache.

import os
from pathlib import Path

from torch_geometric.datasets import Planetoid, TUDataset

ROOT = Path(__file__).resolve().parents[2] / "data" / "processed" / "pyg"
ROOT.mkdir(parents=True, exist_ok=True)

def fetch_planetoid(name: str):
    ds = Planetoid(root=str(ROOT / name), name=name.capitalize())
    print(f"[OK] {name}: {len(ds)} graph(s) -> {ROOT/name}")

def fetch_tu(name: str):
    ds = TUDataset(root=str(ROOT / name), name=name.upper())
    print(f"[OK] {name}: {len(ds)} graph(s) -> {ROOT/name}")

if __name__ == "__main__":
    # Node classification
    for name in ["cora", "citeseer", "pubmed"]:
        fetch_planetoid(name)

    # Graph classification
    for name in ["PROTEINS", "NCI1", "ENZYMES"]:
        fetch_tu(name)

    print("\nAll datasets cached under:", ROOT)
