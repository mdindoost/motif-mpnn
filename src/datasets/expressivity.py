# src/datasets/expressivity.py
"""Synthetic expressivity graphs: CSL benchmark + Shrikhande/rook SRG pair.

All counting is local; no HiPerMotif. These graphs make the 1-WL ceiling and the
substructure-feature payoff concrete (see docs/superpowers/specs).
"""
from __future__ import annotations
from typing import List, Tuple

import torch

CSL_SKIPS = (2, 3, 4, 5, 6, 9, 11, 12, 13, 16)


def _undirected_edge_index(edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Build a [2, 2E] edge_index with both directions, de-duplicated, no self loops."""
    seen = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        seen.add((a, b))
    rows, cols = [], []
    for a, b in sorted(seen):
        rows += [a, b]
        cols += [b, a]
    return torch.tensor([rows, cols], dtype=torch.long)


def make_csl_single(n: int = 41, s: int = 3) -> torch.Tensor:
    """CSL(n, s): cycle on n vertices plus skip-s chords. 4-regular for valid (n, s)."""
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, (i + s) % n))
    return _undirected_edge_index(edges)


def _relabel(edge_index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return perm[edge_index]


def make_csl(n: int = 41, skips=CSL_SKIPS, copies_per_class: int = 15, seed: int = 0):
    """Return a list of (edge_index, label) — `copies_per_class` relabeled copies per skip."""
    g = torch.Generator().manual_seed(int(seed))
    out = []
    for label, s in enumerate(skips):
        base = make_csl_single(n=n, s=s)
        for _ in range(copies_per_class):
            perm = torch.randperm(n, generator=g)
            out.append((_relabel(base, perm), label))
    return out


def make_rook_4x4() -> torch.Tensor:
    """4x4 rook graph: vertices (r,c)->4r+c; adjacent iff same row or same column."""
    edges = []
    for r in range(4):
        for c in range(4):
            u = 4 * r + c
            for c2 in range(4):
                if c2 != c:
                    edges.append((u, 4 * r + c2))
            for r2 in range(4):
                if r2 != r:
                    edges.append((u, 4 * r2 + c))
    return _undirected_edge_index(edges)


def make_shrikhande() -> torch.Tensor:
    """Shrikhande graph: Cayley graph on Z4 x Z4, connection set +/-{(1,0),(0,1),(1,1)}."""
    S = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    idx = lambda a, b: 4 * a + b
    edges = []
    for a in range(4):
        for b in range(4):
            for (da, db) in S:
                edges.append((idx(a, b), idx((a + da) % 4, (b + db) % 4)))
    return _undirected_edge_index(edges)


def _edge_index_to_igraph(edge_index: torch.Tensor, num_nodes: int):
    import igraph as ig
    pairs = set()
    for u, v in edge_index.t().tolist():
        a, b = (u, v) if u < v else (v, u)
        if a != b:
            pairs.add((a, b))
    g = ig.Graph(n=num_nodes, edges=sorted(pairs), directed=False)
    g.simplify()
    return g


# --------------------------------------------------------------------------
# CSL registered dataset (graph task) — mirrors src/datasets/tu.py structure
# --------------------------------------------------------------------------
from typing import Any, Dict, List as _List

from src.utils.registry import DATASET_REGISTRY
from src.datasets.motif_loader import build_or_load_tu_motif_list
from src.datasets.tu_wrapper import TUWithMotifs


def _csl_data_list(seed: int = 0):
    """Build the 150 CSL graphs as a list of PyG Data with constant features."""
    from torch_geometric.data import Data
    graphs = make_csl(seed=seed)
    data_list = []
    for ei, label in graphs:
        n = int(ei.max().item()) + 1
        d = Data(x=torch.ones(n, 1, dtype=torch.float32),
                 edge_index=ei,
                 y=torch.tensor([label], dtype=torch.long))
        d.num_nodes = n
        data_list.append(d)
    return data_list


def _stratified_indices(labels: torch.Tensor, seed: int, ratios=(0.6, 0.2, 0.2)) -> Dict[str, _List[int]]:
    g = torch.Generator().manual_seed(int(seed))
    all_idx = torch.arange(labels.numel())
    train_idx, val_idx, test_idx = [], [], []
    for c in labels.unique(sorted=True):
        idx_c = all_idx[labels == c]
        perm = idx_c[torch.randperm(idx_c.numel(), generator=g)]
        n = perm.numel()
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        train_idx.append(perm[:n_train])
        val_idx.append(perm[n_train:n_train + n_val])
        test_idx.append(perm[n_train + n_val:])
    to_list = lambda xs: torch.cat(xs).tolist() if xs else []
    return {"train": to_list(train_idx), "val": to_list(val_idx), "test": to_list(test_idx)}


@DATASET_REGISTRY.register("csl")
class CSLDataset:
    task = "graph"

    def __init__(self, root: str = "data/processed", use_public_split: bool = False,
                 split_seed: int = 42, **kwargs: Any):
        # CSL graph set is fixed (seed 0) so the motif CSV's graph_id aligns with index.
        base = _csl_data_list(seed=0)
        labels = torch.tensor([int(d.y) for d in base], dtype=torch.long)
        self.splits = _stratified_indices(labels, seed=split_seed)

        pre_dir = "data/precompute/csl"
        art = build_or_load_tu_motif_list(dataset="csl", pyg_dataset=base, precompute_dir=pre_dir)
        if art.X_list is not None:
            self.dataset = TUWithMotifs(base, art.X_list)
        else:
            self.dataset = base
        self.motif_stats = art.stats or {}
        self.motif_manifest = art.manifest or {}

        self.num_features = 1
        self.num_classes = 10
