# src/datasets/planetoid.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import index_to_mask

from src.utils.registry import DATASET_REGISTRY


@dataclass
class NodeDatasetBundle:
    # Single-graph, full-batch node classification bundle
    pyg: Any                 # the underlying PyG dataset object
    data: Any                # pyg[0], a Data object
    num_features: int
    num_classes: int
    splits: Dict[str, Tensor]  # {'train': mask, 'val': mask, 'test': mask}


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _make_planetoid_splits(data, num_classes: int, seed: int) -> Dict[str, Tensor]:
    """
    Deterministic Planetoid-style splits:
      - 20 labeled nodes per class for train (standard Planetoid protocol)
      - 500 val, 1000 test (fall back to proportion if graph is small)
    If dataset already ships masks, we still regenerate using the given seed
    to keep runs comparable across datasets.
    """
    N = data.num_nodes
    y = data.y.cpu()
    g = _seeded_generator(seed)

    # Train: 20 per class (or all if class too small)
    train_idx = []
    for c in range(num_classes):
        idx_c = (y == c).nonzero(as_tuple=False).view(-1)
        # shuffle deterministically
        perm = idx_c[torch.randperm(idx_c.numel(), generator=g)]
        take = min(20, perm.numel())
        train_idx.append(perm[:take])
    train_idx = torch.cat(train_idx) if len(train_idx) else torch.empty(0, dtype=torch.long)

    # Remaining pool
    mask_pool = torch.ones(N, dtype=torch.bool)
    if train_idx.numel() > 0:
        mask_pool[train_idx] = False
    pool_idx = mask_pool.nonzero(as_tuple=False).view(-1)

    # Val/Test sizes with fallbacks for tiny graphs
    val_size = min(500, max(1, int(0.15 * N)))
    test_size = min(1000, max(1, int(0.30 * N)))

    # Deterministic shuffle of pool
    pool_idx = pool_idx[torch.randperm(pool_idx.numel(), generator=g)]
    val_idx = pool_idx[:val_size]
    test_idx = pool_idx[val_size:val_size + test_size]

    # Masks
    train_mask = index_to_mask(train_idx, size=N)
    val_mask = index_to_mask(val_idx, size=N)
    test_mask = index_to_mask(test_idx, size=N)
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def _load_planetoid(name: str, root: str, split_seed: int = 42) -> NodeDatasetBundle:
    ds = Planetoid(root=root, name=name.capitalize())  # 'Cora', 'Citeseer', 'Pubmed'
    data = ds[0]
    splits = _make_planetoid_splits(data, ds.num_classes, split_seed)
    return NodeDatasetBundle(
        pyg=ds,
        data=data,
        num_features=ds.num_features,
        num_classes=ds.num_classes,
        splits=splits,
    )


@DATASET_REGISTRY.register("cora")
class CoraDataset:
    task = "node"
    def __init__(self, root: str = "data/processed", split_seed: int = 42, **kwargs: Any):
        self.bundle = _load_planetoid("cora", root, split_seed=split_seed)
        # convenient shortcuts
        self.data = self.bundle.data
        self.splits = self.bundle.splits
        self.num_features = self.bundle.num_features
        self.num_classes = self.bundle.num_classes


@DATASET_REGISTRY.register("citeseer")
class CiteseerDataset:
    task = "node"
    def __init__(self, root: str = "data/processed", split_seed: int = 42, **kwargs: Any):
        self.bundle = _load_planetoid("citeseer", root, split_seed=split_seed)
        self.data = self.bundle.data
        self.splits = self.bundle.splits
        self.num_features = self.bundle.num_features
        self.num_classes = self.bundle.num_classes


@DATASET_REGISTRY.register("pubmed")
class PubmedDataset:
    task = "node"
    def __init__(self, root: str = "data/processed", split_seed: int = 42, **kwargs: Any):
        self.bundle = _load_planetoid("pubmed", root, split_seed=split_seed)
        self.data = self.bundle.data
        self.splits = self.bundle.splits
        self.num_features = self.bundle.num_features
        self.num_classes = self.bundle.num_classes
