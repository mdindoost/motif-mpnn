# src/datasets/tu.py
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset

from src.utils.registry import DATASET_REGISTRY


@dataclass
class GraphDatasetBundle:
    pyg: Any                 # TUDataset
    num_features: int
    num_classes: int
    splits: Dict[str, List[int]]  # {'train': [idx...], 'val': [...], 'test': [...]}


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _stratified_indices(labels: torch.Tensor, seed: int, ratios=(0.8, 0.1, 0.1)) -> Dict[str, List[int]]:
    """Stratified split over graph labels into train/val/test with given ratios."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    g = _seeded_generator(seed)
    labels = labels.cpu()
    all_idx = torch.arange(labels.numel())
    train_idx, val_idx, test_idx = [], [], []
    for c in labels.unique(sorted=True):
        idx_c = all_idx[labels == c]
        perm = idx_c[torch.randperm(idx_c.numel(), generator=g)]
        n = perm.numel()
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        # ensure all samples assigned
        n_test = n - n_train - n_val
        train_idx.append(perm[:n_train])
        val_idx.append(perm[n_train:n_train + n_val])
        test_idx.append(perm[n_train + n_val:])
    to_list = lambda xs: torch.cat(xs).tolist() if xs else []
    return {"train": to_list(train_idx), "val": to_list(val_idx), "test": to_list(test_idx)}


def _load_tu(name: str, root: str, split_seed: int = 42) -> GraphDatasetBundle:
    ds = TUDataset(root=root, name=name.upper())  # 'PROTEINS', 'NCI1', 'ENZYMES'
    y = torch.tensor([ds[i].y.item() for i in range(len(ds))], dtype=torch.long)
    splits = _stratified_indices(y, seed=split_seed)
    return GraphDatasetBundle(
        pyg=ds,
        num_features=ds.num_features,
        num_classes=ds.num_classes,
        splits=splits,
    )


@DATASET_REGISTRY.register("proteins")
class ProteinsDataset:
    task = "graph"
    def __init__(self, root: str = "data/processed", split_seed: int = 42, **kwargs: Any):
        self.bundle = _load_tu("proteins", root, split_seed=split_seed)
        self.dataset = self.bundle.pyg
        self.num_features = self.bundle.num_features
        self.num_classes = self.bundle.num_classes
        self.splits = self.bundle.splits


@DATASET_REGISTRY.register("nci1")
class NCI1Dataset:
    task = "graph"
    def __init__(self, root: str = "data/processed", split_seed: int = 42, **kwargs: Any):
        self.bundle = _load_tu("nci1", root, split_seed=split_seed)
        self.dataset = self.bundle.pyg
        self.num_features = self.bundle.num_features
        self.num_classes = self.bundle.num_classes
        self.splits = self.bundle.splits


@DATASET_REGISTRY.register("enzymes")
class ENZYMESDataset:
    task = "graph"
    def __init__(self, root: str = "data/processed", split_seed: int = 42, **kwargs: Any):
        self.bundle = _load_tu("enzymes", root, split_seed=split_seed)
        self.dataset = self.bundle.pyg
        self.num_features = self.bundle.num_features
        self.num_classes = self.bundle.num_classes
        self.splits = self.bundle.splits
