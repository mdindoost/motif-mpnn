# src/datasets/__init__.py
from typing import Any, Dict
from src.utils.registry import DATASET_REGISTRY

# --- keep a minimal dummy for quick smoke tests ---
class DummyData:
    def __init__(self, x_dim: int = 1, y_dim: int = 2):
        self.metadata: Dict[str, Any] = {"x_dim": x_dim, "y_dim": y_dim}

@DATASET_REGISTRY.register("dummy_node")
class DummyNodeDataset:
    task = "node"
    def __init__(self, root: str = "data/processed", **kwargs: Any):
        self.root = root
        self.data = DummyData()

# --- import real adapters to populate the registry keys ---
# These imports have side effects: they register 'cora', 'citeseer', 'pubmed', 'proteins', 'nci1', 'enzymes'
from .planetoid import *  # noqa: F401,F403
from .tu import *         # noqa: F401,F403
