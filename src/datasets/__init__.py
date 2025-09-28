# src/datasets/__init__.py
from typing import Any, Dict
from src.utils.registry import DATASET_REGISTRY

class DummyData:
    def __init__(self, x_dim: int = 1, y_dim: int = 2):
        self.metadata: Dict[str, Any] = {"x_dim": x_dim, "y_dim": y_dim}

class _DummyDataset:
    task = "node"
    def __init__(self, root: str = "data/processed", **kwargs: Any):
        self.root = root
        self.data = DummyData()

# Register placeholders for all the keys you listed.
@DATASET_REGISTRY.register("dummy_node")
class DummyNodeDataset(_DummyDataset):
    pass

for _k in ["cora", "citeseer", "pubmed", "proteins", "nci1", "enzymes"]:
    # Make separate classes so registry keys are distinct types
    DATASET_REGISTRY._store[_k] = type(f"DummyDataset_{_k}", (_DummyDataset,), {"task": "graph" if _k in {"proteins","nci1","enzymes"} else "node"})

