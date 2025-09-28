# src/models/__init__.py
from typing import Any
from src.utils.registry import MODEL_REGISTRY

class IdentityModel:
    def __init__(self, in_dim: int = 1, out_dim: int = 1, **kwargs: Any):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kwargs = kwargs
    def __repr__(self):
        return f"IdentityModel(in={self.in_dim}, out={self.out_dim})"

@MODEL_REGISTRY.register("identity")
class IdentityFactory(IdentityModel):
    pass

# Map all planned model keys to identity for Phase A smoke
for _k in ["gcn", "gat", "sage", "concat", "gate", "mix", "motif_adj", "graphlet_concat"]:
    MODEL_REGISTRY._store[_k] = IdentityFactory
