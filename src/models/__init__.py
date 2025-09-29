# src/models/__init__.py
from typing import Any
from src.utils.registry import MODEL_REGISTRY
from .concat import * # noqa: F401,F403

# Identity stays only for in-progress motif variants
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

# Import real baselines (registers: gcn, sage, gat)
from .gcn import *   # noqa: F401,F403
from .sage import *  # noqa: F401,F403
from .gat import *   # noqa: F401,F403
from .gate import *   # noqa: F401,F403

# Keep placeholders ONLY for upcoming motif variants
for _k in ["concat", "gate", "mix", "motif_adj", "graphlet_concat"]:
    if _k not in MODEL_REGISTRY:
        MODEL_REGISTRY._store[_k] = IdentityFactory
