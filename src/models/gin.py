# src/models/gin.py
from typing import Any
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from src.utils.registry import MODEL_REGISTRY
from .common import LayerNorm1d, MLPHead


class GIN(nn.Module):
    """Graph Isomorphism Network — the maximally-1-WL-expressive MPNN (Xu et al. 2019).

    Used as the tight ceiling baseline: on CSL it provably cannot beat 10%.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, layer_norm: bool = True, residual: bool = False,
                 task: str = "graph"):
        super().__init__()
        assert num_layers >= 1
        self.task = task
        self.dropout = nn.Dropout(dropout)
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]), nn.ReLU(),
                nn.Linear(dims[i + 1], dims[i + 1]),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.lns.append(LayerNorm1d(dims[i + 1]) if layer_norm else nn.Identity())
        self.act = nn.ReLU()
        self.head = MLPHead(dims[-1], out_dim)

    def encode(self, x, edge_index):
        for conv, ln in zip(self.convs, self.lns):
            x = conv(x, edge_index)
            x = ln(x)
            x = self.act(x)
            x = self.dropout(x)
        return x

    def forward(self, data):
        x = self.encode(data.x, data.edge_index)
        if self.task == "node":
            return self.head(x)
        x = global_add_pool(x, data.batch)
        return self.head(x)


@MODEL_REGISTRY.register("gin")
class GINFactory:
    def __new__(cls, in_dim: int, out_dim: int, **kwargs: Any):
        task = kwargs.pop("task", "graph")
        # GIN ignores `residual` (no residual in canonical GIN); drop if passed.
        kwargs.pop("residual", None)
        return GIN(in_dim=in_dim, out_dim=out_dim, task=task, **kwargs)
