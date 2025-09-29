# src/models/gat.py
from typing import Any
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


from src.utils.registry import MODEL_REGISTRY
from .common import LayerNorm1d, ResidualBlock, MLPHead




class GAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 2,
        dropout: float = 0.5, layer_norm: bool = True, residual: bool = True,
        heads: int = 4, task: str = "node"):
        
        super().__init__()
        assert num_layers >= 1
        self.task = task
        self.dropout = nn.Dropout(dropout)


        # For GAT, we keep hidden_dim as the per-head output dim.
        dims_in = [in_dim] + [hidden_dim * heads] * (num_layers - 1)
        dims_out = [hidden_dim * heads] * (num_layers - 1) + [hidden_dim]
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.res = nn.ModuleList()


        # first layer: in_dim -> hidden_dim * heads
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=True))
        self.lns.append(LayerNorm1d(hidden_dim * heads) if layer_norm else nn.Identity())
        self.res.append(ResidualBlock(in_dim, hidden_dim * heads, residual))


        # middle layers
        for i in range(1, num_layers):
            self.convs.append(GATv2Conv(dims_in[i], hidden_dim, heads=heads, dropout=dropout, add_self_loops=True))
            self.lns.append(LayerNorm1d(hidden_dim * heads) if layer_norm else nn.Identity())
            self.res.append(ResidualBlock(dims_in[i], hidden_dim * heads, residual))


        self.act = nn.ELU()
        self.head = MLPHead(hidden_dim * heads, out_dim)


    def encode(self, x, edge_index):
        for conv, ln, res in zip(self.convs, self.lns, self.res):
            h = conv(x, edge_index)
            h = ln(h)
            h = self.act(h)
            h = self.dropout(h)
            x = res(x, h)
        return x


    def forward(self, data):
        if self.task == "node":
            x = self.encode(data.x, data.edge_index)
            return self.head(x)
        else:
            x = self.encode(data.x, data.edge_index)
            x = global_mean_pool(x, data.batch)
            return self.head(x)




@MODEL_REGISTRY.register("gat")
class GATFactory:
    def __new__(cls, in_dim: int, out_dim: int, **kwargs: Any):
        task = kwargs.pop("task", "node")
        return GAT(in_dim=in_dim, out_dim=out_dim, task=task, **kwargs)
