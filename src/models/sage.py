# src/models/sage.py
from typing import Any
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool


from src.utils.registry import MODEL_REGISTRY
from .common import LayerNorm1d, ResidualBlock, MLPHead




class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 2,
        dropout: float = 0.5, layer_norm: bool = True, residual: bool = True,
        task: str = "node"):
        
        super().__init__()
        assert num_layers >= 1
        self.task = task
        self.dropout = nn.Dropout(dropout)


        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.res = nn.ModuleList()


        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.lns.append(LayerNorm1d(dims[i+1]) if layer_norm else nn.Identity())
            self.res.append(ResidualBlock(dims[i], dims[i+1], residual))


        self.act = nn.ReLU()
        self.head = MLPHead(dims[-1], out_dim)


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




@MODEL_REGISTRY.register("sage")
class SAGEFactory:
    def __new__(cls, in_dim: int, out_dim: int, **kwargs: Any):
        task = kwargs.pop("task", "node")
        return GraphSAGE(in_dim=in_dim, out_dim=out_dim, task=task, **kwargs)
