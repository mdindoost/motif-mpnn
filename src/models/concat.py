# src/models/concat.py
from typing import Any
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from src.utils.registry import MODEL_REGISTRY
from .gcn import GCN  # reuse encoder stack

class ConcatModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        super().__init__()
        enc_in = in_dim + max(0, motif_dim)
        # use GCN encoder; kwargs include hidden_dim, num_layers, dropout, task, etc.
        self.encoder = GCN(in_dim=enc_in, out_dim=out_dim, **kwargs)
        self.task = kwargs.get("task", "node")
        self.motif_dim = motif_dim

    def forward(self, data):
        if self.task == "node":
            x = data.x
            if hasattr(data, "motif_x") and data.motif_x is not None and data.motif_x.numel() > 0:
                x = torch.cat([x, data.motif_x.to(x.device, dtype=x.dtype)], dim=1)
            proxy = type("Obj", (), {"x": x, "edge_index": data.edge_index, "batch": getattr(data, "batch", None)})
            return self.encoder.forward(proxy)
        else:
            # TU graph task: we’ll attach per-graph motif_x in Phase C.1
            return self.encoder(data)

@MODEL_REGISTRY.register("concat")
class ConcatFactory:
    def __new__(cls, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        return ConcatModel(in_dim=in_dim, out_dim=out_dim, motif_dim=motif_dim, **kwargs)
