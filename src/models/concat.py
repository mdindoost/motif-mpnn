# src/models/concat.py
from typing import Any
from torch import nn
from torch_geometric.nn import global_mean_pool


from src.utils.registry import MODEL_REGISTRY
from .gcn import GCN # reuse GCN encoder stack


class ConcatModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        super().__init__()
        enc_in = in_dim + max(0, motif_dim)
        # reuse GCN as a strong encoder backbone for now
        self.encoder = GCN(in_dim=enc_in, out_dim=out_dim, **kwargs)
        self.task = kwargs.get('task', 'node')


    def forward(self, data):
        # Expect `data` to provide .x and optional .motif_x for node task
        if self.task == 'node':
            x = data.x
            if hasattr(data, 'motif_x') and data.motif_x is not None:
                x = x.new_zeros((x.size(0), x.size(1) + data.motif_x.size(1))).to(x.device)
            return self.encoder.forward(type('Obj', (), {'x': x, 'edge_index': data.edge_index, 'batch': getattr(data, 'batch', None)}))
        else:
            # For TU, we’ll attach per-batch motif tensors later; placeholder path for now
            return self.encoder(data)


@MODEL_REGISTRY.register("concat")
class ConcatFactory:
    def __new__(cls, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        return ConcatModel(in_dim=in_dim, out_dim=out_dim, motif_dim=motif_dim, **kwargs)
