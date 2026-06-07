# src/models/concat.py
import warnings
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
        self._warned_no_motif = False

    def forward(self, data):
        x = data.x
        has_motif = hasattr(data, "motif_x") and data.motif_x is not None and data.motif_x.numel() > 0
        if self.motif_dim > 0 and not has_motif and not self._warned_no_motif:
            warnings.warn(
                "ConcatModel: motif_dim > 0 but data.motif_x is absent — running as plain GCN. "
                "Provide a node_motifs.csv under data/precompute/<dataset>/ to enable motif features.",
                UserWarning, stacklevel=2
            )
            self._warned_no_motif = True
        if has_motif:
            x = torch.cat([x, data.motif_x.to(x.device, dtype=x.dtype)], dim=1)
        # pad to expected input dim if motif features are absent but model was built with motif_dim
        elif self.motif_dim > 0:
            pad = torch.zeros(x.size(0), self.motif_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        proxy = type("Obj", (), {"x": x, "edge_index": data.edge_index, "batch": getattr(data, "batch", None)})
        return self.encoder.forward(proxy)

@MODEL_REGISTRY.register("concat")
class ConcatFactory:
    def __new__(cls, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        return ConcatModel(in_dim=in_dim, out_dim=out_dim, motif_dim=motif_dim, **kwargs)
