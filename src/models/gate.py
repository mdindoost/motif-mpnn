# src/models/gate.py
from typing import Any, Optional
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.utils.registry import MODEL_REGISTRY
from .common import LayerNorm1d, ResidualBlock, MLPHead


class EdgeGate(nn.Module):
    """Compute a scalar gate per edge from motif features of (u, v)."""
    def __init__(self, motif_dim: int, hidden: int = 64, dropout: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        in_dim = 4 * motif_dim  # [m_u, m_v, |m_u - m_v|, m_u * m_v]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, motif_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if motif_x is None or motif_x.numel() == 0:
            return torch.ones(edge_index.size(1), device=edge_index.device,
                              dtype=(motif_x.dtype if motif_x is not None else torch.float32))
        src, dst = edge_index[0], edge_index[1]
        mu = motif_x[src]
        mv = motif_x[dst]
        feats = torch.cat([mu, mv, (mu - mv).abs(), mu * mv], dim=1)
        g = self.net(feats).squeeze(-1)  # [E]
        return torch.sigmoid(g / self.tau).clamp(0.0, 1.0)


class GatedGCN(nn.Module):
    """
    GCN encoder where each layer receives edge weights produced from motif context.
    We compute a single gate from motifs and reuse across layers for efficiency.
    """
    def __init__(self, in_dim: int, out_dim: int, motif_dim: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, layer_norm: bool = True, residual: bool = True,
                 task: str = "node",
                 # new gate knobs (optional)
                 gate_hidden: int = 64, gate_dropout: float = 0.0, gate_temp: float = 1.0,
                 **_ignore: Any):
        super().__init__()
        assert num_layers >= 1
        self.task = task
        self.dropout = nn.Dropout(dropout)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.res = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i+1], normalize=True, add_self_loops=True))
            self.lns.append(LayerNorm1d(dims[i+1]) if layer_norm else nn.Identity())
            self.res.append(ResidualBlock(dims[i], dims[i+1], residual))

        self.edge_gate = EdgeGate(motif_dim, hidden=gate_hidden, dropout=gate_dropout, tau=gate_temp) \
                         if motif_dim > 0 else None

        self.act = nn.ReLU()
        self.head = MLPHead(dims[-1], out_dim)

    def encode(self, x, edge_index, edge_weight):
        for conv, ln, res in zip(self.convs, self.lns, self.res):
            h = conv(x, edge_index, edge_weight=edge_weight)
            h = ln(h)
            h = self.act(h)
            h = self.dropout(h)
            x = res(x, h)
        return x

    def forward(self, data):
        gate = None
        if self.edge_gate is not None and hasattr(data, "motif_x") and data.motif_x is not None:
            gate = self.edge_gate(data.motif_x, data.edge_index)  # [E]
        if self.task == "node":
            x = self.encode(data.x, data.edge_index, edge_weight=gate)
            return self.head(x)
        else:
            x = self.encode(data.x, data.edge_index, edge_weight=gate)
            x = global_mean_pool(x, data.batch)
            return self.head(x)


@MODEL_REGISTRY.register("gate")
class GateFactory:
    def __new__(cls, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        # task may be in kwargs; pop nothing critical here
        return GatedGCN(in_dim=in_dim, out_dim=out_dim, motif_dim=motif_dim, **kwargs)
