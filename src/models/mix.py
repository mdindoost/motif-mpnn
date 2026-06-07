# src/models/mix.py
import warnings
from typing import Any, Optional
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.utils.registry import MODEL_REGISTRY
from .common import LayerNorm1d, ResidualBlock, MLPHead


def _motif_edge_sim(motif_x: torch.Tensor,
                    edge_index: torch.Tensor,
                    metric: str = "cosine",
                    topk: Optional[int] = None) -> torch.Tensor:
    """
    Compute similarity s_uv along existing edges.
    - metric: "cosine" or "dot"
    - topk: keep top-k outgoing edges per source node; zero the rest
    Returns: [E] float tensor in [ -1, 1 ] (cosine) or unbounded (dot), not yet mixed.
    """
    src, dst = edge_index[0], edge_index[1]
    mu = motif_x[src]  # [E, M]
    mv = motif_x[dst]

    if metric == "cosine":
        eps = 1e-8
        mu = mu / (mu.norm(dim=1, keepdim=True) + eps)
        mv = mv / (mv.norm(dim=1, keepdim=True) + eps)
        s = (mu * mv).sum(dim=1)  # in [-1, 1]
        # map to [0,1] for convex mixing if desired:
        s = (s + 1.0) * 0.5
    elif metric == "dot":
        s = (mu * mv).sum(dim=1)
        # normalize to [0,1] with min-max per-batch for stability (optional but helpful)
        if s.numel() > 0:
            s_min, s_max = s.min(), s.max()
            rng = (s_max - s_min).clamp_min(1e-8)
            s = (s - s_min) / rng
    else:
        raise ValueError(f"Unknown sim_metric: {metric}")

    if topk is not None and topk > 0:
        # Vectorized per-source topk — O(E log E) via two stable sorts; no Python loop.
        # SCALING NOTE: the old implementation used `for u in torch.unique(src)` which is
        # O(N) Python loop and stalls on large batched TU graphs. This replaces it.
        E = s.size(0)
        device = s.device
        if E > 0:
            # Lexicographic sort: (src ASC, s DESC)
            # Step 1: sort by s descending (stable)
            perm1 = torch.argsort(s, descending=True, stable=True)
            # Step 2: stable sort by src; within same src, order from step 1 is preserved
            perm2 = torch.argsort(src[perm1], stable=True)
            perm = perm1[perm2]  # edges in (src ASC, s DESC) order

            sorted_src = src[perm]
            # True at each position where a new source group begins
            boundaries = torch.cat([
                torch.ones(1, dtype=torch.bool, device=device),
                sorted_src[1:] != sorted_src[:-1],
            ])
            # group_id[i] = which source group position i belongs to
            group_id = boundaries.long().cumsum(0) - 1
            # position of first edge in each group (the boundary positions)
            group_start_pos = boundaries.nonzero(as_tuple=False).view(-1)
            # within-group rank: 0 = best score in group, 1 = second best, ...
            within_rank = torch.arange(E, device=device) - group_start_pos[group_id]

            keep_sorted = within_rank < topk
            keep = torch.zeros(E, dtype=torch.bool, device=device)
            keep[perm] = keep_sorted
            s = s * keep.float()

    return s.clamp(0.0, 1.0)


class MixGCN(nn.Module):
    """
    GCN with mixed edge weights:
      w = (1 - lambda_mix) * 1 + lambda_mix * sim(motif_x[u], motif_x[v])
    Sim is computed once per forward over existing edges.
    """
    def __init__(self, in_dim: int, out_dim: int, motif_dim: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, layer_norm: bool = True, residual: bool = True,
                 task: str = "node",
                 # mix knobs
                 lambda_mix: float = 0.25,
                 sim_metric: str = "cosine",
                 motif_topk: int = 10,
                 self_loop: bool = True,
                 **_ignore: Any):
        super().__init__()
        assert num_layers >= 1
        self.task = task
        self.dropout = nn.Dropout(dropout)

        self.lambda_mix = float(lambda_mix)
        self.sim_metric = sim_metric
        self.motif_topk = int(motif_topk) if motif_topk is not None else None
        self.keep_self_loop = bool(self_loop)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.res = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i+1], normalize=True, add_self_loops=self.keep_self_loop))
            self.lns.append(LayerNorm1d(dims[i+1]) if layer_norm else nn.Identity())
            self.res.append(ResidualBlock(dims[i], dims[i+1], residual))

        self.act = nn.ReLU()
        self.head = MLPHead(dims[-1], out_dim)
        self._warned_no_motif = False

    def _mixed_edge_weight(self, data):
        if not hasattr(data, "motif_x") or data.motif_x is None or data.motif_x.numel() == 0:
            if not self._warned_no_motif:
                warnings.warn(
                    "MixGCN: data.motif_x is absent — running as plain GCN (lambda_mix has no effect). "
                    "Provide a node_motifs.csv under data/precompute/<dataset>/ to enable adjacency mixing.",
                    UserWarning, stacklevel=3
                )
                self._warned_no_motif = True
            return None  # behaves like standard GCN
        s = _motif_edge_sim(data.motif_x, data.edge_index, metric=self.sim_metric, topk=self.motif_topk)
        # Blend with structural weight=1.0
        return (1.0 - self.lambda_mix) + self.lambda_mix * s

    def encode(self, x, edge_index, edge_weight):
        for conv, ln, res in zip(self.convs, self.lns, self.res):
            h = conv(x, edge_index, edge_weight=edge_weight)
            h = ln(h)
            h = self.act(h)
            h = self.dropout(h)
            x = res(x, h)
        return x

    def forward(self, data):
        edge_weight = self._mixed_edge_weight(data)
        if self.task == "node":
            x = self.encode(data.x, data.edge_index, edge_weight=edge_weight)
            return self.head(x)
        else:
            x = self.encode(data.x, data.edge_index, edge_weight=edge_weight)
            x = global_mean_pool(x, data.batch)
            return self.head(x)


@MODEL_REGISTRY.register("mix")
class MixFactory:
    def __new__(cls, in_dim: int, out_dim: int, motif_dim: int = 0, **kwargs: Any):
        return MixGCN(in_dim=in_dim, out_dim=out_dim, motif_dim=motif_dim, **kwargs)
