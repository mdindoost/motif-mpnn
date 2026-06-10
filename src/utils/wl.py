# src/utils/wl.py
"""Dependency-free 1-dimensional Weisfeiler-Leman color refinement.

Two graphs with different stable color signatures are distinguishable by 1-WL and
hence by any message-passing GNN; equal signatures means they are NOT (the ceiling).
"""
from __future__ import annotations
from collections import Counter
from typing import Dict

import torch


def _adjacency(edge_index: torch.Tensor, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        if u != v:
            adj[u].append(v)
    return adj


def wl_refine(edge_index: torch.Tensor, num_nodes: int, num_iters: int | None = None):
    """Return a list[int] stable coloring (canonicalized to 0..k-1 per iteration)."""
    adj = _adjacency(edge_index, num_nodes)
    colors = [0] * num_nodes  # all start identical
    max_iters = num_iters if num_iters is not None else num_nodes
    for _ in range(max_iters):
        signatures = []
        for v in range(num_nodes):
            nbr = tuple(sorted(colors[u] for u in adj[v]))
            signatures.append((colors[v], nbr))
        # canonicalize signatures -> new integer colors (stable order)
        order = {sig: i for i, sig in enumerate(sorted(set(signatures)))}
        new_colors = [order[s] for s in signatures]
        if new_colors == colors:
            break
        colors = new_colors
    return colors


def wl_graph_signature(edge_index: torch.Tensor, num_nodes: int) -> Dict[int, int]:
    """Histogram {color: count} of the stable coloring — a 1-WL graph invariant."""
    colors = wl_refine(edge_index, num_nodes)
    return dict(Counter(colors))
