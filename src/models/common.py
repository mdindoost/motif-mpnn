# src/models/common.py
from typing import Optional
import torch
from torch import nn


class LayerNorm1d(nn.Module):
    """LayerNorm over feature dim of shape [N, F] (node) or [B, F] (graph)."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
    def forward(self, x, out):
        return out + x if self.use_residual else out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.lin(x)
