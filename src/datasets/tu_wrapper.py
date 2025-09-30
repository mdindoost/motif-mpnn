# src/datasets/tu_wrapper.py
from __future__ import annotations
from typing import List, Optional
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except Exception:
    Data = object  # type: ignore


class TUWithMotifs(Dataset):
    """
    Wrap a PyG TUDataset-like object and attach per-graph motif matrices.

    Args:
        base: a torch_geometric dataset where base[i] returns a Data with .num_nodes
        motif_list: list of tensors [num_nodes_i, M] aligned with base indices
        strict: if True, raises when num_nodes mismatch; if False, pads/crops as needed
    """
    def __init__(self, base: Dataset, motif_list: List[torch.Tensor], strict: bool = False):
        self.base = base
        self.motif_list = motif_list
        self.strict = strict
        if len(self.base) != len(self.motif_list):
            raise ValueError(f"motif_list length {len(self.motif_list)} != dataset length {len(self.base)}")

        # infer motif_dim for convenience
        self.motif_dim = int(self.motif_list[0].size(1)) if len(self.motif_list) > 0 else 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        data = self.base[idx]
        # clone to avoid mutating the underlying dataset object across epochs
        if hasattr(data, "clone"):
            data = data.clone()

        Xg = self.motif_list[idx]
        n_data = int(getattr(data, "num_nodes", Xg.size(0)))
        n_motif = int(Xg.size(0))
        M = int(Xg.size(1))

        if n_motif != n_data:
            if self.strict:
                raise ValueError(f"[TUWithMotifs] num_nodes mismatch at idx={idx}: data={n_data}, motif={n_motif}")
            # non-strict: pad/crop to match graph size
            if n_motif < n_data:
                pad = torch.zeros((n_data - n_motif, M), dtype=Xg.dtype)
                Xg = torch.cat([Xg, pad], dim=0)
            else:
                Xg = Xg[:n_data, :]

        # attach; trainer/model will move to device as needed
        data.motif_x = Xg
        return data

    # PyG DataLoader sometimes calls .get on datasets; pass through when present
    def get(self, idx: int):
        return self.__getitem__(idx)

    def __repr__(self) -> str:
        return f"TUWithMotifs(base={self.base.__class__.__name__}, motif_dim={self.motif_dim}, len={len(self)})"
