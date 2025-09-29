# src/datasets/motif_loader.py

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import torch
import pandas as pd

@dataclass
class MotifArtifacts:
    X: Optional[torch.Tensor] # [N, M] for node datasets; None for TU here
    stats: Dict
    manifest: Dict
    # For TU datasets we also return a per-graph list of motif matrices
    X_list: Optional[List[torch.Tensor]] = None




def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)




def _read_json(p: Path) -> Optional[Dict]:
    return json.loads(p.read_text()) if p.exists() else None




def _write_json(p: Path, obj: Dict):
    p.write_text(json.dumps(obj, indent=2))
    
    
def _log1p_zscore_dense(X: torch.Tensor, stats: Optional[Dict]) -> Tuple[torch.Tensor, Dict]:
    # X is raw counts (float or int)
    X = torch.log1p(X)
    if stats is None:
        # column-wise mean/std (include zeros to remain consistent across nodes)
        mean = X.mean(dim=0)
        std = X.std(dim=0)
        # avoid div by zero
        std = torch.where(std > 0, std, torch.ones_like(std))
        stats = {
        "log1p_mean": mean.tolist(),
        "log1p_std": std.tolist(),
        }
    else:
        mean = torch.tensor(stats.get("log1p_mean", [0.0]*X.size(1)), dtype=X.dtype)
        std = torch.tensor(stats.get("log1p_std", [1.0]*X.size(1)), dtype=X.dtype)
        std = torch.where(std > 0, std, torch.ones_like(std))
    X = (X - mean) / std
    return X, stats

def _flatten_manifest(manifest: Dict) -> Dict[Tuple[int, int], int]:
    """Support either {"k=3": {"175": 0, ...}} or {"(3,175)": 0} styles.
    Returns a map (k, motif_id)->col_idx.
    """
    if not manifest:
        return {}
    flat: Dict[Tuple[int,int], int] = {}
    for k_str, inner in manifest.items():
        # try "k=3" style first
        if isinstance(inner, dict) and str(k_str).startswith("k="):
            k = int(str(k_str).split("=",1)[1])
            for m_str, col in inner.items():
                flat[(k, int(m_str))] = int(col)
        elif isinstance(inner, dict):
            # maybe already grouped differently; attempt to parse keys as ints
            try:
                k = int(k_str)
                for m_str, col in inner.items():
                    flat[(k, int(m_str))] = int(col)
            except Exception:
                # final fallback: assume top-level keys are "(k,m)" strings
                for k2m, col in manifest.items():
                    if isinstance(k2m, str) and k2m.strip().startswith("("):
                        k2, m2 = k2m.strip("() ").split(",")
                        flat[(int(k2), int(m2))] = int(col)
                break
    return flat


def _infer_manifest_from_csv(df: pd.DataFrame) -> Dict[Tuple[int,int], int]:
    uniq = df[["k","motif_id"]].drop_duplicates().sort_values(["k","motif_id"]).values.tolist()
    mapping = {(int(k), int(m)): i for i, (k, m) in enumerate(uniq)}
    # write in {"k=K": {"motif": col}} form for readability
    grouped: Dict[str, Dict[str, int]] = {}
    for (k, m), col in mapping.items():
        grouped.setdefault(f"k={k}", {})[str(m)] = col
    return grouped


def build_or_load_node_motif_X(dataset: str, num_nodes: int, precompute_dir: str | Path) -> MotifArtifacts:
    root = Path(precompute_dir)
    _ensure_dir(root)
    csv_p = root / "node_motifs.csv"
    cache_p = root / "motif_x.pt"
    manifest_p = root / "manifest.json"
    stats_p = root / "stats.json"


    if not csv_p.exists():
        # No motifs present; return graceful None
        return MotifArtifacts(X=None, stats={}, manifest={})

    # Try cache first
    if cache_p.exists() and manifest_p.exists() and stats_p.exists():
        X = torch.load(cache_p)
        stats = _read_json(stats_p) or {}
        manifest = _read_json(manifest_p) or {}
        return MotifArtifacts(X=X, stats=stats, manifest=manifest)


    df = pd.read_csv(csv_p)
    required_cols = {"node_id","motif_id","k","count"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"node_motifs.csv missing columns: {missing}")

    manifest = _read_json(manifest_p)
    if manifest is None or len(manifest) == 0:
        manifest = _infer_manifest_from_csv(df)
        _write_json(manifest_p, manifest)
    flat = _flatten_manifest(manifest)
    M = 1 + max(flat.values()) if flat else 0
    if M == 0:
        # No motifs? create dummy zero matrix
        X = torch.zeros((num_nodes, 0), dtype=torch.float32)
        stats = {"log1p_mean": [], "log1p_std": []}
        torch.save(X, cache_p)
        _write_json(stats_p, stats)
        return MotifArtifacts(X=X, stats=stats, manifest=manifest)


    # Build sparse COO
    rows, cols, vals = [], [], []
    for _, r in df.iterrows():
        node = int(r["node_id"]) ; k = int(r["k"]) ; motif = int(r["motif_id"]) ; c = float(r["count"])
        col = flat.get((k, motif))
        if col is None:
            # unseen (k,motif) -> extend mapping
            col = M
            flat[(k, motif)] = col
            M += 1
        rows.append(node)
        cols.append(col)
        vals.append(c)


    X = torch.zeros((num_nodes, M), dtype=torch.float32)
    if len(rows) > 0:
        X.index_put_((torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)),
            torch.tensor(vals, dtype=torch.float32), accumulate=True)


    # Normalize
    stats = _read_json(stats_p)
    X, stats = _log1p_zscore_dense(X, stats)


    # Save cache
    torch.save(X, cache_p)
    _write_json(stats_p, stats)
    # Also update manifest to reflect any new columns seen
    # rewrite in grouped style
    grouped: Dict[str, Dict[str, int]] = {}
    for (k, m), col in flat.items():
        grouped.setdefault(f"k={k}", {})[str(m)] = col
    _write_json(manifest_p, grouped)


    return MotifArtifacts(X=X, stats=stats, manifest=grouped)


def build_or_load_tu_motif_list(dataset: str, pyg_dataset, precompute_dir: str | Path) -> MotifArtifacts:
    """Build a list of per-graph motif matrices aligned to TUDataset order.
    Expects CSV with columns: graph_id,node_id,motif_id,k,count
    """
    root = Path(precompute_dir)
    _ensure_dir(root)
    csv_p = root / "node_motifs.csv"
    cache_p = root / "motif_list.pt"
    manifest_p = root / "manifest.json"
    stats_p = root / "stats.json"



    if not csv_p.exists():
        return MotifArtifacts(X=None, stats={}, manifest={}, X_list=None)


    if cache_p.exists() and manifest_p.exists() and stats_p.exists():
        lst = torch.load(cache_p)
        stats = _read_json(stats_p) or {}
        manifest = _read_json(manifest_p) or {}
        return MotifArtifacts(X=None, stats=stats, manifest=manifest, X_list=lst)
    
    df = pd.read_csv(csv_p)
    required_cols = {"graph_id","node_id","motif_id","k","count"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"node_motifs.csv missing columns: {missing}")


    manifest = _read_json(manifest_p)
    if manifest is None or len(manifest) == 0:
        manifest = _infer_manifest_from_csv(df)
        _write_json(manifest_p, manifest)
    flat = _flatten_manifest(manifest)
    M = 1 + max(flat.values()) if flat else 0


    # Determine per-graph node counts from dataset
    n_graphs = len(pyg_dataset)
    # Build grouped frames per graph
    lst: List[torch.Tensor] = []
    stats = None
    for gid in range(n_graphs):
        gdf = df[df["graph_id"] == gid]
        if len(gdf) == 0:
            # no motifs for this graph
            # infer node count from dataset object
            num_nodes = int(pyg_dataset[gid].num_nodes)
            Xg = torch.zeros((num_nodes, M), dtype=torch.float32)
        else:
            num_nodes = int(pyg_dataset[gid].num_nodes)
            Xg = torch.zeros((num_nodes, M), dtype=torch.float32)
            for _, r in gdf.iterrows():
                node = int(r["node_id"]) ; k = int(r["k"]) ; motif = int(r["motif_id"]) ; c = float(r["count"])
                col = flat.get((k, motif))
                if col is None:
                    col = M
                    flat[(k, motif)] = col
                    M += 1
                    # need to grow all previous Xg tensors if new columns appear late
                    lst = [torch.nn.functional.pad(Xprev, (0,1)) for Xprev in lst]
                    Xg = torch.nn.functional.pad(Xg, (0,1))
                if 0 <= node < num_nodes:
                    Xg[node, col] += c
        lst.append(Xg)
        
    # Stack stats over concatenated graphs for normalization
    if len(lst) > 0:
        big = torch.cat(lst, dim=0)
    else:
        big = torch.zeros((0, M), dtype=torch.float32)
    big, stats = _log1p_zscore_dense(big, _read_json(stats_p))


    # split back to list
    out: List[torch.Tensor] = []
    start = 0
    for Xg in lst:
        n = Xg.size(0)
        out.append(big[start:start+n])
        start += n


    torch.save(out, cache_p)
    _write_json(stats_p, stats)
    grouped: Dict[str, Dict[str, int]] = {}
    for (k, m), col in flat.items():
        grouped.setdefault(f"k={k}", {})[str(m)] = col
    _write_json(manifest_p, grouped)


    return MotifArtifacts(X=None, stats=stats, manifest=grouped, X_list=out)