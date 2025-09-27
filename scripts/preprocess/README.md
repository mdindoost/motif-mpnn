# Precompute with HiPerXplorer — Expected Outputs

For each graph (e.g., `cora`), create a folder:
```
data/precompute/cora/
  motif_x.pt       # Float tensor [N,K], log1p + zscore (train fold stats)
  motif_edge.pt    # (optional) Float tensor [E,K']
  A_motif.pt       # (optional) Sparse COO/CSR adjacency
  manifest.json    # {{ "motif_id_decimal": column_index, ... }}
  stats.json       # mean/std per column, pruning thresholds
```

### Notes
- Keep K small initially (k=3 motifs; curated k=4 set). Prune to top‑k per node if needed.
- Use **log1p** then **z‑score** (compute mean/std on the training fold only; reuse for val/test).
- Record the exact motif whitelist and role policy in `manifest.json`.
