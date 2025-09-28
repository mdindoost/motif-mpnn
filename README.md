# Motif-Augmented MPNN (M-MPNN)

Precompute motif signals (from HiPerXplorer) and inject them into message passing
via **feature concatenation**, **motif-gated messages**, or **Laplacian mixing**.
This repo aims to be *simple to run* and *reproducible* without code bloat.

## Quickstart
1. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate motif-mpnn
   ```
2. Prepare data (download Cora/Citeseer/Pubmed or PROTEINS/NCI1/ENZYMES).
### Get datasets
```bash
python scripts/preprocess/download_datasets.py

3. Precompute motifs with HiPerXplorer and place outputs under `data/precompute/<graph>/`.
   See **`scripts/preprocess/README.md`** for file expectations.
4. Run a curated experiment config:
   ```bash
   python -m src.train.run --config configs/experiments/cora_concat.yml
   ```

> Note: Early versions ship **no heavyweight frameworks**, plain CSV logging, and curated configs (≤10).

## Repository layout
```
configs/          # dataset & experiment configs (YAML)
data/             # raw/processed graphs; precompute outputs (not in git)
docs/             # paper outline, related work bullets
results/          # logs, figures, tables (generated)
scripts/          # preprocessing and launch wrappers
src/              # lightweight modules (datasets, models, train, utils)
```

## Precompute contract (HiPerXplorer → tensors)
Per graph under `data/precompute/<graph>/`:
- `motif_x.pt` : Float tensor [N,K] (log1p + train-fold z-score), **plus** manifest entry.
- `motif_edge.pt` *(optional)* : Float tensor [E,K′].
- `A_motif.pt` *(optional)* : Sparse adjacency (COO/CSR).
- `manifest.json` : maps decimal motif IDs → column indices.
- `stats.json` : per-column mean/std, pruning thresholds.

## Reproducibility
- Deterministic seeds for splits and training.
- Configs saved with results.
- Environment file pinned to known-working versions.

## License & citation
- Licensed under MIT (see LICENSE).
- Cite using `CITATION.cff`.
