# CLAUDE.md — Motif-MPNN Reference

This file is the authoritative reference for contributors and Claude. Update it whenever a design decision changes, a bug is fixed, or new results are recorded.

---

## 1. Research Context

This codebase tests whether subgraph motif statistics (degree, wedge, triangle counts) improve GNN message passing on standard node and graph classification benchmarks. The repo is an evaluation harness; the long-term research contribution is integrating **HiPerXplorer** — a high-performance parallel motif counter written in Chapel and deployed via Arkouda — and showing its features improve GNN performance. Until HiPerXplorer is connected, motifs are computed locally via NetworKit or igraph.

---

## 2. Quick Start

All commands assume the conda environment `motif-mpnn`.

```bash
# Run one experiment
conda run -n motif-mpnn python -m src.train.run --config configs/experiments/cora_concat.yml

# Dry-run: validate and print resolved config without training
conda run -n motif-mpnn python -m src.train.run --config configs/experiments/cora_gcn.yml --dry-run

# Generate motif features for a dataset (igraph backend, recommended)
conda run -n motif-mpnn python scripts/preprocess/generate_motifs.py --dataset cora --tool igraph --verify

# Multi-seed sweep over multiple configs
conda run -n motif-mpnn python scripts/runs/sweep.py \
    --config configs/experiments/cora_gcn.yml \
    --config configs/experiments/cora_concat.yml \
    --seeds 42 123 456 \
    --out results/tables/sweep.tex
```

---

## 3. Architecture Overview

```
YAML config(s)
    │
    ▼
load_config()          ← deep-merges defaults.yml + experiment YAML
    │
    ▼
validate_config()      ← checks variant, dataset, task, patience, monitor
    │
    ├──► DATASET_REGISTRY.get(name) → dataset bundle (data + splits + optional motif_x)
    │
    ├──► MODEL_REGISTRY.get(name)   → model instance (GCN / motif variant)
    │
    ▼
train_node_task()      ← full-batch, node mask loop
  or
train_graph_task()     ← mini-batch DataLoader loop (backward inside per-batch)
    │
    ▼
results/logs/<timestamp>_<run_name>/
    manifest.json      ← config snapshot + motif metadata
    metrics.csv        ← per-epoch train_loss, val_acc, val_macro_f1, test_acc, test_macro_f1
    run_result.json    ← final test_acc, test_macro_f1, best_val_epoch
results/all_runs.csv   ← one row appended per completed run
```

**Key design choices:**

- **Registry pattern** — `MODEL_REGISTRY` and `DATASET_REGISTRY` in `src/utils/registry.py`. All models/datasets self-register with `@MODEL_REGISTRY.register("key")`. Never hardcode lookups outside the registry.
- **YAML-driven config** — all hyperparameters live in `configs/experiments/`. Do not hardcode hyperparameters in model or training code.
- **Flat vs nested YAML** — two config styles are supported; see Section 5.
- **TUWithMotifs wrapper** — `src/datasets/tu_wrapper.py` attaches per-graph `motif_x` tensors to `Data` objects at load time so the DataLoader batching works transparently.
- **Graceful degradation** — all motif-aware models (concat, gate, mix) fall back to plain GCN when no motif CSV is present, emitting a `warnings.warn()`. This lets any config run on any machine without precomputed motifs.

---

## 4. File Map

```
src/
  train/
    engine.py        — EarlyStopper; train_node_task (full-batch); train_graph_task (mini-batch)
    run.py           — entry point: loads config, builds dataset+model, calls engine, writes results
    metrics.py       — accuracy() and macro_f1() (no sklearn dependency)
  models/
    gcn.py           — GCN baseline (node + graph tasks)
    sage.py          — GraphSAGE baseline
    gat.py           — GAT baseline
    concat.py        — Motif-Concat: concatenates motif_x to node features before GCN
    gate.py          — Motif-Gate: EdgeGate modulates GCN message weights from motif_x
    mix.py           — Motif-Mix: blends structural adjacency with motif-similarity adjacency
    common.py        — LayerNorm1d, ResidualBlock, MLPHead shared building blocks
    __init__.py      — imports all models to populate MODEL_REGISTRY; defines IdentityFactory
  datasets/
    planetoid.py     — Cora/Citeseer/Pubmed loaders; seeded + public split support
    tu.py            — PROTEINS/NCI1/ENZYMES loaders with stratified splits
    tu_wrapper.py    — TUWithMotifs: attaches per-graph motif_x tensors to Data objects
    motif_loader.py  — CSV → tensor pipeline (log1p+zscore, caching, manifest tracking)
    __init__.py      — imports all datasets to populate DATASET_REGISTRY
  utils/
    config.py        — load_config, validate_config, ExperimentConfig dataclasses
    registry.py      — MODEL_REGISTRY and DATASET_REGISTRY
    seed.py          — fix_seed(): sets Python, NumPy, PyTorch, CUDA random states

configs/
  defaults.yml                     — base hyperparameters merged into every experiment
  datasets.yml                     — dataset metadata (informational only)
  experiments/<ds>_<model>.yml     — one file per run; overrides defaults
    cora_{gcn,concat,gate,mix,sage,gat}.yml
    citeseer_{gcn,concat,gate,mix,sage}.yml
    pubmed_{gcn,concat,gate,mix,sage}.yml
    proteins_{gcn,concat,gate,mix}.yml
    nci1_{gcn,concat,gate,mix}.yml
    enzymes_{gcn,concat,gate,mix}.yml

scripts/
  preprocess/
    generate_motifs.py  — compute per-node motif CSVs (NetworKit or igraph backend)
    verify_motifs.py    — sanity-check a motif CSV; optional --plot for histograms
    export_edges.py     — export PyG graph to edge list for external tools
    download_datasets.py — pre-download TU datasets
    README.md           — expected output format for HiPerXplorer
  runs/
    sweep.py            — multi-seed sweep; writes per-run rows + LaTeX summary table

data/                              (gitignored entirely)
  precompute/<dataset>/
    node_motifs.csv     — motif counts (Planetoid: node_id,k,motif_id,count; TU: +graph_id)
    motif_x.pt          — cached dense tensor [N, M] (node tasks)
    motif_list.pt       — cached list of per-graph tensors (graph tasks)
    manifest.json       — (k, motif_id) → column index
    stats.json          — log1p mean/std per column (for consistent normalization)
  processed/            — PyG raw + processed dataset files
  raw_export/           — edge/node lists for external motif tools

results/
  logs/<timestamp>_<run_name>/
    manifest.json       — config snapshot + motif metadata
    metrics.csv         — per-epoch metrics
    run_result.json     — final test results
  all_runs.csv          — cumulative log of all completed runs
  tables/               — summary CSVs and LaTeX tables from sweep.py

docs/
  paper_outline.md      — living paper structure
  related_work.md       — related work bullet map
```

---

## 5. Config System

### Two YAML styles

**Flat style** (used by all existing experiment configs):
```yaml
dataset: cora
variant: concat
run_name: cora_concat
train:
  epochs: 200
  patience: 50
  seed: 42
optim:
  lr: 0.01
  weight_decay: 0.0
```

**Nested style** (alternative; has `dataset: {name: ...}`):
```yaml
dataset:
  name: cora
  task: node
model:
  name: concat
  hidden_dim: 64
```

`load_config()` detects the style from the merged YAML. In flat style, model selection priority is: `variant:` key > explicit `model.name` in the experiment file > `identity` fallback. The raw experiment YAML (before merging defaults) is checked first to avoid picking up the `identity` placeholder from `defaults.yml`.

### Dataclass fields and defaults

| Section | Field | Default | Notes |
|---------|-------|---------|-------|
| `DatasetConfig` | `name` | `"dummy_node"` | dataset key in DATASET_REGISTRY |
| | `task` | `"node"` | `"node"` or `"graph"`; inferred from name if absent |
| | `root` | `"data/processed"` | PyG download root |
| | `use_public_split` | `False` | set `True` for canonical Planetoid splits |
| `ModelConfig` | `name` | `"identity"` | model key in MODEL_REGISTRY |
| | `hidden_dim` | `64` | hidden layer width |
| | `num_layers` | `2` | number of GNN layers |
| | `dropout` | `0.5` | dropout rate |
| | `layer_norm` | `True` | enable LayerNorm1d after each layer |
| | `residual` | `True` | enable residual connections |
| `TrainConfig` | `epochs` | `200` | max training epochs |
| | `patience` | `50` | EarlyStopper patience (must come from config, never hardcoded) |
| | `seed` | `0` | random seed |
| | `batch_size` | `0` | 0 = full-batch for node tasks; defaults to 64 for graph tasks |
| | `monitor` | `"val_acc"` | `"val_acc"` or `"val_macro_f1"` |
| `OptimConfig` | `lr` | `0.01` | Adam learning rate |
| | `weight_decay` | `0.0` | L2 regularization |

### validate_config() rules

Raises `ValueError` (hard errors):
- Unknown variant/model name (not in `KNOWN_VARIANTS`)
- Task/dataset mismatch (e.g., `task: node` for `proteins`)
- `train.patience <= 0`
- `train.monitor` not in `{"val_acc", "val_macro_f1"}`

Emits `warnings.warn()` (soft warnings):
- Unknown dataset name (not in `KNOWN_DATASETS`)

### _safe_dataclass() behavior

When constructing any dataclass from a YAML dict, unknown keys emit a `UserWarning` and are silently dropped rather than raising a `TypeError`. This lets experiment YAMLs contain model-specific knobs (e.g., `gate:` or `mix:` sections) without breaking config loading.

### Key flag: use_public_split

`use_public_split: false` (default) — uses seeded Planetoid splits. These are reproducible but NOT the canonical splits from Kipf & Welling (2017). Set `use_public_split: true` in the dataset config for direct comparability with published baselines. Never mix results from the two split modes.

---

## 6. Models / Variants

| variant | class | description | extra kwargs |
|---------|-------|-------------|--------------|
| `gcn` | `GCN` | Standard 2-layer GCN backbone; supports both node and graph tasks | — |
| `sage` | `GraphSAGE` | GraphSAGE with mean aggregation | — |
| `gat` | `GAT` | Graph Attention Network | — |
| `concat` | `ConcatModel` | Concatenates `motif_x` to node features before GCN encoder; falls back to GCN if no motif CSV | `motif_dim` |
| `gate` | `GateModel` | Learns edge-wise gates from motif context to modulate GCN messages | `motif_dim`; `gate:` YAML section |
| `mix` | `MixModel` | Blends structural adjacency with motif-similarity adjacency (vectorized scatter-topk) | `motif_dim`; `mix:` YAML section (`lambda_mix`, `motif_topk`, `sim_metric`, `self_loop`) |
| `identity` | `IdentityFactory` | No-op stub; used as a fallback placeholder | — |

All motif-aware variants (concat, gate, mix) accept `motif_dim=0` and behave identically to GCN when no motif features are available.

---

## 7. Datasets

| dataset | task | benchmark | motif CSV location |
|---------|------|-----------|--------------------|
| `cora` | node | Planetoid | `data/precompute/cora/node_motifs.csv` |
| `citeseer` | node | Planetoid | `data/precompute/citeseer/node_motifs.csv` |
| `pubmed` | node | Planetoid | `data/precompute/pubmed/node_motifs.csv` |
| `proteins` | graph | TU | `data/precompute/proteins/node_motifs.csv` |
| `nci1` | graph | TU | `data/precompute/nci1/node_motifs.csv` |
| `enzymes` | graph | TU | `data/precompute/enzymes/node_motifs.csv` |

TU datasets (proteins/nci1/enzymes) use stratified train/val/test splits. TU downloads may be blocked by network policies; use `scripts/preprocess/download_datasets.py` to pre-download if needed.

---

## 8. Motif Pipeline

### Step-by-step

```
generate_motifs.py          (compute counts via NetworKit or igraph)
    │
    ▼
data/precompute/<dataset>/node_motifs.csv
    │
    ▼
motif_loader.py             (build_or_load_node_motif_X or build_or_load_tu_motif_list)
    │  ├── reads CSV, builds manifest.json (if absent)
    │  ├── constructs dense tensor X [N, M] (sparse COO → dense)
    │  ├── applies log1p → z-score normalization
    │  └── caches to motif_x.pt / motif_list.pt + stats.json
    │
    ▼
dataset bundle (dataset.motif_x for node tasks; TUWithMotifs for graph tasks)
    │
    ▼
run.py attaches motif_x to data.motif_x (node) or wraps dataset (graph)
    │
    ▼
model.forward(data)         reads data.motif_x if present
```

### CSV format

Planetoid (node classification):
```
node_id, k, motif_id, count
```

TU (graph classification):
```
graph_id, node_id, k, motif_id, count
```

First line may be a comment `# motif_topk=N` (skipped by `comment="#"` in pandas).

### Motif ID scheme (undirected, igraph convention)

| k | motif_id | meaning |
|---|----------|---------|
| 1 | 0 | degree (number of neighbors) |
| 3 | 2 | wedge (open path of length 2) |
| 3 | 3 | triangle (3-clique) |

### Normalization

`log1p(X)` applied column-wise, then z-score (`mean`, `std`) computed over all nodes. Statistics are saved to `stats.json` so the same normalization is applied at inference. Do NOT switch to raw counts or min-max — motif count distributions are heavy-tailed and would dominate gradients.

### Cache files

| file | content | invalidate when |
|------|---------|-----------------|
| `motif_x.pt` | dense float32 tensor [N, M] | CSV changes |
| `motif_list.pt` | list of per-graph tensors (TU) | CSV changes |
| `manifest.json` | (k, motif_id) → column index | motif scheme changes |
| `stats.json` | log1p mean/std per column | normalization changes |

Delete `.pt` files to force a rebuild from the CSV.

### generate_motifs.py flags

```
--dataset  cora|citeseer|pubmed|proteins|nci1|enzymes|all
--tool     igraph (default) | networkit  (networkit is faster for large Planetoid graphs)
--force    overwrite existing CSV
--verify   print sanity stats after writing
--k/--topk motif_topk written as header comment (default: 10)
--out-dir  output directory override (default: data/precompute/<dataset>/)
```

### verify_motifs.py

```bash
conda run -n motif-mpnn python scripts/preprocess/verify_motifs.py --dataset cora
conda run -n motif-mpnn python scripts/preprocess/verify_motifs.py --dataset cora --plot
```

Prints per-column stats (min, max, mean, std, all-zero node count). `--plot` saves histogram PNGs to `data/precompute/<dataset>/motif_distributions.png`.

---

## 9. Experiment Results

Results are written per-run to `results/logs/<timestamp>_<run_name>/run_result.json` and appended to `results/all_runs.csv`.

| dataset | variant | test_acc | test_f1 | seed | split | notes |
|---------|---------|----------|---------|------|-------|-------|
| | | | | | | |

*(Fill in results here as runs are completed. Always note seed, split type (seeded vs public), and motif_dim.)*

**Historical results** (pre-rewrite, seeded splits, seed=42, hidden=64, 2-layer GCN backbone, 2026-06-06): see `results/tables/planetoid_results_seed42.csv`. Cora GCN: 0.756/0.743. PROTEINS GCN: 0.741/0.705 (after backward-pass bugfix from 0.714).

---

## 10. Known Limitations / Open Questions

**Hardlines (do not change without strong reason):**
- EarlyStopper patience must come from config — never hardcode it in `engine.py`.
- Graph classification backward pass belongs inside the `for batch in train_loader` loop — not outside.
- Do not add motif CSVs or `.pt` caches to git — `data/` is gitignored.
- Do not compare results across different `split_seed` values without noting it explicitly.

**Open questions:**
1. **Directed vs undirected motif IDs**: existing Cora/Citeseer CSVs use directed 3-node motif IDs; `generate_motifs.py` uses undirected igraph IDs. Results from the two schemes are NOT comparable. Commit to one convention before the paper.
2. **Public vs seeded splits for the main paper table**: use `use_public_split: true` for direct comparability with published baselines (Kipf & Welling 2017); use seeded splits for multi-seed ablations. Do not mix.
3. **EarlyStopper metric for imbalanced TU datasets**: NCI1 and ENZYMES may benefit from `monitor: val_macro_f1`. Test by adding this to those experiment YAMLs.
4. **Mix topk correctness**: the vectorized scatter-topk replaces an O(N) loop. Verify on a small example before large-scale runs.
5. **No multi-GPU support** — training is single-device (CPU or one GPU).

**Not done yet:**
- No motif CSVs for any TU dataset (proteins/nci1/enzymes) — run `generate_motifs.py --dataset proteins --tool igraph` to generate.
- No pubmed motif CSV.
- No multi-seed sweep results — only single seed=42 runs exist.
- No experiment configs for GAT.
- Motif-Mix for graph tasks not yet tested with motif CSVs.
- HiPerXplorer not connected.

---

## 11. Git Rules

- `git add` specific files only — never `git add -A` or `git add .` (risk of committing `data/` files).
- `data/` is gitignored; motif CSVs and `.pt` caches must never be committed.
- Claude performs `git add` on specific files when asked, but **never commits or pushes**. The user does both. Claude provides the push command as text.

---

## 12. HiPerXplorer Integration Plan

When HiPerXplorer (Chapel/Arkouda parallel motif counter) is ready, replace only the counting logic in `scripts/preprocess/generate_motifs.py` — specifically the `_count_motifs_networkit` and `_count_motifs_igraph_single` functions — with a single call to the HiPerXplorer CLI or Python API. The output CSV format (Planetoid: `node_id,k,motif_id,count`; TU: `graph_id,node_id,k,motif_id,count`) must remain identical so that `motif_loader.py` and all model code require zero changes. After swapping the backend, delete the cached `.pt` files under `data/precompute/` so the loader rebuilds from the new CSV.
