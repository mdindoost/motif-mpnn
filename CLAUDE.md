# CLAUDE.md — Motif-MPNN Reference

This file is the authoritative reference for contributors and Claude. Update it whenever a design decision changes, a bug is fixed, or new results are recorded.

---

## 1. Research Context

This codebase tests whether subgraph motif statistics (degree, wedge, triangle counts) improve GNN message passing on standard node and graph classification benchmarks. The repo is an evaluation harness; the long-term research contribution is integrating **HiPerMotif** — a parallel edge-centric subgraph isomorphism engine in the open-source **Arachne** framework (Chapel/Arkouda) — as an *exact substructure-feature extractor* for expressive GNNs, and showing its features improve GNN performance at scale. Until HiPerMotif is connected, motifs are computed locally via NetworKit or igraph.

**Research framing (HPEC 2026 proposal `hipermotif-gnn-hpec2026.tex`):** message-passing GNNs are bounded by the 1-WL test and provably cannot count triangles, cycles, or cliques. Injecting *exact* substructure counts removes this ceiling (cf. GSN, Bouritsas 2022). The thesis is that HiPerMotif makes exact counting feasible on large hosts (up to ~10^8 edges), so expressive substructure-aware GNNs can be demonstrated at a scale prior work has not reached. The eventual feature set is **per-vertex orbit features**: raw isomorphism counts from HiPerMotif normalized by each pattern's automorphism group `|Aut(H)|` (and orbit stabilizer per vertex). The current repo's `concat` variant (motif_x concatenated to node features) is exactly the "Use in a GNN" step of that pipeline, restricted to a 3-feature library (degree, wedge, triangle).

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
    README.md           — expected output format for HiPerMotif
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
| `gin` | `GIN` | Graph Isomorphism Network; maximally-1-WL-expressive MPNN, sum readout; tight ceiling baseline | — |
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
| `csl`     | graph | synthetic (CSL) | `data/precompute/csl/node_motifs.csv` |

TU datasets (proteins/nci1/enzymes) use **stratified 60/20/20** (train/val/test) splits, seeded per training run (split_seed = train.seed). Planetoid datasets use canonical public splits (`use_public_split: true`). **Any random split in this repo is always 60/20/20 stratified seeded.** TU downloads may be blocked by network policies; use `scripts/preprocess/download_datasets.py` to pre-download if needed.

CSL is a synthetic expressivity benchmark (10 classes, 150 graphs, 60/20/20 stratified). Shrikhande/rook live as a test fixture in `tests/test_srg_distinguishability.py`, not in the registry.

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
| 1 | 0 | degree (number of undirected neighbors) |
| 3 | 2 | wedge (node is endpoint of open path of length 2) |
| 3 | 3 | triangle (node participates in a 3-clique) |

motif_id values 2 and 3 match igraph's `motifs_undirected()` isomorphism class numbering for 3-node connected undirected subgraphs. **This convention is finalized — all datasets use undirected igraph IDs.** Do not mix with directed motif IDs.

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

Results are written per-run to `results/logs/<timestamp>_<run_name>/run_result.json` and appended to `results/all_runs.csv`. Full findings and the LaTeX summary table are in `results/findings.md` and `results/tables/summary_table.tex`.

**Run inventory (2026-06-07):** 105 unique runs — 87 main (6 datasets × up to 7 variants × 3 seeds) + 18 rand_concat ablation (6 datasets × 3 seeds). Seeds: 42, 0, 1. Planetoid: public splits. TU: stratified 60/20/20 (seed=train_seed). motif_dim=3 for all motif-aware variants.

### Summary: Test Accuracy (mean ± std, N=3 seeds)

| Dataset  | Task  | GCN           | SAGE          | GAT           | Concat        | Gate          | Mix           | Rand-Concat   |
|----------|-------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Cora     | node  | 0.785 ± 0.004 | 0.779 ± 0.014 | 0.800 ± 0.007 | 0.778 ± 0.018 | **0.797** ± 0.013 | 0.777 ± 0.002 | 0.785 ± 0.003 |
| Citeseer | node  | 0.664 ± 0.006 | 0.659 ± 0.022 | 0.679 ± 0.008 | 0.663 ± 0.011 | 0.648 ± 0.033 | **0.675** ± 0.008 | 0.671 ± 0.005 |
| Pubmed   | node  | 0.753 ± 0.012 | 0.739 ± 0.008 | —             | **0.760** ± 0.001 | 0.746 ± 0.005 | 0.744 ± 0.009 | 0.727 ± 0.006 |
| PROTEINS | graph | 0.695 ± 0.022 | —             | —             | **0.728** ± 0.023 | 0.713 ± 0.017 | 0.706 ± 0.010 | 0.653 ± 0.005 |
| NCI1     | graph | 0.674 ± 0.009 | —             | —             | **0.731** ± 0.018 | 0.692 ± 0.011 | 0.691 ± 0.003 | 0.665 ± 0.010 |
| ENZYMES  | graph | 0.264 ± 0.043 | —             | —             | **0.311** ± 0.027 | 0.308 ± 0.022 | 0.286 ± 0.039 | 0.272 ± 0.067 |

Bold = best motif-aware result per dataset.

### Key findings

1. **Graph classification**: All three motif variants improve over GCN on all TU datasets. Largest gains: NCI1 concat +5.8%, ENZYMES concat +4.7%, PROTEINS concat +3.3%.
2. **Ablation validates graph-task signal**: Rand-Concat is *worse* than GCN on PROTEINS (−4.2%) and NCI1 (−0.9%), confirming gains come from motif structure, not extra feature dimensions.
3. **Node classification is mixed**: Modest gains exist (Cora gate +1.2%, Citeseer mix +1.2%, Pubmed concat +0.8%), but Cora/Citeseer Rand-Concat matches or exceeds Concat — capacity confound not ruled out. Pubmed concat is clean (real > random by 3.3%).
4. **Concat is the most consistent variant**: Best or tied-best across 4/6 datasets, lowest variance on Pubmed (std=0.001).
5. **Citeseer gate is unstable**: std=0.033 (above 0.03 node-task threshold) — exclude from main table.

**Historical results** (pre-rewrite, seeded splits, seed=42, 2026-06-06): Cora GCN: 0.756/0.743. PROTEINS GCN: 0.741/0.705. These used directed motif IDs and are not comparable to current results.

### Expressivity demos (2026-06-09)

All results are on branch `expressivity-demo`. Analytical results are proven deterministically (no training noise) in the test suite; empirical results are single runs (seed=42).

**CSL (10-class, 150 graphs, constant all-ones node features):**

| variant | test_acc | notes |
|---------|----------|-------|
| `gcn`   | 0.1000   | chance (10-way); hits 1-WL ceiling as predicted |
| `gin`   | 0.1000   | chance; even the maximally-1-WL-expressive MPNN cannot separate the classes |
| `concat`| 1.0000   | exact cycle/substructure features (motif_dim=9: degree, wedge, triangle, simple-cycles by length) break the ceiling; converged ~epoch 6 |

CSL contains NO 4-cliques (4-regular); the cycle-length spectrum is what separates the 10 classes. Confirmed analytically in `tests/test_csl_separability.py`: full cycle spectrum (L_MAX=8) separates all 10 classes; cycles ≤ length 4 do not.

**Shrikhande vs. 4×4-rook (test fixture, not a registered dataset):**

Both graphs are strongly regular (16, 6, 2, 2) with identical 1-WL signatures and identical triangle counts (32 each). Their 4-clique counts differ: rook = 8, Shrikhande = 0 — a single exact substructure count separates a cospectral pair. Proven in `tests/test_srg_distinguishability.py`.

**D2 finding:** Aut(H) normalization is a no-op under per-column z-score for single-orbit features. Proven in `tests/test_aut_normalization_noop.py`.

---

## 10. Known Bugs Fixed

All bugs below were identified and fixed during the 2026-06-06/07 audit.

| # | Bug | Impact if unfixed | Fix location |
|---|-----|-------------------|--------------|
| 1 | `backward()` called outside `for batch in train_loader` loop | Only the last batch's gradients were used; all earlier batches discarded — training was effectively broken for graph tasks | `engine.py:train_graph_task` |
| 2 | `EarlyStopper` always used `patience=50` hardcoded | Config `train.patience` was silently ignored; no way to tune stopping | `engine.py:EarlyStopper` |
| 3 | Concat model silently ignored `motif_x` for graph task | Motif features were never used for graph classification even when CSV existed | `models/concat.py` |
| 4 | `torch.load()` missing `weights_only=False` | FutureWarning became an error in PyTorch 2.x; TU motif cache never loaded | `datasets/motif_loader.py` |
| 5 | Motif fallback was silent (no warning) | User couldn't tell if motif CSV was missing; model silently ran as GCN | `models/concat.py`, `gate.py`, `mix.py` |
| 6 | `monitor` metric not configurable | val_acc was always used as EarlyStopper criterion; `val_macro_f1` option had no effect | `engine.py`, `config.py` |
| 7 | Planetoid had no public-split support | Couldn't reproduce Kipf & Welling (2017) canonical splits; comparisons to published results were invalid | `datasets/planetoid.py` |
| 8 | `defaults.yml` indentation broke flat-style config loading | `model.name: identity` from defaults shadowed `variant: gcn` from experiment files after indentation fix | `utils/config.py:load_config` |
| 9 | `model_kwargs` used hardcoded values (hidden=64, layers=2, dropout=0.5) | Config changes to these fields had no effect | `train/run.py` |
| 10 | `batch_size` invalid value silently defaulted to 64 | Misconfigured batch size was hidden; no way to detect the problem | `engine.py:train_graph_task` |
| 11 | Flat-style config loader defaulted `ds_task="node"` for all datasets | All TU (graph-task) experiment configs failed `validate_config()` with task mismatch; PROTEINS/NCI1/ENZYMES were completely broken | `utils/config.py:load_config` |
| 12 | igraph `g.triangles()` API missing in igraph 1.x | `generate_motifs.py` crashed at import for all datasets | `scripts/preprocess/generate_motifs.py` |
| 13 | Existing cora/citeseer motif CSVs used directed motif IDs (2,9,11) | Features from old and new CSVs were incomparable; mixing would silently corrupt experiments | Old CSVs deleted; all 6 datasets regenerated with undirected IDs |

---

## 11. Known Limitations / Open Questions

**Hardlines (do not change without strong reason):**
- EarlyStopper patience must come from config — never hardcode it in `engine.py`.
- Graph classification backward pass belongs inside the `for batch in train_loader` loop — not outside.
- Do not add motif CSVs or `.pt` caches to git — `data/` is gitignored.
- Do not compare results across different `split_seed` values without noting it explicitly.

**Open questions:**
1. **~~Directed vs undirected motif IDs~~**: RESOLVED (2026-06-07). Standardized on undirected igraph IDs (0=degree, 2=wedge, 3=triangle). All CSVs regenerated. Old directed CSVs deleted.
2. **Public vs seeded splits for the main paper table**: use `use_public_split: true` for direct comparability with published baselines (Kipf & Welling 2017); use seeded splits for multi-seed ablations. Do not mix.
3. **EarlyStopper metric for imbalanced TU datasets**: NCI1 and ENZYMES may benefit from `monitor: val_macro_f1`. Test by adding this to those experiment YAMLs.
4. **Mix topk correctness**: the vectorized scatter-topk replaces an O(N) loop. Verify on a small example before large-scale runs.
5. **No multi-GPU support** — training is single-device (CPU or one GPU).

**Not done yet:**
- ~~No multi-seed sweep results~~: DONE (2026-06-07) — 105 unique runs across seeds {42, 0, 1}.
- GAT configs exist only for Cora and Citeseer; Pubmed/TU GAT configs missing.
- SAGE configs exist only for Planetoid datasets; TU SAGE configs missing.
- ~~Motif-Mix for graph tasks not yet tested~~: DONE — all TU (mix, gate, concat) runs complete.
- `sweep.py` does not support `--group paper-core` yet.
- Social datasets (IMDB-B, IMDB-M, REDDIT-B) not registered in DATASET_REGISTRY.
- Large-scale dataset (ogbn-arxiv) not registered in DATASET_REGISTRY.
- HiPerMotif not connected (local igraph/NetworKit stand-in for now).
- Orbit-feature extraction (automorphism normalization, richer pattern library beyond degree/wedge/triangle) not implemented — see Section 13.
- Expressivity benchmarks from the proposal (CSL, Shrikhande vs. 4x4 rook) not registered in DATASET_REGISTRY.

---

## 12. Git Rules

- `git add` specific files only — never `git add -A` or `git add .` (risk of committing `data/` files).
- `data/` is gitignored; motif CSVs and `.pt` caches must never be committed.
- Claude performs `git add` on specific files when asked, but **never commits or pushes**. The user does both. Claude provides the push command as text.

---

## 13. HiPerMotif Integration Plan

### Backend swap (drop-in, preserves all downstream code)

When HiPerMotif (Arachne / Chapel / Arkouda parallel subgraph isomorphism engine) is ready, replace only the counting logic in `scripts/preprocess/generate_motifs.py` — specifically the `_count_motifs_networkit` and `_count_motifs_igraph_single` functions — with a call to the HiPerMotif interface. The output CSV format (Planetoid: `node_id,k,motif_id,count`; TU: `graph_id,node_id,k,motif_id,count`) must remain identical so that `motif_loader.py` and all model code require zero changes. After swapping the backend, delete the cached `.pt` files under `data/precompute/` so the loader rebuilds from the new CSV.

### HiPerMotif API (per the HPEC 2026 proposal)

Invoke via the Arachne property-graph interface:
- `return_isos_as="count"` → raw number of isomorphisms → divide by `|Aut(H)|` for graph-level motif counts.
- `return_isos_as="vertices"` → flattened mappings (length `n·k` for `k` embeddings of an `n`-vertex pattern) + a mapper identifying each pattern vertex → tally host vertices by pattern position, collapse positions within an orbit → per-vertex orbit counts.
- `algorithm_type="si"` selects the edge-centric HiPerMotif search; `reorder_type="structural"` selects structural pattern reordering.

### Orbit-feature normalization (the only nontrivial math)

For each pattern `H`: precompute `Aut(H)` and the partition of `V(H)` into automorphism orbits `O_1..O_r`. Each embedded copy is reported `|Aut(H)|` times, so true copies = raw count / `|Aut(H)|`. For per-vertex features, divide an orbit's count by the stabilizer size `b_j = |Aut(H)| / |O_j|` so each copy contributes once per participating vertex. Final feature is `log(1+X)` (heavy-tailed) — consistent with the existing `log1p`+z-score normalization in `motif_loader.py`. See Algorithm 1 in the proposal.

### Direction this enables (beyond the current 3-feature library)

- Richer pattern library: triangle, wedge, 4-path, 4-cycle, claw, paw, diamond, 4-clique, selected 5-vertex patterns — each as one or more orbit columns. This requires extending the `(k, motif_id)` manifest scheme in `motif_loader.py` and `generate_motifs.py`.
- Expressivity demonstrations as datasets/tests: CSL (10-way, 1-WL caps at 10%) and the Shrikhande vs. 4x4-rook pair (4-clique count = 8 vs. 0 separates a cospectral pair) — useful as unit tests that exact features add signal a plain GNN cannot represent.
- At-scale story: report extraction time / parallel scaling and the per-motif compute-vs-accuracy frontier on large hosts (the proposal's open `\todo` results).

**Naming note:** the engine is **HiPerMotif** (Dindoost et al., HPEC 2025, arXiv:2507.04130). The repo was reconciled from the old "HiPerXplorer" name on 2026-06-09 — do not reintroduce it.
