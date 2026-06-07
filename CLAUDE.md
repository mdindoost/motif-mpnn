# CLAUDE.md — Persistent Memory for Motif-MPNN

This file is the authoritative reference for Claude (and human contributors) about this codebase.
Update it whenever a design decision changes, a bug is fixed, or new results are recorded.

---

## Research Context

**What this is:** An academic research framework testing whether subgraph motif statistics improve
GNN message passing on node and graph classification benchmarks.

**The real contribution:** This repo is the *evaluation harness*. The actual research contribution
is connecting to **HiPerXplorer** — a high-performance parallel motif counter written in Chapel
and deployed via the Arkouda framework. This codebase shows that motif features from HiPerXplorer
improve GNN performance on standard benchmarks.

**Three injection strategies (ablations of how motif information enters the model):**
- **Concat** — Concatenate motif features to node features before message passing.
- **Gate** — Learn edge-wise gates from motif context to modulate messages.
- **Mix** — Blend structural adjacency with motif-derived similarity adjacency.

**Benchmarks:**
- Node classification: Cora, Citeseer, Pubmed (Planetoid)
- Graph classification: PROTEINS, NCI1, ENZYMES (TU datasets)

**Reproducibility requirement:** Results must be comparable to published baselines.
Do not simplify splits or evaluation in ways that break comparability.

---

## Architecture Decisions (do not change without strong reason)

### Motif normalization
The motif loader uses **log1p → z-score** normalization because motif counts are heavy-tailed
(Poisson-like). Do NOT switch to raw counts or min-max normalization — the tails of triangle
distributions span orders of magnitude and would dominate gradient updates.

### Graceful degradation
All motif-aware models (Concat, Gate, Mix) fall back to plain GCN when no motif CSV is present.
This invariant must be preserved — it lets you run any config on any machine without needing
precomputed motifs. The fallback now emits a `warnings.warn()` so users see it clearly.

### Registry pattern
`MODEL_REGISTRY` and `DATASET_REGISTRY` in `src/utils/registry.py` are intentional.
New models and datasets must self-register by decorating with `@MODEL_REGISTRY.register("key")`.
Do not hardcode model/dataset lookups anywhere outside the registry.

### YAML-driven config
All hyperparameters live in YAML configs under `configs/experiments/`. Do not hardcode
hyperparameters in model or train code. The training script reads everything from config.

### Config loading: flat vs nested style
`src/utils/config.py` handles two YAML styles:
- **Nested style** (has `dataset: {name: ...}`): uses dataclasses directly.
- **Flat style** (has `dataset: cora`, `variant: gcn`): most existing experiment configs.

In flat style, `variant:` takes priority over `model.name:` from defaults.yml.
This is critical: fixing defaults.yml indentation in 2025 broke this — the fix is
in `load_config()` which reads the raw experiment cfg to find explicit model.name
before falling back to `variant:`.

### EarlyStopper metric
`monitor: val_acc` is the default (set in `TrainConfig` and `defaults.yml`).
For imbalanced datasets use `monitor: val_macro_f1` in the experiment YAML.
The config key name is `train.monitor`.

---

## Known Limitations / Hardlines

- **Do not add motif CSVs to git** — they belong in `data/` which is gitignored.
- **Do not compare results across different `split_seed` values** without noting it explicitly.
- **EarlyStopper patience must come from config** — never hardcode it in engine.py.
- **Graph classification training must process all batches per epoch** — the backward pass
  belongs inside the `for batch in train_loader` loop.
- **TU dataset downloads may be blocked** by network policies; handle gracefully.
- **`use_public_split: false` by default** — the seeded Planetoid splits are NOT the canonical
  public splits used by Kipf & Welling (2017) and most GNN baselines. Set `use_public_split: true`
  in the dataset config to use canonical splits for direct comparability. Do not mix the two.

---

## File Map

```
src/
  train/
    engine.py        — train_node_task and train_graph_task; EarlyStopper; batch loop
    run.py           — entry point: loads config, builds dataset+model, calls engine
    metrics.py       — accuracy() and macro_f1() (no sklearn dependency)
  models/
    gcn.py           — GCN baseline (node + graph tasks)
    sage.py          — GraphSAGE baseline
    gat.py           — GAT baseline
    concat.py        — Motif-Concat: cat motif_x to node features before GCN
    gate.py          — Motif-Gate: EdgeGate modulates GCN message weights from motif_x
    mix.py           — Motif-Mix: blend A with motif-similarity adjacency
    common.py        — LayerNorm1d, ResidualBlock, MLPHead shared building blocks
    __init__.py      — imports all models to populate MODEL_REGISTRY
  datasets/
    planetoid.py     — Cora/Citeseer/Pubmed loaders; seeded + public split support
    tu.py            — PROTEINS/NCI1/ENZYMES loaders with stratified splits
    tu_wrapper.py    — TUWithMotifs: attaches per-graph motif_x tensors to Data objects
    motif_loader.py  — CSV → tensor pipeline with log1p+zscore, caching, manifest tracking
    __init__.py      — imports all datasets to populate DATASET_REGISTRY
  utils/
    config.py        — YAML config loading; ExperimentConfig, TrainConfig, etc.
    registry.py      — MODEL_REGISTRY and DATASET_REGISTRY
    seed.py          — fix_seed(): sets all random states for reproducibility

configs/
  defaults.yml                    — base hyperparameters merged into every experiment
  datasets.yml                    — dataset metadata (informational)
  experiments/<ds>_<model>.yml    — one file per run; overrides defaults

scripts/
  preprocess/
    generate_motifs.py  — compute per-node motif CSVs (NetworKit or igraph backend)
    export_edges.py     — export PyG graph to edge list for external tools
    download_datasets.py — pre-download TU datasets
    README.md           — expected output format for HiPerXplorer

data/
  precompute/<dataset>/   — motif CSVs + cache (gitignored; generate with generate_motifs.py)
    node_motifs.csv       — node_id, k, motif_id, count (Planetoid) or + graph_id (TU)
    motif_x.pt            — cached dense tensor [N, M]
    manifest.json         — (k, motif_id) → column index
    stats.json            — log1p mean/std per column
  processed/              — PyG raw + processed dataset files (gitignored)
  raw_export/             — edge/node lists for external motif tools

results/
  logs/<timestamp>_<run>/   — per-run manifest.json + metrics.csv
  tables/                   — summary CSVs across runs

docs/
  paper_outline.md   — living paper structure
  related_work.md    — related work bullet map
```

---

## Experiment Results

### Post-bugfix baseline (2026-06-06, seed=42, seeded splits, hidden=64, 2-layer GCN backbone)

| Dataset  | Model   | test_acc | test_f1 | Notes |
|----------|---------|----------|---------|-------|
| Cora     | GCN     | 0.756    | 0.743   | seeded split |
| Cora     | Concat  | 0.764    | 0.755   | 3 motif features |
| Cora     | Gate    | 0.770    | 0.757   | 3 motif features |
| Cora     | Mix     | 0.762    | 0.749   | 3 motif features |
| Cora     | SAGE    | 0.764    | 0.761   | no motif |
| PROTEINS | GCN     | 0.741    | 0.705   | fixed backward pass (was 0.714) |
| PROTEINS | Concat  | 0.741    | 0.705   | no motif CSV yet — runs as GCN |

**Impact of Bug #1 fix (backward pass):** proteins_gcn improved from 0.714 to 0.741 (+2.7 pp)
after ensuring all batches are trained per epoch (not just the last one).

### Pre-bugfix historical runs (from results/tables/planetoid_results_seed42.csv)
See that file for detailed per-run logs. Note: runs before 2026-06-06 may have:
- Hardcoded patience (EarlyStopper always patience=50 regardless of config)
- PROTEINS runs trained on last batch only per epoch (bug #1)
- Silent motif fallback without user-visible warning

---

## Open Questions

1. **Directed vs undirected motifs for Planetoid**: The existing Cora/Citeseer motif CSVs use
   directed 3-node motif IDs (2=Mutual_Pair, 9=Triangle_2mutual, 11=Fully_Mutual). The new
   `generate_motifs.py` uses undirected igraph motif IDs (2=wedge, 3=triangle). Results using
   the two schemes are NOT comparable. Decide which convention to commit to before the paper.

2. **Public vs seeded splits**: Should the paper results use canonical Planetoid public splits
   (`use_public_split: true`) for direct baseline comparability, or seeded splits for multi-seed
   ablations? Recommend: public splits for the main table, seeded splits for ablations.

3. **EarlyStopper metric for TU datasets**: NCI1 and ENZYMES are imbalanced — `val_macro_f1`
   might be more appropriate as the stopping criterion. Currently defaults to `val_acc`.
   Add `monitor: val_macro_f1` to those experiment YAMLs to test.

4. **Mix topk scaling**: The vectorized scatter-topk replaces an O(N) Python loop.
   Verify correctness on a small example before running at scale.

---

## What's Not Done Yet

- **No motif CSVs for TU datasets** (PROTEINS/NCI1/ENZYMES) — run
  `python scripts/preprocess/generate_motifs.py --dataset proteins --tool igraph` to generate them.
- **No pubmed motif CSV** — generate_motifs.py supports it; run with `--dataset pubmed`.
- **No multi-seed sweep scripts** — results are single seed=42 only.
- **No results table for NCI1 or ENZYMES** — only PROTEINS has been tested.
- **GAT model** exists but has no experiment configs yet.
- **Motif-Mix for graph tasks** — not yet tested with motif CSVs (needs TU motif generation first).
- **HiPerXplorer integration** — the actual Chapel/Arkouda pipeline is not connected.

---

## HiPerXplorer Integration Plan

When HiPerXplorer replaces NetworKit/igraph as the motif source, the only change needed is:

1. **Replace the counting logic in `scripts/preprocess/generate_motifs.py`** (the `_count_motifs_*`
   functions) with a single call to the HiPerXplorer CLI or Python API.

2. The output CSV format must remain:
   - Planetoid: `node_id, k, motif_id, count`
   - TU: `graph_id, node_id, k, motif_id, count`

3. The motif ID scheme should match whatever HiPerXplorer uses — update `manifest.json` accordingly.
   The `motif_loader.py` reads the manifest and builds the column mapping at load time, so no
   model or loader code needs to change.

4. Delete the cached `.pt` files under `data/precompute/` so the loader rebuilds from the new CSV.

**Zero other changes needed.** The motif_loader → model pipeline is completely agnostic to the
counting engine.

---

## Git / Workflow Notes

- Always `git add` specific files — never `git add -A` (risk of committing data/ files).
- Data in `data/` is gitignored; motif CSVs and `.pt` caches must stay out of the repo.
- User handles `git push` manually; provide the push command as text rather than running it.
