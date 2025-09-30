# Motif-Augmented MPNN

Motif-Augmented MPNN (Motif-MPNN) is an experimental graph learning framework where **motif statistics** (from HiPerXplorer) are injected into Graph Neural Networks to enrich message passing.  

We build on top of PyTorch Geometric (PyG), extending standard baselines (GCN, GraphSAGE, GAT) with motif-aware variants:

- **Concat** — Concatenate motif counts to node features.  
- **Gate** — Use motif context to gate edge messages.  
- **Mix** (coming soon) — Blend structural adjacency with motif adjacency.  

---

## 🚀 Getting Started

### 1. Clone & setup environment
```bash
git clone https://github.com/mdindoost/motif-mpnn.git
cd motif-mpnn
conda env create -f environment.yml
conda activate motif-mpnn
```

### 2. Datasets
The repo auto-downloads Planetoid datasets (Cora, Citeseer, Pubmed).

For motif features, place HiPerXplorer exports here:
```bash
data/precompute/<dataset>/node_motifs.csv
```

Example for Cora:
```bash
data/precompute/cora/node_motifs.csv
```

On first run, the loader will generate:
- `manifest.json` — motif ID → column index
- `stats.json` — per-column normalization stats
- `motif_x.pt` — cached dense motif feature tensor

---

## 🧪 Running Experiments

Each experiment is described by a YAML config in `configs/experiments/`.

**Example: Cora GCN (baseline)**
```bash
python -m src.train.run --config configs/experiments/cora_gcn.yml
```

**Example: Cora + motif concat**
```bash
python -m src.train.run --config configs/experiments/cora_concat.yml
```

**Example: Citeseer + motif gate**
```bash
python -m src.train.run --config configs/experiments/citeseer_gate.yml
```

Results are written under:
```bash
results/logs/<timestamp>_<run_name>/
  ├── manifest.json   # config + run metadata
  └── metrics.csv     # per-epoch losses and accuracies
```

Final metrics are also printed at the end of each run.

---

## 📊 Current Results (seed=42, hidden=64, 2-layer GCN backbone)

| Dataset   | GCN (acc) | Concat (acc) | Gate (acc) |
|-----------|-----------|--------------|------------|
| Cora      | ~0.756    | ~0.764       | ~0.762     |
| Citeseer  | ~0.635    | ~0.655       | ~0.669     |
| Pubmed    | TBD       | TBD          | TBD        |

*(macro-F1 is also logged; see metrics.csv for details.)*

---

## ⚙️ Configuration

Each config YAML has the structure:

```yaml
dataset: cora         # cora | citeseer | pubmed | proteins | nci1 | enzymes
variant: gcn          # gcn | sage | gat | concat | gate | mix
run_name: cora_gcn

train:
  epochs: 200
  patience: 50
  seed: 42

optim:
  lr: 0.01
  weight_decay: 0.0

# variant-specific knobs
gate:
  gate_hidden: 64
  gate_temp: 1.0
  gate_dropout: 0.0
  gate_alpha: 0.5
```

---

## 🛠 Repo Structure

```
src/
  datasets/      # Planetoid & TU loaders + motif loader
  models/        # GCN, SAGE, GAT, Concat, Gate (Mix coming soon)
  train/         # trainer, engine, metrics
  utils/         # config, registry, seed
configs/
  experiments/   # YAML configs for reproducible runs
data/
  precompute/    # HiPerXplorer motif CSVs + generated caches
results/
  logs/          # per-run manifests + metrics.csv
```

---

## 🔮 Roadmap

- [x] Phase A: config & registry scaffolding
- [x] Phase B: real datasets (Planetoid + TU)
- [x] Phase C: motif loader (CSV → tensor) + Concat variant
- [x] Phase D: baseline models (GCN/SAGE/GAT)
- [x] Phase E: trainer with early stopping & CSV logs
- [x] Gate variant
- [ ] Mix variant
- [ ] TU batching for motifs (PROTEINS/NCI1/ENZYMES)
- [ ] Multi-seed sweeps + results tables

---

## 🙌 Contributing

Pull requests are welcome!

If you'd like to add motif generators, alternative injection strategies, or new benchmarks, fork and PR.

---

## 📜 Citation

If you use this repo, please cite:

```bibtex
@misc{motif-mpnn,
  author = {Mohammad Dindoost},
  title = {Motif-Augmented MPNN},
  year = {2025},
  howpublished = {\url{https://github.com/mdindoost/motif-mpnn}}
}
```