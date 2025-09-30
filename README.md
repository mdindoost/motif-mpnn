# Motif-Augmented MPNN

Motif-Augmented MPNN (Motif-MPNN) is an experimental graph learning framework where **motif statistics** (exported from [HiPerXplorer](https://github.com/your-org/hiperxplorer)) are injected into Graph Neural Networks to enrich message passing.  

We build on top of [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/), extending standard baselines with motif-aware counterparts:

- **Baselines**:  
  - GCN  
  - GraphSAGE  
  - GAT  

- **Motif-aware variants**:  
  - **Concat** — Concatenate motif features to node features before message passing.  
  - **Gate** — Learn edge-wise gates from motif context to modulate messages.  
  - **Mix** — Blend structural adjacency with motif-derived similarity adjacency.  

The framework is designed for both **node classification** (Cora, Citeseer, Pubmed) and **graph classification** (PROTEINS, NCI1, ENZYMES). Motif features are loaded from CSV artifacts and cached as tensors for efficient training.

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

All experiments are described by YAML configs in `configs/experiments/`.  
The runner script loads the config, instantiates the dataset + model, and trains with early stopping.  

### General usage
```bash
python -m src.train.run --config <path-to-yaml>
```

---

### 🔹 Planetoid datasets (Cora, Citeseer, Pubmed) — node classification

**Baselines**

```bash
python -m src.train.run --config configs/experiments/cora_gcn.yml
python -m src.train.run --config configs/experiments/citeseer_gcn.yml
python -m src.train.run --config configs/experiments/pubmed_gcn.yml
```

**Other baseline architectures**

```bash
python -m src.train.run --config configs/experiments/cora_sage.yml
python -m src.train.run --config configs/experiments/cora_gat.yml
# similarly citeseer_sage.yml, pubmed_gat.yml, etc.
```

**Motif-augmented variants**

```bash
# Feature injection
python -m src.train.run --config configs/experiments/cora_concat.yml
python -m src.train.run --config configs/experiments/citeseer_concat.yml
python -m src.train.run --config configs/experiments/pubmed_concat.yml

# Edge gating with motifs
python -m src.train.run --config configs/experiments/cora_gate.yml
python -m src.train.run --config configs/experiments/citeseer_gate.yml
python -m src.train.run --config configs/experiments/pubmed_gate.yml

# Adjacency mixing (topology + motif similarity)
python -m src.train.run --config configs/experiments/cora_mix.yml
python -m src.train.run --config configs/experiments/citeseer_mix.yml
python -m src.train.run --config configs/experiments/pubmed_mix.yml
```

---

### 🔹 TU datasets (PROTEINS, NCI1, ENZYMES) — graph classification

**Baselines**

```bash
python -m src.train.run --config configs/experiments/proteins_gcn.yml
python -m src.train.run --config configs/experiments/nci1_gcn.yml
python -m src.train.run --config configs/experiments/enzymes_gcn.yml
```

**Motif-augmented variants**

```bash
# Feature injection
python -m src.train.run --config configs/experiments/proteins_concat.yml
python -m src.train.run --config configs/experiments/nci1_concat.yml
python -m src.train.run --config configs/experiments/enzymes_concat.yml

# Edge gating with motifs
python -m src.train.run --config configs/experiments/proteins_gate.yml
python -m src.train.run --config configs/experiments/nci1_gate.yml
python -m src.train.run --config configs/experiments/enzymes_gate.yml

# Adjacency mixing
python -m src.train.run --config configs/experiments/proteins_mix.yml
python -m src.train.run --config configs/experiments/nci1_mix.yml
python -m src.train.run --config configs/experiments/enzymes_mix.yml
```

---

### Notes
* All configs set default `epochs`, `patience`, and `seed`. You can edit them to override hyper-parameters.


Results are written under:
```bash
results/logs/<timestamp>_<run_name>/
  ├── manifest.json   # config + run metadata
  └── metrics.csv     # per-epoch losses and accuracies
```

Final metrics are also printed at the end of each run.
* To test reproducibility across seeds, add `train: { seed: <int> }` to any YAML.

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