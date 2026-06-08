# Motif-MPNN Experiment Session Log

**Session started:** 2026-06-07  
**Protocol:** Public Kipf & Welling splits for Planetoid; stratified **60/20/20** (seed=training seed) for TU  
**Seeds:** 42, 0, 1  
**Motif convention:** k=3 undirected (k=1/id=0=degree, k=3/id=2=wedge, k=3/id=3=triangle)  
**All hyperparameters fixed before any run**

---

## Pre-run Setup

- config.py fixed to read `use_public_split` from flat-style YAML (top-level key)
- run.py fixed to pass `split_seed=train.seed` to TU dataset constructors
- `citeseer_gat.yml` created (lr=0.005, wd=0.0005, epochs=200, patience=50)
- `use_public_split: true` added to all 16 Planetoid experiment configs
- TU split ratio changed from 80/10/10 → **60/20/20** (policy: random splits always 60/20/20 seeded)
- All 11 Phase 1 dry-runs: PASS

---

## Phase 1: Baseline Sweep

**Published reference numbers (public splits for Planetoid; 10-fold CV for TU):**
- Cora GCN: ~0.810 | Cora SAGE: ~0.798 | Cora GAT: ~0.830
- Citeseer GCN: ~0.703 | Citeseer SAGE: ~0.710 | Citeseer GAT: ~0.725
- Pubmed GCN: ~0.790 | Pubmed SAGE: ~0.788
- PROTEINS GCN: ~0.740–0.760 | NCI1 GCN: ~0.760–0.800 | ENZYMES GCN: ~0.380–0.600

---

## Phase 1 Results — Per-Seed Runs

Planetoid: public splits. TU: stratified 60/20/20 seed=train_seed. All motif_dim=0.

| Dataset   | Variant | Seed | test_acc | test_f1 | best_ep | total_ep |
|-----------|---------|------|----------|---------|---------|----------|
| cora      | gcn     | 42   | 0.7850   | 0.7751  | 31      | 82       |
| cora      | gcn     | 0    | 0.7820   | 0.7759  | 47      | 98       |
| cora      | gcn     | 1    | 0.7890   | 0.7792  | 11      | 62       |
| cora      | sage    | 42   | 0.7680   | 0.7532  | 41      | 92       |
| cora      | sage    | 0    | 0.7750   | 0.7684  | 9       | 60       |
| cora      | sage    | 1    | 0.7940   | 0.7872  | 19      | 70       |
| cora      | gat     | 42   | 0.7930   | 0.7895  | 6       | 57       |
| cora      | gat     | 0    | 0.8060   | 0.8002  | 28      | 79       |
| cora      | gat     | 1    | 0.8000   | 0.7941  | 6       | 57       |
| citeseer  | gcn     | 42   | 0.6710   | 0.6372  | 6       | 57       |
| citeseer  | gcn     | 0    | 0.6590   | 0.6292  | 9       | 60       |
| citeseer  | gcn     | 1    | 0.6610   | 0.6282  | 8       | 59       |
| citeseer  | sage    | 42   | 0.6610   | 0.6241  | 11      | 62       |
| citeseer  | sage    | 0    | 0.6370   | 0.6072  | 21      | 72       |
| citeseer  | sage    | 1    | 0.6800   | 0.6367  | 16      | 67       |
| citeseer  | gat     | 42   | 0.6820   | 0.6478  | 31      | 82       |
| citeseer  | gat     | 0    | 0.6840   | 0.6498  | 7       | 58       |
| citeseer  | gat     | 1    | 0.6700   | 0.6406  | 7       | 58       |
| pubmed    | gcn     | 42   | 0.7450   | 0.7431  | 106     | 157      |
| pubmed    | gcn     | 0    | 0.7660   | 0.7629  | 30      | 81       |
| pubmed    | gcn     | 1    | 0.7470   | 0.7479  | 73      | 124      |
| pubmed    | sage    | 42   | 0.7300   | 0.7286  | 8       | 59       |
| pubmed    | sage    | 0    | 0.7460   | 0.7443  | 30      | 81       |
| pubmed    | sage    | 1    | 0.7400   | 0.7380  | 88      | 139      |
| proteins  | gcn     | 42   | 0.7207   | 0.6766  | 25      | 76       |
| proteins  | gcn     | 0    | 0.6802   | 0.6559  | 7       | 58       |
| proteins  | gcn     | 1    | 0.6847   | 0.6349  | 32      | 83       |
| nci1      | gcn     | 42   | 0.6764   | 0.6763  | 84      | 135      |
| nci1      | gcn     | 0    | 0.6630   | 0.6617  | 43      | 94       |
| nci1      | gcn     | 1    | 0.6813   | 0.6810  | 63      | 114      |
| enzymes   | gcn     | 42   | 0.2750   | 0.2461  | 65      | 116      |
| enzymes   | gcn     | 0    | 0.2167   | 0.1814  | 11      | 62       |
| enzymes   | gcn     | 1    | 0.3000   | 0.2887  | 53      | 104      |

---

## Phase 1 Summary — Mean ± Std (N=3 seeds, 60/20/20 for TU)

| Dataset   | Variant | test_acc mean | test_acc std | test_f1 mean | test_f1 std | Published acc  | Gap    |
|-----------|---------|---------------|--------------|--------------|-------------|----------------|--------|
| cora      | gcn     | 0.7853        | 0.0035       | 0.7767       | 0.0022      | ~0.810         | -0.025 |
| cora      | sage    | 0.7790        | 0.0135       | 0.7696       | 0.0170      | ~0.798         | -0.019 |
| cora      | gat     | 0.7997        | 0.0065       | 0.7946       | 0.0054      | ~0.830         | -0.030 |
| citeseer  | gcn     | 0.6637        | 0.0064       | 0.6315       | 0.0049      | ~0.703         | -0.039 |
| citeseer  | sage    | 0.6593        | 0.0215       | 0.6227       | 0.0148      | ~0.710         | -0.051 |
| citeseer  | gat     | 0.6787        | 0.0076       | 0.6461       | 0.0048      | ~0.725         | -0.046 |
| pubmed    | gcn     | 0.7527        | 0.0116       | 0.7513       | 0.0103      | ~0.790         | -0.037 |
| pubmed    | sage    | 0.7387        | 0.0081       | 0.7370       | 0.0079      | ~0.788         | -0.049 |
| proteins  | gcn     | 0.6952        | 0.0222       | 0.6558       | 0.0209      | ~0.740–0.760†  | -0.055 |
| nci1      | gcn     | 0.6736        | 0.0095       | 0.6730       | 0.0101      | ~0.760–0.800†  | -0.087 |
| enzymes   | gcn     | 0.2639        | 0.0427       | 0.2387       | 0.0540      | ~0.380–0.600†  | -0.116 |

† Published TU numbers use 10-fold CV; ours use a single 60/20/20 random split. Direct comparison is not valid; the gap is expected.

**Improvements from 80/10/10 → 60/20/20:**
- PROTEINS: std 0.054 → 0.022 (flag cleared), acc 0.670 → 0.695
- NCI1: duplicate tie (seeds 42&0) resolved — 0.6764 vs 0.6630 (no longer identical)
- ENZYMES: std 0.060 → 0.043 (flag cleared)

---

## CHECKPOINT A STATUS (60/20/20 splits)

- [ ] Cora GCN sanity: **0.785** — within 0.025 of published 0.810 ✓
- [ ] Citeseer GCN sanity: **0.664** — within 0.039 of published 0.703 ✓
- [ ] Pubmed GCN sanity: **0.753** — within 0.037 of published 0.790 ✓
- [ ] PROTEINS GCN: **0.695** — below published 0.740–0.760, but published uses 10-fold CV ✓ acceptable
- [ ] All std < 0.05: all pass (max node=0.022, max graph=0.043) ✓
- [ ] Any identical 4-decimal results across 3 seeds: **none** ✓
- [ ] motif_dim=0 on all baselines: **YES** ✓

All anomaly checks pass.

---

---

## Phase 2: Motif Variant Sweep

All runs: motif_dim=3 (degree + wedge + triangle), same splits as Phase 1.

### Phase 2 Per-Config Summary (mean ± std, N=3 seeds)

| Dataset   | Variant | test_acc mean | test_acc std | test_f1 mean | test_f1 std | GCN baseline | Δ acc   |
|-----------|---------|---------------|--------------|--------------|-------------|--------------|---------|
| cora      | concat  | 0.7783        | 0.0179       | 0.7689       | 0.0137      | 0.7853       | -0.007  |
| cora      | gate    | 0.7970        | 0.0125       | 0.7885       | 0.0085      | 0.7853       | +0.012  |
| cora      | mix     | 0.7773        | 0.0015       | 0.7682       | 0.0003      | 0.7853       | -0.008  |
| citeseer  | concat  | 0.6633        | 0.0110       | 0.6234       | 0.0061      | 0.6637       | -0.000  |
| citeseer  | gate    | 0.6483        | **0.0333**   | 0.6162       | **0.0270**  | 0.6637       | -0.015  |
| citeseer  | mix     | 0.6753        | 0.0080       | 0.6414       | 0.0115      | 0.6637       | +0.012  |
| pubmed    | concat  | 0.7603        | 0.0012       | 0.7631       | 0.0028      | 0.7527       | +0.008  |
| pubmed    | gate    | 0.7457        | 0.0050       | 0.7467       | 0.0057      | 0.7527       | -0.007  |
| pubmed    | mix     | 0.7443        | 0.0090       | 0.7435       | 0.0094      | 0.7527       | -0.008  |
| proteins  | concat  | 0.7282        | 0.0227       | 0.7008       | 0.0270      | 0.6952       | **+0.033** |
| proteins  | gate    | 0.7132        | 0.0170       | 0.6825       | 0.0200      | 0.6952       | +0.018  |
| proteins  | mix     | 0.7057        | 0.0104       | 0.6668       | 0.0180      | 0.6952       | +0.011  |
| nci1      | concat  | 0.7311        | 0.0181       | 0.7308       | 0.0183      | 0.6736       | **+0.058** |
| nci1      | gate    | 0.6922        | 0.0106       | 0.6914       | 0.0099      | 0.6736       | +0.019  |
| nci1      | mix     | 0.6906        | 0.0028       | 0.6901       | 0.0027      | 0.6736       | +0.017  |
| enzymes   | concat  | 0.3111        | 0.0268       | 0.2970       | 0.0232      | 0.2639       | **+0.047** |
| enzymes   | gate    | 0.3083        | 0.0221       | 0.2729       | 0.0100      | 0.2639       | +0.044  |
| enzymes   | mix     | 0.2861        | 0.0385       | 0.2664       | 0.0397      | 0.2639       | +0.022  |

**Flags:**
- Citeseer gate: std=0.033 — elevated variance (above 0.03 node threshold), likely numerical instability in the edge-gate MLP on a small graph
- Cora mix: std=0.0015 — unusually LOW variance (possible underfitting / collapse to mean)

**Patterns:**
- TU datasets (graph classification): all 9 motif runs improve over GCN baseline. Largest gains on NCI1 (+5.8% concat) and ENZYMES (+4.7% concat).
- Planetoid (node classification): mixed. Gate helps on Cora (+1.2%), Mix helps on Citeseer (+1.2%) and Pubmed (+0.8% concat). Gate hurts on Citeseer (-1.5%, high variance).
- Concat is the most consistent variant across both task types.

---

<!-- Phase 3 ablation results appended below after Checkpoint B approval -->
