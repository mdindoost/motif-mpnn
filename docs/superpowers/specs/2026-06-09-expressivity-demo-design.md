# Expressivity Demo (CSL + Shrikhande/Rook) — Design Spec

**Date:** 2026-06-09
**Author:** Mohammad Dindoost (with Claude as research assistant)
**Status:** Implemented (Phase 0 complete)
**Phase:** 0 of the HiPerMotif-GNN research roadmap (see §8)

---

## 1. Purpose

Demonstrate, both **analytically** (provable, deterministic) and **empirically** (through the existing GNN harness), that:

1. A plain message-passing GNN sits at the **1-WL ceiling** and cannot solve two tasks/distinctions; and
2. **Exact substructure features** (cycle-length counts, 4-clique counts) break that ceiling.

This is Phase 0 of the larger research program. It is the cheapest concrete step and it **doubles as a correctness test for the new motif-feature library**: if the features are computed wrong, the CSL classes will not separate and the rook 4-clique count will not be 8-vs-0, so the demo fails loudly.

Deliverables are reusable harness components that serve as **both** paper-figure generators and permanent regression tests.

### Non-goals (explicit)

- **No HiPerMotif / Arachne / Arkouda.** All counting is local (igraph / NetworkX) per the project's deferral strategy. HiPerMotif is a drop-in backend swap saved for last.
- No node-classification claims (out of scope; the repo's node-task results are confounded).
- No general k=5 orbit library and no `|Aut(H)|` multi-orbit normalization yet (Phase 1). Phase 0 adds only the specific features the two demos need, plus the size-≤5 library motifs used for the CSL characterization.
- A `gin` baseline IS added (needed for a tight ceiling proof, D3) — small scope, generally reusable.

---

## 2. Background facts (the targets to reproduce)

All numbers below are mathematical facts the implementation must reproduce exactly.

### CSL (Circulant Skip-Link)

- `CSL(n, s)`: cycle on `n` vertices (each `i` joined to `i±1 mod n`) plus skip edges (`i` joined to `i±s mod n`). 4-regular.
- Standard benchmark: `n = 41`, ten skips `s ∈ {2,3,4,5,6,9,11,12,13,16}` → **10 classes**. Full dataset = **150 graphs** (15 relabeled/isomorphic copies per class).
- All CSL graphs are 4-regular and vertex-transitive ⇒ **1-WL assigns every vertex one color** ⇒ all 10 classes share one 1-WL representation ⇒ any message-passing GNN is capped at chance = **10%**.
- **Simple-cycle counts by length** give each class a distinct signature; a linear classifier on these features reaches **~99.5%** (5-fold CV, robust to 2% noise). Triangles alone are insufficient (a triangle exists only for `s=2`), so cycles up to length ~L are required.

### Shrikhande vs. 4×4 Rook

- Both are strongly regular with parameters **(16, 6, 2, 2)**; non-isomorphic; cospectral.
- 1-WL coloring identical for both (regular, same λ, μ) ⇒ indistinguishable to any message-passing GNN and to every spectral invariant.
- **Triangles: 32 each** (per-vertex 6 each) — cannot separate them.
- **4-cliques: rook = 8 (4 rows + 4 columns), Shrikhande = 0.** Per-vertex: rook = 2, Shrikhande = 0. A single 4-clique count separates the pair.

---

## 2.5 Alignment with the .tex proposal (differences addressed)

This spec is checked against `hipermotif-gnn-hpec2026.tex`. Three differences are handled explicitly rather than glossed:

**D1 — CSL needs features outside the paper's own pattern library.** The paper's library (Sec. V) tops out at **size-5 patterns**, but separating `s=11` from `s=13` on `n=41` requires simple cycles far longer than 5. Therefore:
- CSL is positioned as an **expressivity-*existence* proof** computed by a dedicated long-cycle counter — it demonstrates that *some* exact substructure feature breaks the ceiling, but it does **not** test the HiPerMotif orbit-feature *method*. This matches how the .tex computes the 99.5% ("independent of any trained network").
- **SRG (4-clique, in the library) is the example that validates the actual method.** The spec and paper text must say this plainly.
- **Characterization sub-experiment (new):** also compute the size-≤5 library motifs on CSL and report how far they separate the 10 classes. Expected result: *partial* separation, completed only by the full cycle spectrum. This converts the tension into a contribution — it empirically motivates exact counting of larger/longer structures rather than hiding the gap.

**D2 — `|Aut(H)|` normalization is inert under per-column z-score.** Dividing a feature column by any constant `b_j` is fully absorbed by z-score: `(x/b − μ/b)/(σ/b) = (x − μ)/σ`. So for **single-orbit-per-pattern** features (most of the library) the paper's emphasized automorphism normalization changes nothing downstream. It matters only for (a) **multi-orbit patterns** needing relative weighting, or (b) **raw/unnormalized** features. Handling:
- Phase 0 omits `|Aut|` normalization with **zero loss** and adds a unit test proving the no-op (z-scored raw counts == z-scored `|Aut|`-normalized counts for a single-orbit pattern).
- Recorded as a research finding: the paper should either de-emphasize this step or motivate it specifically via the multi-orbit case. Multi-orbit normalization is deferred to Phase 1 where it is actually load-bearing.

**D3 — GCN vs GIN baseline.** The paper frames the ceiling around **GIN** (maximally 1-WL-expressive). The repo baseline is **GCN** (strictly weaker than 1-WL). CSL caps *every* MPNN at 10%, so GCN≈10% is valid, but the tight, reviewer-proof baseline is GIN. Handling: **add a `gin` model to the registry** and run it as the CSL ceiling baseline alongside GCN. This is a generally useful addition beyond this demo (the repo currently has gcn/sage/gat, no gin).

---

## 3. Components

Five units, each independently testable.

### 3.1 Synthetic graph generators — `src/datasets/expressivity.py`

- `make_csl(n=41, skips=(2,3,4,5,6,9,11,12,13,16), copies_per_class=15, seed=...)` → list of `(edge_index, label)` with random vertex relabelings per copy. Constant all-ones node features (vertices are featureless; degree is constant 4 so carries no signal).
- `make_shrikhande()` → `edge_index` for the Shrikhande graph (standard construction, e.g. Cayley graph on Z₄² or the documented adjacency).
- `make_rook_4x4()` → `edge_index` for the 4×4 rook graph (`i,j` adjacent iff same row or same column; = K₄ □ K₄ complement convention — verify).
- Each generator has a unit test asserting regularity, vertex/edge counts, and (for the SRG pair) the (16,6,2,2) parameters.

### 3.2 CSL dataset registration — `src/datasets/expressivity.py` + `DATASET_REGISTRY`

- Register key `csl` as a **graph-task** dataset producing a PyG-style bundle compatible with `train_graph_task` and the existing 60/20/20 stratified seeded split.
- Reuses `TUWithMotifs`-style motif attachment so motif features flow through the standard concat path.
- Shrikhande/rook are **NOT** registered (see §4) — they are a test fixture.

### 3.3 Feature extraction additions — `scripts/preprocess/generate_motifs.py`

Add two new per-node feature families, emitted in the existing CSV format and flowing through `motif_loader.py` unchanged (only new `(k, motif_id)` manifest columns):

- **Simple-cycle participation counts** (CSL existence-proof feature, *outside* the orbit library — see D1): for each vertex, the number of simple cycles of length `L` it lies on, for `L = 3..L_max` (`L_max` configurable, default chosen so all 10 CSL classes separate — expected ≤ ~10). Computed via a bounded-length simple-cycle enumerator (small graphs; igraph/NetworkX or custom bounded DFS). This counter is explicitly labeled as a special-purpose expressivity feature, not part of the HiPerMotif method.
- **4-clique participation count** (in the orbit library — validates the method): per vertex, number of 4-cliques it belongs to (igraph `cliques(min=4, max=4)` or equivalent).
- **Size-≤5 library motifs on CSL** (D1 characterization sub-experiment): the bounded library set (triangle, wedge, 4-path, 4-cycle, claw, paw, diamond, 4-clique) computed per-node on CSL, used only to report *how far* the bounded library separates the 10 classes (expected: partial). Motivates exact counting of longer structures.

**Manifest encoding (to finalize at implementation):**
- 4-clique: `k=4, motif_id =` igraph `motifs_undirected` size-4 isomorphism class for K₄ (connected size-4 classes live at indices ~4–10; **verify exact index**, do not hardcode blindly).
- Cycle length L: reserved namespace `k=L, motif_id = CYCLE_SENTINEL` (e.g. a documented constant ≥ 1000 meaning "simple cycle of length k"), since L-cycle iso-classes are not enumerable via the size-3/4 scheme. The manifest is just `(k, motif_id) → column`, so any consistent documented convention works.

These features obey the existing `log1p` + z-score normalization in `motif_loader.py` (counts are heavy-tailed) — no normalization changes.

### 3.3b GIN baseline — `src/models/gin.py` + `MODEL_REGISTRY["gin"]`

- Standard GIN (Xu et al. 2019): sum aggregation + MLP per layer, the maximally-1-WL-expressive MPNN. Registered like the other baselines; supports the graph task.
- Used as the **tight** CSL ceiling baseline (alongside GCN). Addresses difference D3.
- Reusable beyond this demo (fills the gcn/sage/gat/**gin** set).

### 3.4 1-WL refinement utility — `src/utils/wl.py`

- `wl_refine(edge_index, num_nodes, num_iters=None) → coloring` : dependency-free 1-dimensional Weisfeiler–Leman color refinement to a stable partition (hash-based color update).
- `wl_graph_signature(edge_index, num_nodes) → multiset/histogram of stable colors` : used to compare two graphs for 1-WL distinguishability.
- Pure-Python, no sklearn/torch dependency (consistent with `metrics.py` style).

### 3.5 Analytical + empirical evaluation — `scripts/runs/expressivity_demo.py` and tests under `tests/`

- **Analytical (deterministic, becomes assertions):**
  - CSL: all 10 class representatives collapse to one `wl_graph_signature` ⇒ 1-WL cannot separate.
  - CSL: cycle-count feature vectors per class are distinct and **linearly separable** (small torch linear classifier or rank/separation check) → reproduce the ~99.5% anchor.
  - SRG: `wl_graph_signature(shrikhande) == wl_graph_signature(rook)`; triangle counts equal (32, per-vertex 6); 4-clique counts differ (8 vs 0; per-vertex 2 vs 0).
- **Empirical (through the harness, paper-table row):**
  - CSL `gcn` and `gin` (constant features) ≈ 10% test accuracy — GIN is the tight 1-WL baseline (D3).
  - CSL `concat` (constant features + cycle counts) ≫ 10% (target high, ideally ~99%).
  - Writes results via the standard `run.py` → `results/` machinery so they land in `all_runs.csv`.
- **`|Aut|` no-op check (D2):** a unit test showing z-scored raw counts == z-scored `|Aut|`-normalized counts for a single-orbit pattern (e.g. 4-clique), documenting that Phase 0's omission of automorphism normalization is lossless and that the step is load-bearing only for multi-orbit patterns (Phase 1).
- **CSL library-vs-cycles characterization (D1):** report class-separation achieved by the size-≤5 library motifs vs. the full cycle spectrum on CSL (table/short figure).

---

## 4. Key design decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SRG in `DATASET_REGISTRY`? | **No — test fixture** | 2 graphs can't be split/trained; would be a registry entry that only feeds an assertion |
| CSL in `DATASET_REGISTRY`? | **Yes — graph task** | Real 150-graph 10-way task; runs through existing harness/splits |
| Counting backend | **Local (igraph/NetworkX)** | HiPerMotif deferred; CSV format is the contract |
| CSL feature | **Dedicated long-cycle counter, labeled outside the library (D1)** | Faithful to the .tex's existence-proof; library motifs only characterize the gap |
| Aut(H) normalization | **Omitted in Phase 0 (D2)** | No-op under per-column z-score for single-orbit patterns; deferred to Phase 1 multi-orbit case |
| Ceiling baseline | **GCN + GIN (D3)** | GIN is the tight 1-WL-maximal MPNN; GCN kept for continuity |
| Ceiling proof | **Analytical + empirical** | Analytical = provable/noise-free assertion; empirical = harness story + paper row |
| Feature plumbing | **Through the real CSV pipeline** | Reusable foundation for Phase 1 orbit library; validates the pipeline |
| Split for CSL empirical run | **Existing 60/20/20 stratified seeded** | Consistency with repo conventions (proposal's 5-fold CV is reproduced only in the analytical linear-classifier check) |

---

## 5. Data flow

```
make_csl() / make_shrikhande() / make_rook_4x4()
        │
        ├─► CSL bundle → DATASET_REGISTRY["csl"]
        │
        ▼
generate_motifs.py  (adds cycle-length + 4-clique counts)
        │
        ▼
data/precompute/csl/node_motifs.csv   (graph_id, node_id, k, motif_id, count)
        │
        ▼
motif_loader.py (unchanged) → motif_x   (log1p + z-score, new manifest columns)
        │
        ├─► EMPIRICAL: run.py → train_graph_task → results/  (gcn vs concat on CSL)
        │
        └─► ANALYTICAL: expressivity_demo.py / tests
                ├─ wl.py: 1-WL signatures (CSL collapse; SRG identical)
                ├─ linear separability of CSL cycle features (~99.5%)
                └─ SRG triangle (32=32) and 4-clique (8 vs 0) assertions
```

---

## 6. Testing strategy

Deterministic unit/regression tests (no training noise) under `tests/`:

1. `make_*` generators produce graphs with correct regularity / SRG parameters / counts.
2. `wl.py`: CSL class reps share one signature; Shrikhande and rook share a signature.
3. Cycle-count features separate the 10 CSL classes (distinct & linearly separable).
4. SRG: triangles 32=32; 4-cliques 8 vs 0 (per-vertex 2 vs 0).
5. **`|Aut|` no-op (D2):** z-scored raw == z-scored `|Aut|`-normalized for a single-orbit pattern.
6. **CSL characterization (D1):** size-≤5 library motifs give only partial CSL separation; full cycle spectrum gives complete separation.
7. Empirical (slower, may be a marked/optional test): CSL `gcn` and `gin` ≈ chance; CSL `concat` ≫ chance.

Tests 1–6 are fast and fully deterministic — they are the permanent guard on the thesis and the feature library.

---

## 7. Risks / open implementation details

- **Exact igraph k=4 motif index for K₄** — verify empirically at implementation; do not assume index 10.
- **`L_max` for cycles** — pick the smallest L that separates all 10 CSL classes; determine empirically, document the value.
- **Simple-cycle enumeration cost** — fine for 41-node graphs; the enumerator must take a length bound to stay tractable and must count *simple* cycles (not closed walks).
- **CSL relabeling** — copies must be true isomorphic relabelings (permuted vertex ids) so the task tests structural invariance, not memorized ids.
- **Constant node features for GCN** — confirm the GCN path accepts all-ones features and that the ~10% result is reproducible across seeds (it should be, by the 1-WL argument).

---

## 8. Roadmap context (Phases 1+ — not this spec)

This Phase 0 is the foundation slice. Subsequent phases (each its own spec):

- **#4 (assess before Phase 1):** homomorphism-basis counting (Curticapean–Dell–Marx) — may change the extractor.
- **Phase 1:** full orbit-feature library (k=4, selected k=5, `|Aut(H)|` normalization).
- **Phase 2:** #5 "motifs help more at scale" (graph-size sweep, local counting); #1 cost–accuracy frontier (proxy cost).
- **Phase 3:** #2 exact vs DeSCo (matched budget); #3 cost-aware greedy motif selection.
- **Phase LAST:** swap in HiPerMotif → real at-scale frontier numbers.
