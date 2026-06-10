# Orbit-Feature Library (Phase 1a: ORCA counting infrastructure) — Design Spec

**Date:** 2026-06-10
**Author:** Mohammad Dindoost (with Claude as research assistant)
**Status:** Implemented (2026-06-10) — see plan `docs/superpowers/plans/2026-06-10-orbit-library-1a.md`; all validation tests pass (`tests/test_orca_orbits.py`, `tests/test_generate_motifs_orbit.py`)
**Phase:** 1a of the HiPerMotif-GNN research roadmap. Phase 1b (benchmark + method-validation *experiments*) is a separate spec that depends on this one.

---

## 1. Purpose

Replace the hand-coded 3-feature motif set (degree, wedge, triangle) with a principled, literature-standard **per-node graphlet-orbit feature library**, computed exactly and locally via **ORCA** (Hočevar & Demšar 2014) — the canonical orbit counter and exactly the feature family GSN (Bouritsas 2022) uses and the HPEC proposal's "≤5-node library" describes.

Phase 1a is **counting infrastructure only**: vendor + build ORCA, wrap it, integrate it as a new feature backend in `generate_motifs.py`, map orbits into the existing CSV/manifest scheme, and **validate the counts against hand-computation and published ORCA definitions**. It makes **no accuracy claim** — that is Phase 1b.

### Why ORCA (and why it matters later)
ORCA computes, for each vertex, its count in each automorphism **orbit** of all 2-to-k-node graphlets (15 orbits for k=4, 73 for k=5). This is the exact per-vertex orbit feature the proposal's Algorithm 1 defines. Because the eventual HiPerMotif backend produces raw isomorphism counts that, after `|Aut|`/orbit normalization, must equal these orbit counts, **ORCA also becomes the validation oracle for HiPerMotif** when it is connected (Phase LAST). Per the project's deferral rule, no HiPerMotif/Arachne/Arkouda is involved here.

### Non-goals (explicit — deferred to Phase 1b or later)
- No dataset-loader changes and no selection of the orbit CSV by `DATASET_REGISTRY` datasets.
- No benchmark runs, no rand-ablation, no SRG/CSL *experiments* (the SRG check here is a unit test of orbit *correctness*, not an experiment).
- No 5-node (73-orbit) production runs; the wrapper supports `graphlet_size=5` but 4-node (15 orbits) is the primary path. 5-node is exercised only by a smoke test.
- No HiPerMotif. No removal of the legacy 3-feature path (it stays, untouched, so existing validated results remain reproducible).

---

## 2. Background facts (targets the implementation must reproduce)

ORCA CLI (canonical source): `orca <node|edge> <4|5> <infile> <outfile>`.
- **Input format:** first line `numNodes numEdges`; then one edge per line `u v`, 0-indexed, undirected, each edge once, no self-loops, vertices contiguous `0..N-1`.
- **Output format (`node` mode):** `numNodes` lines, each with `n_orbits` space-separated integers — vertex `i`'s count in each orbit. `n_orbits = 15` for size 4, `73` for size 5.

**ORCA 4-node orbit set (15 orbits, 0–14), with graphlet size of each** (from Hočevar & Demšar 2014, Fig. 1; the implementation MUST verify these against ORCA output, not trust this table blindly):

| orbit | graphlet | size | meaning |
|------|----------|------|---------|
| 0 | edge endpoint | 2 | degree |
| 1 | path end (P3) | 3 | end of a 2-path |
| 2 | path center (P3) | 3 | center of a 2-path (wedge apex) |
| 3 | triangle | 3 | triangle |
| 4 | 3-star (K1,3) leaf | 4 | |
| 5 | P4 end | 4 | |
| 6 | P4 inner | 4 | |
| 7 | 3-star center | 4 | |
| 8 | paw (cycle part) | 4 | |
| 9 | paw (other) | 4 | |
| 10 | 4-cycle | 4 | |
| 11 | diamond (degree-2) | 4 | |
| 12 | diamond (degree-3) | 4 | |
| 13 | 4-clique-minus? / paw-tail | 4 | |
| 14 | 4-clique | 4 | K4 |

The exact semantic of each orbit index is fixed by ORCA; the **only indices the spec hard-asserts** are the ones verified by hand in tests (§6): orbit 0 = degree; the triangle orbit; the 4-clique orbit. The remaining indices are recorded from ORCA's own output during implementation and the table above is corrected if needed. (This is deliberate: per the rigor rule we assert only what we verify.)

**Hand-verifiable check values (computed by hand, used as test oracles):**
- **K4** (complete, 4 nodes): every node is in the 4-clique orbit exactly once; triangle orbit = 3 per node (C(3,2)); degree orbit = 3.
- **C5** (5-cycle): degree orbit = 2 per node; triangle orbit = 0; 4-cycle orbit = 0; 4-clique orbit = 0.
- **Shrikhande vs 4×4 rook** (the method-validation oracle): triangle orbit equal and nonzero for both; **4-clique orbit = 0 (Shrikhande) vs 2 per node (rook)**.

---

## 3. Components

Five units. Counting logic is importable from `src/`; `generate_motifs.py` only orchestrates (mirrors the existing `substructure_counts.py` pattern).

### 3.1 Vendored ORCA — `third_party/orca/`
- `third_party/orca/orca.cpp` — the canonical single-file source (committed).
- `third_party/orca/PROVENANCE.md` — source URL, authors, citation (Hočevar & Demšar, *Bioinformatics* 2014), license note.
- The compiled binary `third_party/orca/orca` is **gitignored** (built locally).

### 3.2 ORCA wrapper — `src/datasets/orca_orbits.py` (importable, tested)
- `ORCA_DIR`, `ORCA_BIN` path constants; `N_ORBITS = {4: 15, 5: 73}`.
- `ensure_orca_built() -> Path`: if `ORCA_BIN` absent, compile `orca.cpp` with `g++ -O2 -std=c++11 -o orca orca.cpp`. Raise a clear `RuntimeError` if g++ is unavailable, naming the fix.
- `count_orbits(edges, num_nodes, graphlet_size=4) -> np.ndarray` of shape `[num_nodes, N_ORBITS[graphlet_size]]`:
  - Normalize `edges` to a deduped, self-loop-free, 0-indexed undirected edge set.
  - Write ORCA input (`N E` header + edges) to a temp file; run the binary; parse the output matrix.
  - Handle graphs with isolated nodes (ORCA still needs contiguous `0..N-1`; isolated nodes get all-zero rows).
- `ORBIT_GRAPHLET_SIZE: list[int]` — orbit index → graphlet node count (length-15 table for the 4-node basis), used for the manifest `k` field. Verified by a test against ORCA's documented structure.
- `orbit_to_km(o) -> tuple[int,int]`: returns `(k = ORBIT_GRAPHLET_SIZE[o], motif_id = ORCA_ORBIT_BASE + o)`, where `ORCA_ORBIT_BASE = 2000`.

### 3.3 CSV manifest encoding
Each orbit `o` becomes CSV row field pair `(k = graphlet_size(o), motif_id = 2000 + o)`. The `2000+` namespace is disjoint from all existing encodings (degree/wedge/triangle igraph ids 0/2/3, cycles `motif_id=1000`, 4-clique `(4,10)`). `motif_loader.py` already extends the manifest for unseen `(k, motif_id)` pairs (verified in Phase 0), so no loader change is needed.

### 3.4 `generate_motifs.py` integration
- New CLI flag `--features {legacy, orbit}` (default `legacy`).
- `legacy` → unchanged behavior (degree/wedge/triangle; existing CSV path).
- `orbit` → for each graph, call `orca_orbits.count_orbits(..., graphlet_size=4)`, emit rows `(graph_id, node_id, k, 2000+o, count)` for every nonzero orbit count, and write to **`data/precompute/<dataset>/node_motifs_orbit.csv`** (distinct filename; legacy CSV/caches untouched).
- Works for Planetoid (single graph, no `graph_id` column), TU, and synthetic (CSL) datasets, reusing the existing graph loaders.
- A `--graphlet-size {4,5}` flag (default 4) is plumbed to the wrapper; 4 is the only one used in production for 1a.

### 3.5 Validation tests — `tests/`
See §6. These are the deliverable's correctness gate.

---

## 4. Data flow

```
graph (edge_index / networkx, from existing loaders)
   │
   ▼
orca_orbits.count_orbits(edges, N, graphlet_size=4)   [vendored ORCA binary, subprocess]
   │  → [N, 15] integer orbit-count matrix
   ▼
orbit o → (k = graphlet_size(o), motif_id = 2000 + o)
   │
   ▼
generate_motifs.py --features orbit
   → data/precompute/<dataset>/node_motifs_orbit.csv   (graph_id?,node_id,k,motif_id,count)
   │
   ▼
motif_loader.py  (UNCHANGED — extends manifest for new (k,motif_id))  → motif_x [N, 15]
   │
   ▼
[Phase 1b: dataset loader selects the orbit CSV; concat vs rand-concat benchmarks; SRG/CSL method validation]
```

---

## 5. Error handling

- **g++ missing:** `ensure_orca_built()` raises `RuntimeError("g++ not found; cannot build ORCA. Install build-essential ...")`.
- **ORCA nonzero exit / malformed output:** wrapper raises with the captured stderr and the offending graph's `(N, E)`.
- **Non-contiguous / out-of-range vertex ids:** wrapper remaps to `0..N-1` (and asserts the mapping is consistent with `num_nodes`).
- **0-edge graph:** allowed; ORCA returns an all-zero matrix (handled, no crash).
- **Determinism:** ORCA is exact and deterministic (combinatorial counting, no RNG, single-threaded) — orbit counts are fully reproducible, unlike the GPU training runs.

---

## 6. Testing strategy

Deterministic unit tests under `tests/` (all fast; ORCA build cached after first test):

1. `ensure_orca_built()` produces a runnable binary; a second call is a no-op.
2. **K4:** orbit 0 (degree) == 3 per node; triangle orbit == 3 per node; 4-clique orbit == 1 per node; all match hand-computed values.
3. **C5:** degree == 2; triangle/4-cycle/4-clique orbits == 0.
4. **Star K1,3:** center vs leaf orbit counts match hand-computed (distinguishes the two 3-star orbits).
5. **orbit 0 == networkx degree** on a random small graph (cross-check against an independent tool).
6. **Method validation (SRG):** triangle orbit equal for Shrikhande and rook; **4-clique orbit == 0 (Shrikhande) and == 2/node (rook)**. (This is the orbit that validates the proposal's actual method.)
7. `ORBIT_GRAPHLET_SIZE` table is length 15, values in {2,3,4}, and consistent with the verified orbits (0→2, triangle→3, 4-clique→4).
8. **End-to-end:** `generate_motifs.py --features orbit` on CSL writes `node_motifs_orbit.csv` with `motif_id` values in the `2000+` range spanning ≥15 orbits across all graphs; `motif_loader` builds a `motif_x` of width 15 from it (round-trip).

Tests that compile/run ORCA over many graphs (the end-to-end on CSL) are marked `@pytest.mark.slow`.

---

## 7. Risks / open implementation details

- **Exact orbit indices beyond {degree, triangle, 4-clique}** — recorded from ORCA's own output during implementation; the §2 table is corrected to match. We assert only hand-verified indices (rigor rule).
- **ORCA source acquisition** — vendor the canonical `orca.cpp`; record exact source URL + commit/version in `PROVENANCE.md`. If the upstream layout differs (e.g. needs `orca.h`), vendor all required files.
- **ORCA edge-orbit mode** (`edge` CLI mode) is **not** used in 1a (node orbits only).
- **5-node (73 orbits)** wrapper path is implemented but only smoke-tested; production use is a Phase 1b ablation.

---

## 8. Roadmap context

- **Phase 1a (this spec):** ORCA orbit-feature library + validation. Deliverable: trusted per-node orbit CSVs.
- **Phase 1b (next spec):** loader selection of orbit CSV; concat vs capacity-matched rand-concat on Planetoid/TU (4-node primary, 5-node ablation); SRG/CSL method-validation experiments; record results. Depends on 1a.
- **Phase 2+:** scale-dependence (#5), cost–accuracy frontier (#1), exact-vs-DeSCo (#2), greedy selection (#3).
- **Phase LAST:** HiPerMotif backend swap; ORCA becomes the correctness oracle for HiPerMotif's normalized orbit counts.
