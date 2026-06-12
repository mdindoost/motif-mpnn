# Project Story & Handoff — motif-mpnn → HiPerMotif-GNN (HPEC 2026)

> **Purpose of this file.** A standalone narrative so a fresh Claude session (or
> a new collaborator) can understand, with zero prior context: where this repo
> started, what changed it ("David's idea"), what we are building now, and the
> plan for writing the paper. It is a *story + state + plan*, not a spec. The
> authoritative engineering reference is `CLAUDE.md`; the formal designs live in
> `docs/superpowers/specs/` and `docs/superpowers/plans/`. This file is
> intentionally **not** tracked in git — it is a working handoff note.
>
> **Date written:** 2026-06-10. **Author:** Mohammad Dindoost (mdindoost), with
> Claude as research assistant. **Repo:** `~/motif-mpnn`, branch `expressivity-demo`.

---

## 0. Cast and one-paragraph orientation

- **Mohammad Dindoost** (`md724@njit.edu`) — PhD student, NJIT Department of Data
  Science. Owns this repo and the GNN side of the work.
- **David A. Bader** (`bader@njit.edu`) — advisor/PI, co-author. His group builds
  **Arachne** (a large-scale graph-analytics package on **Arkouda**/**Chapel**)
  and **HiPerMotif**, a parallel edge-centric *subgraph isomorphism* engine
  (Dindoost et al., HPEC 2025, arXiv:2507.04130). "David's idea" below refers to
  the research direction that reframed this whole repo around HiPerMotif.

**The thesis in one sentence:** message-passing GNNs are capped by the 1-WL test
(they provably cannot count triangles, cycles, or cliques); injecting *exact*
substructure/orbit counts removes that ceiling; and HiPerMotif makes exact
counting feasible on very large graphs (~10^8 edges), so expressive
substructure-aware GNNs can be demonstrated at a scale prior work never reached.

---

## 1. ACT I — "Before David's idea": a motif-MPNN evaluation harness

The repo began life as a **self-contained evaluation harness** answering a modest,
local question:

> Do subgraph **motif statistics** (degree, wedge, triangle counts) improve GNN
> message passing on standard node- and graph-classification benchmarks?

Everything in this act is local and uses off-the-shelf counters (**igraph** /
**NetworKit**) — no HPC, no Arachne. The harness is well-engineered:

- **Registry-driven** models and datasets (`src/utils/registry.py`); YAML-driven
  configs (`configs/`); clean train/eval engine (`src/train/`).
- **Models / variants:** `gcn`, `sage`, `gat`, `gin` baselines; and three
  motif-aware variants — `concat` (motif features concatenated to node features),
  `gate` (edge gates from motif context), `mix` (blend structural adjacency with
  motif-similarity adjacency).
- **Datasets:** Planetoid (Cora/Citeseer/Pubmed, node) and TU
  (PROTEINS/NCI1/ENZYMES, graph).
- **A full multi-seed study:** 105 runs across seeds {42, 0, 1} (see `CLAUDE.md`
  §9 and `results/findings.md`).

**What Act I actually found (real, empirical):**

- **Graph classification: motifs help.** All three motif variants beat plain GCN
  on every TU dataset. Best gains (concat): NCI1 **+5.8%**, ENZYMES **+4.7%**,
  PROTEINS **+3.3%**. A **rand-concat ablation** (random features of the same
  width) is *worse* than GCN on PROTEINS/NCI1, which is the key evidence that the
  gain is **structural**, not just extra capacity.
- **Node classification: mixed/weak.** Small gains exist (e.g. Pubmed concat
  +0.8%, clean vs random), but on Cora/Citeseer the rand-concat control matches
  concat, so the capacity confound is not ruled out.
- A **13-bug audit** (2026-06-06/07) fixed real correctness bugs (graph-task
  backward pass was outside the batch loop; patience/monitor were hardcoded;
  flat-config loader dropped model hyperparameters; directed-vs-undirected motif
  ID confusion). See `CLAUDE.md` §10.

**The honest limitation of Act I:** local motif counting is cheap only on small
graphs and only for a 3-feature library (degree/wedge/triangle). The results are
real but modest, and they do not, by themselves, make a high-performance-computing
story. That is the gap David's idea fills.

---

## 2. ACT II — "David's idea": HiPerMotif as an exact orbit-feature engine at scale

The reframing turns the modest harness into an **HPC + expressivity** contribution:

1. **Use exact parallel subgraph isomorphism (HiPerMotif) as the feature
   extractor.** Instead of cheap local triangle/wedge counts, compute *exact*
   per-vertex **orbit** features for a whole library of small patterns
   (triangle, wedge, 4-path, 4-cycle, claw, paw, diamond, 4-clique, selected
   5-vertex patterns). This is exactly the GSN (Bouritsas 2022) feature family,
   but computed exactly and at scale.
2. **The only nontrivial math is symmetry normalization.** A subgraph-iso search
   reports each embedded copy of pattern `H` exactly `|Aut(H)|` times, so true
   copies = raw count / `|Aut(H)|`; per-vertex orbit counts divide by the orbit
   stabilizer `|Aut(H)|/|orbit|`. Final feature is `log(1+X)` (counts are
   heavy-tailed). This is Algorithm 1 in the paper draft.
3. **The at-scale story is the novelty.** Prior substructure-GNN work either runs
   on tiny molecular graphs or *learns an approximate* counter (DeSCo) because
   exact counting was assumed intractable. HiPerMotif (which has processed a
   connectome with ~10^8 edges) makes exact features feasible on large hosts —
   a regime the GNN expressivity literature has not reached.

So Act I's `concat` variant is literally the "Use in a GNN" step of David's
pipeline, restricted to a 3-feature library. The plan is to **grow the feature
library to exact orbits and grow the host graphs to HPC scale.**

**Standing project rule (important for the next session):** keep HiPerMotif /
Arachne / Arkouda *out* of the experiments for now. Develop and validate the whole
pipeline with a **local exact counter** first, get the science solid, and **swap
HiPerMotif in last** (it is a drop-in backend; see Act III, Phase LAST). This is a
deliberate de-risking choice, not an oversight.

---

## 3. ACT III — What we are doing now: the phased roadmap

The work is organized into phases. Phases 0 and 1a are **done**; 1b is the next
build; HiPerMotif is the final swap.

### Phase 0 — Expressivity demos (DONE, branch `expressivity-demo`)
Prove the thesis *analytically and empirically* on tiny synthetic cases, as unit
tests (deterministic, no training noise):

- **CSL** (circulant skip-link, 10 classes, 150 graphs, constant node features):
  `gcn` and even `gin` (the maximally-1-WL-expressive MPNN) hit **chance (10%)**;
  `concat` with exact cycle/substructure features reaches **100%** (single run,
  seed 42). Proven analytically in `tests/test_csl_separability.py` (full cycle
  spectrum separates all 10 classes; cycles ≤ length 4 do not).
- **Shrikhande vs. 4×4 rook** (cospectral strongly-regular pair, both SRG(16,6,2,2)):
  identical 1-WL signatures, identical 32 triangles, but **4-clique count = 0
  (Shrikhande) vs. 8 total / 2 per node (rook)** — a single exact substructure
  count separates a pair that defeats 1-WL *and every spectral invariant*. Proven
  in `tests/test_srg_distinguishability.py`.

These are the paper's Section III ("The Expressivity Gap, Concretely") and the
numbers in the draft are real.

### Phase 1a — ORCA orbit-feature library (DONE, 2026-06-10 — most recent work)
Replace the hand-coded 3-feature set with a **principled, literature-standard
per-vertex graphlet-orbit library**, computed exactly and locally via **ORCA**
(Hočevar & Demšar 2014) — the canonical orbit counter, and the same feature family
the paper describes. **Phase 1a is counting infrastructure only — no accuracy
claim.**

What shipped (commit `8dfafe3`, all on `expressivity-demo`):
- **Vendored ORCA** at `third_party/orca/orca.cpp` (+ `PROVENANCE.md`); the
  compiled binary is gitignored and built on demand.
- **`src/datasets/orca_orbits.py`** — `count_orbits(edges, num_nodes,
  graphlet_size=4) -> [N, 15]` (15 orbits for size-4, 73 for size-5), built on a
  subprocess call to ORCA, with `orbit_to_km(o) -> (k, motif_id=2000+o)`.
- **`scripts/preprocess/generate_motifs.py --features orbit`** — an *additive*
  backend (legacy degree/wedge/triangle path byte-for-byte unchanged) that writes
  per-node orbit counts to a **separate** `data/precompute/<ds>/node_motifs_orbit.csv`
  in a disjoint `motif_id` namespace (`2000+orbit`). `motif_loader.py` needs no
  change (it auto-extends its manifest for unseen `(k, motif_id)`).
- **Validation against hand-computed oracles** (`tests/test_orca_orbits.py`, all
  pass): K4, C5, isolated node, degree-vs-networkx cross-check, and the
  Shrikhande/rook SRG pair. These *prove* ORCA's orbit columns mean what we claim:
  **orbit 0 = degree, orbit 3 = triangle, orbit 14 = 4-clique.** Full repo suite:
  **33/33 passing.**

Why ORCA matters beyond Phase 1a: when HiPerMotif is connected, its raw iso counts
— after `|Aut|`/orbit normalization — must equal ORCA's orbit counts. **ORCA
therefore becomes the correctness oracle for HiPerMotif.**

### Phase 1b — Benchmarks & method validation (NEXT, not started)
A separate spec, depends on 1a. Scope:
- Wire the dataset loader to *select* the orbit CSV (Phase 1a deliberately did not
  — the loader still reads the fixed filename `node_motifs.csv`).
- `concat` vs **capacity-matched rand-concat** on Planetoid/TU using the richer
  orbit library (4-node primary; 5-node = 73 orbits as an ablation).
- SRG/CSL method-validation *experiments* (Phase 0/1a only did correctness tests).
- Record results into `results/` and the LaTeX tables.

### Phases 2+ — At-scale story
Scale-dependence of the gain; the **cost-vs-accuracy frontier** per motif family;
exact (HiPerMotif) vs. learned-approximate (DeSCo) at matched cost; greedy motif
selection. These map to the paper's open `\todo` results.

### Phase LAST — HiPerMotif backend swap (BUILT & locally validated 2026-06-12)
The backend is **built and unit-tested locally**; only the Wulver equivalence run +
scale experiments remain. `src/datasets/hipermotif_patterns.py` (ORCA-verified orbit→
graphlet table, `|Aut|`, the corrected collapse-then-`÷|Aut|` normalization) and
`src/datasets/hipermotif_backend.py` (`count_orbits_hipermotif`, a drop-in for
`orca_orbits.count_orbits`, arkouda import-guarded) are done; `generate_motifs.py`
gained `--backend {orca,hipermotif}` (default `orca`, identical CSV schema). The full
15-orbit normalization is proven byte-exact equal to ORCA *locally* (no arkouda needed)
via a networkx induced-embedding oracle — `tests/test_hipermotif_patterns.py`, 44 tests
pass. A subtle bug was caught and fixed first: the per-vertex divisor must be the full
`|Aut(H)|`, not the stabilizer `|Aut|/|orbit|` (the latter over-counts by `|orbit|`; the
proposal's Algorithm 1 still has this typo — flagged for paper correction).

**What remains (user, on Wulver):** run `scripts/wulver/verify_hipermotif_equals_orca.py`
FIRST — it computes orbit features both ways (ORCA oracle vs live HiPerMotif) on
K4/C5/Shrikhande/rook/Cora/PROTEINS and asserts exact equality (exits nonzero on
mismatch). Only if it prints ALL-PASS does the scale phase proceed. ORCA stays the oracle.

---

## 4. Where the code actually is right now (concrete state)

- **Branch:** `expressivity-demo` (holds Phase 0 + Phase 1a). Also fast-forwarded
  to local `main`. `origin/expressivity-demo` is pushed and up to date.
- **Recent commits:** `628c3df` (docs: HPEC draft + Phase 1a notes + TU reval),
  `8dfafe3` (Phase 1a ORCA orbit library), `13b8195` (flat-config fix),
  `db0fac2` (Phase 0 CSL+SRG).
- **Tests:** `conda run -n motif-mpnn pytest -q` → **33 passed**.
- **Environment:** conda env `motif-mpnn`. g++ available (needed to build ORCA).
- **The paper draft:** `hipermotif-gnn-hpec2026.tex` (in repo root, tracked).

Key files to read first in a new session: `CLAUDE.md` (authoritative reference),
`hipermotif-gnn-hpec2026.tex` (paper + the "Claude → David" readme header),
`docs/superpowers/specs/2026-06-10-orbit-library-1a-design.md` and
`docs/superpowers/plans/2026-06-10-orbit-library-1a.md` (Phase 1a),
`src/datasets/orca_orbits.py` + `tests/test_orca_orbits.py` (the validated counter).

---

## 5. The paper-writing plan (IEEE HPEC 2026)

**File:** `hipermotif-gnn-hpec2026.tex`. **Title:** "Exact Substructure Features for
Expressive Graph Neural Networks at Scale with Parallel Subgraph Isomorphism."
**Authors:** Mohammad Dindoost and David A. Bader (NJIT). **Venue logistics:** ≤ 6
pages (refs excluded), IEEEtran two-column, **not anonymous**, submission
**Jul 7 2026** via CMT. **Style rule from David: no em dashes.**

**The draft is deliberately split into "real" vs. "must-measure":**

- **Already real and written** (do not re-derive, just refine):
  - Intro + Background: the 1-WL ceiling, GSN, DeSCo, HiPerMotif/Arachne/VF2-PS —
    real citations, correctly attributed (`thebibliography` at end of the .tex).
  - **Section III (the motivating expressivity results) — numbers are real and
    computed in this repo:** CSL 10% (chance) vs. exact features; Shrikhande/rook
    4-clique 8 vs. 0. This is the paper's strongest rhetorical asset and it is
    done.
  - Section IV (Method): the orbit-feature pipeline + Algorithm 1 (Aut/orbit
    normalization), anchored to the real Arachne API.
- **Marked `\todo` / red — require cluster runs and MUST NOT be fabricated**
  (the .tex header is explicit about this; honor it):
  1. HiPerMotif extraction **wall-clock + strong/weak scaling** vs. host size and
     thread count (the headline at-scale result).
  2. **GNN accuracy with vs. without exact features** on the expressivity and
     **large-graph** benchmarks (Table II is a stub).
  3. **Exact (HiPerMotif) vs. learned-approximate (DeSCo)** feature quality at
     matched cost.
  4. The **per-motif compute-vs-accuracy frontier**.
  5. Platform/repro details: Chapel version, Arachne/Arkouda commit, `CHPL_*`
     config; host-graph list with `|V|,|E|` (e.g. the H01 connectome ~147M edges);
     baseline counters (ESCAPE/PGD) if used; author confirmation.
  - **Hardware:** Wulver (NJIT), dual AMD EPYC 7713 (128 cores), 512 GB RAM;
    report mean of 5 runs with 95% CI.

**Critical writing rule (matches the repo's standing rigor hardline):** the red
`\todo` items are gated on real measurements. Section III numbers are real;
everything in Results that needs hardware must come from actual cluster runs.
**Never fabricate or "estimate" scaling/accuracy numbers.** Distinguish
theory-forced results (CSL is a theory-defined construct) from empirically
measured ones.

**Suggested path from here to a submittable paper:**
1. **Finish Phase 1b** so there is a *real* accuracy table (concat/orbit vs.
   baseline vs. rand-concat) on the standard benchmarks — this can populate Table
   II's left half honestly even before HPC.
2. **Pick the large host graph(s)** and get HiPerMotif running on Wulver to
   produce the scaling figure (Fig. 1) and the at-scale accuracy rows — the
   genuine HPEC contribution.
3. **DeSCo comparison** at matched cost (item 3) — the "exact beats approximate"
   argument.
4. Fill platform/repro `\todo`s; finalize `.bib`; verify page count ≤ 6.

---

## 6. Guardrails the next session must respect

These are hard rules (from `CLAUDE.md` §11 and the user's standing preferences):

- **Scientific rigor — never guess, fabricate, simplify away rigor, or randomly
  generate data/results; never propose a hypothesis without a theoretical or
  code-verified basis.** Read the actual code/output and cite it before any claim.
  Synthetic data (CSL) is allowed only as a labeled theory construct, never as
  real-world evidence. Evidence before assertions, always.
- **Git:** the user commits and pushes themselves; Claude stages specific files
  with `git add` (never `git add -A`/`.`, to avoid `data/`), and provides commit
  text. (One-time exceptions are granted explicitly when they occur.) `data/` and
  `third_party/orca/orca` are gitignored — never commit them.
- **HiPerMotif is deferred on purpose** — keep Arachne/Arkouda out of experiments;
  validate everything with local ORCA/igraph first; swap HiPerMotif in last.
- **Motif ID scheme is finalized:** undirected igraph IDs for the legacy library
  (0=degree, 2=wedge, 3=triangle); ORCA orbits live in the disjoint `2000+orbit`
  namespace. Do not mix directed/undirected IDs or split-seed results.
- **Naming:** the engine is **HiPerMotif** (not the old "HiPerXplorer").

---

## 7. TL;DR for a brand-new reader

We built a clean GNN+motif evaluation harness (Act I) that showed local triangle/
wedge features modestly help graph classification. David Bader's idea reframed it
into an HPC + expressivity contribution (Act II): use his group's **exact parallel
subgraph-isomorphism engine, HiPerMotif**, to compute **exact per-vertex orbit
features** that break the 1-WL ceiling, and demonstrate it on graphs far larger
than prior substructure-GNN work. We are de-risking by building the full pipeline
with a **local exact counter first**: Phase 0 proved the expressivity gap on CSL
and the Shrikhande/rook pair (done); Phase 1a vendored and **validated ORCA** as
the exact orbit counter (done, 33/33 tests); Phase 1b will run the benchmark
experiments; the **HiPerMotif backend swap is the final step**, with ORCA as its
correctness oracle. The **HPEC 2026 paper** (`hipermotif-gnn-hpec2026.tex`) is
drafted with the motivating expressivity results real and the at-scale results
clearly marked as pending honest cluster measurements (submission Jul 7 2026).
