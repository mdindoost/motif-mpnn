# Scale / Timing Results Ledger (HiPerMotif, Wulver)

**Purpose:** single source of truth for every HiPerMotif scaling/timing CSV and the
numbers extracted from them, so nothing is lost. Accuracy results live in
`results/all_runs.csv` + `results/findings.md`; this file is ONLY the systems/scale half.

**Hardware:** Wulver general partition, EPYC 7753, 128 cores, 512 GB. Runner: Bartosz.
All HiPerMotif rows use `reorder_type="None"` (correctness path), `--no-mem` (server_mem_gb=-1).
CSV schema (all files): `graph,n_nodes,n_edges,backend,threads_label,maxtaskpar_actual,pattern,run_idx,py_seconds,n_embeddings,server_mem_gb,status,reason`.

Last updated: 2026-06-26.

**Authoritative consolidated source (2026-06-26):** Bartosz's report PDF
`new numbers/beyond_orca_report.pdf` collects STEP 0, STEP 0b, EXP-A (incl. the full reg-1M
cycle sweep), EXP-B, EXP-C, and the reorder None-vs-Structural check. Numbers below that are
not yet backed by a CSV in the repo (reg-1M full cycle sweep; reorder check) are taken from
this report and flagged as such — request the backing CSVs from Bartosz to close the gap.

---

## A. CSV INVENTORY — what each file is, and whether it is canonical

### CANONICAL (use these)

| file | date | experiment | graphs | threads | patterns |
|------|------|-----------|--------|---------|----------|
| `new numbers/hm_scale_exp1_results-NEW - reg-1M.csv` | 06-23 | **EXP1 size-4 strong scaling (post merge-fix)** | reg-1M | 1→128 sweep | 9 size-4 orbits |
| `new numbers/hm_scale_exp1_results-NEW - reg-200k.csv` | 06-23 | EXP1 size-4 strong scaling | reg-200k | 1→128 | size-4 |
| `new numbers/hm_scale_exp1_results-NEW - reg-20k.csv` | 06-23 | EXP1 size-4 strong scaling | reg-20k | 1→128 | size-4 |
| `new numbers/hm_scale_exp1_results-NEW - ba.csv` | 06-23 | EXP1 size-4, scale-free **load-imbalance** case | ba | 1→128 | size-4 |
| `new numbers/hm_scale_exp1_results-NEW - cora.csv` | 06-23 | EXP1 size-4 single point | cora | 128 | size-4 |
| `new numbers/hm_scale_exp1_results-NEW - roadnetca.csv` | 06-23 | EXP1 size-4 real-world headline | roadNet-CA | 128 | size-4 (15 orbits) |
| `new numbers/hm_scale_exp1_results-NEW - ogbn-arxiv.csv` | 06-23 | EXP1 hub-explosion CEILING case | ogbn-arxiv | 128 | only edge/triangle complete |
| `new numbers/hm_scale_exp1_results-NEW - ogbn-products.csv` | 06-23 | EXP1 hub-explosion CEILING case | ogbn-products | 128 | only edge/triangle |
| `hm_scale_exp2_results - csl_cycles-STEP0b.csv` | 06-25 | **STEP 0b** HiPerMotif vs networkx induced cycles on 10 CSL graphs | csl_s* (×10) | 1 | c5–c8 |
| `new numbers/Exp-A/hm_scale_exp2_results - cyc_strong_reg20k-EXP_A.csv` | 06-25 | **EXP-A** cycle C6–C8 strong scaling | reg-20k | 1→128 | c6,c7,c8 |
| `new numbers/Exp-A/hm_scale_exp2_results - cyc_strong_reg200k-EXP_A.csv` | 06-25 | EXP-A cycle strong scaling | reg-200k | 1→128 | c6,c7,c8 |
| `new numbers/Exp-A/hm_scale_exp2_results - cyc_strong_reg1M-EXP_A.csv` | 06-25 | EXP-A cycle strong scaling — **STILL INCOMPLETE** (1t partial only: c6=3584s, c7=19831s, no c8). Full 1→128 sweep EXISTS in the report PDF but its CSV is **NOT in the repo** — request it from Bartosz to back tab:cycle-scale + regenerate Fig D | reg-1M | 1 only | c6,c7 |
| `new numbers/beyond_orca_report.pdf` | 06-26 | **Bartosz's consolidated report** — authoritative for reg-1M full cycle sweep + reorder None-vs-Structural | all | — | — |
| `new numbers/Exp-A/hm_scale_exp2_results - cyc_strong_reg1M-EXP_A-fromreport.csv` | 06-26 | reg-1M cycle full sweep **reconstructed from the report table** (TOTAL+per-pattern, 1c C8=TIMEOUT). Used to plot Fig D's reg-1M curve. Replace with Bartosz's raw per-run CSV if/when he sends it. | reg-1M | 1→128 (no 1c TOTAL) | c6,c7,c8 |
| `new numbers/Exp-A/hm_scale_exp2_results - cyc_roadnetca-EXP_A.csv` | 06-25 | EXP-A cycle real-world point | roadNet-CA | 128 (×3 runs) | c6,c7,c8 |
| `new numbers/Exp-A/hm_scale_exp2_results - reg_33M-EXP_B.csv` | 06-25 | **EXP-B** 10^8-edge bounded-degree extraction (with real mem) | reg-33.3M | 128 | triangle,c4,c6 |
| `new numbers/Exp-A/hm_scale_exp2_results - orca_size4_reg200k-EXP_C.csv` | 06-25 | **EXP-C** ORCA size-4 reference time (coverage crossover) | reg-200k | n/a (ORCA) | 15 size-4 orbits |

### SUPERSEDED / EARLY (keep for provenance, do NOT use for paper numbers)

| file | date | what it was | superseded by |
|------|------|-------------|---------------|
| `hm_scale_exp1_results - Sheet1.csv` | 06-17 | early EXP1 size-4, cora+roadnetca, 128t only (pre strong-scaling sweep) | `new numbers/...NEW` |
| `hm_scale_exp1_results - Sheet2.csv` | 06-18 | early EXP1 size-4, cora+roadnetca | `new numbers/...NEW` |
| `results/scale/bench_cora.csv` | 06-19 | local-staged size-4 EXP1 (cora) that fed figA/B/C | `new numbers/...NEW` |
| `results/scale/bench_roadnetca.csv` | 06-19 | local-staged size-4 EXP1 (roadNet-CA) | `new numbers/...NEW` |
| `results/scale/bench_strong.csv` | 06-19 | BA strong-scaling, **pre merge-fix** (capped at ~8 cores) | `new numbers/...NEW - ba.csv` |

### FIGURES / DERIVED (in `results/scale/`)
`figA_strong_scaling.{png,pdf}`, `figB_size_scaling.{png,pdf}`, `figC_per_pattern.{png,pdf}`,
`table_scale.tex` — generated 06-23 by `make_scale_figures.py` from the PRE-cycle (size-4) data.
**Must be regenerated** once EXP-A/B/C cycle CSVs are wired in.

### NOT scale data (ignore for this ledger)
`results/all_runs.csv` (accuracy, canonical 11-col), `results/all_runs_legacy_malformed.csv`
(archived bad rows), `results/logs/*/metrics.csv` (per-epoch training logs, already aggregated
into all_runs.csv), `results/synthetic_motif_task.csv` (rook/Shrikhande accuracy task),
`results/tables/planetoid_results_seed42.csv`.

---

## B. VERIFIED NUMBERS

### STEP 0 — induced vs all-simple gate (2026-06-25)
**Result: HiPerMotif `subgraph_isomorphism` is INDUCED (chordless).** So the paper's cycle
feature is "induced Cn". Reported by Bartosz on `reg(n=200,d=6,seed=1)`, n_embeddings = count×2n:
| Cn | n_emb | ÷2n | induced oracle | all-simple oracle |
|----|-------|-----|----------------|-------------------|
| C5 | 2790    | 279   | 279 ✓   | 301   |
| C6 | 13524   | 1127  | 1127 ✓  | 1282  |
| C7 | 62048   | 4432  | 4432 ✓  | 5461  |
| C8 | 281952  | 17622 | 17622 ✓ | 23519 |
÷2n matches the INDUCED oracle exactly on all four. Induced/all-simple gap grows ~7% (C5) → ~25%
(C8): the distinction matters for the count, but BOTH still separate CSL 10/10 (STEP 0b + the
local all-simple=1.00 check).

### STEP 0b — HiPerMotif reproduces CSL-separating induced cycles (2026-06-25) — PASS
Independently re-verified by me (`scratchpad/check0b.py` vs
`data/precompute/csl_stage/csl_cycle_reference.json`): **all 40 checks (10 CSL graphs × C5–C8)
match `induced × 2n` exactly**, and rows like csl_s11/s12 C8 actively rule out all-simple
(656 = induced 41×16, not all-simple 287×16; 5248 not 574×16). → the C6–C8 cycles that complete
CSL 10/10 separation are HiPerMotif's OWN output, not networkx's. (C3/C4 not run via HiPerMotif —
they are ORCA-coverable so HiPerMotif is not the necessary tool there.)

Local accuracy chain it backs (`scripts/expressivity/csl_beyond_orca.py`, CSL 10-way linear 5-fold CV):
legacy-3 = 0.20, ORCA-15 = 0.30, ORCA-73 (max ORCA) = 0.50, cycles C3–C4 = 0.30,
**cycles C3–C8 induced = 1.00** (= all-simple 1.00). → ORCA's best (5/10) is completed to 10/10 by induced C6–C8.

### EXP-A — cycle C6/C7/C8 strong scaling (2026-06-25)
Counts are **thread-invariant** (correctness preserved by the per-task merge fix): reg-20k
c6/c7/c8 = 15900/77728/386000 at every thread count; reg-200k = 15900/80612/390512.
TOTAL = c6+c7+c8 wall-time. Speedup vs 1 thread; runs averaged where >1.

**reg-20k (20000 nodes, 60000 edges):**
| thr | mean s | speedup | eff% |
|-----|--------|---------|------|
| 1   | 2683.1 | 1.0×   | 100  |
| 2   | 1158.0 | 2.3×   | 116  |
| 4   | 580.1  | 4.6×   | 116  |
| 8   | 289.4  | 9.3×   | 116  |
| 16  | 168.0  | 16.0×  | 100  |
| 32  | 73.1   | 36.7×  | 115  |
| 64  | 44.4   | 60.4×  | 94   |
| 128 | 22.7   | **118.1×** | 92 |

**reg-200k (200000 nodes, 600000 edges):**
| thr | mean s | speedup | eff% |
|-----|--------|---------|------|
| 1   | 26426.9 | 1.0×  | 100  |
| 2   | 11360.1 | 2.3×  | 116  |
| 4   | 5738.7  | 4.6×  | 115  |
| 8   | 2840.5  | 9.3×  | 116  |
| 16  | 1429.7  | 18.5× | 116  |
| 32  | 720.3   | 36.7× | 115  |
| 64  | 438.1   | 60.3× | 94   |
| 128 | 227.4   | **116.2×** | 91 |

**reg-1M (1000000 nodes, 3000000 edges): FULL SWEEP DONE (Bartosz report, 2026-06-26).**
From `beyond_orca_report.pdf` (CSV NOT yet in repo — request it). TOTAL = c6+c7+c8:
| thr | C6 s | C7 s | C8 s | TOTAL s |
|-----|------|------|------|---------|
| 1   | 3561.4 | 19877.2 | timeout (>24h) | — |
| 2   | 1529.1 | 8517.3  | 46833.5 | 56879.9 |
| 4   | 783.4  | 4320.8  | 23457.6 | 28561.8 |
| 8   | 395.4  | 2166.3  | 11790.0 | 14351.7 |
| 16  | 195.4  | 1072.9  | 5860.3  | 7098.3  |
| 32  | 98.7   | 550.9   | 3065.0  | 3714.6  |
| 64  | 57.6   | 325.4   | 1756.5  | 2139.5  |
| 128 | 38.4   | 162.8   | 897.3   | **1098.6** |
Speedup **2→128 = 51.8×** (56879.9/1098.6, 81% eff over 64× cores; this is the PAPER baseline);
8→128 = 13.1×. 1-core C8 exceeded the 24h job limit, so 2 cores is the smallest baseline feasible
on every host. The earlier 1-thread partial CSV (`...cyc_strong_reg1M-EXP_A.csv`: c6=3584s,
c7=19831s — the c7≈5.5h figure independently backs the paper's "C7 alone exceeds five hours")
is the only RAW repo-resident reg-1M cycle data; the full sweep lives in the report PDF (and the
`-fromreport` reconstructed CSV).
**Paper:** tab:cycle-scale reg-1M row FILLED (56879.9 / 1098.6 / 52×, 2-core baseline);
Fig D reg-1M curve DONE.

**roadNet-CA (1965206 nodes, 2766607 edges) — real-world headline, 128t, 3 runs:**
TOTAL mean = **45.5 s** (runs 47.9 / 44.1 / 44.5). Embeddings: c6 = 1307760, c7 = 1412586,
c8 = 2360992 (total 5.08M). → induced C6–C8 on a 2M-node road network in ~45 s @128t.

**Headline takeaways:** cycle strong scaling (~116–118× @128t, 91–92% eff, monotonic) is
**better than the size-4 patterns** (50×/80×/68×) because cycles are bounded-degree, uniform
work with no claw/p4 hub load-imbalance. c8 dominates cost (c8 ≈ 5×c7 ≈ 25×c6).

### EXP-B — 10^8-edge bounded-degree extraction (2026-06-25) — closes claimed-vs-shown scale gap
`reg`, **33,333,333 nodes / 99,999,999 edges (~10^8)**, 128 threads, 1 run, memory recorded:
| pattern | time | embeddings |
|---------|------|-----------|
| triangle | 101.5 s | 114 |
| c4 | 43.1 s | 560 |
| c6 | 982.8 s | 15912 |
| **TOTAL** | **1127.4 s ≈ 18.8 min** | 16586 |
**Peak server mem = 12 GB** (drop-in-the-bucket of the 512 GB node). TOTAL = exact sum of patterns.
Low counts are expected (random 6-regular is locally tree-like → few short cycles); the result is
SCALE/time/memory, not the count. Only triangle/c4/c6 run (the bounded-degree-safe set).

### EXP-C — ORCA size-4 reference + coverage/cost crossover (2026-06-25)
ORCA size-4 on reg-200k, 3 runs: **mean 1.955 s** (1.982/1.952/1.932), **0.807 GB**, 86,195,908
orbit counts. Crossover table (vs EXP-A HiPerMotif cycles on the SAME reg-200k @128t = 227.4 s TOTAL):
| engine | size-4 (<=5-vertex) reg-200k | C6–C8 reg-200k |
|--------|------------------------------|----------------|
| ORCA | **1.96 s** (oracle, wins) | **unsupported** (caps at 5 vertices) |
| HiPerMotif | ~200× slower (enumerates instances) | **227 s @128t** (only engine that can) |
Message = COVERAGE, not a speed race. ORCA honestly wins where it applies; HiPerMotif is the only
tool that runs above the size-5 wall.

### Reorder None vs Structural — reg-200k C6–C8 @128t (2026-06-26) — resolves [O-B]
From `beyond_orca_report.pdf` (CSV not in repo). 3 runs each:
| reorder_type | run0 | run1 | run2 | mean s |
|--------------|------|------|------|--------|
| None       | 227.6 | 222.0 | 232.5 | **227.4** |
| Structural | 227.1 | 222.6 | 290.8 | 246.8 |
Structural ≈ None; the 246.8 mean is inflated by a single run-2 C8 spike (runs 0/1 within 0.5 s
of None). → **Confirms reporting the None (correctness-path) numbers does NOT understate cycle
performance** — structural reordering is a no-op for unlabeled, vertex-transitive cycle patterns
(nothing to reorder). This is the empirical close of [O-B]; the paper already removed all
reorder_type mention, so no paper edit needed, just provenance.

### EXP1 size-4 strong scaling (post merge-fix) — already in CLAUDE.md §14
reg-1M 50×, reg-200k 80×, reg-20k 68× @128t; roadNet-CA 15 orbits in 12.3 s @128t; BA = load-
imbalance case (claw on hubs). Source: `new numbers/hm_scale_exp1_results-NEW - *.csv`. Not
re-tabulated here to avoid drift — CLAUDE.md §14 is authoritative for size-4.

---

### EXP-D — HiPerMotif vs Glasgow Subgraph Solver (enumeration baseline) — SPEC'D, awaiting Bartosz
**Status (2026-06-27): spec FINAL, not yet run.** Spec: `scripts/wulver/GLASGOW_BASELINE_SPEC.md`.
This is **strong-accept lever #1** — the head-to-head baseline all three external reviews flagged as
the gap keeping the paper at weak-accept. Comparator = Glasgow Subgraph Solver (`mccreesh2020glasgow`),
a general SI solver in HiPerMotif's category (NOT ORCA/ESCAPE/PGD, which are counters).
- **Task:** count ALL induced embeddings, both tools, solve-only wall-clock.
- **Hosts (scale ladder):** calibration reg-20k/reg-200k (both finish) → scale roadNet-CA/reg-1M
  (Glasgow expected TIMEOUT/OOM) → headline reg-33M/10^8 edges (HiPerMotif's EXP-B host).
- **Patterns:** size-4 overlap (triangle,p4,claw,paw,c4,diamond,k4) + C6,C7,C8.
- **Knobs LOCKED (lead, 2026-06-26):** 128 threads primary + 1-core on reg-20k/reg-200k; **24 h**
  wall timeout; full ladder; cheap first gate = Glasgow induced count == HiPerMotif n_embeddings.
- **Oracle bonus:** Glasgow's induced count independently validates HiPerMotif's C6–C8 counts.
- **Output CSV (NEW, when Bartosz runs):** `tool,graph,n_nodes,n_edges,threads,pattern,run_idx,seconds,n_solutions,status,reason` — re-run HiPerMotif on the same hosts/patterns/threads in the same session (don't reuse old rows).

## C. PENDING / GAPS

- **reg-1M cycle TABLE + FIGURE both DONE (2026-06-26).** tab:cycle-scale + Fig D use a
  **2-core baseline** (NOT 8 — corrected after the lead flagged that 8 silently discarded the
  measured 2c/4c points). Rationale: single-core C6–C8 does not complete on reg-1M (1c C8
  timed out >24h), so 2 cores is the smallest count feasible on EVERY host; the superlinear
  cache artifact is only the 1→2 step, so 2→128 is near-linear. Table: reg-20k 1158.0→22.7
  (**51×**), reg-200k 11360.1→227.4 (**50×**), reg-1M 56879.9→1098.6 (**52×**), all ~80% eff
  over 2→128. Fig D = all three host curves, 2→128. `make_cycle_figures.py` gained
  `--strong-baseline N`. reg-1M points from the `-fromreport` CSV above. Nice-to-have only:
  Bartosz's raw per-run reg-1M cycle CSV. Regen cmd: `make_cycle_figures.py --strong <reg20k>
  <reg200k> <reg1M-fromreport> --strong-baseline 2 --out Paper --also-out results/scale`.
- **EXP-A reg-1M** — DONE (report 2026-06-26): full 1→128 sweep, 13.1× @8→128. See Section B.
- **EXP-B** — DONE (2026-06-25): 10^8 edges, 18.8 min @128t, 12 GB. See Section B.
- **EXP-C** — DONE (2026-06-25): ORCA 1.96 s reg-200k; crossover table assembled. See Section B.
- **Reorder None-vs-Structural [O-B]** — DONE (report 2026-06-26): None 227.4 ≈ Structural 246.8
  (one C8 spike); reporting None does not understate perf. See Section B.
- **all-simple CSL claim** — VERIFIED (2026-06-26, `results/expressivity_csl_beyond_orca.json`):
  induced AND all-simple C3–C8 both = 1.00 (10/10); paper sentence is grounded.
- **Figures**: BOTH now academic IEEE style via `scripts/wulver/make_cycle_figures.py` —
  `figD_cycle_scaling` (cycles, reg-20k/200k) AND `figA_strong_scaling` REGENERATED (size-4,
  reg-20k/200k/1M, `--size4` mode; replaced the old plain twin-axis version; matches tab:scale 68/80/50x).
  Both in `Paper/` + `results/scale/`. OVERLEAF: re-upload BOTH figA + figD PNGs.
- **MERGED into `Paper/main-new.tex` (2026-06-25)**: §III CSL paragraph replaced (distinct-signatures +
  spectral + O-G framing); new §VI-D "Beyond size-5 graphlets" subsection with cycle prose, the cycle
  strong-scaling Table (`tab:cycle-scale`), Fig D (`fig:cycle-scaling`), capacity para (EXP-B), coverage
  Table (`tab:coverage`). Compiles clean, 7pp, 0 undefined cites. `beyond_orca_draft.tex` now redundant
  (kept as reference). **3 red `\todo` placeholders pending Bartosz**: line ~569 structural-vs-None
  timing (O-B); line ~606 reg-1M table row (8c/128c/speedup); line ~613 reg-1M 3rd curve in Fig D.

### Full-paper review (HPC + GNN sub-agents, 2026-06-25)
Verdicts: HPC = **weak accept → clear path to accept**; GNN = **accept (lean-clear)**.
DONE this round: removed all `reorder_type` mention (kills HPC CRITICAL-1 method/results
contradiction AND the O-B pending item); justified the |Aut| divisor in §IV (+K4 calibration,
GNN's #1); named tab:acc feature set as induced C3-C8 not the 15-orbit library (both reviewers).
REMAINING (see "Review action list" below).

### Fresh-eyes integrity audit (2 NEW unbiased reviewers + my CSV cross-check, 2026-06-25)
My audit: EVERY table/figure number reconciles EXACTLY with source CSVs (68/80/50x size-4;
12.8/12.5x cycles; roadNet 45.5s; EXP-B 18.8min/12GB; ORCA 1.96s). No fabricated numbers.
6 real issues found by fresh reviewers, ALL FIXED:
1. [my error] §VI-C said size-4 8->128 eff "~80%" — FALSE (that's the CYCLE number; size-4 is 35-55%).
   Corrected + honest non-monotonicity story (1->8 ~9x all hosts; mid-size reg-200k peaks at 80x;
   reg-1M 50x = worse high-core scaling on bigger working set, NOT a baseline artifact).
2. Coverage table: HiPerMotif size-5 + labeled/directed were undemonstrated -> marked with dagger
   "supported by engine, not measured here" (size-4 + C6-C8 ARE demonstrated).
3. "validated throughout against ORCA" overclaim (intro+discussion) -> scoped: ORCA for size-4,
   independent enumerator for C6-C8.
4. BA "host of the same size" -> "of comparable edge count" (BA 10k/50k vs reg-20k 20k/60k).
5. chordless-cycle sentence imprecise -> "every CONNECTED <(k) induced subgraph is a path".
6. morris2019weisfeiler mis-cited as TU-dataset source -> swapped to morris2020tudataset (added to
   refs.bib; user may replace w/ Google-Scholar bibtex). morris2019 correctly KEPT for 1-WL (3 uses).
Fresh verdicts: HPC "accept w/ required revisions (now done)"; GNN "sound". PAGE COUNT now 8 (was 7) -
   watch HPEC limit, esp. once abstract+conclusion added.

### Review action list (post first reviews) — DONE 2026-06-25
- DONE: scaling-table Option A (kept 1-core 50x/80x headlines + added superlinear caveat, 8->128 ~80%
  regime, non-monotonicity-is-not-regression explanation, per-task merge-lock-fix sentence, runs-per-point note).
- CPU SKU: user CONFIRMED "EPYC 7753" is correct (on the NJIT Wulver website) — NOT a typo, leave as-is.
- DONE batch: tab:phase1b caption (unrounded-means + "not a significance test at n=3"); EXP-B clause
  "C6 representative, C7/C8 omitted for cost at 10^8"; arxiv clause (degree=orbit0 + random control rules out capacity).
- STILL OPEN (small, when ready): Glasgow/VF3 single size-4 anchor (optional); move tab:acc out of
  scaling subsec (cosmetic); add figure error bars (roadNet has 3 runs) if desired.
- ABSTRACT (when written): "at scale" = feasibility ONLY, never accuracy-at-scale; the commented
  "99% under 2% noise" clause is unsupported in body (we have 100% at 2% noise) -> fix or drop.

### Superseded expert open items (pre-merge, kept for history)
- **[O-A] External serial baseline** (HPC C2) — **RESOLVED/DOWNGRADED (HPC expert + lead, 2026-06-25)**.
  Scaling claim defended INTRINSICALLY: same engine binary at every thread count (only worker-thread
  count varies, no separate serial version) + bit-identical counts + 8→128c efficiency. No external
  baseline needed; networkx race is OFF-NARRATIVE and OUT. OPTIONAL belt-and-suspenders: ONE size-4
  enumerator-to-enumerator point vs Glasgow/VF3 folded into the coverage table (NOT PGD/ESCAPE = counters).
  Private local sanity check: 1-core HiPerMotif (2683s reg-20k) is FASTER than serial networkx
  (~4000s extrapolated; networkx reg-2000=70s, reg-5000>120s) → baseline not padded. networkx also
  independently cross-validated the cycle counts (reg-200 induced C8=17622 == STEP 0 spec; reg-2000
  induced C8=23588 ≈ reg-20k bench emb/16=24125, confirming counts ~n-independent).
- **[O-B] Timing used reorder_type="None"** (HPC C3) — **DOWNGRADED from CRITICAL (2026-06-25)**.
  Verified in `SubgraphSearch (2).chpl`: reorderType branches ONLY at setup (~1049-1086); the
  search recursion + result assembly (incl. the merge fix) are downstream and identical for both
  modes. So the merge fix applies to None AND structural. structural's speedup is label/degree-driven
  search pruning; cycle patterns are unlabeled + vertex-transitive (nothing to reorder) → None ~=
  structural here. Resolution: get ONE None-vs-structural wall-time on reg-200k C6-C8 @128t; if the
  gap is small, report None (it has the identity-mapper correctness guarantee) and state structural
  is a no-op for unlabeled patterns. NOT a re-run-everything item. (Bonus: structural+labels is the
  regime ORCA/ESCAPE/PGD can't touch at all — strengthen the coverage discussion with it.)
- **[O-C] Honest efficiency framing** (HPC C4, done in draft). 118x is inflated by a superlinear
  (cache-bound) 1-core baseline; report ~80% efficiency over 8→128c. Draft already does this.
- **[O-D] "ORCA-verified" overclaim** (HPC C1, fixed in draft). ORCA can't validate C6+; C6–C8 are
  validated vs networkx (STEP 0b). Draft says this; make sure final text doesn't say "ORCA-verified" for cycles.
- **[O-E] Variance/std** (HPC S5). Only 16–128c have 3 runs; 1–8c single-run; roadNet has 3. Add error
  bars where available; explain the reg-20k 16c efficiency dip (likely CCD/thread-placement).
- **[O-F] Spectral-baseline + novelty pre-emption** (GNN C1/C2, fixed in draft §III). Lead with
  distinct-signatures ceiling; cite murphy/chen/bouritsas; novelty = coverage-boundary + at-scale.
- **[O-G] Synthetic-only differentiator** (GNN S3) — **ADDRESSED in draft (2026-06-25)**. CSL reframed
  as a controlled probe (not sole evidence); §III paragraph cites GSN \cite{bouritsas2022improving},
  CW Networks/rings \cite{bodnar2021weisfeiler}, LRGB \cite{dwivedi2022long} for real-task ring utility.
  Both new keys added to refs.bib + compile-clean. rook/Shrikhande kept OUT of the beyond-5 argument
  (it's size-4, ORCA-visible). corso2020principal (PNA) is NOT relevant — not used.

---

## D. RERUN ASK FOR BARTOSZ (reg-1M cycles, option 2)
See the message in the conversation; the gist: rerun reg-1M c6/c7/c8 starting at 8 threads
(skip the ~28 h single-thread c8 baseline), threads {8,16,32,64,128}, report speedup vs 8t.

---

## E. SESSION HANDOFF 2026-06-25 (paper drafting state)
Paper is DRAFTED end-to-end and compiles at **8 pages** (`Paper/main-new.tex`): Abstract + Conclusion
both ACTIVE; beyond-ORCA story MERGED into main-new.tex (beyond_orca_draft.tex now redundant). All
expert + fresh-eyes integrity reviews done and ALL fixes applied (reorder_type fully REMOVED; size-4
8->128 efficiency corrected to ~35-55%, NOT 80% which is the cycle number; coverage table daggers
undemonstrated HiPerMotif size-5/labeled cells; ORCA-scope language fixed; morris2020tudataset added
for TU datasets). §II/§III/§V/Discussion all tightened.

**0 red \todo REMAIN** (2026-06-26): the beyond-ORCA cycle story is COMPLETE in the paper.
tab:cycle-scale reg-1M row FILLED (**2-core baseline**: 56879.9 / 1098.6 / **52×**) and Fig D
regenerated with all three host curves (2-core baseline, matching the table — corrected from the
earlier 8-core draft after the lead flagged that 8 silently dropped the measured 2c/4c points).
Paper builds clean, no overfull. Bartosz's raw per-run reg-1M cycle CSV is the only nice-to-have
left (cosmetic provenance).

**IN-PROGRESS, NOT YET APPLIED: tighten §VI Results prose** (cut ONLY redundant). Specific §VI-B arxiv
cuts identified: "diagnostic, not real-world accuracy" stated ~3x -> keep once + caption; drop the
"same capacity-independent effect... now confirmed on 1.7e5 nodes" cross-link; drop explicit "+0.195
gap" number (keep "0.407 vs 0.213", avoids 0.194/0.195 rounding nit). Then ranked: inline tab:acc
1-row CSL table as a sentence, light §VI prose, §I opening sentence.

**Overleaf re-upload:** main-new.tex, refs.bib, figA_strong_scaling.png, figD_cycle_scaling.png.
