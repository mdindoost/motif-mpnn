# EXP-D — HiPerMotif vs Glasgow Subgraph Solver (enumeration baseline)

**For: Bartosz (Wulver). Status: FINAL — knobs locked (§8), ready to run.**

## 1. Why this experiment (the framing — read first)
The paper currently has **no head-to-head baseline**; all three external reviews flagged this as
the gap that keeps it at weak-accept. ORCA/ESCAPE/PGD are *counters* (category-mismatched, and they
cannot do C6–C8), so they are NOT the baseline. The right comparator is a **general subgraph-isomorphism
solver in the same category as HiPerMotif**: the **Glasgow Subgraph Solver** (McCreesh et al.,
`mccreesh2020glasgow`), shared-memory parallel, supports arbitrary patterns including long cycles.

This is **calibration, NOT "we beat Glasgow."** The honest goals:
1. Show HiPerMotif is **competitive** with a state-of-the-art solver on the overlapping regime
   (exact induced enumeration/counting of small patterns).
2. Show both can do **beyond-graphlet** patterns (C6–C8) that the counters cannot — and where
   HiPerMotif's at-scale design pays off (large hosts where Glasgow times out).
3. **Bonus oracle:** Glasgow's induced solution count is an *independent* check on HiPerMotif's
   C6–C8 counts (ORCA cannot validate those).

**Outcome is genuinely unknown.** Glasgow is a very strong solver and may win on some small/dense
cases. We report whatever we measure. HiPerMotif's robust win is the **at-scale / timeout regime**
(large hosts where Glasgow does not finish). Do not tune to make HiPerMotif win; tune for fairness.

## 2. The task (must be identical for both tools)
**Count ALL induced embeddings** of each pattern H in host G.
- **INDUCED** matching on both sides (every pattern non-edge maps to a host non-edge). This matches
  what our orbit features require and what HiPerMotif uses (`reorder_type="None"`, induced).
- **Count-all** (enumerate every embedding / mapping), NOT "find first". This is the same search work
  both engines must do; it is the fair, I/O-free metric (don't print mappings — counting avoids
  output-size bias).

## 3. Fairness controls (a reviewer WILL scrutinize these)
- Same Wulver node (EPYC 7753), same core count, same 3-repeat protocol, same timeout.
- **Same timing scope:** wall-clock of the *solve only* (exclude graph loading / format parsing on
  both sides), to match what `bench_orbits.py` reports for HiPerMotif. State this in the CSV.
- Same host graph (identical edge set), undirected, simple (no self-loops/multi-edges).
- Same pattern definitions (see §5), induced on both.
- **Counting convention must be reconciled:** Glasgow `--count-solutions` counts *labeled* mappings.
  HiPerMotif `n_embeddings` = (induced copies) × |Aut(H)| = also labeled mappings. So
  **Glasgow count should EQUAL HiPerMotif `n_embeddings`** for each (H, G). Record both; a mismatch
  is either a convention bug or a correctness bug — flag, do not paper over (this is also the oracle
  cross-check in §1.3).

## 4. Glasgow invocation (confirm exact flags for the installed version)
Binary: `glasgow_subgraph_solver` (github.com/ciaranm/glasgow-subgraph-solver). Confirm the flag
names against `--help` on the build you install; conceptually we need:
- `--induced` (induced subgraph isomorphism)
- `--count-solutions` (enumerate/count all, not first)
- parallel/threads control (e.g. `--parallel` or a thread/threads flag) — match the HiPerMotif core
  count exactly; if Glasgow's parallelism is not a simple thread count, record what was set.
- pattern file, target file in Glasgow's input format (LAD / DIMACS / CSV — convert hosts once).
Record the FULL command line per run in the CSV `reason` field or a sidecar log.

## 5. Patterns
- **Size-4 overlap set** (same library the paper uses): triangle, p4 (4-path), claw, paw, c4 (4-cycle),
  diamond, k4 (4-clique). (p3/edge optional; they are cheap.)
- **Beyond-graphlet set:** C6, C7, C8 (induced).
Provide each as a Glasgow pattern file; keep the same pattern vertex set HiPerMotif uses.

## 6. Hosts — a scale LADDER (this IS the story; see §9)
HiPerMotif's design point is **large graphs**, so the comparison must climb to where that shows.
But keep the small head-to-head anchors too, or the large-scale result reads as cherry-picking the
regime where the competitor fails (see §9). Match patterns to what HiPerMotif itself completes at
each tier.
- **Calibration tier (both tools expected to FINISH — the fair head-to-head):**
  - `reg-20k`  (2e4 nodes, 6e4 edges) — full size-4 set + C6–C8
  - `reg-200k` (2e5, 6e5)             — full size-4 set + C6–C8
- **Scale tier (HiPerMotif's strength; Glasgow expected to TIMEOUT/OOM):**
  - `roadNet-CA` (1.97e6 nodes, 2.77e6 edges) — real-world; C6–C8 (we have HiPerMotif numbers) + triangle/C4/C6
  - `reg-1M`     (1e6, 3e6)                    — C6–C8 and/or triangle/C4/C6
- **Headline tier (10^8 edges — HiPerMotif's EXP-B host; Glasgow almost certainly cannot load/finish):**
  - `reg-33M` (3.33e7 nodes, 1e8 edges) — triangle, C4, C6 (the EXP-B set HiPerMotif completes in 18.8 min / 12 GB)

If Glasgow cannot even *load* a host (memory), record `OOM` — that is itself a fair, strong result
(HiPerMotif traverses 10^8 edges in 12 GB). Do not omit a tier because Glasgow fails on it; the
failure IS the measurement.

## 7. Output CSV (so it slots into our tooling / make_scale_figures style)
One row per (tool, host, pattern, threads, run):
```
tool,graph,n_nodes,n_edges,threads,pattern,run_idx,seconds,n_solutions,status,reason
```
- `tool` ∈ {glasgow, hipermotif}. Re-run HiPerMotif on the SAME hosts/patterns/threads here too
  (don't reuse old rows) so the comparison is apples-to-apples on one node/session.
- `status` ∈ {OK, TIMEOUT, OOM, ERROR}; `reason` = full command / notes.
- `n_solutions` = Glasgow count (labeled) / HiPerMotif `n_embeddings`. These two MUST match per (H,G).

## 8. Knobs — LOCKED (lead sign-off 2026-06-26)
- **Thread counts: 128 primary + 1-core on reg-20k and reg-200k** (serial head-to-head anchor;
  cheap, strengthens the story). No full sweep — 128 + 1 is enough for a baseline.
- **Timeout: 24 h wall per (tool, host, pattern)** (matches the Wulver job wall; record TIMEOUT
  rows — they are results).
- **Host set: the full LADDER (§6)** — reg-20k, reg-200k (anchors), roadNet-CA, reg-1M (scale),
  reg-33M / 10^8 (headline).
- **Pattern set: size-4 overlap + C6–C8** (as in §5).

**Node-hour note (sequencing, not a knob):** run the **calibration tier first** (cheap, the real
head-to-head). On the scale/headline tiers Glasgow DNFs are *expected and are the point* — but a 24 h
wall per cell can burn many node-days if Glasgow grinds. Once a Glasgow large-host cell is clearly
hopeless (already orders of magnitude past HiPerMotif's seconds with no end in sight), you may cut it
early and record `TIMEOUT` with the elapsed wall in `reason`; "did not finish in N h while HiPerMotif
did it in seconds" is already conclusive. Don't spend a full 24 h just to confirm the obvious.

## 9. What we will report (paper) — the narrative IS the ladder
A compact table in §VI: per (host, pattern), HiPerMotif vs Glasgow wall-clock @128c (+ 1-core if run),
with a "counts match" column (oracle cross-check) and TIMEOUT/OOM marked plainly.

**Both tiers must appear, in this order, or the argument fails:**
(a) **Calibration tier** — both tools finish; show we are competitive head-to-head (report honestly
    even if Glasgow wins some small/dense cases; "competitive" is enough here).
(b) **Scale + headline tiers** — HiPerMotif completes where Glasgow times out / OOMs, up to 10^8 edges.

The contrast (a)→(b) is the whole point. **Large-scale-only would read as cherry-picking the
competitor's failure regime**, so the small anchors are non-negotiable for credibility. The honest
headline is: *competitive with a state-of-the-art solver where both apply, and the only one of the
two that reaches HiPerMotif's target scale.*

## 10. First gate (cheap, do before the full matrix)
Run ONE small case both ways and confirm **Glasgow induced count == HiPerMotif n_embeddings**
(e.g. C6 on reg-20k, triangle on reg-20k). If counts disagree, stop and reconcile conventions
(induced? labeled vs up-to-symmetry?) before spending node-hours on the full matrix.
