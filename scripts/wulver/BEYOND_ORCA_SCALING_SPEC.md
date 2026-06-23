# Wulver runbook: beyond-ORCA long-cycle extraction at scale (best-paper experiments)

Owner: Bartosz (Wulver general partition, 128-core EPYC 7753, 512 GB). Everything below is
TURN-KEY: the cycle patterns, the `--patterns` flag, the correctness oracle, and the
strong-scaling `PATTERNS` pass-through are already built and locally tested (pure-python parts).
You fill only the SLURM header + arkouda server launch you already filled for the size-4 runs.

**Hardline:** report measured numbers only. If a pattern hits an unexpected cost wall on a host,
record it (the bench already writes a FAILED ceiling row) — do not hide it.

What was built this turn (all committed in the repo):
- `src/datasets/hipermotif_patterns.py`: added `c5,c6,c7,c8` to `PATTERNS` (|Aut(Cn)|=2n, single
  orbit, verified). `PATTERN_NAMES` pinned to the 9 size-4 graphlets so the orbit path is unchanged.
- `scripts/wulver/bench_orbits.py`: new `--patterns c6 c7 c8` flag (default = the 9 size-4).
- `scripts/wulver/run_strong_scaling.sh`: new `PATTERNS` env var passes through to `--patterns`.
- `scripts/wulver/cycle_oracle.py`: networkx oracle (induced AND all-simple Cn counts) — the
  beyond-ORCA correctness gate, since ORCA cannot validate C5+.

---

## STEP 0 (REQUIRED FIRST) — correctness gate, resolves induced-vs-all-simple
HiPerMotif's `ar.subgraph_isomorphism` is INDUCED (chordless). The local CSL accuracy demo used
ALL simple cycles. On a random-regular host these DIFFER (e.g. reg(200,6): induced C8 = 17622 vs
all-simple = 23519). This gate tells us which one HiPerMotif returns, on an identical graph
(`nx.random_regular_graph(6,200,seed=1)` is built the same way by both tools).

```
# (a) oracle — runs anywhere, no arkouda:
conda run -n motif-mpnn python scripts/wulver/cycle_oracle.py --graph reg --n 200 --d 6 --seed 1
# (b) HiPerMotif on the SAME graph (1 thread, 1 run):
python scripts/wulver/bench_orbits.py --backend hipermotif --graph reg --n 200 --m 6 --seed 1 \
  --patterns c5 c6 c7 c8 --threads 1 --runs 1 --out results/scale/gate_cycles.csv --no-mem
```
Check: for each Cn, `n_embeddings / (2*n)` from (b) must equal the **induced_Cn** column from (a)
(expected, induced engine) OR the **all-simple_Cn** column (would mean monomorphism). Report which.
This decides whether the paper's cycle feature is "induced Cn" or "all simple Cn" — both are
beyond ORCA; we just need to state the right one and re-run the local CSL demo to match.

## EXP-A (RANK 1) — strong scaling of C6/C7/C8 extraction
Cycles are bounded-degree-safe (won't hub-explode), so they scale where size-4 ran. Reuse the
strong-scaling runner three times (your server-relaunch TODO-2 is already filled):
```
PATTERNS="c6 c7 c8" NARG="--n 20000   --m 6" OUT=results/scale/cyc_strong_reg20k.csv  ./scripts/wulver/run_strong_scaling.sh
PATTERNS="c6 c7 c8" NARG="--n 200000  --m 6" OUT=results/scale/cyc_strong_reg200k.csv ./scripts/wulver/run_strong_scaling.sh
PATTERNS="c6 c7 c8" NARG="--n 1000000 --m 6" OUT=results/scale/cyc_strong_reg1M.csv   ./scripts/wulver/run_strong_scaling.sh
```
Plus one real-world point on roadNet-CA at 128 threads (stage the edge-file as for the size-4 run):
```
python scripts/wulver/bench_orbits.py --backend hipermotif --graph roadnetca \
  --edge-file <staged_roadNetCA_edges> --patterns c6 c7 c8 --threads 128 --runs 3 \
  --out results/scale/cyc_roadnetca.csv --no-mem
```
Deliver: the CSVs (same schema as size-4) — they drop straight into `make_scale_figures.py`.

## EXP-B (RANK 2) — one ~10^8-edge bounded-degree extraction (close the claimed-vs-shown gap)
A d=6 regular graph with ~33.3M nodes has ~10^8 undirected edges (m = n*d/2). Server at 128
threads, 512 GB. Run only the bounded-degree-safe patterns:
```
python scripts/wulver/bench_orbits.py --backend hipermotif --graph reg --n 33333333 --m 6 \
  --patterns triangle c4 c6 --threads 128 --runs 1 --out results/scale/bench_1e8.csv
```
Deliver: one TOTAL row with wall-time + peak memory at 10^8 edges. (Drop `--no-mem` here so we
get the memory number; if `get_mem_used` is missing, add `--no-mem` and report RAM separately.)

## EXP-C (RANK 3) — coverage + cost crossover (ORCA on the SAME node)
HiPerMotif C6/C7/C8 times come free from EXP-A. Add ORCA's size-4 time on the same node:
```
python scripts/wulver/bench_orbits.py --backend orca --graph reg --n 200000 --m 6 \
  --out results/scale/orca_size4_reg200k.csv
```
We (md724, local) assemble the table: ORCA/ESCAPE/PGD honestly WIN <=5-vertex (oracle), and are
"unsupported" for C6+ (ESCAPE/PGD coverage cited from their papers, not run); only HiPerMotif has
a C6/C7/C8 time. Message is COVERAGE, not a speed race.

## EXP-D (RANK 4, optional) — weak scaling
Pick host sizes so embeddings/core is ~constant across threads {1..128}; report near-flat time.
Largely a re-plot of EXP-A plus 1-2 fill-in points.

---

**Priority if compute is tight:** STEP 0 then EXP-A (the differentiator — makes HiPerMotif
necessary). EXP-B is the cheap systems-credibility close. EXP-C is nearly free. EXP-D is a bonus.
