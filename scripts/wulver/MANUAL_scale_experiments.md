# HiPerMotif scale experiments — run manual (Wulver)

This runs the **mandatory 6-page HPEC spine**: HiPerMotif orbit-extraction scaling on **one
shared-memory node** (HiPerMotif is shared-memory; not distributed), with ORCA as the
single-threaded crossover reference. Everything writes raw CSV; figures are pure groupings.

Scripts (all in `scripts/wulver/`):
- `bench_orbits.py` — times extraction for ONE graph; appends CSV rows.
- `run_scale_experiments.sh` — driver for EXP1/EXP2/EXP3.
- `make_scale_figures.py` — Fig A/B/C + LaTeX table from the CSV (run anywhere).

`reorder_type="None"` is used everywhere (the validated-correct path). Correctness itself is
the small-graph gate `verify_hipermotif_equals_orca.py` (run once, separately) — we do **not**
re-verify on the big graphs.

---

## 0. Prerequisites
- `git pull origin expressivity-demo` (gets these scripts).
- Arachne rebuilt with the `appendBlock` fix (the parallel-race fix).
- conda env with arkouda + arachne + torch_geometric + networkx + (optional) `ogb`; `g++` for ORCA.
- Whole node, exclusive, all memory (`--exclusive --mem=0`) — shared-memory scaling needs all cores+RAM.

## 1. Stage the graphs (NO internet on compute nodes)
Loading is excluded from timing, but the data must be on disk first:
- **cora** — already in `data/processed/` (validation point).
- **ogbn-arxiv / ogbn-products** — pre-download with `ogb` on a login node into
  `data/processed/ogb/` (the loader reads from there; it will NOT download on a compute node).
- **SNAP sparse hosts** (recommended for the headline: a road network / web graph) — download the
  edge-list (`.txt`, `u v` per line, `#` comments ok) to `/scratch/$USER/` and pass via
  `--edge-file`. (Dense social graphs Orkut/Friendster are NEXT-PRIORITY stress, not the spine.)

## 2. Fill the three TODO blocks in `run_scale_experiments.sh`
The script is ready except for cluster-specific bits (marked `TODO-1/2/3`):
- **TODO-1**: SLURM header (or delete if interactive).
- **TODO-2**: `start_server <threads>` / `stop_server` — how you launch the arkouda server with
  `CHPL_RT_NUM_THREADS_PER_LOCALE=<threads>` and how you set `AK_HOST`. **This is the key bit:**
  the HiPerMotif thread count is fixed at server launch (it shows up as `maxTaskPar` in
  `ak.get_config()`), so the strong-scaling sweep **relaunches the server at each thread count**.
- **TODO-3**: the graph list (`SPARSE_LADDER`, in increasing size order) and `--edge-file` paths.

## 3. Run
```bash
REPO=$HOME/motif-mpnn OUTDIR=results/scale RUNS=3 AK_PORT=5555 \
  bash scripts/wulver/run_scale_experiments.sh
```
What it does:
- **EXP1** — HiPerMotif on the ladder at 128 threads (capability + ceiling). It continues past a
  graph that OOMs and records a `status=FAILED` row — **that's the point**, we want the ceiling.
- **EXP2** — ORCA on the same ladder bottom-up until ORCA runs out of time/memory (the crossover).
- **EXP3** — HiPerMotif strong scaling: threads ∈ {1,2,4,8,16,32,64,128}, **relaunching the
  server each time**. This is the headline HPC figure.

All rows go to `results/scale/bench.csv`.

### Running one graph manually (handy for testing / a single point)
```bash
PYTHONPATH=$REPO python scripts/wulver/bench_orbits.py \
  --backend hipermotif --graph ogbn-arxiv --threads 128 --runs 3 \
  --out results/scale/bench.csv --ak-host <server_host> --ak-port 5555
# arbitrary staged edge-list:
PYTHONPATH=$REPO python scripts/wulver/bench_orbits.py \
  --backend hipermotif --graph roadnet --edge-file /scratch/$USER/roadNet-CA.txt \
  --threads 128 --runs 3 --out results/scale/bench.csv --ak-host <server_host> --ak-port 5555
```

## 4. Figures + table (run on Wulver or copy the CSV back and run locally)
```bash
python scripts/wulver/make_scale_figures.py --csv results/scale/bench.csv \
  --out results/scale --strong-graph ogbn-products
```
Produces `figA_strong_scaling`, `figB_size_scaling`, `figC_per_pattern` (PNG+PDF) and
`table_scale.tex`.

## 5. Send back
- `results/scale/bench.csv` (the raw data — everything else is derived).
- The three figures + `table_scale.tex`.
- The `server_*.log` files if anything OOMed (to see where/why the ceiling hit).

---

## Notes / gotchas
- **Thread count is a server property.** `bench_orbits.py --threads N` is only a *label*; it reads
  `maxTaskPar` from the running server and **warns** if they differ. The driver enforces N by
  relaunching the server. If Fig A looks wrong, check the `maxtaskpar_actual` column actually
  varied across the EXP3 rows.
- **Memory is server-side.** `server_mem_gb` comes from arkouda (`ak.get_mem_used()`). If your
  arkouda build exposes it under a different name, tell me and I'll adjust `server_mem_gb()` in
  `bench_orbits.py` (it returns −1 if it can't find the API — not fatal, just no memory column).
- **We don't pull embeddings to the client.** The bench reads `result[0].size` (a pdarray
  attribute), never `.to_ndarray()`, so a billion-embedding result won't OOM the *client*; the
  *server* still materializes it (that's the memory we measure and the real ceiling).
- **`py_seconds` ≈ engine time.** The `subgraph_isomorphism` call is synchronous on the server and
  we don't transfer the array, so the Python-side timer is dominated by engine compute.
- **FAILED rows are expected at the top of the ladder** — that's the capability ceiling, a result.
- **Cost axis is work, not |E|.** Fig B plots time vs total #embeddings (the real cost driver),
  with |V|,|E| in the table.

## Not in this harness (NEXT PRIORITY, if space/time)
PGD/ESCAPE head-to-head baseline; dense-social stress (Orkut/Friendster); the at-scale accuracy
demo (synthetic motif-defined task + molecular). `bench_orbits.py` is backend-agnostic so a
`pgd`/`escape` backend is a small add when we get there.

## What I still need from you (to finalize the driver)
1. How you launch the arkouda server (interactive vs sbatch; server binary path; module loads) and
   how you obtain its host — so `start_server`/`stop_server` can be filled exactly.
2. Confirm `ak.get_mem_used()` exists in your arkouda build (else the memory column is −1).
3. Which sparse large host(s) you'll stage for the headline (road/web/citation) + their paths.
