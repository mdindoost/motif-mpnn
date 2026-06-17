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

## RUNBOOK — do these in order. The ONLY environment-specific action is starting your
## arkouda server (your usual `arkouda_server` launch). Everything else is exact commands.
Set once: `export REPO=$HOME/motif-mpnn; export PYTHONPATH=$REPO; cd $REPO; export GR=/scratch/$USER/hm_graphs`

### Step 1 — stage graphs (LOGIN NODE, has internet; compute nodes don't)
```bash
python scripts/wulver/stage_datasets.py --out $GR
```
This downloads OGB (ogbn-arxiv, ogbn-products) into `data/processed/ogb/` and the SNAP sparse hosts
(web-BerkStan, roadNet-CA) into `$GR`. Idempotent. (If you ever want a dataset that isn't here,
tell me and I'll add it to `stage_datasets.py` — don't hand-fetch.)

### Step 2 — EXP1: HiPerMotif capability ladder (128 threads)
Start your arkouda server with `CHPL_RT_NUM_THREADS_PER_LOCALE=128`; set `H`/`P` to its host/port.
```bash
H=<server_host>; P=5555; OUT=results/scale/bench.csv
for cmd in \
  "--graph cora" \
  "--graph ogbn-arxiv" \
  "--graph webberkstan --edge-file $GR/web-BerkStan.txt" \
  "--graph ogbn-products" \
  "--graph roadnetca --edge-file $GR/roadNet-CA.txt" ; do
  python scripts/wulver/bench_orbits.py --backend hipermotif $cmd \
    --threads 128 --runs 3 --out $OUT --ak-host $H --ak-port $P --no-mem
done
```
FAILED rows on the big graphs are EXPECTED (that's the ceiling) — the loop continues. Drop `--no-mem`
only if you want the memory column and `ak.get_mem_used()` works on your build.

### Step 3 — EXP2: ORCA crossover (no server; ORCA is local)
```bash
for cmd in "--graph cora" "--graph ogbn-arxiv" "--graph webberkstan --edge-file $GR/web-BerkStan.txt" \
           "--graph ogbn-products" "--graph roadnetca --edge-file $GR/roadNet-CA.txt" ; do
  python scripts/wulver/bench_orbits.py --backend orca $cmd --runs 3 --out $OUT
done
```

### Step 4 — EXP3: strong scaling (the headline HPC figure). Synthetic BA graph, no staging.
For each thread count: **restart your arkouda server** with that `CHPL_RT_NUM_THREADS_PER_LOCALE`,
update `H`, then run the one command:
```bash
for T in 1 2 4 8 16 32 64 128; do
  # >>> restart arkouda server with CHPL_RT_NUM_THREADS_PER_LOCALE=$T, set H=<host> <<<
  python scripts/wulver/bench_orbits.py --backend hipermotif --graph ba --n 150000 --m 6 \
    --threads $T --runs 3 --out $OUT --ak-host $H --ak-port $P --no-mem
done
```

(Alternative: `run_scale_experiments.sh` automates all of EXP1/2/3 — it only needs you to fill the
`start_server`/`stop_server` functions with your launch command. The runbook above is the manual
equivalent if you'd rather not edit bash.)

## Figures + table (run on Wulver or copy `bench.csv` back and run locally)
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

## You don't decide anything here
The graphs, order, thread sweep, runs, and feature path are all fixed in the runbook. The ONLY
thing that's yours is your normal `arkouda_server` launch command (with the given
`CHPL_RT_NUM_THREADS_PER_LOCALE`) — that's procedure, not a choice. If a dataset you need isn't
staged by `stage_datasets.py`, don't hand-fetch it — tell us and we'll add it to the script.

## Not in this harness (NEXT PRIORITY, if space/time)
PGD/ESCAPE head-to-head baseline; dense-social stress (Orkut/Friendster); the molecular accuracy
benchmark. `bench_orbits.py` is backend-agnostic so a `pgd`/`escape` backend is a small add later.
