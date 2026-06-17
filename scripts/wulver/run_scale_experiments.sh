#!/usr/bin/env bash
# =============================================================================
# HiPerMotif scale-experiment driver (shared-memory, ONE Wulver node).
# Runs the MANDATORY 6-page spine: EXP1 capability ladder, EXP2 ORCA crossover,
# EXP3 strong-scaling thread sweep. PGD/ESCAPE + dense-host + accuracy are NEXT
# PRIORITY (not here). See MANUAL_scale_experiments.md.
#
# >>> BARTOSZ: fill the three TODO blocks (SLURM header, server launch, paths). <<<
# Everything else is ready. All timing goes to CSV under $OUTDIR.
# =============================================================================
set -u  # (no -e: we WANT to continue past a graph that OOMs — that's the ceiling)

# ---- TODO-1: SLURM header (if using sbatch). Delete if running interactively. ----
# #SBATCH --job-name=hipermotif-scale
# #SBATCH --nodes=1
# #SBATCH --exclusive               # whole node: shared-memory scaling needs all cores+RAM
# #SBATCH --mem=0                    # all node memory (~500GB)
# #SBATCH --time=24:00:00
# #SBATCH --partition=<your_partition>

REPO="${REPO:-$HOME/motif-mpnn}"
cd "$REPO"
export PYTHONPATH="$REPO"
OUTDIR="${OUTDIR:-results/scale}"
RUNS="${RUNS:-3}"
AK_PORT="${AK_PORT:-5555}"
mkdir -p "$OUTDIR"

# ---- TODO-2: arkouda server launch/relaunch. HiPerMotif thread count is fixed at
#      server start via CHPL_RT_NUM_THREADS_PER_LOCALE. Implement these two for YOUR
#      cluster (module loads, server binary path, how you get the server host). ----
AK_HOST="localhost"   # set to the server's node hostname after launch
start_server() {       # $1 = threads-per-locale
  local threads="$1"
  echo "[driver] starting arkouda server with CHPL_RT_NUM_THREADS_PER_LOCALE=$threads"
  # TODO: e.g.
  #   export CHPL_RT_NUM_THREADS_PER_LOCALE="$threads"
  #   /path/to/arkouda_server -nl 1 --ServerPort=$AK_PORT &> "$OUTDIR/server_${threads}.log" &
  #   sleep 20                       # wait for it to come up
  #   AK_HOST=$(hostname)            # or parse from the server log
  :
}
stop_server() {
  echo "[driver] stopping arkouda server"
  # TODO: e.g.  python -c "import arkouda as ak; ak.connect('$AK_HOST',$AK_PORT); ak.shutdown()"
  :
}

# ---- TODO-3: graph specs. Sparse hosts LEAD (capability headline); dense social are
#      NEXT-PRIORITY stress (not here). Point --edge-file at STAGED files (no downloads).
#      Provide them in increasing expected size-4-subgraph-count (cost) order. ----
# name|loader-args   (use --edge-file for staged SNAP lists; ogbn-* via ogb; gnp/ba synthetic)
SPARSE_LADDER=(
  "cora|"                                              # small validation point
  "ogbn-arxiv|"                                        # ogb (pre-stage under data/processed/ogb)
  "roadnetca|--edge-file ${GR:-/scratch/$USER/hm_graphs}/roadNet-CA.txt"  # bounded-degree, tractable
  "ogbn-products|"                                     # the large headline host
)
# NOTE: web/social graphs (web-BerkStan, Orkut, Friendster) are hub-heavy -> claw/size-4 explosion
# (cost ~ size-4 count, NOT |E|). They are a STRESS/ceiling case, NOT the capability ladder; the
# bench auto-skips exploding patterns as FAILED ceiling rows (--max-embeddings). Run them separately.
STRONG_SCALE_GRAPH="ogbn-products"                     # representative sparse mid/large host
STRONG_SCALE_ARGS=""

run_bench() {  # $1=backend $2=threads $3=name $4=loader-args
  local backend="$1" threads="$2" name="$3" args="$4"
  echo "[driver] bench backend=$backend threads=$threads graph=$name"
  python scripts/wulver/bench_orbits.py \
    --backend "$backend" --graph "$name" $args \
    --threads "$threads" --runs "$RUNS" \
    --out "$OUTDIR/bench.csv" --ak-host "$AK_HOST" --ak-port "$AK_PORT" \
    || echo "[driver] bench returned nonzero for $name ($backend) — continuing"
}

# =================== EXP1: HiPerMotif capability ladder (128 threads) ===================
echo "########## EXP1: HiPerMotif capability ladder ##########"
start_server 128
for spec in "${SPARSE_LADDER[@]}"; do
  name="${spec%%|*}"; args="${spec#*|}"
  run_bench hipermotif 128 "$name" "$args"
done

# =================== EXP2: ORCA crossover (single-threaded oracle) ===================
# ORCA is local (no server) — same ladder bottom-up until it fails (time/memory).
# Gives the crossover point; correctness itself is the small-graph gate
# (scripts/wulver/verify_hipermotif_equals_orca.py), run separately, not here.
echo "########## EXP2: ORCA crossover ##########"
for spec in "${SPARSE_LADDER[@]}"; do
  name="${spec%%|*}"; args="${spec#*|}"
  run_bench orca 0 "$name" "$args"
done

# =================== EXP3: strong-scaling thread sweep (THE HPC figure) ===================
# Thread count is set by RELAUNCHING the server. One representative sparse host.
echo "########## EXP3: strong scaling (server relaunch per thread count) ##########"
stop_server
for T in 1 2 4 8 16 32 64 128; do
  start_server "$T"
  run_bench hipermotif "$T" "$STRONG_SCALE_GRAPH" "$STRONG_SCALE_ARGS"
  stop_server
done

echo "[driver] all experiments done. CSV: $OUTDIR/bench.csv"
echo "[driver] make figures:  python scripts/wulver/make_scale_figures.py --csv $OUTDIR/bench.csv --out $OUTDIR"
