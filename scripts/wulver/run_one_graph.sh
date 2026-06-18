#!/usr/bin/env bash
# ONE graph on ONE node (HiPerMotif @128 threads + ORCA), writing its OWN CSV.
# Submit one of these per graph IN PARALLEL — each gets its own node, its own arkouda server,
# and its own output file, so wall-clock = the slowest single graph (not the sum).
# DO NOT point two parallel jobs at the same --out file (concurrent flushes corrupt it).
#
# >>> BARTOSZ: fill TODO-1 (SLURM header) and TODO-2 (your arkouda server launch). <<<
# Everything else is fixed. reorder_type="None"; bench prints live progress + writes each row
# immediately, so a hang/kill never loses completed-pattern rows.
set -u

# ---- TODO-1: SLURM header (one node, exclusive, all memory). Delete if interactive. ----
# #SBATCH --job-name=hm-onegraph
# #SBATCH --nodes=1
# #SBATCH --exclusive
# #SBATCH --mem=0
# #SBATCH --time=08:00:00
# #SBATCH --partition=<your_partition>

REPO="${REPO:-$HOME/motif-mpnn}"; cd "$REPO"; export PYTHONPATH="$REPO"
GRAPH="${GRAPH:?set GRAPH, e.g. GRAPH=roadnetca}"   # cora | ogbn-arxiv | ogbn-products | roadnetca | ...
EARGS="${EARGS:-}"                                  # e.g. EARGS="--edge-file /scratch/$USER/hm_graphs/roadNet-CA.txt"  (empty for cora/ogbn-*)
THREADS="${THREADS:-128}"
AK_PORT="${AK_PORT:-5555}"
OUT="${OUT:-results/scale/bench_${GRAPH}.csv}"      # PER-GRAPH file (do not share across jobs)
mkdir -p "$(dirname "$OUT")"

# ---- TODO-2: launch/stop your arkouda server (your usual command). Set AK_HOST to its node. ----
AK_HOST="localhost"
start_server() {   # $1 = threads-per-locale
  echo "[one-graph] start arkouda server, CHPL_RT_NUM_THREADS_PER_LOCALE=$1"
  # export CHPL_RT_NUM_THREADS_PER_LOCALE="$1"
  # <your arkouda_server launch> --ServerPort=$AK_PORT &> "$(dirname "$OUT")/server_${GRAPH}.log" &
  # sleep 20; AK_HOST=$(hostname)
  :
}
stop_server() { : ; }   # e.g. python -c "import arkouda as ak; ak.connect('$AK_HOST',$AK_PORT); ak.shutdown()"

echo "=================================================================="
echo " EXP1+EXP2  graph=$GRAPH  (HiPerMotif @${THREADS}t + ORCA crossover)"
echo " node=$(hostname)  out=$OUT"
echo "=================================================================="
start_server "$THREADS"
# HiPerMotif (engine, needs the server)
python scripts/wulver/bench_orbits.py --backend hipermotif --graph "$GRAPH" $EARGS \
  --threads "$THREADS" --runs 3 --out "$OUT" --ak-host "$AK_HOST" --ak-port "$AK_PORT" --no-mem
stop_server
# ORCA crossover (local; no server). Skips itself if ORCA OOMs/times out (FAILED row).
python scripts/wulver/bench_orbits.py --backend orca --graph "$GRAPH" $EARGS \
  --runs 3 --out "$OUT"
echo "[one-graph] done: $OUT"
