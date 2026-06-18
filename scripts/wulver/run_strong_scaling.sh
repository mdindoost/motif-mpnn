#!/usr/bin/env bash
# ONE node: strong-scaling thread sweep for Fig A (the headline HPC figure).
# Threads {1,2,4,8,16,32,64,128} on ONE fixed node (same hardware = valid speedup curve), each
# by RELAUNCHING the arkouda server with that CHPL_RT_NUM_THREADS_PER_LOCALE. Own CSV.
# Run this on a dedicated node WHILE the per-graph EXP1 jobs run on other nodes.
#
# IMPORTANT: the 1-thread point is the long pole. Use a SMALL graph so it finishes in a sane
# time (default synthetic BA ~10k nodes). If the low-thread points are too slow, shrink --n or
# set RUNS_LOW=1 below.
#
# >>> BARTOSZ: fill TODO-1 (SLURM header) and TODO-2 (arkouda server launch). <<<
set -u

# ---- TODO-1: SLURM header (one node, exclusive, all memory). ----
# #SBATCH --job-name=hm-strong --nodes=1 --exclusive --mem=0 --time=12:00:00 --partition=<...>

REPO="${REPO:-$HOME/motif-mpnn}"; cd "$REPO"; export PYTHONPATH="$REPO"
GRAPH="${GRAPH:-ba}"; NARG="${NARG:---n 10000 --m 5}"   # small synthetic; 1-thread must be tractable
AK_PORT="${AK_PORT:-5555}"
OUT="${OUT:-results/scale/bench_strong.csv}"
RUNS="${RUNS:-3}"; RUNS_LOW="${RUNS_LOW:-$RUNS}"        # set RUNS_LOW=1 to speed up the slow low-thread points
mkdir -p "$(dirname "$OUT")"

# ---- TODO-2: launch/stop your arkouda server with N threads; set AK_HOST. ----
AK_HOST="localhost"
start_server() {   # $1 = threads-per-locale
  echo "[strong] start arkouda server, CHPL_RT_NUM_THREADS_PER_LOCALE=$1"
  # export CHPL_RT_NUM_THREADS_PER_LOCALE="$1"
  # <your arkouda_server launch> --ServerPort=$AK_PORT &> "$(dirname "$OUT")/server_strong_$1.log" &
  # sleep 20; AK_HOST=$(hostname)
  :
}
stop_server() { : ; }

for T in 1 2 4 8 16 32 64 128; do
  if [ "$T" -le 8 ]; then R="$RUNS_LOW"; else R="$RUNS"; fi
  start_server "$T"
  echo "[strong] threads=$T runs=$R graph=$GRAPH $NARG -> $OUT"
  python scripts/wulver/bench_orbits.py --backend hipermotif --graph "$GRAPH" $NARG \
    --threads "$T" --runs "$R" --out "$OUT" --ak-host "$AK_HOST" --ak-port "$AK_PORT" --no-mem
  stop_server
done
echo "[strong] done: $OUT  (check maxtaskpar_actual varied 1..128 in the CSV)"
