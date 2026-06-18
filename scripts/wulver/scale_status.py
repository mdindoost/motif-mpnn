#!/usr/bin/env python
"""At-a-glance status of the scale experiments: what's done, running, and left.

Reads the bench CSV(s) (results/scale/bench_*.csv — including the live-flushed partial files,
since bench writes each row immediately) and prints coverage so you/Bartosz can see exactly which
experiment + graph is complete vs in progress, without reading raw rows.

  EXP1  = HiPerMotif at a single (128) thread count, per graph  (capability ladder)
  EXP2  = ORCA, per graph                                       (crossover / oracle)
  EXP3  = HiPerMotif with a THREAD SWEEP (>1 thread count)      (strong scaling, Fig A)

Run anywhere (no arkouda): python scripts/wulver/scale_status.py [--csv results/scale/bench_*.csv]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

EXP1_GRAPHS = ["cora", "ogbn-arxiv", "roadnetca", "ogbn-products"]   # capability ladder (sparse-led)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", default=["results/scale/bench_*.csv", "results/scale/bench.csv"],
                    help="CSV file(s) or glob(s)")
    args = ap.parse_args()
    files = sorted({f for pat in args.csv for f in glob.glob(pat)})
    if not files:
        print(f"no CSVs found for {args.csv}"); return
    print(f"[status] reading {len(files)} file(s): {', '.join(Path(f).name for f in files)}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("py_seconds", "n_embeddings", "threads_label"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- per (graph, backend) coverage -----------------------------------------------------
    print("\n" + "=" * 86)
    print(f"{'graph':14}{'backend':11}{'threads':14}{'runs':5}{'OK pat':7}{'CEIL':6}{'mean TOTAL s':13}")
    print("-" * 86)
    seen = {}   # (graph,kind) -> True   kind in {hm_single, orca, hm_sweep}
    for (g, b), sub in df.groupby(["graph", "backend"]):
        threads = sorted(t for t in sub["threads_label"].dropna().unique())
        is_sweep = (b == "hipermotif" and len(threads) > 1)
        tcol = (f"sweep[{int(min(threads))}..{int(max(threads))}]" if is_sweep
                else (str(int(threads[0])) if threads else "?"))
        tot = sub[(sub["pattern"] == "TOTAL") & (sub["status"] == "OK")]
        nruns = tot["run_idx"].nunique() if "run_idx" in tot else len(tot)
        ok_pat = sub[(sub["pattern"] != "TOTAL") & (sub["status"] == "OK")].shape[0]
        ceil = sub[(sub["pattern"] != "TOTAL") & (sub["status"] == "FAILED")].shape[0]
        mean_s = tot["py_seconds"].mean() if not tot.empty else float("nan")
        print(f"{g:14}{b:11}{tcol:14}{nruns:<5}{ok_pat:<7}{ceil:<6}{mean_s:<13.1f}")
        if is_sweep:
            seen[(g, "hm_sweep")] = True
        elif b == "hipermotif":
            seen[(g, "hm_single")] = True
        elif b == "orca":
            seen[(g, "orca")] = True

    # ---- experiment checklist --------------------------------------------------------------
    def mark(present): return "DONE" if present else "----"
    print("\n" + "=" * 86 + "\nEXPERIMENT CHECKLIST\n" + "-" * 86)
    print("EXP1  HiPerMotif @128, per graph:")
    for g in EXP1_GRAPHS:
        print(f"   {mark((g,'hm_single') in seen)}  {g}")
    print("EXP2  ORCA crossover, per graph:")
    for g in EXP1_GRAPHS:
        print(f"   {mark((g,'orca') in seen)}  {g}")
    sweeps = [g for (g, k) in seen if k == "hm_sweep"]
    print(f"EXP3  strong-scaling thread sweep (Fig A):  "
          f"{'DONE on ' + ', '.join(sweeps) if sweeps else '---- (no multi-thread sweep yet)'}")
    extra = sorted({g for (g, _) in seen} - set(EXP1_GRAPHS) - set(sweeps))
    if extra:
        print(f"other graphs present (stress/optional): {', '.join(extra)}")
    print("=" * 86)


if __name__ == "__main__":
    main()
