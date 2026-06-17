#!/usr/bin/env python
"""Figures + LaTeX table for the HiPerMotif scale experiments. Pure grouping of bench CSV(s).

Reads the CSV(s) written by bench_orbits.py and produces (matplotlib Agg, PNG+PDF, no display):
  Fig A : strong scaling  — speedup & efficiency vs threads (ideal-linear reference).
  Fig B : size scaling    — total extraction time vs total #embeddings (log-log work axis,
          NOT |E|), HiPerMotif, with ORCA points overlaid up to its ceiling (the crossover).
  Fig C : per-pattern cost breakdown on the largest completed HiPerMotif graph.
  Table : LaTeX summary (graph | nodes | edges | ORCA s | HiPerMotif s @128t | server mem GB |
          speedup-vs-ORCA where both ran).

Runs locally (no arkouda needed) — copy the CSV back from Wulver, or run on Wulver.
Usage: python scripts/wulver/make_scale_figures.py --csv results/scale/bench.csv --out results/scale
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ok_total(df):
    """OK TOTAL rows, mean over runs, keyed by (graph, backend, threads_label)."""
    t = df[(df["pattern"] == "TOTAL") & (df["status"] == "OK")].copy()
    g = (t.groupby(["graph", "backend", "threads_label"])
           .agg(py_seconds=("py_seconds", "mean"),
                n_embeddings=("n_embeddings", "mean"),
                n_nodes=("n_nodes", "first"), n_edges=("n_edges", "first"),
                server_mem_gb=("server_mem_gb", "max"))
           .reset_index())
    return g


def fig_strong_scaling(df, outdir, graph):
    hm = df[(df["backend"] == "hipermotif") & (df["graph"] == graph) &
            (df["pattern"] == "TOTAL") & (df["status"] == "OK")]
    s = hm.groupby("threads_label")["py_seconds"].mean().sort_index()
    if len(s) < 2 or 1 not in s.index:
        print("[fig A] insufficient thread points (need threads=1 baseline); skipping")
        return
    threads = s.index.values.astype(float)
    speedup = s.loc[1] / s.values
    eff = speedup / threads
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(threads, speedup, "o-", label="measured speedup")
    ax1.plot(threads, threads, "k--", alpha=0.5, label="ideal (linear)")
    ax1.set_xlabel("threads (cores)"); ax1.set_ylabel("speedup vs 1 thread")
    ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=2)
    ax1.set_title(f"Strong scaling — {graph}")
    ax2 = ax1.twinx()
    ax2.plot(threads, eff, "s:", color="tab:green", alpha=0.7, label="efficiency")
    ax2.set_ylabel("parallel efficiency"); ax2.set_ylim(0, 1.05)
    ax1.legend(loc="upper left"); ax2.legend(loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figA_strong_scaling.{ext}", dpi=150)
    plt.close(fig)
    print(f"[fig A] strong scaling for {graph}: threads={list(s.index)} speedup_max={speedup.max():.1f}")


def fig_size_scaling(df, outdir):
    g = _ok_total(df)
    hm = g[(g["backend"] == "hipermotif") & (g["threads_label"] == g["threads_label"].max())]
    orca = g[g["backend"] == "orca"]
    if hm.empty:
        print("[fig B] no hipermotif TOTAL rows; skipping"); return
    fig, ax = plt.subplots(figsize=(5, 4))
    hm2 = hm[hm["n_embeddings"] > 0]
    ax.plot(hm2["n_embeddings"], hm2["py_seconds"], "o-", label="HiPerMotif (max threads)")
    o2 = orca[orca["n_embeddings"] > 0]
    if not o2.empty:
        ax.plot(o2["n_embeddings"], o2["py_seconds"], "s--", color="tab:red", label="ORCA (1 thread)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("# size-4 subgraph embeddings (work)"); ax.set_ylabel("extraction time (s)")
    ax.set_title("Extraction time vs structure (work), w/ ORCA crossover")
    ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figB_size_scaling.{ext}", dpi=150)
    plt.close(fig)
    print(f"[fig B] hipermotif graphs={list(hm2['graph'])}; orca graphs={list(o2['graph'])}")


def fig_per_pattern(df, outdir):
    hm = df[(df["backend"] == "hipermotif") & (df["status"] == "OK") &
            (df["pattern"] != "TOTAL")]
    if hm.empty:
        print("[fig C] no per-pattern rows; skipping"); return
    # largest completed graph = most total embeddings
    tot = _ok_total(df)
    tot = tot[tot["backend"] == "hipermotif"]
    graph = tot.sort_values("n_embeddings").iloc[-1]["graph"]
    sub = hm[hm["graph"] == graph].groupby("pattern")["py_seconds"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sub.index, sub.values, color="tab:purple")
    ax.set_ylabel("extraction time (s)"); ax.set_xlabel("graphlet pattern")
    ax.set_title(f"Per-pattern cost — {graph}")
    plt.xticks(rotation=45, ha="right"); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figC_per_pattern.{ext}", dpi=150)
    plt.close(fig)
    print(f"[fig C] per-pattern on {graph}: costliest={sub.index[-1]} ({sub.iloc[-1]:.2f}s)")


def latex_table(df, outdir):
    g = _ok_total(df)
    hm = g[g["backend"] == "hipermotif"]
    if not hm.empty:
        hm = hm[hm["threads_label"] == hm["threads_label"].max()]
    orca = g[g["backend"] == "orca"].set_index("graph")
    lines = [
        "% HiPerMotif scale results (mean over runs). Generated by make_scale_figures.py",
        r"\begin{table}[t]\centering",
        r"\caption{Exact size-4 orbit extraction on one shared-memory node. "
        r"ORCA is the single-threaded oracle; HiPerMotif at max threads.}",
        r"\label{tab:scale}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Graph & Nodes & Edges & ORCA (s) & HiPerMotif (s) & Mem (GB) & Speedup \\",
        r"\midrule",
    ]
    for _, r in hm.sort_values("n_edges").iterrows():
        gname = r["graph"]
        o = float(orca.loc[gname]["py_seconds"]) if gname in orca.index else None
        sp = f"{o / r['py_seconds']:.1f}$\\times$" if (o and r["py_seconds"] > 0) else "--"
        ostr = f"{o:.2f}" if o else "--"
        lines.append(f"{gname} & {int(r['n_nodes'])} & {int(r['n_edges'])} & {ostr} & "
                     f"{r['py_seconds']:.2f} & {r['server_mem_gb']:.1f} & {sp} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "table_scale.tex").write_text("\n".join(lines) + "\n")
    print(f"[table] wrote {outdir/'table_scale.tex'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="one or more bench CSV files")
    ap.add_argument("--out", required=True, help="output dir for figures + table")
    ap.add_argument("--strong-graph", default="ogbn-products",
                    help="graph used for the strong-scaling figure (must have a threads sweep)")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    # numeric coercion (FAILED rows carry -1 / strings)
    for col in ("py_seconds", "n_embeddings", "server_mem_gb", "threads_label", "n_nodes", "n_edges"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[figs] {len(df)} rows; backends={sorted(df['backend'].unique())}; "
          f"FAILED rows={int((df['status']=='FAILED').sum())}")
    fig_strong_scaling(df, outdir, args.strong_graph)
    fig_size_scaling(df, outdir)
    fig_per_pattern(df, outdir)
    latex_table(df, outdir)
    print("[figs] done.")


if __name__ == "__main__":
    main()
