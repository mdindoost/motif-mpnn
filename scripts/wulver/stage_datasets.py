#!/usr/bin/env python
"""Stage all scale-experiment graphs to disk. RUN ON A LOGIN NODE (needs internet).

Compute nodes have NO internet, so download everything once here first. Idempotent: skips
anything already present. After this, bench_orbits.py reads from disk with no downloads.

Stages:
  * OGB node graphs (ogbn-arxiv, ogbn-products) into <ogb-root> (default data/processed/ogb)
    -> bench uses them via `--graph ogbn-arxiv` / `--graph ogbn-products` (no --edge-file).
  * SNAP sparse hosts (web-BerkStan, roadNet-CA) into <out> as plain edge lists
    -> bench uses them via `--edge-file <out>/<name>.txt`.
  (Synthetic BA/gnp graphs need no staging — bench generates them.)

Usage (login node):
  python scripts/wulver/stage_datasets.py --out /scratch/$USER/hm_graphs
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# name -> SNAP .gz URL. (Verify a URL if a download 404s — SNAP occasionally reorganizes.)
SNAP = {
    "web-BerkStan": "https://snap.stanford.edu/data/web-BerkStan.txt.gz",   # ~685K nodes, ~7.6M edges (sparse)
    "roadNet-CA":   "https://snap.stanford.edu/data/roadNet-CA.txt.gz",     # ~2M nodes, ~2.8M edges (ultra-sparse)
    # dense social — NEXT-PRIORITY stress only, big downloads; uncomment when needed:
    # "com-orkut":  "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
}
OGB = ["ogbn-arxiv", "ogbn-products"]


def stage_snap(name, url, out: Path):
    dest = out / f"{name}.txt"
    if dest.exists():
        print(f"[stage] SNAP {name}: already at {dest} (skip)")
        return dest
    gz = out / f"{name}.txt.gz"
    print(f"[stage] SNAP {name}: downloading {url}")
    urllib.request.urlretrieve(url, gz)
    print(f"[stage] SNAP {name}: gunzip -> {dest}")
    with gzip.open(gz, "rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    gz.unlink()
    print(f"[stage] SNAP {name}: ready. Use:  --edge-file {dest}")
    return dest


def stage_ogb(name, ogb_root: Path):
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except Exception as e:
        print(f"[stage] OGB {name}: SKIP — `ogb` not installed ({e}). `pip install ogb` on the "
              f"login node, then rerun. (bench uses --graph {name} once it's on disk.)")
        return
    print(f"[stage] OGB {name}: downloading/preparing under {ogb_root} ...")
    NodePropPredDataset(name=name, root=str(ogb_root))   # downloads + processes on first call
    print(f"[stage] OGB {name}: ready. Use:  --graph {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data/scale_graphs"),
                    help="dir for SNAP edge lists (use /scratch on Wulver)")
    ap.add_argument("--ogb-root", default=str(REPO / "data/processed/ogb"))
    ap.add_argument("--snap-only", action="store_true")
    ap.add_argument("--ogb-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ogb_root = Path(args.ogb_root); ogb_root.mkdir(parents=True, exist_ok=True)

    print("[stage] LOGIN NODE ONLY (needs internet). Compute nodes read from disk.\n")
    if not args.ogb_only:
        for name, url in SNAP.items():
            try:
                stage_snap(name, url, out)
            except Exception as e:
                print(f"[stage] SNAP {name}: FAILED ({e}) — check the URL / network and rerun.")
    if not args.snap_only:
        for name in OGB:
            try:
                stage_ogb(name, ogb_root)
            except Exception as e:
                print(f"[stage] OGB {name}: FAILED ({e}).")

    print("\n[stage] done. Paths for the runbook:")
    print(f"  OGB root      : {ogb_root}   (--graph ogbn-arxiv / ogbn-products)")
    for name in SNAP:
        print(f"  {name:13}: {out / (name + '.txt')}   (--edge-file ...)")


if __name__ == "__main__":
    main()
