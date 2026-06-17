#!/usr/bin/env python
"""Synthetic motif-defined node task — the decisive, confound-free accuracy utility result.

THEORY-FORCED CONSTRUCT (labeled as such, per the project's rigor rule). The graph is a disjoint
union of `--copies` copies of the 4x4 rook's graph (label 1) and `--copies` copies of the
Shrikhande graph (label 0). Both are strongly regular SRG(16,6,2,2): every node has degree 6 and
the SAME per-node triangle count, and the two graphs are 1-WL-indistinguishable. They differ ONLY
in 4-cliques: rook has 2 per node, Shrikhande 0. So node label == "is this node in a 4-clique?":

  * A 1-WL GNN (GCN) with constant features CANNOT separate the classes (identical WL colorings).
  * Cheap legacy features (degree, wedge, triangle) are IDENTICAL between the classes -> provably
    at chance.
  * The exact 15-orbit feature set contains the 4-clique orbit (ORCA orbit 14 = 2 vs 0) -> the
    classes are linearly separable -> ~100%.
  * A dimension-matched RANDOM control -> chance (capacity is not the cause).

This is the mechanism proof the at-scale accuracy claim rests on: exact higher-order structure
carries label information that cheap features and 1-WL GNNs provably cannot. It scales by adding
copies (local runs use ORCA for features; at-scale runs use HiPerMotif). Reuses the validated
rook/Shrikhande fixtures and the repo's GCN/Concat models + ORCA oracle.

Usage:
  conda run -n motif-mpnn python scripts/experiments/synthetic_motif_task.py \
      --copies 60 --seeds 42 0 1 --out results/synthetic_motif_task.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.expressivity import make_rook_4x4, make_shrikhande
from src.datasets.orca_orbits import count_orbits, CLIQUE4_ORBIT, DEGREE_ORBIT, TRIANGLE_ORBIT
from src.models.gcn import GCN
from src.models.concat import ConcatModel

LEGACY3 = [DEGREE_ORBIT, 2, TRIANGLE_ORBIT]   # degree, wedge(p3-center), triangle: the cheap set


def _edges_from_ei(ei):
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in ei.t().tolist() if int(u) != int(v)}


def build_task(copies: int):
    """Disjoint union: `copies` rook (label 1) + `copies` Shrikhande (label 0). Returns
    (edge_index[2,E], y[N], num_nodes)."""
    rook = _edges_from_ei(make_rook_4x4())
    shrik = _edges_from_ei(make_shrikhande())
    edges, y, off = [], [], 0
    for _ in range(copies):                       # label 1: rook (has 4-cliques)
        edges += [(u + off, v + off) for (u, v) in rook]
        y += [1] * 16; off += 16
    for _ in range(copies):                       # label 0: Shrikhande (no 4-cliques)
        edges += [(u + off, v + off) for (u, v) in shrik]
        y += [0] * 16; off += 16
    return edges, np.array(y, dtype=np.int64), off


def normalize(mat):
    """log1p + per-column z-score (the repo's motif convention)."""
    x = np.log1p(mat.astype(np.float64))
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def gadget_split(copies, seed):
    """Split by WHOLE 16-node gadget (component), stratified by class. A gadget never straddles
    train/test, so test components are truly unseen — otherwise distinct per-node random features
    leak component labels across the split via message passing (the rand control would look like
    it 'works'). Each gadget g occupies nodes [16g, 16g+16); gadgets 0..copies-1 are rook (label
    1), copies..2*copies-1 are Shrikhande (label 0)."""
    rng = np.random.default_rng(seed)
    n = 32 * copies
    tr = np.zeros(n, bool); va = np.zeros(n, bool); te = np.zeros(n, bool)
    for gids in (np.arange(copies), np.arange(copies, 2 * copies)):
        gids = gids.copy(); rng.shuffle(gids)
        a, b = int(0.6 * len(gids)), int(0.8 * len(gids))
        for split, arr in ((tr, gids[:a]), (va, gids[a:b]), (te, gids[b:])):
            for g in arr:
                split[g * 16:(g + 1) * 16] = True
    return tr, va, te


def train_eval(data, model, seed, epochs=200, patience=40, lr=0.01):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val, best_test, wait = 0.0, 0.0, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(1)
            va = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            te = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if va > best_val:
            best_val, best_test, wait = va, te, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_test


def make_data(edges, y, num_nodes, motif_x):
    ei = torch.tensor([[u for u, v in edges] + [v for u, v in edges],
                       [v for u, v in edges] + [u for u, v in edges]], dtype=torch.long)
    d = type("D", (), {})()
    d.x = torch.ones(num_nodes, 1)                # constant features: structure is the only signal
    d.edge_index = ei
    d.y = torch.tensor(y, dtype=torch.long)
    d.motif_x = None if motif_x is None else torch.tensor(motif_x)
    return d


def run(copies, seeds, out):
    edges, y, n = build_task(copies)
    print(f"[task] {n} nodes ({y.sum()} in-4-clique / {n - y.sum()} not), {len(edges)} edges, "
          f"{2*copies} gadgets")
    orb = count_orbits(edges, n, 4)                                  # ORCA oracle [n,15]
    # sanity: the two classes are degree/triangle-IDENTICAL but 4-clique-DIFFERENT
    c1, c0 = orb[y == 1], orb[y == 0]
    print(f"[check] degree  class1={c1[:,DEGREE_ORBIT].mean():.1f} class0={c0[:,DEGREE_ORBIT].mean():.1f} "
          f"(equal => not predictive)")
    print(f"[check] triangle class1={c1[:,TRIANGLE_ORBIT].mean():.1f} class0={c0[:,TRIANGLE_ORBIT].mean():.1f} "
          f"(equal => not predictive)")
    print(f"[check] 4-clique class1={c1[:,CLIQUE4_ORBIT].mean():.1f} class0={c0[:,CLIQUE4_ORBIT].mean():.1f} "
          f"(differ => the label)")
    density = float((orb[:, CLIQUE4_ORBIT] > 0).mean())
    print(f"[check] 4-clique-orbit participation density = {density:.3f}")

    feats = {
        "gcn":      None,
        "legacy3":  normalize(orb[:, LEGACY3]),
        "orbit15":  normalize(orb),
        "rand15":   None,   # filled per-seed (random)
    }
    rows = []
    for cond in ["gcn", "legacy3", "orbit15", "rand15"]:
        accs = []
        for s in seeds:
            if cond == "rand15":
                mx = np.random.default_rng(s).standard_normal((n, 15)).astype(np.float32)
            else:
                mx = feats[cond]
            data = make_data(edges, y, n, mx)
            tr, va, te = gadget_split(copies, s)
            data.train_mask = torch.tensor(tr); data.val_mask = torch.tensor(va); data.test_mask = torch.tensor(te)
            if cond == "gcn":
                model = GCN(in_dim=1, out_dim=2, hidden_dim=32, num_layers=2, task="node")
            else:
                model = ConcatModel(in_dim=1, out_dim=2, motif_dim=mx.shape[1], hidden_dim=32,
                                    num_layers=2, task="node")
            acc = train_eval(data, model, s)
            accs.append(acc)
            rows.append({"condition": cond, "seed": s, "test_acc": round(acc, 4),
                         "copies": copies, "n_nodes": n, "clique_density": round(density, 4)})
        print(f"  {cond:8} test_acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}  (seeds {seeds})")

    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "seed", "test_acc", "copies", "n_nodes", "clique_density"])
        w.writeheader(); w.writerows(rows)
    print(f"[task] wrote {out}")
    # the headline assertion the paper rests on
    by = {c: np.mean([r["test_acc"] for r in rows if r["condition"] == c]) for c in feats}
    print(f"\nVERDICT: orbit15={by['orbit15']:.3f}  vs  legacy3={by['legacy3']:.3f}  "
          f"gcn={by['gcn']:.3f}  rand15={by['rand15']:.3f}")
    print("Expected: orbit15 ~1.0 ; gcn/legacy3/rand15 ~0.5 (chance) — exact 4-clique orbit is the "
          "only feature that carries the label.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copies", type=int, default=60, help="copies per class (node count = 32*copies)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1])
    ap.add_argument("--out", default="results/synthetic_motif_task.csv")
    args = ap.parse_args()
    run(args.copies, args.seeds, args.out)


if __name__ == "__main__":
    main()
