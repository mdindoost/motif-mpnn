"""CSL cycle-spectrum linear-separability experiment (paper Section III).

CSL (circulant skip-link graphs) is a THEORY-DEFINED SYNTHETIC construct, used here
purely as a controlled expressivity demonstration — never as real-world evidence. The
10 CSL classes share a single 1-WL coloring, so any message-passing GNN is capped at
chance (1/10 = 10%). This experiment shows that EXACT substructure features (counts of
simple cycles by length) make the 10 classes linearly separable, independent of any
trained network — reproducing the paper's Section III claim with an actual experiment.

Protocol:
  - Features: per-graph cycle spectrum = [#simple cycles of length L, for L=3..L_MAX].
  - Classifier: multinomial logistic regression (a LINEAR model) on standardized features.
  - Evaluation: stratified 5-fold cross-validation, clean and under 2% noise.
  - Noise model (per the draft's intent): each feature is multiplied by (1 + 0.02 * z),
    z ~ N(0,1) i.i.d. per (sample, feature); the noisy result is averaged over N_NOISE
    independent noise realizations (each itself 5-fold CV) for robustness.
  - Fully deterministic: fixed SEED throughout (CSL build, fold split, classifier, noise).

Run:  conda run -n motif-mpnn python scripts/expressivity/csl_linear_cv.py
Writes results/expressivity_csl_linear_cv.json and prints the actual numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.datasets.expressivity import make_csl  # noqa: E402

# --- protocol constants ---------------------------------------------------
L_MAX = 8       # cycle-length cutoff; matches CYCLE_L_MAX and test_csl_separability
N_FOLDS = 5
NOISE_STD = 0.02
N_NOISE = 20    # noise realizations averaged for the noisy report
SEED = 0
CSL_N = 41


def cycle_spectrum(edge_index, n: int, l_max: int = L_MAX):
    """Per-graph feature vector: count of simple cycles of each length L in 3..l_max."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u, v in edge_index.t().tolist():
        if u != v:
            g.add_edge(int(u), int(v))
    counts = {L: 0 for L in range(3, l_max + 1)}
    for cyc in nx.simple_cycles(g, length_bound=l_max):
        L = len(cyc)
        if 3 <= L <= l_max:
            counts[L] += 1
    return [counts[L] for L in range(3, l_max + 1)]


def build_features(seed: int = SEED, n: int = CSL_N, l_max: int = L_MAX):
    data = make_csl(n=n, seed=seed)  # 150 (edge_index, label): 10 classes x 15 copies
    X = np.asarray([cycle_spectrum(ei, n, l_max) for ei, _ in data], dtype=float)
    y = np.asarray([lab for _, lab in data], dtype=int)
    return X, y


def _linear_clf():
    # LINEAR classifier: standardize then near-unregularized multinomial logistic regression.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1e4, random_state=SEED),
    )


def run(seed: int = SEED):
    X, y = build_features(seed=seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    clean = cross_val_score(_linear_clf(), X, y, cv=skf)

    rng = np.random.default_rng(SEED)
    noisy_acc = []
    for _ in range(N_NOISE):
        Xn = X * (1.0 + NOISE_STD * rng.standard_normal(X.shape))
        noisy_acc.extend(cross_val_score(_linear_clf(), Xn, y, cv=skf).tolist())
    noisy = np.asarray(noisy_acc)

    return {
        "construct": "CSL (synthetic, theory-defined; expressivity demo, not real-world evidence)",
        "n_graphs": int(len(y)), "n_classes": int(y.max() + 1),
        "n_features": int(X.shape[1]), "feature": f"simple-cycle counts, lengths 3..{L_MAX}",
        "classifier": "multinomial logistic regression (linear)",
        "clean_mean": float(clean.mean()), "clean_std": float(clean.std()),
        "clean_folds": [float(a) for a in clean],
        "noisy_mean": float(noisy.mean()), "noisy_std": float(noisy.std()),
        "noise_model": "feature *= (1 + 0.02 * N(0,1)) per element",
        "noise_std": NOISE_STD, "n_noise_realizations": N_NOISE,
        "n_folds": N_FOLDS, "seed": SEED, "l_max": L_MAX,
    }


def main():
    res = run()
    out = ROOT / "results" / "expressivity_csl_linear_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("CSL cycle-spectrum linear separability (SYNTHETIC expressivity demo)")
    print(f"  graphs={res['n_graphs']}  classes={res['n_classes']}  "
          f"features={res['n_features']} ({res['feature']})  classifier={res['classifier']}")
    print(f"  CLEAN  5-fold CV acc = {res['clean_mean']:.4f} +/- {res['clean_std']:.4f}   "
          f"folds={['%.3f' % a for a in res['clean_folds']]}")
    print(f"  NOISY  5-fold CV acc = {res['noisy_mean']:.4f} +/- {res['noisy_std']:.4f}   "
          f"(2% mult noise; {N_NOISE} realizations x {N_FOLDS} folds)")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
