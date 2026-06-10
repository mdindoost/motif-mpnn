"""Vendored-ORCA per-vertex graphlet-orbit counter (Phase 1a).

Pure Python wrapper around the ORCA C++ binary (third_party/orca/). ORCA
(Hocevar & Demsar, Bioinformatics 2014) computes, for each vertex, its count in
each automorphism orbit of all 2..k-node graphlets: 15 orbits for k=4, 73 for k=5.

Counting is exact, deterministic, and single-threaded (no RNG) -- orbit counts
are fully reproducible. This module is the local stand-in / future correctness
oracle for HiPerMotif's normalized orbit counts.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# --- paths --------------------------------------------------------------
ORCA_DIR = Path(__file__).resolve().parents[2] / "third_party" / "orca"
ORCA_SRC = ORCA_DIR / "orca.cpp"
ORCA_BIN = ORCA_DIR / "orca"

# --- orbit scheme -------------------------------------------------------
N_ORBITS = {4: 15, 5: 73}
ORCA_ORBIT_BASE = 2000  # disjoint from legacy ids (0/2/3), cycles (1000), 4-clique (4,10)

# Hand-verified ORCA size-4 orbit indices (validated against K4/C5/SRG in Task 3):
DEGREE_ORBIT = 0     # edge endpoint == degree
TRIANGLE_ORBIT = 3   # triangle
CLIQUE4_ORBIT = 14   # K4

# orbit index -> graphlet node count (size-4 basis). Standard ORCA numbering:
# orbit 0 lives in the 2-node graphlet; orbits 1-3 in 3-node graphlets;
# orbits 4-14 in 4-node graphlets. Validated by test_orbit_graphlet_size_table
# and (for 0/3/14) by the hand-computed graphs in Task 3.
ORBIT_GRAPHLET_SIZE = [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def orbit_to_km(o: int) -> Tuple[int, int]:
    """Map an ORCA orbit index to the repo's (k, motif_id) CSV encoding."""
    return ORBIT_GRAPHLET_SIZE[o], ORCA_ORBIT_BASE + o


def ensure_orca_built() -> Path:
    """Compile third_party/orca/orca.cpp into ORCA_BIN if absent. Idempotent."""
    if ORCA_BIN.exists():
        return ORCA_BIN
    if not ORCA_SRC.exists():
        raise RuntimeError(
            f"ORCA source not found at {ORCA_SRC}. Vendor it first (see Task 1: "
            "curl https://raw.githubusercontent.com/thocevar/orca/master/orca.cpp)."
        )
    gxx = shutil.which("g++")
    if gxx is None:
        raise RuntimeError(
            "g++ not found; cannot build ORCA. Install build-essential "
            "(e.g. `sudo apt-get install -y build-essential`) and retry."
        )
    proc = subprocess.run(
        [gxx, "-O2", "-std=c++11", "-o", str(ORCA_BIN), str(ORCA_SRC)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not ORCA_BIN.exists():
        raise RuntimeError(f"ORCA build failed:\n{proc.stderr}")
    return ORCA_BIN


def _normalize_edges(edges: Iterable[Tuple[int, int]], num_nodes: int):
    """Return a sorted list of undirected, self-loop-free, deduped (u, v) with u < v."""
    seen = set()
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge ({u},{v}) out of range for num_nodes={num_nodes}")
        seen.add((u, v) if u < v else (v, u))
    return sorted(seen)


def count_orbits(edges: Iterable[Tuple[int, int]], num_nodes: int,
                 graphlet_size: int = 4) -> np.ndarray:
    """Per-vertex orbit counts, shape [num_nodes, N_ORBITS[graphlet_size]].

    Isolated vertices get all-zero rows. 0-edge graphs short-circuit to zeros
    (avoids relying on ORCA's behavior with m=0).
    """
    if graphlet_size not in N_ORBITS:
        raise ValueError(f"graphlet_size must be 4 or 5, got {graphlet_size}")
    k = N_ORBITS[graphlet_size]
    norm = _normalize_edges(edges, num_nodes)
    if not norm:
        return np.zeros((num_nodes, k), dtype=np.int64)

    ensure_orca_built()
    with tempfile.TemporaryDirectory() as td:
        fin = Path(td) / "in.txt"
        fout = Path(td) / "out.txt"
        with open(fin, "w") as f:
            f.write(f"{num_nodes} {len(norm)}\n")
            for u, v in norm:
                f.write(f"{u} {v}\n")
        proc = subprocess.run(
            [str(ORCA_BIN), "node", str(graphlet_size), str(fin), str(fout)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not fout.exists():
            raise RuntimeError(
                f"ORCA failed (N={num_nodes}, E={len(norm)}):\n{proc.stderr}\n{proc.stdout}"
            )
        mat = np.loadtxt(fout, dtype=np.int64)
    mat = np.atleast_2d(mat)
    if mat.shape != (num_nodes, k):
        raise RuntimeError(
            f"ORCA output shape {mat.shape}, expected ({num_nodes}, {k})"
        )
    return mat
