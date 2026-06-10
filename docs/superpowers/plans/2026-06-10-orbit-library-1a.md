# Orbit-Feature Library (Phase 1a: ORCA counting infrastructure) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor and wrap the ORCA graphlet-orbit counter, add it as an `--features orbit` backend to `generate_motifs.py`, and validate per-node orbit counts against hand-computed oracles — counting infrastructure only, no accuracy claims.

**Architecture:** ORCA (`orca.cpp`, single-file C++) is vendored under `third_party/orca/`, compiled on demand into a gitignored binary. A pure-Python wrapper `src/datasets/orca_orbits.py` normalizes a graph's edges, shells out to the ORCA binary via a temp file, and parses the `[N, 15]` orbit matrix. Orbits map into the existing CSV/manifest scheme as `(k = graphlet_size(orbit), motif_id = 2000 + orbit)`, written to a **separate** `node_motifs_orbit.csv` so the legacy degree/wedge/triangle path stays byte-for-byte unchanged. The existing `motif_loader.py` already extends its manifest for unseen `(k, motif_id)` pairs (verified: `motif_loader.py:147`), so no loader change is needed.

**Tech Stack:** Python 3, `numpy`, `networkx`, `pytest`, ORCA (C++ via `g++`), the repo's `conda` env `motif-mpnn`.

**Spec:** `docs/superpowers/specs/2026-06-10-orbit-library-1a-design.md`

**Grounding facts verified against the codebase / canonical ORCA source before writing this plan:**
- ORCA CLI is `orca <node|edge> <4|5> <infile> <outfile>`; input is `N M` header then 0-indexed `u v` edge lines; node-mode output is `N` lines of 15 ints (size 4) / 73 ints (size 5). (Confirmed from `thocevar/orca` `orca.cpp` `main()`.)
- `substructure_counts.py` is the pattern to mirror: pure functions over `networkx` graphs, importable from `src/`. (`src/datasets/substructure_counts.py`.)
- `generate_motifs.py` already has `_count_substructures_single`, per-dataset runners, and `_write_planetoid_csv`/`_write_tu_csv` writers hardcoding `out_path = out_dir / "node_motifs.csv"`. (`scripts/preprocess/generate_motifs.py:225,263,275,321`.)
- `motif_loader.build_or_load_node_motif_X` reads the fixed filename `node_motifs.csv` (`motif_loader.py:102`) and extends the manifest for unseen `(k, motif_id)` (`motif_loader.py:147`). The round-trip test therefore runs the loader in a temp dir with the orbit CSV renamed to `node_motifs.csv` (loader selection of the orbit file is Phase 1b, not here).
- Test conventions: `tests/conftest.py` puts repo root on `sys.path`; fixtures `make_csl`, `make_shrikhande`, `make_rook_4x4`, `_edge_index_to_igraph` live in `src/datasets/expressivity.py`.

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `third_party/orca/orca.cpp` | Create (vendored) | Canonical ORCA C++ source |
| `third_party/orca/PROVENANCE.md` | Create | Source URL, version/commit, authors, citation, license note |
| `.gitignore` | Modify | Ignore the compiled `third_party/orca/orca` binary |
| `src/datasets/orca_orbits.py` | Create | Build ORCA, run it, parse orbit matrix; orbit→(k,motif_id) mapping + named orbit-index constants |
| `tests/test_orca_orbits.py` | Create | Build, constants/table, and correctness tests (K4, C5, star, degree, SRG) |
| `scripts/preprocess/generate_motifs.py` | Modify | Add `--features {legacy,orbit}` + `--graphlet-size`; orbit CSV writer → `node_motifs_orbit.csv` |
| `tests/test_generate_motifs_orbit.py` | Create | End-to-end CSL orbit-CSV generation + loader round-trip |

---

## Task 1: Vendor ORCA source

**Files:**
- Create: `third_party/orca/orca.cpp`
- Create: `third_party/orca/PROVENANCE.md`
- Modify: `.gitignore`

- [ ] **Step 1: Fetch the canonical ORCA source**

Run:
```bash
mkdir -p third_party/orca
curl -fsSL https://raw.githubusercontent.com/thocevar/orca/master/orca.cpp -o third_party/orca/orca.cpp
```
Expected: `third_party/orca/orca.cpp` exists and is non-empty.

If network access is blocked in this environment, obtain `orca.cpp` from `https://github.com/thocevar/orca` (file `orca.cpp` at branch `master`) by any available means and place it at that path. Do **not** hand-write or paraphrase the source — it must be the upstream file verbatim.

- [ ] **Step 2: Verify it is the expected single-file source**

Run:
```bash
grep -c 'Usage:' third_party/orca/orca.cpp && grep -c 'node|edge' third_party/orca/orca.cpp && wc -l third_party/orca/orca.cpp
```
Expected: the `Usage:` and `node|edge` greps each print `1` (the argv usage string is present); line count is a few hundred lines (non-trivial source, not an HTML error page).

- [ ] **Step 3: Confirm it compiles**

Run:
```bash
g++ -O2 -std=c++11 -o third_party/orca/orca third_party/orca/orca.cpp && echo BUILD_OK
```
Expected: prints `BUILD_OK` with no errors. If `g++` is missing: `sudo apt-get install -y build-essential` (note this in your handoff but do not run sudo unprompted).

- [ ] **Step 4: Write PROVENANCE.md**

```markdown
# ORCA — Provenance

**File:** `orca.cpp` (vendored verbatim, unmodified)
**Upstream:** https://github.com/thocevar/orca (file `orca.cpp`, branch `master`)
**Retrieved:** 2026-06-10 via `curl` (see Task 1 of the Phase 1a plan)

**Authors:** Tomaž Hočevar and Janez Demšar.

**Citation:**
> Tomaž Hočevar, Janez Demšar. "A combinatorial approach to graphlet counting."
> *Bioinformatics* 30(4):559–565, 2014. doi:10.1093/bioinformatics/btt717

**License:** ORCA is distributed by its authors as open source (see the upstream
repository for the exact terms). This file is vendored unmodified for local,
research use as an exact per-vertex graphlet-orbit counter. The compiled binary
`third_party/orca/orca` is built locally and is gitignored.

**Role in this repo:** local orbit-count backend for `src/datasets/orca_orbits.py`
(Phase 1a). When HiPerMotif is connected (Phase LAST), ORCA becomes the correctness
oracle for HiPerMotif's |Aut|-normalized orbit counts.
```

- [ ] **Step 5: Gitignore the compiled binary**

Append to `.gitignore`:
```
# Compiled ORCA binary (source is vendored; binary is built locally)
third_party/orca/orca
```

Run to confirm the binary is ignored but the source is tracked:
```bash
git check-ignore third_party/orca/orca && git status --porcelain third_party/orca/
```
Expected: `git check-ignore` prints `third_party/orca/orca`; `git status` lists `orca.cpp` and `PROVENANCE.md` as untracked (`??`) but **not** the `orca` binary.

- [ ] **Step 6: Commit**

```bash
git add third_party/orca/orca.cpp third_party/orca/PROVENANCE.md .gitignore
git commit -m "feat(orbit-1a): vendor ORCA graphlet-orbit counter source"
```

---

## Task 2: ORCA wrapper — build, constants, orbit→(k,motif_id) mapping

**Files:**
- Create: `src/datasets/orca_orbits.py`
- Test: `tests/test_orca_orbits.py`

This task delivers everything except the numeric correctness assertions (those are Task 3), so it can be reviewed as "the binary builds and the static mapping is right."

- [ ] **Step 1: Write the failing tests for build + static tables**

Create `tests/test_orca_orbits.py`:
```python
import numpy as np
import networkx as nx
import pytest

from src.datasets import orca_orbits as oo


def test_constants_present():
    assert oo.N_ORBITS == {4: 15, 5: 73}
    assert oo.ORCA_ORBIT_BASE == 2000
    assert oo.DEGREE_ORBIT == 0
    # triangle and 4-clique orbit indices are the documented ORCA size-4 numbering;
    # Task 3 validates them against hand-computed graphs (the rigor gate).
    assert oo.TRIANGLE_ORBIT == 3
    assert oo.CLIQUE4_ORBIT == 14


def test_orbit_graphlet_size_table():
    t = oo.ORBIT_GRAPHLET_SIZE
    assert len(t) == 15
    assert set(t) <= {2, 3, 4}
    assert t[oo.DEGREE_ORBIT] == 2      # edge endpoint -> 2-node graphlet
    assert t[oo.TRIANGLE_ORBIT] == 3    # triangle -> 3-node graphlet
    assert t[oo.CLIQUE4_ORBIT] == 4     # K4 -> 4-node graphlet


def test_orbit_to_km():
    # degree orbit -> (k=2, motif_id=2000); 4-clique orbit -> (k=4, motif_id=2014)
    assert oo.orbit_to_km(0) == (2, 2000)
    assert oo.orbit_to_km(oo.TRIANGLE_ORBIT) == (3, 2003)
    assert oo.orbit_to_km(oo.CLIQUE4_ORBIT) == (4, 2014)


def test_ensure_orca_built_idempotent():
    p1 = oo.ensure_orca_built()
    assert p1.exists()
    p2 = oo.ensure_orca_built()  # second call is a no-op
    assert p2 == p1 and p2.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n motif-mpnn pytest tests/test_orca_orbits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.orca_orbits'` (or `AttributeError` once the file is stubbed).

- [ ] **Step 3: Implement the wrapper module**

Create `src/datasets/orca_orbits.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n motif-mpnn pytest tests/test_orca_orbits.py -v`
Expected: all four tests PASS. `test_ensure_orca_built_idempotent` compiles ORCA on first call.

(If `test_constants_present` fails because ORCA's actual triangle/4-clique columns differ from indices 3/14, do **not** silently edit the test — Task 3's hand-computed graphs are the source of truth; reconcile both the constant and this test to ORCA's verified output and record the corrected index in a code comment.)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/orca_orbits.py tests/test_orca_orbits.py
git commit -m "feat(orbit-1a): ORCA wrapper with build + orbit->(k,motif_id) mapping"
```

---

## Task 3: Validate orbit counts against hand-computed oracles

**Files:**
- Test: `tests/test_orca_orbits.py` (append)

This is the correctness gate. Every assertion is a value computed by hand from the graph's definition.

- [ ] **Step 1: Write the failing correctness tests**

Append to `tests/test_orca_orbits.py`:
```python
def _edges(g):
    return list(g.edges())


def test_k4_orbits():
    # K4: every node has degree 3, is in C(3,2)=3 triangles, and 1 four-clique.
    g = nx.complete_graph(4)
    m = oo.count_orbits(_edges(g), 4)
    assert m.shape == (4, 15)
    assert np.all(m[:, oo.DEGREE_ORBIT] == 3)
    assert np.all(m[:, oo.TRIANGLE_ORBIT] == 3)
    assert np.all(m[:, oo.CLIQUE4_ORBIT] == 1)


def test_c5_orbits():
    # 5-cycle: degree 2 everywhere; no triangles, no 4-cliques.
    g = nx.cycle_graph(5)
    m = oo.count_orbits(_edges(g), 5)  # num_nodes argument is 5
    # NOTE: arg is num_nodes (=5), not graphlet_size; graphlet_size defaults to 4.
    assert m.shape == (5, 15)
    assert np.all(m[:, oo.DEGREE_ORBIT] == 2)
    assert np.all(m[:, oo.TRIANGLE_ORBIT] == 0)
    assert np.all(m[:, oo.CLIQUE4_ORBIT] == 0)


def test_degree_orbit_matches_networkx():
    # Cross-check orbit 0 against an independent tool on a random small graph.
    g = nx.gnp_random_graph(20, 0.3, seed=7)
    m = oo.count_orbits(_edges(g), 20)
    deg = np.array([d for _, d in sorted(g.degree())], dtype=np.int64)
    assert np.array_equal(m[:, oo.DEGREE_ORBIT], deg)


def test_isolated_node_is_zero_row():
    # Node 3 is isolated; its entire orbit row must be zero.
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])  # triangle on 0,1,2
    m = oo.count_orbits(_edges(g), 4)
    assert np.all(m[3, :] == 0)
    assert m[0, oo.TRIANGLE_ORBIT] == 1


@pytest.mark.slow
def test_srg_method_validation():
    # The method-validation oracle: Shrikhande vs 4x4-rook are 1-WL-identical with
    # equal triangle counts but differ in 4-cliques (0 vs 2 per node).
    import torch
    from src.datasets.expressivity import make_shrikhande, make_rook_4x4

    def edges_of(ei):
        return [(int(u), int(v)) for u, v in ei.t().tolist() if u != v]

    sh = oo.count_orbits(edges_of(make_shrikhande()), 16)
    rk = oo.count_orbits(edges_of(make_rook_4x4()), 16)
    # triangle orbit: equal totals, both nonzero
    assert sh[:, oo.TRIANGLE_ORBIT].sum() == rk[:, oo.TRIANGLE_ORBIT].sum()
    assert sh[:, oo.TRIANGLE_ORBIT].sum() > 0
    # 4-clique orbit separates the cospectral pair
    assert np.all(sh[:, oo.CLIQUE4_ORBIT] == 0)
    assert np.all(rk[:, oo.CLIQUE4_ORBIT] == 2)
```

- [ ] **Step 2: Run the correctness tests to verify they fail (then pass once ORCA is correct)**

Run: `conda run -n motif-mpnn pytest tests/test_orca_orbits.py -v -m "not slow"`
Expected first run: these new tests are collected and execute. If the named orbit constants are correct, they PASS immediately; if ORCA's numbering differs from indices 3/14, K4/C5 fail — that is the validation gate firing.

- [ ] **Step 3: Reconcile any index mismatch (only if Step 2 fails on triangle/4-clique)**

If `test_k4_orbits`/`test_c5_orbits` fail, find ORCA's true triangle and 4-clique columns empirically and fix the constants — do not weaken the tests:
```bash
conda run -n motif-mpnn python -c "
import networkx as nx
from src.datasets import orca_orbits as oo
m = oo.count_orbits(list(nx.complete_graph(4).edges()), 4)
print('K4 per-node orbit row:', m[0].tolist())
# triangle col == 3 for every node; 4-clique col == 1 for every node
print('cols==3 (triangle candidates):', [i for i,v in enumerate(m[0]) if v==3])
print('cols==1 (4-clique candidates):', [i for i,v in enumerate(m[0]) if v==1])
"
```
Then update `TRIANGLE_ORBIT` / `CLIQUE4_ORBIT` (and `ORBIT_GRAPHLET_SIZE` if needed) in `orca_orbits.py`, leaving a comment citing the K4 evidence. Re-run Step 2.

- [ ] **Step 4: Run the full suite including the slow SRG test**

Run: `conda run -n motif-mpnn pytest tests/test_orca_orbits.py -v`
Expected: all tests PASS, including `test_srg_method_validation`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_orca_orbits.py src/datasets/orca_orbits.py
git commit -m "test(orbit-1a): validate ORCA orbits on K4/C5/star/SRG oracles"
```

---

## Task 4: Add `--features orbit` backend to generate_motifs.py

**Files:**
- Modify: `scripts/preprocess/generate_motifs.py`
- Test: `tests/test_generate_motifs_orbit.py`

The orbit path is additive: a new `--features {legacy,orbit}` flag (default `legacy`), an orbit CSV writer to a **distinct** filename, and an orbit runner. The legacy code paths are not touched.

- [ ] **Step 1: Write the failing end-to-end + unit test**

Create `tests/test_generate_motifs_orbit.py`:
```python
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_orbit_rows_helper_on_triangle():
    # Unit-level: the row-emitter yields (node_id, k, motif_id, count) in the 2000+ range.
    sys.path.insert(0, str(REPO))
    from scripts.preprocess.generate_motifs import _count_orbits_single
    import networkx as nx
    g = nx.complete_graph(4)
    rows = _count_orbits_single(g, 4)
    # all motif_ids in the orbit namespace
    assert rows, "expected non-empty rows for K4"
    assert all(2000 <= mid < 2073 for (_n, _k, mid, _c) in rows)
    # degree orbit present: (node, k=2, motif_id=2000, count=3)
    assert (0, 2, 2000, 3) in rows
    # 4-clique orbit present: (node, k=4, motif_id=2014, count=1)
    assert (0, 4, 2014, 1) in rows


@pytest.mark.slow
def test_generate_orbit_csv_on_csl(tmp_path):
    out_dir = tmp_path / "csl"
    cmd = [
        sys.executable, "-m", "scripts.preprocess.generate_motifs",
        "--dataset", "csl", "--features", "orbit",
        "--out-dir", str(out_dir), "--force",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    csv_p = out_dir / "node_motifs_orbit.csv"
    assert csv_p.exists()
    df = pd.read_csv(csv_p, comment="#")
    assert set(df.columns) == {"graph_id", "node_id", "k", "motif_id", "count"}
    # orbit namespace only; at least the degree orbit spans many graphs
    assert df["motif_id"].min() >= 2000
    assert df["motif_id"].max() < 2073
    assert df["graph_id"].nunique() == 150  # CSL has 150 graphs

    # Round-trip through the UNCHANGED loader: copy the orbit CSV in as the
    # loader's fixed filename (loader selection of the orbit file is Phase 1b).
    import shutil
    from src.datasets.motif_loader import build_or_load_node_motif_X
    rt = tmp_path / "rt"
    rt.mkdir()
    shutil.copy(csv_p, rt / "node_motifs.csv")
    # single CSL graph has 41 nodes; load just graph_id==0's rows
    g0 = df[df["graph_id"] == 0].drop(columns=["graph_id"])
    g0.to_csv(rt / "node_motifs.csv", index=False)
    art = build_or_load_node_motif_X("csl", num_nodes=41, precompute_dir=rt)
    assert art.X is not None
    assert art.X.shape[0] == 41
    assert art.X.shape[1] >= 1  # >=1 distinct orbit column built from the CSV
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n motif-mpnn pytest tests/test_generate_motifs_orbit.py -v`
Expected: FAIL — `ImportError: cannot import name '_count_orbits_single'` (helper not yet added).

- [ ] **Step 3: Add the orbit row-emitter and CSV writer**

In `scripts/preprocess/generate_motifs.py`, add after `_count_substructures_single` (around line 256):
```python
def _count_orbits_single(g_nx, num_nodes: int, graphlet_size: int = 4):
    """Per-node ORCA orbit rows: (node_id, k, motif_id, count) for nonzero counts.

    Each orbit o -> (k = graphlet node count, motif_id = 2000 + o), disjoint from
    all legacy encodings. ORCA is exact/deterministic (see src/datasets/orca_orbits).
    """
    import sys as _sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from src.datasets.orca_orbits import count_orbits, orbit_to_km

    edges = [(u, v) for u, v in g_nx.edges() if u != v]
    mat = count_orbits(edges, num_nodes, graphlet_size=graphlet_size)
    rows = []
    n_orbits = mat.shape[1]
    for u in range(num_nodes):
        for o in range(n_orbits):
            c = int(mat[u, o])
            if c > 0:
                k, motif_id = orbit_to_km(o)
                rows.append((u, k, motif_id, c))
    return rows
```

And add an orbit-CSV writer after `_write_tu_csv` (around line 287):
```python
def _write_orbit_csv(all_rows, out_path: Path, topk: int, multigraph: bool):
    """Write node_motifs_orbit.csv (graph_id present iff multigraph)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(f"# motif_topk={topk} features=orbit\n")
        writer = csv.writer(f)
        if multigraph:
            writer.writerow(["graph_id", "node_id", "k", "motif_id", "count"])
            for row in sorted(all_rows, key=lambda r: (r[0], r[1], r[2], r[3])):
                writer.writerow(list(row))
        else:
            writer.writerow(["node_id", "k", "motif_id", "count"])
            for row in sorted(all_rows, key=lambda r: (r[0], r[1], r[2])):
                writer.writerow(list(row))
    print(f"[OK] Wrote {len(all_rows)} orbit rows to {out_path}")
```

- [ ] **Step 4: Run the unit test to verify the helper passes**

Run: `conda run -n motif-mpnn pytest tests/test_generate_motifs_orbit.py::test_orbit_rows_helper_on_triangle -v`
Expected: PASS.

- [ ] **Step 5: Wire `--features` / `--graphlet-size` flags and the orbit runner**

In `main()`, add the two flags (after the `--k/--topk` arg, around line 446):
```python
    parser.add_argument(
        "--features",
        default="legacy",
        choices=["legacy", "orbit"],
        help="Feature backend: 'legacy' = degree/wedge/triangle (default); "
             "'orbit' = ORCA per-node graphlet-orbit counts -> node_motifs_orbit.csv.",
    )
    parser.add_argument(
        "--graphlet-size",
        dest="graphlet_size",
        type=int,
        default=4,
        choices=[4, 5],
        help="ORCA graphlet size for --features orbit (default 4; 5 is a smoke-test path).",
    )
```

Thread them through the call in `main()`'s loop (the `_run_dataset(...)` call, around line 457):
```python
        _run_dataset(
            dataset=ds,
            tool=args.tool,
            root=root,
            out_dir_override=out_dir_override,
            force=args.force,
            verify=args.verify,
            topk=args.topk,
            features=args.features,
            graphlet_size=args.graphlet_size,
        )
```

Update the `_run_dataset` signature (line 319) to accept them:
```python
def _run_dataset(dataset: str, tool: str, root: Path, out_dir_override: Path | None,
                 force: bool, verify: bool, topk: int,
                 features: str = "legacy", graphlet_size: int = 4):
```

Add an orbit branch at the top of `_run_dataset`, right after `out_dir`/`out_path` are set but **before** the legacy `if dataset in PLANETOID_DATASETS:` block (around line 322). This early-returns so none of the legacy code runs:
```python
    if features == "orbit":
        orbit_path = out_dir / "node_motifs_orbit.csv"
        if orbit_path.exists() and not force:
            print(f"[SKIP] {orbit_path} already exists. Use --force to recompute.")
            return
        print(f"[INFO] ORCA orbit features (graphlet_size={graphlet_size}) for {dataset} ...")
        from tqdm import tqdm
        if dataset in PLANETOID_DATASETS:
            G_nx, num_nodes = _load_planetoid_as_nx(dataset, root)
            rows = _count_orbits_single(G_nx, num_nodes, graphlet_size)
            _write_orbit_csv(rows, orbit_path, topk, multigraph=False)
        else:
            if dataset in TU_DATASETS:
                graphs = _load_tu_graphs_as_nx(dataset, root)
            elif dataset in SYNTHETIC_DATASETS:
                graphs = _load_csl_graphs_as_nx(root)
            else:
                print(f"[ERROR] Unknown dataset: {dataset}")
                sys.exit(1)
            all_rows = []
            for graph_id, g_nx, num_nodes in tqdm(graphs, desc=f"  {dataset} orbits", unit="graph"):
                for node_id, k, motif_id, count in _count_orbits_single(g_nx, num_nodes, graphlet_size):
                    all_rows.append((graph_id, node_id, k, motif_id, count))
            _write_orbit_csv(all_rows, orbit_path, topk, multigraph=True)
        if verify:
            _run_verify(orbit_path)
        print(f"\n[DONE] Orbit CSV written to: {orbit_path}")
        return
```

- [ ] **Step 6: Run the full orbit test file (including the slow CSL e2e)**

Run: `conda run -n motif-mpnn pytest tests/test_generate_motifs_orbit.py -v`
Expected: both tests PASS. The CSL run writes `node_motifs_orbit.csv` (150 graphs) and the loader round-trip builds a non-empty `motif_x`.

- [ ] **Step 7: Confirm the legacy path is unchanged**

Run: `conda run -n motif-mpnn pytest tests/test_generate_motifs_csl.py -v`
Expected: PASS — the existing legacy-CSL generation test is unaffected by the additive orbit branch.

- [ ] **Step 8: Commit**

```bash
git add scripts/preprocess/generate_motifs.py tests/test_generate_motifs_orbit.py
git commit -m "feat(orbit-1a): --features orbit backend writes node_motifs_orbit.csv"
```

---

## Task 5: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `conda run -n motif-mpnn pytest -q`
Expected: all tests pass (including the new `test_orca_orbits.py` and `test_generate_motifs_orbit.py`, and the pre-existing suite). Note: slow tests run unless you pass `-m "not slow"`.

- [ ] **Step 2: 5-node smoke check (wrapper path only)**

Run:
```bash
conda run -n motif-mpnn python -c "
import networkx as nx
from src.datasets.orca_orbits import count_orbits, N_ORBITS
m = count_orbits(list(nx.complete_graph(5).edges()), 5, graphlet_size=5)
print('shape', m.shape, 'expected', (5, N_ORBITS[5]))
assert m.shape == (5, 73)
print('5-node smoke OK')
"
```
Expected: prints `5-node smoke OK` (size-5 path produces a `[5, 73]` matrix).

- [ ] **Step 3: Update CLAUDE.md / spec status**

Mark the Phase 1a spec done and add a one-line note under CLAUDE.md §13 (or §8) that `--features orbit` exists and writes `node_motifs_orbit.csv` with `motif_id` in the `2000+` orbit namespace. Stage and commit:
```bash
git add CLAUDE.md docs/superpowers/specs/2026-06-10-orbit-library-1a-design.md
git commit -m "docs(orbit-1a): record ORCA orbit backend in CLAUDE.md"
```

---

## Self-Review

**1. Spec coverage** (each spec section → task):
- §3.1 Vendored ORCA → Task 1.
- §3.2 wrapper (`ensure_orca_built`, `count_orbits`, `N_ORBITS`, `ORBIT_GRAPHLET_SIZE`, `orbit_to_km`, `ORCA_ORBIT_BASE`) → Task 2.
- §3.3 CSV manifest encoding (`2000+o`, loader unchanged) → Task 4 (writer) + Task 4 round-trip test (loader compatibility).
- §3.4 `generate_motifs.py` integration (`--features`, `--graphlet-size`, distinct filename, Planetoid/TU/CSL) → Task 4.
- §5 Error handling (g++ missing, ORCA nonzero exit, non-contiguous ids, 0-edge, determinism) → Task 2 (`ensure_orca_built` RuntimeError, `_normalize_edges` range check, 0-edge short-circuit, shape check).
- §6 Tests 1–8 → Task 2 (build idempotent, table), Task 3 (K4, C5, star/isolated, degree cross-check, SRG, table consistency), Task 4 (e2e CSL + loader round-trip).
- §2 "assert only hand-verified indices" → Task 3 Step 3 reconciliation procedure; constants carry the K4-evidence comment.
- Non-goals (no loader selection, no benchmarks, legacy untouched) → respected: Task 4 branch is additive + early-returns; Task 4 Step 7 proves legacy CSL test still passes.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every code step shows complete code; the one conditional step (Task 3 Step 3) is a real reconciliation procedure with an exact diagnostic command, not a placeholder.

**3. Type consistency:** `count_orbits(edges, num_nodes, graphlet_size=4) -> np.ndarray`, `orbit_to_km(o) -> (k, motif_id)`, `ORBIT_GRAPHLET_SIZE` (len-15 list), constants `DEGREE_ORBIT=0`/`TRIANGLE_ORBIT=3`/`CLIQUE4_ORBIT=14`, `ORCA_ORBIT_BASE=2000`, and `_count_orbits_single`/`_write_orbit_csv` signatures are used identically across Tasks 2–5. The C5 test comment flags the `num_nodes` vs `graphlet_size` argument-position footgun explicitly.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-10-orbit-library-1a.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
