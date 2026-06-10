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
    # orbit namespace only; CSL has 150 graphs
    assert df["motif_id"].min() >= 2000
    assert df["motif_id"].max() < 2073
    assert df["graph_id"].nunique() == 150

    # Round-trip through the UNCHANGED loader: write graph 0's rows as the loader's
    # fixed filename (loader selection of the orbit file is Phase 1b, not here).
    from src.datasets.motif_loader import build_or_load_node_motif_X
    rt = tmp_path / "rt"
    rt.mkdir()
    g0 = df[df["graph_id"] == 0].drop(columns=["graph_id"])
    g0.to_csv(rt / "node_motifs.csv", index=False)
    art = build_or_load_node_motif_X("csl", num_nodes=41, precompute_dir=rt)
    assert art.X is not None
    assert art.X.shape[0] == 41
    assert art.X.shape[1] >= 1  # >=1 distinct orbit column built from the CSV
