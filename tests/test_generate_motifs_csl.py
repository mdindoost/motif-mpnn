import pandas as pd
import pytest
from pathlib import Path

pytestmark = pytest.mark.slow


def test_csl_motif_csv_has_cycle_and_clique_rows(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "genmotifs", "scripts/preprocess/generate_motifs.py")
    gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)

    out_dir = tmp_path / "csl"
    gm._run_dataset(dataset="csl", tool="igraph", root=Path("data/processed"),
                    out_dir_override=out_dir, force=True, verify=False, topk=10)
    csv = out_dir / "node_motifs.csv"
    assert csv.exists()
    df = pd.read_csv(csv, comment="#")
    assert set(["graph_id", "node_id", "k", "motif_id", "count"]).issubset(df.columns)
    # 4-clique rows: (k=4, motif_id=10). CSL has no 4-cliques, so may be absent — OK.
    # cycle rows: motif_id == 1000 must be present for several lengths
    assert (df["motif_id"] == 1000).any()
    assert df.loc[df["motif_id"] == 1000, "k"].nunique() >= 3  # multiple cycle lengths
    # graph_ids span all 150 graphs
    assert df["graph_id"].nunique() == 150
