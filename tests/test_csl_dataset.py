from src.utils.registry import DATASET_REGISTRY
import src.datasets  # populate registry


def test_csl_registered_and_bundle_shape():
    assert "csl" in DATASET_REGISTRY
    ds = DATASET_REGISTRY.get("csl")(root="data/processed", split_seed=0)
    assert ds.task == "graph"
    assert ds.num_features == 1           # constant all-ones features
    assert ds.num_classes == 10
    assert len(ds.dataset) == 150
    # splits cover all 150 indices disjointly
    idx = ds.splits["train"] + ds.splits["val"] + ds.splits["test"]
    assert sorted(idx) == list(range(150))
    # a Data object has x and y
    d0 = ds.dataset[0]
    assert d0.x.shape[1] == 1
    assert int(d0.y) in range(10)
