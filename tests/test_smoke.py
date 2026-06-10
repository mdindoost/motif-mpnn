def test_imports():
    import src.utils.registry as reg
    assert hasattr(reg, "MODEL_REGISTRY")
    assert hasattr(reg, "DATASET_REGISTRY")
