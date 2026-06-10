import torch
from src.utils.registry import MODEL_REGISTRY
import src.models  # populate registry


def test_gin_registered_and_forward_graph():
    assert "gin" in MODEL_REGISTRY
    Factory = MODEL_REGISTRY.get("gin")
    model = Factory(in_dim=1, out_dim=10, hidden_dim=16, num_layers=2,
                    dropout=0.0, layer_norm=False, residual=False, task="graph")
    # tiny 2-graph batch: 3 nodes each
    x = torch.ones(6, 1)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                               [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    data = type("D", (), {"x": x, "edge_index": edge_index, "batch": batch})
    out = model(data)
    assert out.shape == (2, 10)
