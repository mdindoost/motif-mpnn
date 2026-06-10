"""Regression tests for flat-style YAML model-hyperparameter plumbing.

Bug: flat-style configs silently dropped model.hidden_dim/num_layers/dropout/
layer_norm/residual (only `name` was read). Fix: build ModelConfig from the merged
`model:` block, with the resolved variant name taking priority.
"""
import textwrap
from pathlib import Path

from src.utils.config import load_config


DEFAULTS = textwrap.dedent(
    """
    run_name: dev
    save_dir: results/logs
    train: {epochs: 200, patience: 50, seed: 0, batch_size: 0, monitor: val_acc}
    optim: {lr: 0.01, weight_decay: 0.0}
    model:
      name: identity
      hidden_dim: 64
      num_layers: 2
      dropout: 0.5
      layer_norm: true
      residual: true
    """
)


def _write(tmp_path: Path, body: str) -> Path:
    # load_config reads defaults from p.parents[1]/defaults.yml
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "defaults.yml").write_text(DEFAULTS)
    exp_dir = tmp_path / "configs" / "experiments"
    exp_dir.mkdir(exist_ok=True)
    p = exp_dir / "cora_gcn.yml"
    p.write_text(body)
    return p


def test_flat_config_model_overrides_are_applied(tmp_path):
    body = textwrap.dedent(
        """
        dataset: cora
        variant: gcn
        run_name: cora_gcn
        model:
          hidden_dim: 128
          num_layers: 4
          dropout: 0.1
          layer_norm: false
        """
    )
    exp = load_config(str(_write(tmp_path, body)))
    assert exp.model.name == "gcn"          # variant priority preserved
    assert exp.model.hidden_dim == 128      # override now applied (was 64 before fix)
    assert exp.model.num_layers == 4        # override now applied (was 2 before fix)
    assert exp.model.dropout == 0.1
    assert exp.model.layer_norm is False


def test_flat_config_without_model_block_uses_defaults(tmp_path):
    # This is the TU-config case: no model block -> defaults (64/2/0.5) apply.
    body = textwrap.dedent(
        """
        dataset: cora
        variant: concat
        run_name: cora_concat
        """
    )
    exp = load_config(str(_write(tmp_path, body)))
    assert exp.model.name == "concat"
    assert exp.model.hidden_dim == 64
    assert exp.model.num_layers == 2
    assert exp.model.dropout == 0.5
    assert exp.model.layer_norm is True
    assert exp.model.residual is True
