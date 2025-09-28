# src/utils/config.py  (only the load_config and helpers changed)
import copy
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml

# (dataclasses same as before)
@dataclass
class OptimConfig:
    lr: float = 0.01
    weight_decay: float = 0.0

@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 50
    seed: int = 0
    batch_size: int = 0

@dataclass
class ModelConfig:
    name: str = "identity"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    layer_norm: bool = True
    residual: bool = True

@dataclass
class DatasetConfig:
    name: str = "dummy_node"
    task: str = "node"
    root: str = "data/processed"

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run_name: str = "dev"
    save_dir: str = "results/logs"
    variant: Optional[str] = None
    use_motif_edge: Optional[bool] = None
    use_A_motif: Optional[bool] = None
    pruning: Dict[str, Any] = field(default_factory=dict)
    normalize: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base or {})
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _infer_dataset_name_from_filename(path: pathlib.Path) -> Optional[str]:
    # e.g., cora_concat.yml → cora
    stem = path.stem  # "cora_concat"
    if not stem:
        return None
    return stem.split("_", 1)[0] or None

def _task_by_name(name: str) -> str:
    return "graph" if name in {"proteins", "nci1", "enzymes"} else "node"

def load_config(path: str) -> ExperimentConfig:
    p = pathlib.Path(path)
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Project defaults (optional)
    defaults_path = p.parents[1] / "defaults.yml"
    defaults = {}
    if defaults_path.exists():
        with open(defaults_path, "r") as f:
            defaults = yaml.safe_load(f) or {}

    merged = _deep_update(defaults, cfg)

    # ---- NESTED STYLE ----
    if isinstance(merged.get("dataset"), dict) and "name" in merged["dataset"]:
        ds = DatasetConfig(**merged.get("dataset", {}))
        md = ModelConfig(**merged.get("model", {}))
        tr = TrainConfig(**merged.get("train", {}))
        op = OptimConfig(**merged.get("optim", {}))
        return ExperimentConfig(
            dataset=ds, model=md, train=tr, optim=op,
            run_name=merged.get("run_name", "dev"),
            save_dir=merged.get("save_dir", "results/logs"),
            pruning=merged.get("pruning", {}) or {},
            normalize=merged.get("normalize", {}) or {},
            variant=merged.get("variant"),
            use_motif_edge=merged.get("use_motif_edge"),
            use_A_motif=merged.get("use_A_motif"),
            raw=merged
        )

    # ---- FLAT STYLE (robust) ----
    flat_ds = merged.get("dataset", None)
    ds_name: Optional[str] = None

    if isinstance(flat_ds, str) and flat_ds.strip():
        ds_name = flat_ds.strip()
    elif isinstance(flat_ds, dict) and flat_ds.get("name"):
        ds_name = str(flat_ds["name"])
    else:
        # Fallback: infer from filename (e.g., cora_concat.yml → cora)
        ds_name = _infer_dataset_name_from_filename(p)

    if not ds_name:
        # Last resort: don't crash; make it explicit in run manifest
        ds_name = "dummy_node"

    ds_task = "node"
    ds_root = "data/processed"
    # Respect optional hints in flat dict
    if isinstance(flat_ds, dict):
        ds_task = flat_ds.get("task", ds_task)
        ds_root = flat_ds.get("root", ds_root)
    # Or infer task by known dataset name
    ds_task = ds_task or _task_by_name(ds_name)
    if ds_task not in {"node", "graph"}:
        ds_task = _task_by_name(ds_name)

    # Model selection: prefer explicit model.name; else use variant; else identity
    model_block = merged.get("model")
    model_name = None
    if isinstance(model_block, dict):
        model_name = model_block.get("name")
    if not model_name:
        model_name = merged.get("variant") or "identity"

    ds = DatasetConfig(name=ds_name, task=ds_task, root=ds_root)
    md = ModelConfig(name=model_name)
    tr = TrainConfig(**(merged.get("train") or {}))
    op = OptimConfig(**(merged.get("optim") or {}))

    return ExperimentConfig(
        dataset=ds, model=md, train=tr, optim=op,
        run_name=merged.get("run_name", "dev"),
        save_dir=merged.get("save_dir", "results/logs"),
        pruning=merged.get("pruning", {}) or {},
        normalize=merged.get("normalize", {}) or {},
        variant=merged.get("variant"),
        use_motif_edge=merged.get("use_motif_edge"),
        use_A_motif=merged.get("use_A_motif"),
        raw=merged
    )
