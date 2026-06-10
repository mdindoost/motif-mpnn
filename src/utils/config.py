# src/utils/config.py  (only the load_config and helpers changed)
import copy
import dataclasses
import pathlib
import warnings
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
    # 'val_acc' for balanced datasets; 'val_macro_f1' for imbalanced (NCI1, ENZYMES)
    monitor: str = "val_acc"

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
    # When True, use canonical Planetoid public splits for comparability with published baselines
    use_public_split: bool = False

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

# FIX: safe dataclass construction — filters unknown keys and warns instead of crashing
def _safe_dataclass(cls, d: dict, section: str):
    """Construct dataclass cls from dict d, warning about and dropping unknown keys."""
    if not isinstance(d, dict):
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    extra = set(d.keys()) - known
    if extra:
        warnings.warn(
            f"Config section '{section}' has unknown keys (will be ignored): {sorted(extra)}. "
            f"Known fields: {sorted(known)}",
            UserWarning, stacklevel=3
        )
    return cls(**{k: v for k, v in d.items() if k in known})


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
    return "graph" if name in {"proteins", "nci1", "enzymes", "csl"} else "node"

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
        # FIX: use _safe_dataclass to warn about unknown keys instead of crashing
        ds = _safe_dataclass(DatasetConfig, merged.get("dataset", {}), "dataset")
        md = _safe_dataclass(ModelConfig, merged.get("model", {}), "model")
        tr = _safe_dataclass(TrainConfig, merged.get("train", {}), "train")
        op = _safe_dataclass(OptimConfig, merged.get("optim", {}), "optim")
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

    ds_root = "data/processed"
    # FIX: always infer task from known dataset name first; explicit YAML hint overrides
    ds_task = _task_by_name(ds_name)
    if isinstance(flat_ds, dict):
        ds_root = flat_ds.get("root", ds_root)
        if flat_ds.get("task") in {"node", "graph"}:
            ds_task = flat_ds["task"]

    # Model selection priority: variant > explicit model.name in experiment cfg > identity
    # We check the raw experiment cfg (not merged) to avoid picking up the defaults placeholder.
    explicit_model_block = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model_name = (explicit_model_block.get("name") or merged.get("variant") or "identity")

    # FIX: read use_public_split from top-level flat-style YAML key
    use_public_split_flag = bool(merged.get("use_public_split", False))
    ds = DatasetConfig(name=ds_name, task=ds_task, root=ds_root, use_public_split=use_public_split_flag)
    # FIX: build ModelConfig from the merged `model:` block so flat-style configs can
    # override hidden_dim/num_layers/dropout/layer_norm/residual (previously these were
    # silently dropped — only name was read). The resolved name (variant priority) wins.
    md = _safe_dataclass(ModelConfig, merged.get("model") or {}, "model")
    md.name = model_name
    # FIX: use _safe_dataclass to warn about unknown keys instead of crashing
    tr = _safe_dataclass(TrainConfig, merged.get("train") or {}, "train")
    op = _safe_dataclass(OptimConfig, merged.get("optim") or {}, "optim")

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


# FIX: known-value sets for validate_config
KNOWN_VARIANTS = {"gcn", "sage", "gat", "gin", "concat", "gate", "mix", "identity"}
KNOWN_DATASETS = {"cora", "citeseer", "pubmed", "proteins", "nci1", "enzymes", "csl"}
GRAPH_DATASETS = {"proteins", "nci1", "enzymes", "csl"}
NODE_DATASETS = {"cora", "citeseer", "pubmed"}


def validate_config(exp: ExperimentConfig) -> None:
    """Validate experiment config at startup. Raises ValueError for hard errors, warns for soft ones."""
    errors = []

    # Unknown variant
    variant = exp.variant or exp.model.name
    if variant not in KNOWN_VARIANTS:
        errors.append(f"Unknown variant/model '{variant}'. Known: {sorted(KNOWN_VARIANTS)}")

    # Unknown dataset
    ds_name = exp.dataset.name
    if ds_name not in KNOWN_DATASETS and ds_name != "dummy_node":
        warnings.warn(f"Dataset '{ds_name}' is not a recognized benchmark dataset. "
                      f"Known: {sorted(KNOWN_DATASETS)}", UserWarning)

    # Task/dataset mismatch
    expected_task = "graph" if ds_name in GRAPH_DATASETS else "node"
    if ds_name in KNOWN_DATASETS and exp.dataset.task != expected_task:
        errors.append(
            f"Dataset '{ds_name}' expects task='{expected_task}' but config has task='{exp.dataset.task}'."
        )

    # Patience sanity
    if exp.train.patience <= 0:
        errors.append(f"train.patience must be > 0, got {exp.train.patience}")

    # Monitor sanity
    if exp.train.monitor not in ("val_acc", "val_macro_f1"):
        errors.append(f"train.monitor must be 'val_acc' or 'val_macro_f1', got '{exp.train.monitor}'")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
