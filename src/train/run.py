# src/train/run.py
import argparse
import json
import os

import src.datasets  # noqa: F401
import src.models    # noqa: F401

from datetime import datetime
from pathlib import Path

from src.utils.config import load_config
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    args = parser.parse_args()

    exp = load_config(args.config)
    print("[cfg]", {"dataset": exp.dataset.name, "model": exp.model.name, "variant": exp.variant})

    # Dataset by key (Phase A: dummy for all; Phase B will swap real loaders)
    DatasetCls = DATASET_REGISTRY.get(exp.dataset.name)
    dataset = DatasetCls(root=exp.dataset.root)

    # Model by key; for now identity for everything (Phase D will replace)
    ModelCls = MODEL_REGISTRY.get(exp.model.name)
    model = ModelCls(in_dim=1, out_dim=1)

    save_dir = Path(exp.save_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp.run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_path": os.path.abspath(args.config),
        "dataset": exp.dataset.name,
        "task": exp.dataset.task,
        "model": exp.model.name,
        "variant": exp.variant,
        "normalize": exp.normalize,
        "pruning": exp.pruning,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(save_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Phase A sanity check ===")
    print("Dataset:", exp.dataset.name, "→", dataset.__class__.__name__)
    print("Model:", exp.model.name, "→", model)
    print("Run dir:", str(save_dir))

if __name__ == "__main__":
    main()
