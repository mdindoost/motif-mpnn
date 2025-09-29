# src/train/run.py
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from src.utils.seed import fix_seed

import torch

from src.utils.config import load_config
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY

# Ensure registries populate via import side-effects
import src.datasets  # noqa: F401
import src.models    # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    args = parser.parse_args()

    exp = load_config(args.config)
    fix_seed(getattr(exp.train, "seed", 0))

    print("[cfg]", {"dataset": exp.dataset.name, "model": exp.model.name, "variant": exp.variant})

    # ---------------- Dataset ----------------
    DatasetCls = DATASET_REGISTRY.get(exp.dataset.name)
    dataset = DatasetCls(root=exp.dataset.root)

    # Infer dims + task from dataset bundle
    if getattr(dataset, "task", "node") == "node":
        in_dim = int(dataset.num_features if hasattr(dataset, "num_features")
                     else dataset.data.x.size(-1))
        out_dim = int(dataset.num_classes)
        task = "node"
    else:
        in_dim = int(dataset.num_features)
        out_dim = int(dataset.num_classes)
        task = "graph"

    # Motif dim (node task only for now)
    motif_dim = 0
    if task == "node" and getattr(dataset, "motif_x", None) is not None:
        motif_dim = int(dataset.motif_x.size(1))

    # ---------------- Model ----------------
    ModelCls = MODEL_REGISTRY.get(exp.model.name)
    model_kwargs = dict(
        in_dim=in_dim, out_dim=out_dim,
        hidden_dim=64, num_layers=2, dropout=0.5,
        layer_norm=True, residual=True, task=task,
    )
    # Only pass motif_dim to motif-aware variants
    if exp.model.name in {"concat", "gate", "mix"}:
        model_kwargs["motif_dim"] = motif_dim
    model = ModelCls(**model_kwargs)

    # ---------------- Run dir + manifest ----------------
    save_dir = Path(exp.save_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp.run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    motif_manifest = getattr(dataset, "motif_manifest", {})
    manifest = {
        "config_path": os.path.abspath(args.config),
        "dataset": exp.dataset.name,
        "task": task,
        "model": exp.model.name,
        "variant": exp.variant,
        "normalize": exp.normalize,
        "pruning": exp.pruning,
        "motif_dim": motif_dim,
        "motif_manifest_preview_keys": list(motif_manifest.keys())[:5] if motif_manifest else [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(save_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ---------------- Sanity prints ----------------
    print("=== Phase A/D sanity check ===")
    print("Dataset:", exp.dataset.name, "→", dataset.__class__.__name__)
    if task == "node":
        print(f"Task: {task} | in_dim: {in_dim} out_dim: {out_dim} motif_dim: {motif_dim}")
    else:
        print(f"Task: {task} | in_dim: {in_dim} out_dim: {out_dim}")
    print("Model:", exp.model.name, "→", model)
    print("Run dir:", str(save_dir))

    # ---------------- Phase E: training ----------------
    epochs = int(getattr(exp.train, 'epochs', 200))
    patience = int(getattr(exp.train, 'patience', 50))  # used by EarlyStopper default
    lr = float(getattr(exp.optim, 'lr', 0.01))
    wd = float(getattr(exp.optim, 'weight_decay', 0.0))

    # Graph tasks need a positive batch size; node tasks can be full-batch (0)
    raw_bs = int(getattr(exp.train, 'batch_size', 0))
    batch_size = raw_bs if (task == "node" or raw_bs > 0) else 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} for up to {epochs} epochs (patience={patience})...")

    from src.train.engine import train_node_task, train_graph_task

    if task == 'node':
        masks = getattr(dataset, 'splits', None)
        if masks is None:
            raise RuntimeError('Node dataset missing splits masks')
        # Attach motif_x if present so motif-aware models can consume it
        if getattr(dataset, 'motif_x', None) is not None:
            dataset.data.motif_x = dataset.motif_x
        final = train_node_task(model, dataset.data, masks,
                                epochs=epochs, lr=lr, weight_decay=wd,
                                save_dir=save_dir, num_classes=out_dim, device=device)
    else:
        splits = getattr(dataset, 'splits', None)
        ds_obj = getattr(dataset, 'dataset', None)
        if splits is None or ds_obj is None:
            raise RuntimeError('Graph dataset missing splits or dataset object')
        final = train_graph_task(model, ds_obj, splits,
                                 epochs=epochs, lr=lr, weight_decay=wd,
                                 save_dir=save_dir, num_classes=out_dim,
                                 batch_size=batch_size, device=device)

    print('Final test:', final)


if __name__ == "__main__":
    main()
