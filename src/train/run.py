# src/train/run.py
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from src.utils.seed import fix_seed

import pandas as pd
import torch

from src.utils.config import load_config, validate_config
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY

# Ensure registries populate via import side-effects
import src.datasets  # noqa: F401
import src.models    # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print resolved config, then exit without training")
    args = parser.parse_args()

    exp = load_config(args.config)
    validate_config(exp)  # FIX: validate config at startup before any work begins
    fix_seed(getattr(exp.train, "seed", 0))

    # ---- Dry-run: print resolved config and exit immediately ----
    if args.dry_run:
        print("Resolved config:")
        print(f"  dataset      : {exp.dataset.name}")
        print(f"  variant      : {exp.variant or exp.model.name}")
        print(f"  model        : {exp.model.name}")
        print(f"  epochs       : {exp.train.epochs}")
        print(f"  patience     : {exp.train.patience}")
        print(f"  lr           : {exp.optim.lr}")
        print(f"  weight_decay : {exp.optim.weight_decay}")
        print(f"  monitor      : {exp.train.monitor}")
        print(f"  seed         : {exp.train.seed}")
        print(f"  save_dir     : {exp.save_dir}")
        print(f"  config_path  : {os.path.abspath(args.config)}")
        print("DRY RUN — no training will be performed")
        raise SystemExit(0)

    print("[cfg]", {"dataset": exp.dataset.name, "model": exp.model.name, "variant": exp.variant})

    # ---------------- Dataset ----------------
    DatasetCls = DATASET_REGISTRY.get(exp.dataset.name)
    dataset = DatasetCls(root=exp.dataset.root,
                         use_public_split=getattr(exp.dataset, 'use_public_split', False))

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
    elif task == "graph":  # FIX: read motif_dim from TUWithMotifs wrapper for graph tasks
        ds_obj_inner = getattr(dataset, 'dataset', None)
        if ds_obj_inner is not None and hasattr(ds_obj_inner, 'motif_dim'):
            motif_dim = int(ds_obj_inner.motif_dim)

    # FIX: warn if a motif variant is requested but no motif features were found
    if exp.model.name in {"concat", "gate", "mix"} and motif_dim == 0:
        import warnings
        warnings.warn(
            f"Running '{exp.model.name}' variant but motif_dim=0 — no motif CSV found for '{exp.dataset.name}'. "
            f"Model will behave identically to plain GCN. Run scripts/preprocess/generate_motifs.py to generate motifs.",
            UserWarning
        )

    # ---------------- Model ----------------
    ModelCls = MODEL_REGISTRY.get(exp.model.name)
    model_kwargs = dict(
        in_dim=in_dim, out_dim=out_dim,
        hidden_dim=exp.model.hidden_dim,   # FIX: use config value, not hardcoded 64
        num_layers=exp.model.num_layers,   # FIX: use config value, not hardcoded 2
        dropout=exp.model.dropout,         # FIX: use config value, not hardcoded 0.5
        layer_norm=exp.model.layer_norm,   # FIX: use config value, not hardcoded True
        residual=exp.model.residual,       # FIX: use config value, not hardcoded True
        task=task,
    )
    # Only pass motif_dim to motif-aware variants
    if exp.model.name in {"concat", "gate", "mix"}:
        model_kwargs["motif_dim"] = motif_dim

    # Pass gate-specific knobs if present
    gate_cfg = {}
    if exp.model.name == "gate":
        # exp.raw holds the raw merged YAML (you added this in ExperimentConfig)
        gate_cfg = exp.raw.get("gate", {}) or {}

    # Pass mix-specific knobs if present

    mix_cfg = {}
    if exp.model.name == "mix":
        mix_cfg = (getattr(exp, "raw", {}) or {}).get("mix", {}) or {}
        
        # Back-compat: map old keys if present
        if not mix_cfg:
            raw = getattr(exp, "raw", {}) or {}
            if raw.get("use_A_motif", False):
                lam_raw = raw.get("lambda")
                lam = (lam_raw.get("value") if isinstance(lam_raw, dict) else lam_raw)
                if lam is not None:
                    mix_cfg["lambda_mix"] = float(lam)
        
        # Set defaults if not provided
        mix_cfg.setdefault("lambda_mix", 0.25)
        mix_cfg.setdefault("motif_topk", 10)
        mix_cfg.setdefault("sim_metric", "cosine")
        mix_cfg.setdefault("self_loop", True)

    # Merge all config dicts into model_kwargs
    model = ModelCls(**{**model_kwargs, **gate_cfg, **mix_cfg})    
    
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
    patience = int(getattr(exp.train, 'patience', 50))
    monitor = str(getattr(exp.train, 'monitor', 'val_acc'))
    lr = float(getattr(exp.optim, 'lr', 0.01))
    wd = float(getattr(exp.optim, 'weight_decay', 0.0))

    # Graph tasks need a positive batch size; node tasks can be full-batch (0)
    raw_bs = int(getattr(exp.train, 'batch_size', 0))
    batch_size = raw_bs if (task == "node" or raw_bs > 0) else 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device} for up to {epochs} epochs (patience={patience}, monitor={monitor})...")

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
                                save_dir=save_dir, num_classes=out_dim,
                                patience=patience, monitor=monitor, device=device)
    else:
        splits = getattr(dataset, 'splits', None)
        ds_obj = getattr(dataset, 'dataset', None)
        if splits is None or ds_obj is None:
            raise RuntimeError('Graph dataset missing splits or dataset object')
        final = train_graph_task(model, ds_obj, splits,
                                 epochs=epochs, lr=lr, weight_decay=wd,
                                 save_dir=save_dir, num_classes=out_dim,
                                 patience=patience, monitor=monitor,
                                 batch_size=batch_size, device=device)

    best_val_epoch = final.pop('best_val_epoch', 'N/A')
    total_epochs = len(list(pd.read_csv(save_dir / 'metrics.csv').iterrows())) \
        if (save_dir / 'metrics.csv').exists() else 'N/A'

    # ---------------- Task B: write run_result.json ----------------
    run_ts = datetime.now().isoformat(timespec="seconds")
    run_result = {
        "dataset":        exp.dataset.name,
        "variant":        exp.variant or exp.model.name,
        "motif_dim":      motif_dim,
        "test_acc":       final.get('test_acc', None),
        "test_macro_f1":  final.get('test_macro_f1', None),
        "best_val_epoch": best_val_epoch,
        "total_epochs":   total_epochs,
        "seed":           int(getattr(exp.train, 'seed', 0)),
        "config_path":    os.path.abspath(args.config),
        "timestamp":      run_ts,
    }
    with open(save_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2)

    # ---------------- Task C: append to results/all_runs.csv ----------------
    all_runs_path = Path(exp.save_dir).parent / "all_runs.csv"
    # Ensure the parent directory exists (it should — save_dir was just created above)
    all_runs_path.parent.mkdir(parents=True, exist_ok=True)
    _CSV_COLUMNS = [
        "timestamp", "config_path", "dataset", "variant", "motif_dim",
        "test_acc", "test_macro_f1", "best_val_epoch", "total_epochs", "seed",
    ]
    row_df = pd.DataFrame([{col: run_result[col] for col in _CSV_COLUMNS}])
    write_header = not all_runs_path.exists()
    row_df.to_csv(all_runs_path, mode="a", header=write_header, index=False)

    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"  dataset      : {exp.dataset.name}")
    print(f"  variant      : {exp.variant or exp.model.name}")
    print(f"  motif_dim    : {motif_dim}")
    print(f"  test_acc     : {final.get('test_acc', 'N/A'):.4f}" if isinstance(final.get('test_acc'), float) else f"  test_acc     : {final.get('test_acc', 'N/A')}")
    print(f"  test_f1      : {final.get('test_macro_f1', 'N/A'):.4f}" if isinstance(final.get('test_macro_f1'), float) else f"  test_f1      : {final.get('test_macro_f1', 'N/A')}")
    print(f"  best_val_ep  : {best_val_epoch}")
    print(f"  total_epochs : {total_epochs}")
    print(f"  run_dir      : {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
