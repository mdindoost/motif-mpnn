"""Multi-seed sweep runner for motif-mpnn experiments.

Usage:
    python scripts/runs/sweep.py \
        --config configs/experiments/cora_gcn.yml \
        --config configs/experiments/cora_concat.yml \
        --seeds 42 123 456 \
        --out results/tables/sweep.tex
"""
import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    import copy
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _write_temp_config(config_path: str, seed: int) -> str:
    """Write a temp YAML with train.seed overridden; return temp file path."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Deep-merge seed override into train section
    seed_override = {"train": {"seed": seed}}
    merged = _deep_merge(cfg, seed_override)

    # Deterministic temp filename based on config + seed
    key = f"{config_path}:{seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    tmp_path = f"/tmp/sweep_{h}.yml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, allow_unicode=True)
    return tmp_path


def _parse_summary(stdout: str) -> Dict[str, Optional[float]]:
    """Extract test_acc, test_f1, best_val_epoch from RUN SUMMARY block."""
    result: Dict[str, Optional[float]] = {
        "test_acc": None,
        "test_macro_f1": None,
        "best_val_epoch": None,
    }
    # Match lines like:  test_acc     : 0.7563
    acc_m = re.search(r"test_acc\s*:\s*([0-9]+\.[0-9]+)", stdout)
    if acc_m:
        result["test_acc"] = float(acc_m.group(1))

    f1_m = re.search(r"test_f1\s*:\s*([0-9]+\.[0-9]+)", stdout)
    if f1_m:
        result["test_macro_f1"] = float(f1_m.group(1))

    ep_m = re.search(r"best_val_ep\s*:\s*([0-9]+)", stdout)
    if ep_m:
        result["best_val_epoch"] = int(ep_m.group(1))

    return result


def _config_label(config_path: str) -> str:
    """Human-readable label from config filename, e.g. cora_gcn."""
    return Path(config_path).stem


def _append_all_runs(row: dict, results_dir: str) -> None:
    """Append one completed run to results/all_runs.csv (create if absent)."""
    all_runs_path = Path(results_dir).parent / "all_runs.csv"
    fieldnames = ["timestamp", "config", "seed", "dataset", "variant",
                  "test_acc", "test_macro_f1", "best_val_epoch"]
    write_header = not all_runs_path.exists()
    with open(all_runs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Plain-text summary table
# ---------------------------------------------------------------------------

def _print_summary_table(
    results: List[dict],
    configs: List[str],
) -> None:
    """Print mean ± std of test_acc and test_f1 per config to stdout."""
    import statistics

    print("\n" + "=" * 72)
    print("SWEEP SUMMARY")
    print("=" * 72)
    header = f"{'Config':<30}  {'test_acc':>14}  {'test_f1':>14}  {'N':>3}"
    print(header)
    print("-" * 72)

    for cfg_path in configs:
        label = _config_label(cfg_path)
        rows = [r for r in results if r["config"] == cfg_path
                and r["test_acc"] is not None]
        n = len(rows)
        if n == 0:
            print(f"{label:<30}  {'N/A':>14}  {'N/A':>14}  {0:>3}")
            continue

        accs = [r["test_acc"] for r in rows]
        f1s  = [r["test_macro_f1"] for r in rows if r["test_macro_f1"] is not None]

        acc_mean = statistics.mean(accs)
        acc_std  = statistics.stdev(accs) if n > 1 else 0.0
        f1_mean  = statistics.mean(f1s)  if f1s  else float("nan")
        f1_std   = statistics.stdev(f1s) if len(f1s) > 1 else 0.0

        acc_str = f"{acc_mean:.4f} ± {acc_std:.4f}"
        f1_str  = f"{f1_mean:.4f} ± {f1_std:.4f}" if f1s else "N/A"
        print(f"{label:<30}  {acc_str:>14}  {f1_str:>14}  {n:>3}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _write_latex_table(
    results: List[dict],
    configs: List[str],
    out_path: str,
) -> None:
    """Write a booktabs-style LaTeX table to out_path."""
    import statistics

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Multi-seed sweep results (mean $\pm$ std)}",
        r"  \label{tab:sweep}",
        r"  \begin{tabular}{lrrc}",
        r"    \toprule",
        r"    Config & Test Acc & Test F1 & $N$ \\",
        r"    \midrule",
    ]

    for cfg_path in configs:
        label = _config_label(cfg_path).replace("_", r"\_")
        rows = [r for r in results if r["config"] == cfg_path
                and r["test_acc"] is not None]
        n = len(rows)
        if n == 0:
            lines.append(f"    {label} & N/A & N/A & 0 \\\\")
            continue

        accs = [r["test_acc"] for r in rows]
        f1s  = [r["test_macro_f1"] for r in rows if r["test_macro_f1"] is not None]

        acc_mean = statistics.mean(accs)
        acc_std  = statistics.stdev(accs) if n > 1 else 0.0
        f1_mean  = statistics.mean(f1s)  if f1s  else float("nan")
        f1_std   = statistics.stdev(f1s) if len(f1s) > 1 else 0.0

        acc_str = f"{acc_mean:.4f} $\\pm$ {acc_std:.4f}"
        f1_str  = f"{f1_mean:.4f} $\\pm$ {f1_std:.4f}" if f1s else "N/A"
        lines.append(f"    {label} & {acc_str} & {f1_str} & {n} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX table written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed sweep runner for motif-mpnn experiments."
    )
    parser.add_argument(
        "--config", dest="configs", action="append", required=True,
        metavar="CONFIG",
        help="Path to experiment YAML (can be repeated for multiple configs).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        metavar="SEED",
        help="Seeds to sweep over (default: 42 123 456).",
    )
    parser.add_argument(
        "--results-dir", default="results/logs",
        metavar="DIR",
        help="Directory where run logs are written (default: results/logs).",
    )
    parser.add_argument(
        "--out", default=None,
        metavar="FILE",
        help="Optional path for output LaTeX table file (default: stdout only).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands that would be run without executing them.",
    )
    args = parser.parse_args()

    configs: List[str] = args.configs
    seeds: List[int] = args.seeds
    total = len(configs) * len(seeds)

    print(f"Sweep: {len(configs)} config(s) × {len(seeds)} seed(s) = {total} run(s)")
    if args.dry_run:
        print("[DRY RUN] commands that would be executed:")

    results: List[dict] = []
    run_idx = 0

    for cfg_path in configs:
        for seed in seeds:
            run_idx += 1
            label = _config_label(cfg_path)
            print(f"\n[{run_idx}/{total}] {label} seed={seed} ...")

            tmp_path: Optional[str] = None
            try:
                tmp_path = _write_temp_config(cfg_path, seed)
                cmd = [sys.executable, "-m", "src.train.run", "--config", tmp_path]

                if args.dry_run:
                    print("  " + " ".join(cmd))
                    print(f"  (temp config: {tmp_path} — would be deleted after run)")
                    continue

                proc = subprocess.run(
                    cmd,
                    capture_output=False,   # let stdout/stderr stream to terminal
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = proc.stdout or ""
                # Stream output to terminal as well
                print(output, end="")

                parsed = _parse_summary(output)
                row = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "config": cfg_path,
                    "seed": seed,
                    "dataset": label.split("_")[0] if "_" in label else label,
                    "variant": label.split("_", 1)[1] if "_" in label else label,
                    "test_acc": parsed["test_acc"],
                    "test_macro_f1": parsed["test_macro_f1"],
                    "best_val_epoch": parsed["best_val_epoch"],
                }
                results.append(row)
                _append_all_runs(row, args.results_dir)

                if proc.returncode != 0:
                    print(f"  [WARNING] run exited with code {proc.returncode}")

            finally:
                if tmp_path and os.path.exists(tmp_path) and not args.dry_run:
                    os.remove(tmp_path)

    if args.dry_run:
        return

    if not results:
        print("\nNo results collected.")
        return

    _print_summary_table(results, configs)

    if args.out:
        _write_latex_table(results, configs, args.out)


if __name__ == "__main__":
    main()
