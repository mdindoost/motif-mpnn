# src/train/metrics.py
from typing import Dict
import torch
import numpy as np


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.ndim == 1:
        pred = (logits > 0).long()
    else:
        pred = logits.argmax(dim=-1)
    y = y.to(pred.device)
    valid = y >= 0
    if valid.sum() == 0:
        return 0.0
    return (pred[valid] == y[valid]).float().mean().item()


@torch.no_grad()
def macro_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    # simple macro-F1 without sklearn
    if logits.ndim == 1:
        pred = (logits > 0).long()
    else:
        pred = logits.argmax(dim=-1)
    y = y.to(pred.device)
    valid = y >= 0
    if valid.sum() == 0:
        return 0.0
    pred = pred[valid]
    y = y[valid]
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))
