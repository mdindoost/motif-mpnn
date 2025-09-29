# src/train/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import time
from pathlib import Path


import torch
from torch import nn
from torch.optim import Adam
import pandas as pd
from torch_geometric.loader import DataLoader


from .metrics import accuracy, macro_f1


@dataclass
class EarlyStopper:
    patience: int = 50
    best_val: float = float('-inf')
    best_state: Dict[str, Any] | None = None
    epochs_since: int = 0


    def step(self, val_score: float, model: nn.Module) -> bool:
        if val_score > self.best_val:
            self.best_val = val_score
            self.epochs_since = 0
            # store CPU state dict to keep GPU memory free
            self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.epochs_since += 1
            return self.epochs_since > self.patience



def _to_device(d, device):
    if hasattr(d, 'to'):
        return d.to(device)
    return d


def train_node_task(model: nn.Module, data, masks: Dict[str, torch.Tensor], *,
    epochs: int, lr: float, weight_decay: float, save_dir: Path,
    num_classes: int, device: str = 'cpu') -> Dict[str, float]:
    
    model.to(device)
    data = _to_device(data, device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=50)


    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data)
        loss = loss_fn(logits[masks['train']], data.y[masks['train']])
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data)
            tr_acc = accuracy(logits[masks['train']], data.y[masks['train']])
            va_acc = accuracy(logits[masks['val']], data.y[masks['val']])
            te_acc = accuracy(logits[masks['test']], data.y[masks['test']])
            va_f1 = macro_f1(logits[masks['val']], data.y[masks['val']], num_classes)
            te_f1 = macro_f1(logits[masks['test']], data.y[masks['test']], num_classes)
            
        log_rows.append({
            'epoch': epoch,
            'train_loss': float(loss.item()),
            'train_acc': tr_acc,
            'val_acc': va_acc,
            'val_macro_f1': va_f1,
            'test_acc': te_acc,
            'test_macro_f1': te_f1,
        })


        if stopper.step(va_acc, model):
            break
        
    # write CSV
    df = pd.DataFrame(log_rows)
    (save_dir / 'metrics.csv').parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / 'metrics.csv', index=False)


    # restore best
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)


    # final evaluation on test
    model.eval()
    with torch.no_grad():
        logits = model(data)
        te_acc = accuracy(logits[masks['test']], data.y[masks['test']])
        te_f1 = macro_f1(logits[masks['test']], data.y[masks['test']], num_classes)
    return {'test_acc': te_acc, 'test_macro_f1': te_f1}

def train_graph_task(model: nn.Module, dataset, splits: Dict[str, list], *,
        epochs: int, lr: float, weight_decay: float, save_dir: Path,
        num_classes: int, batch_size: int = 64, device: str = 'cpu') -> Dict[str, float]:
    
    batch_size = int(batch_size) if (batch_size and batch_size > 0) else 64


    model.to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=50)


    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            sampler=torch.utils.data.SubsetRandomSampler(splits['train']))
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            sampler=torch.utils.data.SubsetRandomSampler(splits['val']))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            sampler=torch.utils.data.SubsetRandomSampler(splits['test']))


    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = _to_device(batch, device)
        opt.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item())


        # eval
        def _eval(loader):
            model.eval()
            all_logits = []
            all_y = []
            with torch.no_grad():
                for b in loader:
                    b = _to_device(b, device)
                    all_logits.append(model(b).cpu())
                    all_y.append(b.y.cpu())
            logits = torch.cat(all_logits, dim=0)
            y = torch.cat(all_y, dim=0)
            return accuracy(logits, y), macro_f1(logits, y, num_classes)
        
        va_acc, va_f1 = _eval(val_loader)
        te_acc, te_f1 = _eval(test_loader)


        log_rows.append({
            'epoch': epoch,
            'train_loss': total_loss,
            'val_acc': va_acc,
            'val_macro_f1': va_f1,
            'test_acc': te_acc,
            'test_macro_f1': te_f1,
        })


        if stopper.step(va_acc, model):
            break
        


    # write CSV
    df = pd.DataFrame(log_rows)
    (save_dir / 'metrics.csv').parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / 'metrics.csv', index=False)


    # restore best
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)


    # final test
    te_acc, te_f1 = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        all_logits, all_y = [], []
        for b in test_loader:
            b = _to_device(b, device)
            all_logits.append(model(b).cpu())
            all_y.append(b.y.cpu())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        te_acc = accuracy(logits, y)
        te_f1 = macro_f1(logits, y, num_classes)
    return {'test_acc': te_acc, 'test_macro_f1': te_f1}