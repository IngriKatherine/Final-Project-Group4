"""Training loop with early stopping, plus a prediction helper."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn


def _loss_fn(name: str) -> nn.Module:
    return {"mse": nn.MSELoss(), "huber": nn.HuberLoss(delta=1.0)}[name]


def train_model(model: nn.Module, train_loader, val_loader, cfg: dict, device,
                verbose: bool = True) -> tuple[nn.Module, dict]:
    """Mini-batch Adam training; keeps the weights with the lowest validation loss.

    Overfitting control: early stopping (patience), dropout, optional weight decay,
    gradient clipping. The validation set is the chronological block after training.
    """
    t = cfg["training"]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(1, t["patience"] // 3))
    loss_fn = _loss_fn(t["loss"])

    best_val, best_state, wait = np.inf, None, 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    start = time.time()

    for epoch in range(1, t["max_epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if t["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            running += loss.item() * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        sched.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}  best {best_val:.4f}")
        if wait >= t["patience"]:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.update(best_val_loss=float(best_val), epochs_run=epoch, train_seconds=time.time() - start)
    return model, history


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += loss_fn(model(xb), yb).item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i: i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
