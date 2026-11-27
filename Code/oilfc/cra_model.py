"""Chronos-anchored gated residual LSTM (issue #10).

Inputs per sample: a window x (lookback, n_features), the scaled prior features
p (4,) and the prior quantiles in volatility units mz = m / sigma_h (3,).

    w_s   = softmax(MLP(x_s)),  x~_s = n_features * w_s * x_s     (variable selection)
    h_s   = LSTM(x~_s),  a = sum_s beta_s h_s                       (attention pooling)
    c     = GRN([a, p])
    g     = sigmoid(w_g c + b_g)                                     (b_g = gate_bias)
    q50   = mz50 + g * delta
    q10   = q50 - softplus(inv_softplus(mz50 - mz10) + d_lo)         (non-crossing)
    q90   = q50 + softplus(inv_softplus(mz90 - mz50) + d_hi)

All quantiles are in units of the ex-ante h-step volatility.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QUANTILES = (0.1, 0.5, 0.9)


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class VariableSelection(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ELU(), nn.Linear(hidden, n_features))
        self.n = n_features

    def forward(self, x):
        w = torch.softmax(self.net(x), dim=-1)
        return self.n * w * x, w                    # factor n keeps the input scale at uniform weights


class GRN(nn.Module):
    """Gated residual network (Lim et al. 2021)."""

    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.skip = nn.Linear(d_in, d)
        self.fc1 = nn.Linear(d_in, d)
        self.fc2 = nn.Linear(d, 2 * d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.drop(self.fc2(F.elu(self.fc1(x))))
        return self.norm(self.skip(x) + F.glu(h, dim=-1))


class GatedResidualLSTM(nn.Module):
    def __init__(self, n_features: int, n_prior: int = 4, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.1, feature_dropout: float = 0.1, gate_bias: float = -3.0):
        super().__init__()
        self.vsn = VariableSelection(n_features, hidden_size)
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.att = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, 1, bias=False))
        self.fuse = GRN(hidden_size + n_prior, hidden_size, dropout)
        self.head = nn.Linear(hidden_size, 3)       # delta, d_lo, d_hi
        self.gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)               # start exactly at the prior quantiles
        nn.init.constant_(self.gate.bias, gate_bias)
        self.feature_dropout = feature_dropout

    def forward(self, x, p, mz):
        if self.training and self.feature_dropout > 0:
            keep = torch.bernoulli(torch.full((x.shape[0], 1, x.shape[2]), 1 - self.feature_dropout,
                                              device=x.device))
            x = x * keep / (1 - self.feature_dropout)
        x, w = self.vsn(x)
        hs, _ = self.lstm(x)
        beta = torch.softmax(self.att(hs), dim=1)    # (B, T, 1)
        a = (beta * hs).sum(dim=1)
        c = self.fuse(torch.cat([a, p], dim=-1))
        delta, d_lo, d_hi = self.head(c).unbind(-1)
        g = torch.sigmoid(self.gate(c)).squeeze(-1)

        q50 = mz[:, 1] + g * delta
        lo = inv_softplus((mz[:, 1] - mz[:, 0]).clamp(min=1e-3))
        hi = inv_softplus((mz[:, 2] - mz[:, 1]).clamp(min=1e-3))
        q10 = q50 - F.softplus(lo + d_lo)
        q90 = q50 + F.softplus(hi + d_hi)
        return torch.stack([q10, q50, q90], dim=-1), g, w.mean(dim=1)


def build_cra_model(cfg: dict, n_features: int) -> GatedResidualLSTM:
    m = cfg["model"]
    return GatedResidualLSTM(n_features, 4, m["hidden_size"], m["num_layers"], m["dropout"],
                             m["feature_dropout"], m["gate_bias"])


def cra_loss(q: torch.Tensor, z: torch.Tensor, g: torch.Tensor, t: dict) -> torch.Tensor:
    """Pinball over the three quantiles + Huber on the median + gate penalty."""
    pin = 0.0
    for k, tau in enumerate(QUANTILES):
        e = z - q[:, k]
        pin = pin + torch.maximum(tau * e, (tau - 1) * e).mean()
    hub = F.huber_loss(q[:, 1], z, delta=t["huber_delta"])
    return pin + t["lambda_huber"] * hub + t["lambda_gate"] * g.mean()


def _tensors(split, device):
    mz = split.m_raw / split.sigma_h[:, None]
    return [torch.as_tensor(a, dtype=torch.float32, device=device)
            for a in (split.X, split.prior, mz, split.z)]


@torch.no_grad()
def _eval_loss(model, tensors, t):
    model.eval()
    x, p, mz, z = tensors
    q, g, _ = model(x, p, mz)
    return float(cra_loss(q, z, g, t))


def train_cra(model: GatedResidualLSTM, train, val, cfg: dict, device, seed: int,
              verbose: bool = False) -> tuple[GatedResidualLSTM, dict]:
    """AdamW + cosine schedule, early stopping on the validation loss."""
    t = cfg["training"]
    model.to(device)
    tr, va = _tensors(train, device), _tensors(val, device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["max_epochs"])
    gen = torch.Generator().manual_seed(seed)
    n = len(tr[0])

    best_val, best_state, wait = np.inf, None, 0
    history = {"train_loss": [], "val_loss": []}
    start = time.time()
    for epoch in range(1, t["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        running = 0.0
        for i in range(0, n, t["batch_size"]):
            b = perm[i: i + t["batch_size"]]
            q, g, _ = model(tr[0][b], tr[1][b], tr[2][b])
            loss = cra_loss(q, tr[3][b], g, t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            running += float(loss) * len(b)
        sched.step()
        val_loss = _eval_loss(model, va, t)
        history["train_loss"].append(running / n)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:4d}  train {running / n:.4f}  val {val_loss:.4f}")
        if wait >= t["patience"]:
            break
    model.load_state_dict(best_state)
    history.update(best_val_loss=float(best_val), epochs_run=epoch, train_seconds=time.time() - start)
    return model, history


@torch.no_grad()
def predict_cra(model: GatedResidualLSTM, split, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantiles of the h-step log return (N, 3), gate (N,), mean variable-selection weights (F,)."""
    model.eval()
    x, p, mz, _ = _tensors(split, device)
    q, g, w = model(x, p, mz)
    r = q.cpu().numpy() * split.sigma_h[:, None]
    return r, g.cpu().numpy(), w.mean(dim=0).cpu().numpy()
