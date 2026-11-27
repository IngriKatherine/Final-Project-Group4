"""Chronos-feature LSTM (issue #10 follow-up).

Chronos-2 is an input, not the starting forecast. The LSTM predicts quantiles of
the volatility-normalized return z = r / sigma_h directly:

    x~    = VSN(x),  h_s = LSTM(x~_s),  a = sum_s beta_s h_s
    e     = GRN(p)                        p: Chronos q10/q50/q90 / sigma_h, log sigma_h (scaled)
    g     = sigmoid(W_g [a, e])           per-unit gate on the Chronos embedding
    c     = GRN(a + g * e)
    q50   = w_m c + b_m
    q10   = q50 - softplus(w_lo c + b_lo),  q90 = q50 + softplus(w_hi c + b_hi)

The head starts at a zero-mean N(0, 1) forecast (the random walk with EWMA
intervals). During training the whole Chronos block is dropped per sample with
probability prior_dropout, so the model also learns to forecast without it and
the same network can be evaluated with Chronos masked.

Forward signature matches GatedResidualLSTM, so cra_model.train_cra and
predict_cra are reused unchanged (mz is ignored).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cra_model import GRN, VariableSelection, inv_softplus

Z90 = 1.2815516                                    # N(0,1) 90% quantile


class ChronosFeatureLSTM(nn.Module):
    def __init__(self, n_features: int, n_prior: int = 4, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.1, feature_dropout: float = 0.1, prior_dropout: float = 0.2):
        super().__init__()
        self.vsn = VariableSelection(n_features, hidden_size)
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.att = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, 1, bias=False))
        self.prior_enc = GRN(n_prior, hidden_size, dropout)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.fuse = GRN(hidden_size, hidden_size, dropout)
        self.head = nn.Linear(hidden_size, 3)       # median, lower spread, upper spread
        nn.init.zeros_(self.head.weight)
        spread = float(inv_softplus(torch.tensor(Z90)))
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0, spread, spread]))
        self.feature_dropout = feature_dropout
        self.prior_dropout = prior_dropout

    def forward(self, x, p, mz=None):
        if self.training and self.feature_dropout > 0:
            keep = torch.bernoulli(torch.full((x.shape[0], 1, x.shape[2]), 1 - self.feature_dropout,
                                              device=x.device))
            x = x * keep / (1 - self.feature_dropout)
        if self.training and self.prior_dropout > 0:
            p = p * torch.bernoulli(torch.full((p.shape[0], 1), 1 - self.prior_dropout, device=p.device))
        x, w = self.vsn(x)
        hs, _ = self.lstm(x)
        beta = torch.softmax(self.att(hs), dim=1)
        a = (beta * hs).sum(dim=1)
        e = self.prior_enc(p)
        g = torch.sigmoid(self.gate(torch.cat([a, e], dim=-1)))
        c = self.fuse(a + g * e)
        m, s_lo, s_hi = self.head(c).unbind(-1)
        q = torch.stack([m - F.softplus(s_lo), m, m + F.softplus(s_hi)], dim=-1)
        return q, g.mean(dim=-1), w.mean(dim=1)


def build_followup_model(fu: dict, n_features: int) -> ChronosFeatureLSTM:
    m = fu["model"]
    return ChronosFeatureLSTM(n_features, 4, m["hidden_size"], m["num_layers"], m["dropout"],
                              m["feature_dropout"], m["prior_dropout"])
