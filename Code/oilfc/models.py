"""Recurrent forecasters in PyTorch.

Both models read a window x of shape (batch, lookback, n_features) and output
one number per sample: the (scaled) h-step-ahead target.

    h_t = RNN(x_1 .. x_T)            # LSTM or GRU, num_layers stacked
    y_hat = W * dropout(h_T) + b     # linear head on the last hidden state

TODO (modeling owner):
  * try an attention / mean-pool head instead of the last hidden state
  * try predicting all horizons at once (multi-output head) vs. one model per horizon
"""
from __future__ import annotations

import torch
from torch import nn


class RNNForecaster(nn.Module):
    def __init__(self, n_features: int, cell: str = "lstm", hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)            # (batch, T, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(cfg: dict, n_features: int) -> RNNForecaster:
    m = cfg["model"]
    return RNNForecaster(
        n_features=n_features,
        cell=m["type"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        bidirectional=m.get("bidirectional", False),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
