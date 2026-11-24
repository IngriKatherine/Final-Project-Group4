# Code sketches

These modules plug into `Code/oilfc/` without changing `data.py`, the splits or the scalers. Every block below was run on synthetic data: the GASF and GADF images match their arccos definition, the layout check catches a planted `reshape` bug, every optimizer × schedule pair trains, divergence is caught, and an EMA checkpoint loads back into a plain model. They have not been run on the real data yet.

## Image encoding (`oilfc/imaging.py`)

Name the module `imaging.py`, not `encodings.py`, which would shadow a Python standard-library package.

```python
"""Per-window encodings. Input: windows standardized with TRAIN stats, (B, T, F).
Output: images (B, F, T, T). Only looks inside the window, so there is no leakage."""
import numpy as np
import torch
from torch import nn


def to_unit_interval(x, clip=3.0):
    # Fixed bounds (+-3 train std). Per-window min-max would erase the volatility level.
    return (x / clip).clamp(-1 + 1e-6, 1 - 1e-6)


def _split(u):
    ut = u.transpose(1, 2)                      # (B, F, T)
    return ut, torch.sqrt(1 - ut ** 2)          # cos(phi), sin(phi)


def gasf(u):                                    # cos(phi_i + phi_j)
    ut, st = _split(u)
    return ut[..., :, None] * ut[..., None, :] - st[..., :, None] * st[..., None, :]


def gadf(u):                                    # sin(phi_i - phi_j)
    ut, st = _split(u)
    return st[..., :, None] * ut[..., None, :] - ut[..., :, None] * st[..., None, :]


def recurrence(u):                              # unthresholded: no epsilon to tune
    ut = u.transpose(1, 2)
    return -(ut[..., :, None] - ut[..., None, :]).abs()


ENCODERS = {"gasf": gasf, "gadf": gadf, "rp": recurrence}


class MTF(nn.Module):
    """Markov Transition Field; bins and transition matrix fitted on TRAIN rows only."""

    def __init__(self, edges, W):
        super().__init__()
        self.register_buffer("edges", edges)    # (F, Q-1)
        self.register_buffer("W", W)            # (F, Q, Q)

    @staticmethod
    def fit(train_rows, Q=8):                   # (N, F) standardized rows dated <= train_end
        F = train_rows.shape[1]
        edges = np.quantile(train_rows, np.linspace(0, 1, Q + 1)[1:-1], axis=0).T
        W = np.zeros((F, Q, Q))
        for f in range(F):
            q = np.searchsorted(edges[f], train_rows[:, f], side="left")
            np.add.at(W[f], (q[:-1], q[1:]), 1.0)
        W = (W + 1e-6) / (W + 1e-6).sum(-1, keepdims=True)
        return MTF(torch.tensor(np.ascontiguousarray(edges), dtype=torch.float32),
                   torch.tensor(W, dtype=torch.float32))

    def forward(self, x):
        xt = x.transpose(1, 2).contiguous()
        q = torch.stack([torch.bucketize(xt[:, f].contiguous(), self.edges[f].contiguous())
                         for f in range(xt.shape[1])], 1)                  # (B, F, T)
        Wb = self.W.unsqueeze(0).expand(q.shape[0], -1, -1, -1)
        rows = torch.gather(Wb, 2, q.unsqueeze(-1).expand(-1, -1, -1, Wb.shape[-1]))
        return torch.gather(rows, 3, q.unsqueeze(-2).expand(-1, -1, q.shape[-1], -1))
```

## CNN architectures (add to `oilfc/models.py`)

Both take `(B, T, F)` windows and return `(B,)`, like `RNNForecaster`, so training, SHAP and the DM code work unchanged. The 1D CNN has about 11–12k parameters and the GAF-CNN about 15–16k, against 38.7k for the GRU.

```python
from .imaging import ENCODERS, to_unit_interval


class CNN1DForecaster(nn.Module):
    """Causal dilated 1D CNN. Receptive field 1 + 4*(1+2+4) = 29 of the 30 days.
    No BatchNorm, so the batch-size study isn't confounded by BN statistics."""

    def __init__(self, n_features, channels=32, kernel=5, dilations=(1, 2, 4),
                 dropout=0.2, out_dim=1):
        super().__init__()
        layers, c_in = [], n_features
        for d in dilations:
            layers += [nn.ConstantPad1d(((kernel - 1) * d, 0), 0.0),   # left pad = causal
                       nn.Conv1d(c_in, channels, kernel, dilation=d),
                       nn.ReLU(), nn.Dropout(dropout)]
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels, out_dim)

    def forward(self, x):                          # (B, T, F) -> (B,)
        h = self.net(x.transpose(1, 2))            # (B, C, T)
        return self.head(h[:, :, -1]).squeeze(-1)  # last step = "today"


class GAFCNN(nn.Module):
    """Images built inside forward(); coordinate channels plus a 4x4 pooled grid keep
    track of where "today" is (bottom-right). Global average pooling would lose it."""

    def __init__(self, n_features, encodings=("gasf", "gadf"), clip=3.0, width=16,
                 dropout=0.2, out_dim=1):
        super().__init__()
        self.encodings, self.clip = tuple(encodings), clip
        c_in = n_features * len(self.encodings) + 2
        self.body = nn.Sequential(
            nn.Conv2d(c_in, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 30x30 -> 15x15
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(2 * width * 16, out_dim))

    def encode(self, x):
        u = to_unit_interval(x, self.clip)
        imgs = [ENCODERS[e](u) for e in self.encodings]              # each (B, F, T, T)
        r = torch.linspace(-1, 1, x.shape[1], device=x.device)
        coords = torch.stack(torch.meshgrid(r, r, indexing="ij")).expand(x.shape[0], -1, -1, -1)
        return torch.cat(imgs + [coords], dim=1)

    def forward(self, x):
        return self.head(self.body(self.encode(x))).squeeze(-1)


def build_model(cfg, n_features):
    m = cfg["model"]
    if m["type"] in ("lstm", "gru"):
        return RNNForecaster(n_features, m["type"], m["hidden_size"], m["num_layers"], m["dropout"])
    if m["type"] == "cnn1d":
        return CNN1DForecaster(n_features, dropout=m["dropout"])
    if m["type"] == "gafcnn":
        return GAFCNN(n_features, encodings=m.get("encodings", ("gasf", "gadf")), dropout=m["dropout"])
    raise ValueError(m["type"])
```

Also widen `choices=["lstm", "gru"]` in `04_train_rnn.py`. For direction classification, keep `out_dim=1` as a logit and swap `nn.MSELoss` for `nn.BCEWithLogitsLoss`.

## Augmentation (`oilfc/augment.py`)

```python
from .checks import mixup_variance_preserving


class Augmenter:
    """Applied to TRAINING batches only. Its own generator, so augmentation noise
    never changes the initialization or the batch order."""

    def __init__(self, kinds=(), jitter_sigma=0.05, scale_sigma=0.1, seed=0, device="cpu"):
        self.kinds, self.js, self.ss = set(kinds), jitter_sigma, scale_sigma
        self.strength = 1.0                    # set per epoch; fades to 0 over the last 10%
        self.g = torch.Generator(device=device).manual_seed(10_000 + seed)

    def __call__(self, x, y):                  # x (B, T, F), y (B,), both standardized
        if "scale" in self.kinds:              # one s for inputs AND target = a volatility regime
            s = torch.exp(self.strength * self.ss * torch.randn(x.shape[0], 1, 1, generator=self.g, device=x.device))
            x, y = x * s, y * s.view(-1)       # fine in standardized space: means are ~0
        if "jitter" in self.kinds:
            x = x + self.strength * self.js * torch.randn(x.shape, generator=self.g, device=x.device)
        if "mixup" in self.kinds:              # variance-preserving version
            x, y = mixup_variance_preserving(x, y, g=self.g)
        if "reverse" in self.kinds:            # NEGATIVE CONTROL ONLY
            x = x.flip(1)
        return x, y


def block_bootstrap_train(feats_train, lookback, horizon, block_len=20, rng=None, target_col=0):
    """feats_train: (N, F) UNSCALED daily rows dated <= train_end; column 0 = WTI log return.
    Whole rows are resampled, so cross-asset correlation inside a block survives.
    Targets are recomputed inside the synthetic series. Scale the output with the
    ORIGINAL train x_scaler / y_scaler."""
    rng = rng or np.random.default_rng(0)
    N = len(feats_train)
    starts = rng.integers(0, N - block_len + 1, size=int(np.ceil(N / block_len)))
    S = np.concatenate([feats_train[s:s + block_len] for s in starts])[:N]
    csum = np.concatenate([[0.0], np.cumsum(S[:, target_col])])
    idx = np.arange(lookback - 1, N - horizon)
    X = np.stack([S[i - lookback + 1:i + 1] for i in idx])
    y = csum[idx + 1 + horizon] - csum[idx + 1]      # = log P(i+h) - log P(i)
    return X.astype(np.float32), y
```

## Optimizer comparison (`oilfc/optim.py` and `11_optimizer_study.py`)

```python
def make_optimizer(name, params, lr, wd=1e-2):
    if name == "sgd":      return torch.optim.SGD(params, lr=lr)
    if name == "momentum": return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "nesterov": return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":  return torch.optim.RMSprop(params, lr=lr, alpha=0.99)
    if name == "adam":     return torch.optim.Adam(params, lr=lr)
    if name == "adamw":    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def make_scheduler(name, opt, total_steps, warmup_frac=0.05):
    if name == "constant":
        return None, None
    if name == "cosine":
        warm = max(1, int(warmup_frac * total_steps))
        lam = lambda s: (s + 1) / warm if s < warm else \
            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))
        return torch.optim.lr_scheduler.LambdaLR(opt, lam), "step"
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=opt.param_groups[0]["lr"], total_steps=total_steps), "step"
    if name == "plateau":                      # current default: patience = 20 // 3
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6), "epoch_val"
    raise ValueError(name)


def train_fixed_budget(model, train_loader, val_loader, opt_name, lr, sched_name, epochs,
                       device, grad_clip=1.0, augmenter=None, ema=None):
    """No early stopping. Returns the last-epoch model AND the best-val state."""
    model.to(device)
    opt = make_optimizer(opt_name, model.parameters(), lr)
    sched, when = make_scheduler(sched_name, opt, epochs * len(train_loader))
    loss_fn = nn.MSELoss()
    hist = {k: [] for k in ("train_loss", "val_loss", "lr", "grad_p50", "grad_p95", "clip_frac")}
    best_val, best_state, best_epoch = float("inf"), None, -1
    for ep in range(epochs):
        model.train()
        if augmenter is not None:              # fade augmentation over the last 10% of epochs
            augmenter.strength = min(1.0, (epochs - ep) / max(1, 0.1 * epochs))
        losses, norms = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if augmenter is not None:
                xb, yb = augmenter(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            start = hist["train_loss"][0] if hist["train_loss"] else float("inf")
            if not torch.isfinite(loss) or loss.item() > 10 * start:
                hist["diverged_at_epoch"] = ep              # a result, not a crash
                hist.update(best_val=best_val, best_epoch=best_epoch)
                return model, hist, best_state
            loss.backward()
            norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip)))  # pre-clip
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
            if when == "step":
                sched.step()
            losses.append(loss.item())
        val = evaluate_loss(ema if ema is not None else model, val_loader, loss_fn, device)
        if when == "epoch_val":
            sched.step(val)
        n = np.asarray(norms)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(val)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["grad_p50"].append(float(np.median(n)))
        hist["grad_p95"].append(float(np.percentile(n, 95)))
        hist["clip_frac"].append(float((n > grad_clip).mean()))
        if val < best_val:
            src = ema.module if ema is not None else model   # plain state_dict keys
            best_val, best_epoch, best_state = val, ep, copy.deepcopy(src.state_dict())
    hist.update(best_val=best_val, best_epoch=best_epoch)
    return model, hist, best_state


def run_metrics(hist, steps_per_epoch, pred_ret, true_ret):
    """pred_ret / true_ret: test h-day log returns, e.g. np.log(pred_price / base_price)."""
    tl = np.asarray(hist["train_loss"])
    out = {"best_val": hist["best_val"], "best_epoch": hist["best_epoch"],
           "diverged": "diverged_at_epoch" in hist,
           "pred_std_ratio": float(np.std(pred_ret) / np.std(true_ret)),   # ~0 = collapsed to RW
           "pred_corr": float(np.corrcoef(pred_ret, true_ret)[0, 1])}
    if len(tl):
        out["updates_to_within_5pct"] = int(np.argmax(tl <= 1.05 * tl.min()) + 1) * steps_per_epoch
        out["clip_frac"] = float(np.mean(hist["clip_frac"]))
        be = hist["best_epoch"]
        out["gen_gap"] = hist["val_loss"][be] - tl[be] if be >= 0 else float("nan")
    return out


def seed_disagreement(preds_by_seed):
    """Mean pairwise RMSE between seed forecasts: stability that 3 seeds can estimate."""
    P = np.stack(preds_by_seed)
    return float(np.mean([np.sqrt(np.mean((P[i] - P[j]) ** 2))
                          for i in range(len(P)) for j in range(i + 1, len(P))]))


OPTS = ["sgd", "momentum", "nesterov", "rmsprop", "adam", "adamw"]
LRS = {"sgd": [1e-2, 3e-2, 1e-1, 3e-1, 1.0],
       "momentum": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "nesterov": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
       "rmsprop": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
       "adamw": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}

# ---- 11_optimizer_study.py, stage E3a (E3b and E3e rerun it with each optimizer's best LR) ----
cfg, device, EPOCHS = load_config(), get_device(), 60
prep = build_datasets(cfg, "shap_selected", horizon=5)
tr, va = prep.splits["train"], prep.splits["val"]
train_ds = TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.y))
val_loader = DataLoader(TensorDataset(torch.from_numpy(va.X), torch.from_numpy(va.y)), batch_size=256)

for model_type, opt_name, seed in itertools.product(["gru", "cnn1d"], OPTS, [0, 1, 2]):
    for lr in LRS[opt_name]:
        set_seed(seed)                                  # identical init for every optimizer
        model = build_model(override(cfg, **{"model.type": model_type}), len(prep.feature_names))
        loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))   # identical batch order
        model, hist, best_state = train_fixed_budget(model, loader, val_loader, opt_name, lr,
                                                     "constant", EPOCHS, device)
        # Evaluate the last-epoch model and best_state on test in price space (as in
        # 04_train_rnn.py), then save hist + run_metrics(...) under
        # run_name(model_type, "shap_selected", 5, seed, tag=f"opt-{opt_name}-lr{lr:g}-constant").
```

## Checks and diagnostics (`oilfc/checks.py`)

On synthetic data, the split-half check shows no held-out gain on pure noise (0.5023 tuned against 0.5025 default), while a real difference wins in all 200 splits.

```python
import hashlib
import json

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .imaging import gasf, to_unit_interval


# ---- Pipeline checks (Day 1, E0) -------------------------------------------------
def check_encoder_layout(model, n_features=5, T=30, atol=1e-6):
    """Perturbing series f must change only series f's image channels (catches a
    reshape-instead-of-permute bug), and the GASF diagonal must equal 2u^2 - 1."""
    x = torch.randn(2, T, n_features)
    u = to_unit_interval(x)
    diag = torch.diagonal(gasf(u), dim1=-2, dim2=-1)
    assert torch.allclose(diag, 2 * u.transpose(1, 2) ** 2 - 1, atol=1e-5), "GASF diagonal wrong"
    imgs = model.encode(x)
    x2 = x.clone(); x2[:, :, 0] += 0.5                     # perturb series 0 only
    changed = (model.encode(x2) - imgs).abs().amax(dim=(0, 2, 3)) > atol
    expected = torch.zeros_like(changed)
    expected[[i * n_features for i in range(len(model.encodings))]] = True   # channel = enc*F + f
    assert torch.equal(changed, expected), f"layout bug: {changed.nonzero().ravel().tolist()}"
    return True


def predict_direction(logits):
    """Threshold PROBABILITIES at 0.5, never raw logits."""
    return (torch.sigmoid(logits) > 0.5).long()


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---- Weight averaging (E3e) ---------------------------------------------------------
def make_ema(model, decay=0.995):
    """After each opt.step(): ema.update_parameters(model). Evaluate `ema`, not `model`."""
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))


# ---- Soft direction labels (E2s) --------------------------------------------------
def soft_direction_labels(y_raw, sigma_train, kappa=0.5):
    """Phi(r / (kappa * sigma)); sigma from TRAINING h-day returns. Evaluate on hard labels."""
    from scipy.stats import norm
    return norm.cdf(y_raw / (kappa * sigma_train)).astype(np.float32)


# ---- Error diversity (E6) ---------------------------------------------------------
def error_correlation(preds, truth):
    """preds: {model name: seed-averaged forecast}. Returns (names, correlation matrix)."""
    names = list(preds)
    return names, np.corrcoef(np.stack([preds[n] - truth for n in names]))


def shared_large_errors(preds, truth, q=0.95):
    """Days on which EVERY model is in its own worst 5%, relative to each model's worst-5% count."""
    big = np.stack([np.abs(p - truth) >= np.quantile(np.abs(p - truth), q) for p in preds.values()])
    return float(big.all(0).sum() / big.sum(1).mean())


def equal_weight_combination(preds):
    return np.mean(np.stack(list(preds.values())), 0)


# ---- Does tuning beat noise? -------------------------------------------------------
def split_half_check(val_loss_by_option, default, n_splits=200, block=20, seed=0):
    """val_loss_by_option: {option: per-day validation losses}, same days for every option.
    Tune on a random half of the 20-day blocks, score on the other half, 200 times."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(val_loss_by_option.values())))
    blocks = np.arange(n) // block
    ub = np.unique(blocks)
    tuned, dflt, same = [], [], 0
    for _ in range(n_splits):
        a = np.isin(blocks, rng.choice(ub, len(ub) // 2, replace=False))
        pick = min(val_loss_by_option, key=lambda k: val_loss_by_option[k][a].mean())
        tuned.append(val_loss_by_option[pick][~a].mean())
        dflt.append(val_loss_by_option[default][~a].mean())
        same += pick == default
    tuned, dflt = np.array(tuned), np.array(dflt)
    return {"heldout_tuned": tuned.mean(), "heldout_default": dflt.mean(),
            "share_tuned_better": (tuned < dflt).mean(), "share_pick_is_default": same / n_splits}


def paired_block_bootstrap_win_rate(err_a, err_b, block=20, n_boot=2000, seed=0):
    """Share of block-bootstrap test resamples in which A has the lower MSE (next to DM)."""
    rng = np.random.default_rng(seed)
    d = err_a ** 2 - err_b ** 2
    n, k = len(d), int(np.ceil(len(d) / block))
    starts = np.arange(0, n - block + 1)
    wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:n]
        wins += d[idx].mean() < 0
    return wins / n_boot


# ---- Augmentation preview and variance-preserving mixup (E4) -----------------------
def stylized_facts(r):
    from scipy.stats import kurtosis
    a = np.abs(r - r.mean())
    acf = lambda z, k: float(np.corrcoef(z[:-k], z[k:])[0, 1])
    return {"std": r.std(), "excess_kurtosis": kurtosis(r), "acf_r_1": acf(r, 1),
            "acf_absr_1": acf(a, 1), "acf_absr_5": acf(a, 5)}


def mixup_variance_preserving(x, y, alpha=0.2, g=None):
    """Plain mixup rescaled by 1/sqrt(lam^2 + (1-lam)^2) so mixed windows keep their volatility."""
    lam = torch.distributions.Beta(alpha, alpha).sample((x.shape[0],)).to(x.device)
    perm = torch.randperm(x.shape[0], generator=g).to(x.device)
    c = 1 / torch.sqrt(lam ** 2 + (1 - lam) ** 2)
    xm = (lam.view(-1, 1, 1) * x + (1 - lam).view(-1, 1, 1) * x[perm]) * c.view(-1, 1, 1)
    return xm, (lam * y + (1 - lam) * y[perm]) * c
```

## Hooks in the existing code

```python
# 04_train_rnn.py, run_single: never reuse a stale or unfinished run
h = config_hash({**cfg, "features": fs_name, "horizon": horizon, "seed": seed, "tag": tag})
if summary_file.exists() and not force:
    old = load_json(summary_file)
    if old.get("config_hash") == h and old.get("status") == "complete":
        return old
    print(f"[rerun] {stem}: cached run is stale or incomplete")
...
summary.update(config_hash=h, status="complete")   # last line before save_json

# train.py, inside the batch loop: augmentation on training batches only
if augmenter is not None:
    xb, yb = augmenter(xb, yb)

# classifier: start the output bias at the training up-share
p_up = float((y_train_raw > 0).mean())
model.head[-1].bias.data.fill_(np.log(p_up / (1 - p_up)))   # model.head.bias for CNN1D
```

## Pesaran–Timmermann test

P̂ is the share of correct sign predictions, P̂y the share of actual up moves and P̂x the share of predicted up moves. It takes about 15 lines in `metrics.py`.

```latex
\hat P_* = \hat P_y \hat P_x + (1-\hat P_y)(1-\hat P_x), \qquad \widehat{V}(\hat P) = \hat P_*(1-\hat P_*)/n
\widehat{V}(\hat P_*) = \tfrac{(2\hat P_y-1)^2 \hat P_x(1-\hat P_x)}{n} + \tfrac{(2\hat P_x-1)^2 \hat P_y(1-\hat P_y)}{n} + \tfrac{4\hat P_y \hat P_x(1-\hat P_y)(1-\hat P_x)}{n^2}
\mathrm{PT} = (\hat P - \hat P_*)\,/\,\sqrt{\widehat{V}(\hat P) - \widehat{V}(\hat P_*)} \;\sim\; N(0,1) \text{ under no skill (one-sided)}
```
