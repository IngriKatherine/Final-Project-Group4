"""Leakage and correctness checks for issue #10. Run from Code/: python -m pytest tests -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from oilfc.cra_data import (asof_on_calendar, build_refit, ewma_sigma, load_cra_data, load_prior,
                            loss_mask, make_features, refit_bounds)
from oilfc.cra_model import GatedResidualLSTM
from oilfc.cra_prior import chronos_inputs
from oilfc.utils import CODE_DIR, load_config

CFG = load_config(CODE_DIR / "cra_config.yaml")


@pytest.fixture(scope="module")
def data():
    lv, feats = load_cra_data(CFG)
    return lv, feats, load_prior(CFG)


def perturb_after(lv: pd.DataFrame, t0: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = lv.copy()
    cols = [c for c in lv.columns if c != "interp"]
    after = out.index > t0
    out.loc[after, cols] = out.loc[after, cols].values * rng.uniform(0.5, 1.5, (after.sum(), len(cols)))
    return out


def test_features_and_sigma_are_causal(data):
    lv = data[0]
    t0 = pd.Timestamp("2021-06-30")
    lv2 = perturb_after(lv, t0)
    a, b = make_features(lv).loc[:t0], make_features(lv2).loc[:t0]
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(ewma_sigma(lv, 0.94).loc[:t0], ewma_sigma(lv2, 0.94).loc[:t0])


def test_asof_respects_staleness_limit():
    cal = pd.bdate_range("2024-01-01", periods=10)
    s = pd.Series([1.0, 2.0], index=[cal[0], cal[2]])
    out = asof_on_calendar(s, cal, limit=3)
    assert out.iloc[1] == 1.0 and out.iloc[2] == 2.0
    assert out.iloc[5] == 2.0 and np.isnan(out.iloc[6])


def test_gpr_is_lagged(data):
    raw = pd.read_csv(CODE_DIR / CFG["paths"]["raw_dir"] / "gpr.csv", index_col=0, parse_dates=True)["GPR"]
    lv = data[0]
    d = pd.Timestamp("2024-03-13")                         # a Wednesday
    assert np.isclose(lv.loc[d, "GPR"], raw.loc[d - pd.Timedelta(days=1)])


@pytest.mark.parametrize("h", [1, 20])
@pytest.mark.parametrize("year", [2022, 2026])
def test_refit_split_dates(data, h, year):
    lv, feats, prior = data
    r = build_refit(CFG, lv, feats, prior, h, year)
    train_end, val_end, eval_end = refit_bounds(year, CFG["evaluation"]["val_years"])
    s = r.splits
    assert s["train"].target_dates.max() <= train_end
    assert s["val"].target_dates.min() > train_end and s["val"].target_dates.max() <= val_end
    assert s["eval"].target_dates.min() > val_end and s["eval"].target_dates.max() <= eval_end
    assert (s["eval"].target_dates.year == year).all()
    assert s["val"].origin_dates.max() < s["eval"].target_dates.min()


def test_scalers_use_training_rows_only(data):
    lv, feats, prior = data
    year = 2023
    train_end = refit_bounds(year, CFG["evaluation"]["val_years"])[0]
    lv2 = perturb_after(lv, train_end)
    r1 = build_refit(CFG, lv, feats, prior, 5, year)
    r2 = build_refit(CFG, lv2, make_features(lv2), prior, 5, year)
    np.testing.assert_allclose(r1.x_scaler.mean_, r2.x_scaler.mean_)
    np.testing.assert_allclose(r1.splits["train"].X, r2.splits["train"].X)


def test_interpolated_price_excluded_from_loss(data):
    lv, feats, prior = data
    h, after = 20, CFG["data"]["interp_exclusion_rows"]
    i = int(np.flatnonzero(lv["interp"].values == 1)[0])
    banned = set(lv.index[i - h: i + after + 1])          # origins whose target or inputs use it
    r = build_refit(CFG, lv, feats, prior, h, 2022)        # 2020 lies in its training years
    for name in ("train", "val"):
        assert not banned & set(r.splits[name].origin_dates)
    assert not loss_mask(lv, h, after)[i - h: i + after + 1].any()


def test_no_nonpositive_prices_in_evaluation(data):
    lv = data[0]
    assert (lv.loc["2022":, "WTI"] > 0).all()


def test_chronos_context_ends_at_origin(data):
    lv = data[0]
    i = lv.index.get_loc(pd.Timestamp("2023-05-15"))
    inp = chronos_inputs(lv, np.array([i]), 512, CFG["chronos"]["covariates"])[0]
    assert len(inp["target"]) == 512
    assert np.isclose(inp["target"][-1], np.log(lv["WTI"].iloc[i]))
    for c, v in inp["past_covariates"].items():
        assert len(v) == 512 and np.isclose(v[-1], np.log(lv[c].iloc[i]))


def test_quantiles_never_cross():
    torch.manual_seed(0)
    m = GatedResidualLSTM(15).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=2.0)               # extreme weights
    x, p = torch.randn(256, 30, 15), torch.randn(256, 4)
    mz = torch.sort(torch.randn(256, 3), dim=1).values
    q, g, _ = m(x, p, mz)
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
    assert ((g >= 0) & (g <= 1)).all()


def test_model_starts_at_prior():
    m = GatedResidualLSTM(15).eval()
    mz = torch.tensor([[-1.0, 0.1, 1.2]]).repeat(8, 1)
    q, _, _ = m(torch.randn(8, 30, 15), torch.randn(8, 4), mz)
    torch.testing.assert_close(q, mz, atol=1e-5, rtol=0)
