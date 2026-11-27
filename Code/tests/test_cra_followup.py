"""Checks for the issue #10 follow-up. Run from Code/: python -m pytest tests -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from oilfc.cra_data import build_refit, load_cra_data, load_prior
from oilfc.cra_followup_model import Z90, ChronosFeatureLSTM
from oilfc.cra_gpr import gpr_releases, load_followup_data, release_dates
from oilfc.utils import CODE_DIR, load_config

CFG = load_config(CODE_DIR / "cra_config.yaml")
FU = load_config(CODE_DIR / "cra_followup.yaml")


@pytest.fixture(scope="module")
def gpr():
    final = pd.read_csv(CODE_DIR / CFG["paths"]["raw_dir"] / "gpr.csv", index_col=0, parse_dates=True)["GPR"]
    vint = pd.read_csv(CODE_DIR / CFG["paths"]["raw_dir"] / "gpr_vintages.csv", parse_dates=["vintage", "date"])
    return final, vint


def test_release_schedule():
    d = pd.to_datetime(["2026-11-17", "2026-11-23", "2026-10-07", "2021-12-30"])
    got = release_dates(d)
    # Tue -> next Mon; Mon -> same day; Columbus Day 2026-10-12 -> Tue; Mon 2022-01-03
    assert list(got.strftime("%Y-%m-%d")) == ["2026-11-23", "2026-11-23", "2026-10-13", "2022-01-03"]


def test_schedule_matches_archived_vintages(gpr):
    """The schedule (used only before 2022-02-24) agrees with the actual first release for most days."""
    _, vint = gpr
    first = vint.groupby("date").vintage.min()
    first = first[first.index >= "2022-03-01"]       # the archive starts mid-week
    lag = (first.values - release_dates(first.index)).days
    assert (lag == 0).mean() > 0.85 and (lag > 0).mean() < 0.05


def test_vintages_end_at_release(gpr):
    _, vint = gpr
    last = vint.groupby("vintage").date.max()
    assert (last <= last.index).all()


def test_gpr_used_only_after_release(gpr):
    final, vint = gpr
    lv, _ = load_followup_data(CFG, FU)
    rel = gpr_releases(final, vint)
    usable = rel.release + pd.Timedelta(days=FU["gpr"]["usable_after_days"])
    for t in pd.to_datetime(["2015-06-10", "2022-06-14", "2024-03-13", "2026-06-22"]):
        known = rel[usable <= t]
        expected = known.loc[known.release.idxmax(), "GPR"]
        assert np.isclose(lv.loc[t, "GPR"], expected)
        assert known.date.max() < t


def test_followup_keeps_samples_and_baseline_inputs():
    lv0, f0 = load_cra_data(CFG)
    lv1, f1 = load_followup_data(CFG, FU)
    assert lv0.index.equals(lv1.index)
    prior = load_prior(CFG)
    r0 = build_refit(CFG, lv0, f0, prior, 5, 2024)
    r1 = build_refit(CFG, lv1, f1, prior, 5, 2024)
    j = r0.features.index("r_wti")
    for k in ("train", "val", "eval"):
        assert r0.splits[k].origin_dates.equals(r1.splits[k].origin_dates)
        np.testing.assert_allclose(r0.splits[k].X[:, :, j], r1.splits[k].X[:, :, j])
        np.testing.assert_allclose(r0.splits[k].prior, r1.splits[k].prior)


def test_followup_model_starts_at_random_walk():
    m = ChronosFeatureLSTM(15).eval()
    q, _, _ = m(torch.randn(8, 30, 15), torch.randn(8, 4))
    torch.testing.assert_close(q, torch.tensor([-Z90, 0.0, Z90]).repeat(8, 1), atol=1e-5, rtol=0)


def test_followup_quantiles_never_cross():
    torch.manual_seed(0)
    m = ChronosFeatureLSTM(15).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=2.0)
    q, g, _ = m(torch.randn(256, 30, 15), torch.randn(256, 4))
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
    assert ((g >= 0) & (g <= 1)).all()
