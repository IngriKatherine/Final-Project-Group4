"""
21_cra_followup_evaluate.py  -  Chronos-feature LSTM vs the issue #10 baselines (follow-up)

The follow-up was designed after the 18_cra_evaluate.py results, including
2026, were seen. Both periods are reported, but 2026 is labelled as previously
seen and is not an untouched holdout for this model.

Compared on identical origins (seed-averaged):
  Chronos-feature LSTM, the same network with Chronos masked, the original LSTM,
  the gated residual LSTM (first design; used the 1-day GPR lag), Chronos-2, random walk.
  * Primary: DM-HLN, Chronos-feature LSTM vs original LSTM, Holm over horizons
  * Also: DM vs Chronos-2, vs the gated residual LSTM, and vs the masked network;
    Clark-West vs RW; Pesaran-Timmermann on non-overlapping origins

Outputs
  results/cra/followup/tables/metrics.csv, tests.csv, rmse_by_year.csv, gpr_alignment.csv, summary.md
  results/cra/followup/figures/rmse_vs_rw.png

Usage
  python 21_cra_followup_evaluate.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilfc.cra_data import load_cra_data, load_prior
from oilfc.cra_gpr import gpr_releases, load_followup_data
from oilfc.cra_metrics import holm, interval_coverage, pesaran_timmermann
from oilfc.metrics import all_metrics, clark_west, diebold_mariano, oos_r2
from oilfc.utils import CODE_DIR, import_script, load_config, path

ev18 = import_script("18_cra_evaluate.py")
LABELS = {**ev18.LABELS, "cfl": "Chronos-feature LSTM", "cfl0": "Chronos-feature LSTM, Chronos masked"}


def gpr_alignment(cfg, fu) -> pd.DataFrame:
    """How the original 1-day lag differs from the point-in-time series, by year."""
    lv_old, _ = load_cra_data(cfg)
    lv_new, _ = load_followup_data(cfg, fu)
    final = pd.read_csv(path(cfg, "raw_dir", "gpr.csv"), index_col=0, parse_dates=True)["GPR"]
    vint = pd.read_csv(path(cfg, "raw_dir", "gpr_vintages.csv"), parse_dates=["vintage", "date"])
    rel = gpr_releases(final, vint)
    rel = rel[rel.source == "vintage"]
    first_seen = vint.sort_values("vintage").groupby("date").GPR.first()   # first published value
    revision = (np.log1p(final.reindex(first_seen.index)) - np.log1p(first_seen)).dropna()
    rows = []
    for yr in range(2022, 2027):
        o, n = np.log1p(lv_old.loc[str(yr), "GPR"]), np.log1p(lv_new.loc[str(yr), "GPR"])
        rv = revision[revision.index.year == yr]
        rows.append({"year": yr, "corr_old_vs_pit": o.corr(n), "mean_abs_diff_log": float((o - n).abs().mean()),
                     "share_days_differ": float((o != n).mean()),
                     "mean_abs_revision_log": float(rv.abs().mean()), "share_revised": float((rv.abs() > 1e-6).mean())})
    print(f"GPR releases from vintages: {len(rel)}; last date covered == release date in "
          f"{(rel.date == rel.release).mean():.0%} of them; max gap between releases "
          f"{rel.release.diff().dt.days.max():.0f} days")
    return pd.DataFrame(rows)


def main():
    cfg = load_config(CODE_DIR / "cra_config.yaml")
    fu = load_config(CODE_DIR / "cra_followup.yaml")
    fcfg = {**cfg, "paths": {**cfg["paths"], **fu["paths"]}}
    ev = cfg["evaluation"]
    years = ev["test_years"] + ev["holdout_years"]
    periods = {f"test {ev['test_years'][0]}-{ev['test_years'][-1]}": ev["test_years"],
               "2026 (seen before; not a holdout)": ev["holdout_years"]}
    prior = load_prior(cfg)
    metric_rows, test_rows, year_rows = [], [], []

    for h in cfg["framing"]["horizons"]:
        seeds = {m: ev18.load_runs(cfg, m, h, years) for m in ("lstm", "cra")}
        seeds["cfl"] = ev18.load_runs(fcfg, "cfl", h, years)
        seeds["cfl0"] = [fr.assign(pred_price=fr.pred_price_noprior)[["origin_date", "target_date", "base_price",
                                                                       "true_price", "pred_price"]]
                         for fr in seeds["cfl"]]
        f = {m: ev18.seed_mean(fr) for m, fr in seeds.items()}
        for m in f:
            assert f[m].origin_date.equals(f["lstm"].origin_date), f"{m}: origins differ"
        base = f["cfl"][["origin_date", "target_date", "base_price", "true_price"]]
        q = prior.xs(h, level="horizon").reindex(base.origin_date)
        f["rw"] = base.assign(pred_price=base.base_price)
        f["chronos"] = base.assign(pred_price=base.base_price.values * np.exp(q.q50.values),
                                   q10_price=base.base_price.values * np.exp(q.q10.values),
                                   q90_price=base.base_price.values * np.exp(q.q90.values))

        for pname, pyears in periods.items():
            sel = base.target_date.dt.year.isin(pyears).values
            g = {m: d.loc[sel].reset_index(drop=True) for m, d in f.items()}
            y, b = g["rw"].true_price.values, g["rw"].base_price.values
            act = np.log(y / b)
            for m, d in g.items():
                pr = np.log(d.pred_price.values / b)
                row = {"period": pname, "horizon": h, "model": LABELS[m], "n": int(sel.sum()),
                       **all_metrics(y, d.pred_price, b), "R2_oos_vs_RW": oos_r2(y, d.pred_price, b),
                       "pred_std_ratio": pr.std() / act.std(),
                       "corr_pred_actual": np.corrcoef(pr, act)[0, 1] if pr.std() > 0 else np.nan}
                if m in seeds:
                    rm = [np.sqrt(np.mean((fr.loc[sel].true_price - fr.loc[sel].pred_price) ** 2)) for fr in seeds[m]]
                    row["seed_RMSE_std"] = float(np.std(rm))
                if "q10_price" in d:
                    row["coverage80"] = interval_coverage(y, d.q10_price, d.q90_price)
                if "gate" in d:
                    row["gate_mean"] = float(d.gate.mean())
                metric_rows.append(row)

            def add(model, vs, nested):
                r = {"period": pname, "horizon": h, "model": LABELS[model], "vs": LABELS[vs],
                     **diebold_mariano(y, g[model].pred_price, g[vs].pred_price, h=h)}
                if nested:
                    r.update(clark_west(y, g[vs].pred_price, g[model].pred_price, h=h))
                test_rows.append(r)

            for vs in ("lstm", "chronos", "cra", "cfl0"):
                add("cfl", vs, False)
            add("cfl", "rw", True)
            step = np.arange(0, len(y), h)
            for m in ("cfl", "lstm", "chronos"):
                pt = pesaran_timmermann(act[step], np.log(g[m].pred_price.values / b)[step])
                test_rows.append({"period": pname, "horizon": h, "model": LABELS[m], "vs": "direction (PT)",
                                  "PT": pt["PT"], "PT_p_value": pt["PT_p_value"], "hit_rate": pt["hit_rate"],
                                  "n_nonoverlap": pt["n"]})

        for m, d in f.items():
            for yr, dd in d.groupby(d.target_date.dt.year):
                year_rows.append({"horizon": h, "year": yr, "model": LABELS[m],
                                  "RMSE": np.sqrt(np.mean((dd.true_price - dd.pred_price) ** 2))})

    metrics, tests = pd.DataFrame(metric_rows), pd.DataFrame(test_rows)
    primary = (tests.model == LABELS["cfl"]) & (tests.vs == LABELS["lstm"])
    tests["p_holm_primary"] = np.nan
    for pname in periods:
        k = primary & (tests.period == pname)
        tests.loc[k, "p_holm_primary"] = holm(tests.loc[k, "p_value"].values)
    by_year = pd.DataFrame(year_rows).pivot_table(index=["horizon", "year"], columns="model", values="RMSE")
    gpr = gpr_alignment(cfg, fu)

    tab = path(fcfg, "results_dir", "tables")
    tab.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tab / "metrics.csv", index=False)
    tests.to_csv(tab / "tests.csv", index=False)
    by_year.to_csv(tab / "rmse_by_year.csv")
    gpr.to_csv(tab / "gpr_alignment.csv", index=False)
    cols = ["period", "horizon", "model", "RMSE", "MAE", "DirAcc", "R2_oos_vs_RW", "pred_std_ratio",
            "corr_pred_actual", "seed_RMSE_std", "coverage80", "gate_mean"]
    dm_cols = ["period", "horizon", "model", "vs", "DM", "p_value", "p_holm_primary", "CW", "CW_p_value"]
    pt_cols = ["period", "horizon", "model", "hit_rate", "PT", "PT_p_value", "n_nonoverlap"]
    with open(tab / "summary.md", "w") as fh:
        fh.write("Follow-up designed after the first results were seen; 2026 is not an untouched holdout.\n\n")
        fh.write(metrics[cols].round(4).to_markdown(index=False) + "\n\n")
        fh.write(tests[tests.DM.notna()][dm_cols].round(4).to_markdown(index=False) + "\n\n")
        fh.write(tests[tests.vs == "direction (PT)"][pt_cols].round(4).to_markdown(index=False) + "\n\n")
        fh.write(gpr.round(4).to_markdown(index=False) + "\n")
    plot(fcfg, metrics, periods)
    pd.set_option("display.width", 250)
    print(metrics[cols].round(4).to_string(index=False))
    print(tests[tests.DM.notna()][dm_cols].round(4).to_string(index=False))
    print(tests[tests.vs == "direction (PT)"][pt_cols].round(4).to_string(index=False))
    print(gpr.round(4).to_string(index=False))


def plot(cfg, metrics, periods):
    fig_dir = path(cfg, "results_dir", "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = metrics.copy()
    rw = m[m.model == LABELS["rw"]].set_index(["period", "horizon"]).RMSE
    m["ratio"] = m.RMSE.values / rw.reindex(pd.MultiIndex.from_frame(m[["period", "horizon"]])).values
    m = m[m.model != LABELS["rw"]]
    fig, axes = plt.subplots(1, len(periods), figsize=(5.5 * len(periods), 3.8), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), periods):
        m[m.period == p].pivot(index="horizon", columns="model", values="ratio").plot.bar(ax=ax, rot=0)
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_title(p); ax.set_ylabel("RMSE / random-walk RMSE"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(fig_dir / "rmse_vs_rw.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
